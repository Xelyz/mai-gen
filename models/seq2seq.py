import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from simai_tokenizer import build_vocab

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 20000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class AudioEncoder(nn.Module):
    def __init__(self, n_mels=128, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        # Conv stem (similar to Whisper)
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
    def forward(self, x):
        """
        x: (batch, n_mels, time)
        returns: (time_downsampled, batch, d_model)
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x)) # (B, C, T')
        x = x.permute(2, 0, 1) # (T', B, C)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        return output

class ChartDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        """
        tgt: (tgt_seq_len, batch)
        memory: (memory_seq_len, batch, d_model)
        """
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(tgt.device)
            
        output = self.transformer_decoder(
            tgt_emb, memory, 
            tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.fc_out(output)

class ChartGenerator(pl.LightningModule):
    def __init__(self, 
                 n_mels=128, d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=6, 
                 dim_feedforward=1024, dropout=0.1, lr=1e-4, warmup_steps=4000):
        super().__init__()
        self.save_hyperparameters()
        
        vocab = build_vocab()
        self.vocab_size = len(vocab)
        self.pad_idx = vocab['<pad>']
        
        self.lr = lr
        self.warmup_steps = warmup_steps
        
        self.encoder = AudioEncoder(n_mels, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.decoder = ChartDecoder(self.vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, dropout)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        
    def forward(self, mel, tgt):
        # mel: (B, 128, T)
        # tgt: (B, L)
        memory = self.encoder(mel) # (T', B, d_model)
        tgt_transposed = tgt.transpose(0, 1) # (L, B)
        
        # Shift target for input
        tgt_input = tgt_transposed[:-1, :]
        
        tgt_key_padding_mask = (tgt_input == self.pad_idx).transpose(0, 1) # (B, L-1)
        
        logits = self.decoder(tgt_input, memory, tgt_key_padding_mask=tgt_key_padding_mask) # (L-1, B, vocab_size)
        return logits
        
    def training_step(self, batch, batch_idx):
        mel = batch['audio']
        tgt = batch['tokens']
        
        logits = self(mel, tgt) # (L-1, B, V)
        
        # Expected tgt: tgt_transposed[1:, :]
        tgt_expected = tgt.transpose(0, 1)[1:, :]
        
        loss = self.criterion(logits.reshape(-1, self.vocab_size), tgt_expected.reshape(-1))
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mel = batch['audio']
        tgt = batch['tokens']
        
        logits = self(mel, tgt)
        tgt_expected = tgt.transpose(0, 1)[1:, :]
        loss = self.criterion(logits.reshape(-1, self.vocab_size), tgt_expected.reshape(-1))
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Simple warmup scheduler
        def lr_foo(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return 1.0 # Constant after warmup, could also use cosine decay
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_foo)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
        
    @torch.no_grad()
    def generate(self, mel, bos_idx, eos_idx, max_len=10000):
        self.eval()
        device = mel.device
        batch_size = mel.shape[0]
        
        memory = self.encoder(mel) # (T', B, d_model)
        
        ys = torch.ones(1, batch_size).fill_(bos_idx).type(torch.long).to(device)
        
        for i in range(max_len - 1):
            tgt_mask = self.decoder.generate_square_subsequent_mask(ys.size(0)).to(device)
            out = self.decoder(ys, memory, tgt_mask=tgt_mask) # (L, B, V)
            prob = F.softmax(out[-1, :, :], dim=-1) # (B, V)
            _, next_word = torch.max(prob, dim=1) # (B,)
            
            ys = torch.cat([ys, next_word.unsqueeze(0)], dim=0) # (L+1, B)
            
            # Simple early stopping if all generated eos
            if (ys == eos_idx).any(dim=0).all():
                break
                
        return ys.transpose(0, 1) # (B, L)
