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
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class AudioEncoder(nn.Module):
    def __init__(self, n_mels=128, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        # Conv stem (similar to Whisper)
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
    def forward(self, x):
        """
        x: (batch, n_mels, time)
        returns: (batch, time_downsampled, d_model)
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x)) # (B, C, T')
        x = x.transpose(1, 2) # (B, T', C)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        return output

class ChartDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, tgt_is_causal=False):
        """
        tgt: (batch, tgt_seq_len)
        memory: (batch, memory_seq_len, d_model)
        """
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
        kwargs = {'tgt_mask': tgt_mask}
        if tgt_is_causal:
            kwargs['tgt_is_causal'] = True

        output = self.transformer_decoder(
            tgt_emb, memory, 
            tgt_key_padding_mask=tgt_key_padding_mask,
            **kwargs
        )
        return self.fc_out(output)

class ChartGenerator(pl.LightningModule):
    def __init__(self, 
                 n_mels=128, d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=6, 
                 dim_feedforward=1024, dropout=0.1, lr=1e-4, warmup_steps=4000,
                 max_token_len=4096):
        super().__init__()
        self.save_hyperparameters()
        
        vocab = build_vocab()
        self.vocab_size = len(vocab)
        self.pad_idx = vocab['<pad>']
        self.max_token_len = max_token_len
        
        self.lr = lr
        self.warmup_steps = warmup_steps
        
        self.encoder = AudioEncoder(n_mels, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.decoder = ChartDecoder(self.vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, dropout)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        
    def forward(self, mel, tgt):
        # mel: (B, 128, T)
        # tgt: (B, L)
        memory = self.encoder(mel) # (B, T', d_model)
        
        # Shift target for input
        tgt_input = tgt[:, :-1]
        
        tgt_key_padding_mask = (tgt_input == self.pad_idx) # (B, L-1)
        
        logits = self.decoder(tgt_input, memory, tgt_key_padding_mask=tgt_key_padding_mask, tgt_is_causal=True) # (B, L-1, vocab_size)
        return logits
        
    def training_step(self, batch, batch_idx):
        mel = batch['audio']
        tgt = batch['tokens'][:, :self.max_token_len]
        
        logits = self(mel, tgt) # (B, L-1, V)
        
        # Expected tgt
        tgt_expected = tgt[:, 1:]
        
        loss = self.criterion(logits.reshape(-1, self.vocab_size), tgt_expected.reshape(-1))
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mel = batch['audio']
        tgt = batch['tokens'][:, :self.max_token_len]
        
        logits = self(mel, tgt)
        tgt_expected = tgt[:, 1:]
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
    def generate(self, mel, bos_idx, eos_idx, max_len=None):
        if max_len is None:
            max_len = self.max_token_len
        self.eval()
        device = mel.device
        batch_size = mel.shape[0]
        
        memory = self.encoder(mel) # (B, T', d_model)
        
        ys = torch.ones(batch_size, 1).fill_(bos_idx).type(torch.long).to(device)
        
        for i in range(max_len - 1):
            tgt_mask = self.decoder.generate_square_subsequent_mask(ys.size(1)).to(device)
            out = self.decoder(ys, memory, tgt_mask=tgt_mask) # (B, L, V)
            prob = F.softmax(out[:, -1, :], dim=-1) # (B, V)
            _, next_word = torch.max(prob, dim=1) # (B,)
            
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1) # (B, L+1)
            
            # Simple early stopping if all generated eos
            if (ys == eos_idx).any(dim=1).all():
                break
                
        return ys # (B, L)
