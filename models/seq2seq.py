import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import logging
from simai_tokenizer import build_vocab

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Mamba availability check (auto-fallback)
# ──────────────────────────────────────────────
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

# ──────────────────────────────────────────────
# RoPE  (Rotary Positional Embedding)
# ──────────────────────────────────────────────

class RotaryPositionalEmbedding(nn.Module):
    """Pre-computes cos/sin tables for Rotary Position Embedding."""

    def __init__(self, head_dim: int, max_len: int = 20000, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)  # (head_dim/2,)
        # Pre-build cache for common lengths
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)         # (seq_len, head_dim)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]  # (L, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, nhead, L, head_dim), cos/sin: (L, head_dim) → broadcast"""
    return x * cos.unsqueeze(0).unsqueeze(0) + _rotate_half(x) * sin.unsqueeze(0).unsqueeze(0)

# ──────────────────────────────────────────────
# Flash Attention + RoPE  Multi-Head Attention
# ──────────────────────────────────────────────

class RoPEAttention(nn.Module):
    """
    Multi-head attention with:
    - RoPE on Q/K  (optional, controlled by `use_rope`)
    - Flash Attention via F.scaled_dot_product_attention
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, use_rope: bool = True):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, q_in, k_in, v_in,
                rope_cos=None, rope_sin=None,
                is_causal: bool = False,
                key_padding_mask=None):
        """
        q_in: (B, Lq, D)   k_in/v_in: (B, Lk, D)
        rope_cos/sin: (L, head_dim) – only used when use_rope=True
        key_padding_mask: (B, Lk) bool – True = padded (ignored)
        """
        B, Lq, _ = q_in.shape
        Lk = k_in.size(1)

        q = self.q_proj(q_in).view(B, Lq, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_in).view(B, Lk, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(v_in).view(B, Lk, self.nhead, self.head_dim).transpose(1, 2)
        # shapes: (B, nhead, L, head_dim)

        # Apply RoPE to Q and K
        if self.use_rope and rope_cos is not None:
            q = _apply_rope(q, rope_cos[:Lq], rope_sin[:Lq])
            k = _apply_rope(k, rope_cos[:Lk], rope_sin[:Lk])

        # Convert key_padding_mask to attn_mask for SDPA
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk), True=pad → attn_mask -inf for pad positions
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
            attn_mask = attn_mask.expand(-1, self.nhead, Lq, -1)   # (B, nhead, Lq, Lk)
            attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(attn_mask, float('-inf'))

        # SDPA — auto-dispatches to Flash Attention 2 when conditions are met
        # (float16/bfloat16, CUDA, no custom mask when is_causal=True)
        dropout_p = self.dropout_p if self.training else 0.0

        # When using is_causal, attn_mask must be None for Flash path
        if is_causal and attn_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False)

        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.out_proj(out)

# ──────────────────────────────────────────────
# Encoder Layer  (RoPE + Flash Attention)
# ──────────────────────────────────────────────

class RoPEEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = RoPEAttention(d_model, nhead, dropout, use_rope=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, rope_cos, rope_sin):
        # Pre-norm style
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, rope_cos=rope_cos, rope_sin=rope_sin)
        x = x + self.ffn(self.norm2(x))
        return x

# ──────────────────────────────────────────────
# Decoder Layer  (RoPE self-attn + Flash cross-attn)
# ──────────────────────────────────────────────

class RoPEDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        # Causal self-attention with RoPE
        self.self_attn = RoPEAttention(d_model, nhead, dropout, use_rope=True)
        # Cross-attention without RoPE (different position spaces)
        self.cross_attn = RoPEAttention(d_model, nhead, dropout, use_rope=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tgt, memory, rope_cos, rope_sin,
                tgt_key_padding_mask=None, is_causal=True):
        # 1. Causal self-attention with RoPE
        h = self.norm1(tgt)
        # For causal + padding: we need to handle both
        if tgt_key_padding_mask is not None and is_causal:
            # Build combined causal + padding mask manually
            L = tgt.size(1)
            causal = torch.triu(torch.full((L, L), float('-inf'), device=tgt.device, dtype=tgt.dtype), diagonal=1)
            # causal: (L, L) → expand to (B, nhead, L, L)
            pad_mask = tgt_key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            pad_mask = torch.zeros_like(pad_mask, dtype=tgt.dtype).masked_fill(pad_mask, float('-inf'))
            # Combine: broadcast add
            combined = causal.unsqueeze(0).unsqueeze(0) + pad_mask  # (B, 1, L, L)
            # Pass as attn_mask directly through the attention module
            # We bypass the standard path and call SDPA ourselves
            tgt = tgt + self._self_attn_with_mask(h, rope_cos, rope_sin, combined)
        else:
            tgt = tgt + self.self_attn(h, h, h, rope_cos=rope_cos, rope_sin=rope_sin, is_causal=is_causal)

        # 2. Cross-attention (no RoPE, no causal)
        h = self.norm2(tgt)
        tgt = tgt + self.cross_attn(h, memory, memory)

        # 3. FFN
        tgt = tgt + self.ffn(self.norm3(tgt))
        return tgt

    def _self_attn_with_mask(self, h, rope_cos, rope_sin, attn_mask):
        """Self-attention with pre-built combined causal+padding mask."""
        B, L, D = h.shape
        sa = self.self_attn
        q = sa.q_proj(h).view(B, L, sa.nhead, sa.head_dim).transpose(1, 2)
        k = sa.k_proj(h).view(B, L, sa.nhead, sa.head_dim).transpose(1, 2)
        v = sa.v_proj(h).view(B, L, sa.nhead, sa.head_dim).transpose(1, 2)

        if rope_cos is not None:
            q = _apply_rope(q, rope_cos[:L], rope_sin[:L])
            k = _apply_rope(k, rope_cos[:L], rope_sin[:L])

        dropout_p = sa.dropout_p if self.training else 0.0
        # attn_mask already (B, 1, L, L), will broadcast over nhead
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return sa.out_proj(out)

# ──────────────────────────────────────────────
# Mamba Block  (wrapper with norm + residual)
# ──────────────────────────────────────────────

class MambaBlock(nn.Module):
    """Wraps mamba_ssm.Mamba with pre-norm and residual connection."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        """x: (B, L, D)"""
        return x + self.mamba(self.norm(x))

# ──────────────────────────────────────────────
# Audio Encoder  (Hybrid: Mamba + RoPE Transformer)
# ──────────────────────────────────────────────

class AudioEncoder(nn.Module):
    def __init__(self, n_mels=128, d_model=256, nhead=8,
                 num_mamba_layers=2, num_attn_layers=2,
                 dim_feedforward=1024, dropout=0.1,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        # Conv stem (similar to Whisper)
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2)

        # Mamba layers (O(n) sequence modeling)
        self.use_mamba = MAMBA_AVAILABLE and num_mamba_layers > 0
        if self.use_mamba:
            self.mamba_layers = nn.ModuleList([
                MambaBlock(d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
                for _ in range(num_mamba_layers)
            ])
            logger.info(f"AudioEncoder: using {num_mamba_layers} Mamba layers "
                        f"(d_state={mamba_d_state}, d_conv={mamba_d_conv}, expand={mamba_expand})")
        else:
            if num_mamba_layers > 0 and not MAMBA_AVAILABLE:
                logger.warning("mamba-ssm not installed — falling back to extra Transformer encoder layers. "
                               "Install with: pip install mamba-ssm causal-conv1d")
            # Fallback: use additional Transformer layers instead
            self.mamba_layers = nn.ModuleList([
                RoPEEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_mamba_layers)
            ])

        # RoPE Transformer layers (global attention)
        self.attn_layers = nn.ModuleList([
            RoPEEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_attn_layers)
        ])

        # Shared RoPE for all Transformer-based layers in encoder
        head_dim = d_model // nhead
        self.rope = RotaryPositionalEmbedding(head_dim)

    def forward(self, x):
        """
        x: (batch, n_mels, time)
        returns: (batch, time_downsampled, d_model)
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))  # (B, d_model, T')
        x = x.transpose(1, 2)      # (B, T', d_model)

        rope_cos, rope_sin = self.rope(x.size(1))

        # Mamba layers (or fallback Transformer layers)
        for layer in self.mamba_layers:
            if self.use_mamba:
                if self.training:
                    x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
                else:
                    x = layer(x)
            else:
                # Fallback Transformer layers need rope
                if self.training:
                    x = torch.utils.checkpoint.checkpoint(layer, x, rope_cos, rope_sin, use_reentrant=False)
                else:
                    x = layer(x, rope_cos, rope_sin)

        # Attention layers
        for layer in self.attn_layers:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, rope_cos, rope_sin, use_reentrant=False)
            else:
                x = layer(x, rope_cos, rope_sin)

        return x

# ──────────────────────────────────────────────
# Chart Decoder  (RoPE + Flash Attention)
# ──────────────────────────────────────────────

class ChartDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8,
                 num_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_scale = math.sqrt(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            RoPEDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

        # RoPE for decoder
        head_dim = d_model // nhead
        self.rope = RotaryPositionalEmbedding(head_dim)

    def forward(self, tgt, memory, tgt_key_padding_mask=None, is_causal=True):
        """
        tgt: (batch, tgt_seq_len)  – token ids
        memory: (batch, memory_seq_len, d_model)
        """
        tgt_emb = self.embedding(tgt) * self.embed_scale
        tgt_emb = self.embed_dropout(tgt_emb)

        rope_cos, rope_sin = self.rope(tgt.size(1))

        output = tgt_emb
        for layer in self.layers:
            if self.training:
                def _ckpt_fn(o, m, rc, rs):
                    return layer(o, m, rc, rs,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 is_causal=is_causal)
                output = torch.utils.checkpoint.checkpoint(
                    _ckpt_fn, output, memory, rope_cos, rope_sin,
                    use_reentrant=False
                )
            else:
                output = layer(output, memory, rope_cos, rope_sin,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               is_causal=is_causal)

        output = self.norm(output)
        return self.fc_out(output)

# ──────────────────────────────────────────────
# Chart Generator  (Lightning Module)
# ──────────────────────────────────────────────

class ChartGenerator(pl.LightningModule):
    def __init__(self,
                 n_mels=128, d_model=256, nhead=8,
                 num_mamba_layers=2, num_encoder_attn_layers=2,
                 num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 lr=1e-4, warmup_steps=4000,
                 max_token_len=4096,
                 # Legacy compat: silently accept old param name
                 num_encoder_layers=None):
        super().__init__()

        # Handle legacy config: if old name is provided, use it as attn layers
        if num_encoder_layers is not None and num_encoder_attn_layers == 2:
            logger.warning(f"Legacy param 'num_encoder_layers={num_encoder_layers}' detected. "
                           f"Interpreting as num_encoder_attn_layers={num_encoder_layers}, num_mamba_layers=0.")
            num_encoder_attn_layers = num_encoder_layers
            num_mamba_layers = 0

        self.save_hyperparameters()

        vocab = build_vocab()
        self.vocab_size = len(vocab)
        self.pad_idx = vocab['<pad>']
        self.max_token_len = max_token_len

        self.lr = lr
        self.warmup_steps = warmup_steps

        self.encoder = AudioEncoder(
            n_mels, d_model, nhead,
            num_mamba_layers=num_mamba_layers,
            num_attn_layers=num_encoder_attn_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            mamba_d_state=mamba_d_state, mamba_d_conv=mamba_d_conv, mamba_expand=mamba_expand,
        )
        self.decoder = ChartDecoder(
            self.vocab_size, d_model, nhead,
            num_decoder_layers, dim_feedforward, dropout,
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        # Debug: log model info on init
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"ChartGenerator initialized: "
                    f"total_params={total_params:,}, trainable={trainable_params:,}")
        logger.info(f"  Encoder: {num_mamba_layers} Mamba + {num_encoder_attn_layers} Attention layers "
                    f"(Mamba available: {MAMBA_AVAILABLE})")
        logger.info(f"  Decoder: {num_decoder_layers} RoPE+Flash Attention layers")
        logger.info(f"  d_model={d_model}, nhead={nhead}, ff={dim_feedforward}, dropout={dropout}")

        self._logged_shapes = False

    def forward(self, mel, tgt):
        # mel: (B, 128, T)
        # tgt: (B, L)
        memory = self.encoder(mel)  # (B, T', d_model)

        # Shift target for teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_key_padding_mask = (tgt_input == self.pad_idx)  # (B, L-1)

        logits = self.decoder(tgt_input, memory,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              is_causal=True)  # (B, L-1, vocab_size)

        # Debug: log shapes once
        if not self._logged_shapes:
            logger.info(f"[shapes] mel={mel.shape}, memory={memory.shape}, "
                        f"tgt_input={tgt_input.shape}, logits={logits.shape}")
            if torch.cuda.is_available():
                logger.info(f"[memory] peak GPU memory: "
                            f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
            self._logged_shapes = True

        return logits

    def training_step(self, batch, batch_idx):
        mel = batch['audio']
        tgt = batch['tokens'][:, :self.max_token_len]

        logits = self(mel, tgt)  # (B, L-1, V)

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
            return 1.0  # Constant after warmup, could also use cosine decay

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

        memory = self.encoder(mel)  # (B, T', d_model)

        ys = torch.ones(batch_size, 1).fill_(bos_idx).type(torch.long).to(device)

        for i in range(max_len - 1):
            logits = self.decoder(ys, memory, is_causal=True)  # (B, L, V)
            prob = F.softmax(logits[:, -1, :], dim=-1)  # (B, V)
            _, next_word = torch.max(prob, dim=1)  # (B,)

            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)  # (B, L+1)

            # Simple early stopping if all generated eos
            if (ys == eos_idx).any(dim=1).all():
                break

        return ys  # (B, L)
