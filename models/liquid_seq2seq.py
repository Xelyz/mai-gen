"""
Liquid Chart Generator — CfC-based direct audio-to-chart generation.

Architecture:
  Audio Mel (B,128,T) → Conv stem → CfC Encoder → Memory
  Memory → CfC Decoder (with cross-attention) → 72-channel chart matrix (B,C_out,T')

Uses the ncps library (Neural Circuit Policies) for CfC networks.
Output format is compatible with Mug-Diffusion's chart feature matrix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import logging

from ncps.torch import CfC
from ncps.wirings import AutoNCP

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# CfC Encoder Block (norm + CfC + residual)
# ──────────────────────────────────────────────

class CfCBlock(nn.Module):
    """Single CfC layer with LayerNorm and residual connection."""

    def __init__(self, d_model, cfc_units):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        wiring = AutoNCP(units=cfc_units, output_size=d_model)
        self.cfc = CfC(d_model, wiring, batch_first=True)

    def forward(self, x, hx=None):
        """
        x: (B, L, D)
        hx: optional hidden state
        returns: (output, hidden_state)
        """
        residual = x
        out, hn = self.cfc(self.norm(x), hx)
        return residual + out, hn


# ──────────────────────────────────────────────
# Cross-Attention (lightweight, for decoder)
# ──────────────────────────────────────────────

class CrossAttention(nn.Module):
    """Lightweight multi-head cross-attention for decoder conditioning."""

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory):
        """x: (B, L, D), memory: (B, L_mem, D)"""
        h = self.norm(x)
        out, _ = self.attn(h, memory, memory)
        return x + out


# ──────────────────────────────────────────────
# Liquid Decoder Layer (CfC + CrossAttn + FFN)
# ──────────────────────────────────────────────

class LiquidDecoderLayer(nn.Module):
    """Single decoder layer: CfC → Cross-Attention → FFN"""

    def __init__(self, d_model, cfc_units, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.cfc_block = CfCBlock(d_model, cfc_units)
        self.cross_attn = CrossAttention(d_model, nhead, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, memory, hx=None):
        """
        x: (B, L, D)
        memory: (B, L_mem, D)
        hx: optional CfC hidden state
        returns: (output, hidden_state)
        """
        # 1. CfC temporal modeling
        x, hn = self.cfc_block(x, hx)
        # 2. Cross-attention to encoder memory
        x = self.cross_attn(x, memory)
        # 3. FFN
        x = x + self.ffn(self.norm(x))
        return x, hn


# ──────────────────────────────────────────────
# Liquid Audio Encoder
# ──────────────────────────────────────────────

class LiquidAudioEncoder(nn.Module):
    """Audio encoder: Conv stem + stacked CfC layers."""

    def __init__(self, n_mels=128, d_model=256, cfc_units=256, num_layers=3, dropout=0.1):
        super().__init__()
        # Conv stem (downsample time dimension)
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2)
        self.dropout = nn.Dropout(dropout)

        # CfC layers
        self.layers = nn.ModuleList([
            CfCBlock(d_model, cfc_units) for _ in range(num_layers)
        ])

    def forward(self, mel):
        """
        mel: (B, n_mels, T)
        returns: (B, T', d_model)
        """
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))   # (B, d_model, T')
        x = self.dropout(x)
        x = x.transpose(1, 2)        # (B, T', d_model)

        for layer in self.layers:
            x, _ = layer(x)

        return x


# ──────────────────────────────────────────────
# Liquid Chart Decoder
# ──────────────────────────────────────────────

class LiquidChartDecoder(nn.Module):
    """
    Decoder: produces chart feature matrix from encoder memory.
    Input: time-aligned query (from audio downsampling) + encoder memory.
    Output: (B, output_channels, T') chart feature matrix.
    """

    def __init__(self, d_model=256, output_channels=72,
                 cfc_units=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.1,
                 decoder_output_dim=None):
        super().__init__()
        # decoder_output_dim: raw decoder output size (may be > output_channels due to categorical logits)
        self.output_dim = decoder_output_dim if decoder_output_dim else output_channels

        # Input projection (from d_model to d_model, for the query)
        self.input_proj = nn.Linear(d_model, d_model)

        # Decoder layers
        self.layers = nn.ModuleList([
            LiquidDecoderLayer(d_model, cfc_units, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        # Output projection: d_model → raw output dim
        self.output_proj = nn.Linear(d_model, self.output_dim)

    def forward(self, memory):
        """
        memory: (B, T', d_model) — encoder output acts as both query and key/value.
        For direct generation, we use the encoder memory as the starting query.
        returns: (B, output_dim, T')
        """
        x = self.input_proj(memory)

        for layer in self.layers:
            x, _ = layer(x, memory)

        x = self.norm(x)
        x = self.output_proj(x)  # (B, T', output_dim)
        return x.transpose(1, 2)  # (B, output_dim, T') — match Mug-Diffusion format


# ──────────────────────────────────────────────
# Liquid Chart Generator (Lightning Module)
# ──────────────────────────────────────────────

class LiquidChartGenerator(pl.LightningModule):
    """
    End-to-end audio → chart matrix generator using CfC networks.
    Output format is compatible with Mug-Diffusion's autoencoder.
    """

    def __init__(self,
                 # Audio
                 n_mels=128,
                 # Model
                 d_model=256,
                 nhead=8,
                 encoder_cfc_units=256,
                 num_encoder_layers=3,
                 decoder_cfc_units=256,
                 num_decoder_layers=4,
                 dim_feedforward=1024,
                 dropout=0.1,
                 # Chart output
                 input_channels=72,
                 z_channels=48,
                 # Loss config (passed through to loss function)
                 lossconfig=None,
                 # Training
                 lr=1e-4,
                 warmup_steps=4000,
                 # Audio params
                 audio_note_window_ratio=8,
                 max_audio_frame=32768,
                 monitor="val/total_weighted_loss",
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.input_channels = input_channels
        self.audio_note_window_ratio = audio_note_window_ratio
        self.max_audio_frame = max_audio_frame

        # Compute decoder output dim from loss config
        decoder_output_dim = input_channels  # default: same as input
        if lossconfig and 'params' in lossconfig:
            fc = lossconfig['params'].get('feature_config', {})
            dol = fc.get('decoder_output_lengths', {})
            if dol:
                decoder_output_dim = sum(dol.values())
                logger.info(f"Decoder output dim from lossconfig: {decoder_output_dim}")

        # Encoder
        self.encoder = LiquidAudioEncoder(
            n_mels=n_mels, d_model=d_model,
            cfc_units=encoder_cfc_units, num_layers=num_encoder_layers,
            dropout=dropout,
        )

        # Decoder
        self.decoder = LiquidChartDecoder(
            d_model=d_model, output_channels=input_channels,
            cfc_units=decoder_cfc_units, nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            decoder_output_dim=decoder_output_dim,
        )

        # Loss function — reuse Mug-Diffusion's NewMaimaiReconstructLoss
        self.loss = None
        if lossconfig:
            self.loss = self._build_loss(lossconfig)

        if monitor is not None:
            self.monitor = monitor

        # Debug logging
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"LiquidChartGenerator initialized: total_params={total_params:,}")
        logger.info(f"  Encoder: {num_encoder_layers} CfC layers (units={encoder_cfc_units})")
        logger.info(f"  Decoder: {num_decoder_layers} CfC+CrossAttn layers (units={decoder_cfc_units})")
        logger.info(f"  d_model={d_model}, nhead={nhead}, output_dim={decoder_output_dim}")
        self._logged_shapes = False

    def _build_loss(self, lossconfig):
        """Instantiate loss function from config dict."""
        target = lossconfig['target']
        params = lossconfig.get('params', {})
        # Dynamic import
        module_path, class_name = target.rsplit('.', 1)
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls(**params)

    def forward(self, mel):
        """
        mel: (B, n_mels, T)
        returns: (B, decoder_output_dim, T') — raw decoder output (logits for binary, etc.)
        """
        memory = self.encoder(mel)  # (B, T', d_model)

        # Audio is at higher time resolution than chart.
        # Downsample memory to chart time resolution.
        # audio_note_window_ratio=8, conv stride=4 → need additional 2x downsample
        chart_len = self.max_audio_frame // self.audio_note_window_ratio
        if memory.size(1) != chart_len:
            # Adaptive pooling to match chart time dimension
            memory_t = memory.transpose(1, 2)  # (B, D, T')
            memory_t = F.adaptive_avg_pool1d(memory_t, chart_len)
            memory = memory_t.transpose(1, 2)  # (B, chart_len, D)

        output = self.decoder(memory)  # (B, decoder_output_dim, chart_len)

        # Debug logging
        if not self._logged_shapes:
            logger.info(f"[shapes] mel={mel.shape}, memory_pooled=({memory.shape}), "
                        f"output={output.shape}")
            if torch.cuda.is_available():
                logger.info(f"[memory] peak GPU: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
            self._logged_shapes = True

        return output

    def _compute_loss(self, batch):
        """Compute loss from batch."""
        mel = batch['audio']
        notes = batch['note']        # (B, 72, T_chart)
        valid_flag = batch['valid_flag']  # (B, T_chart)

        recon = self(mel)  # (B, decoder_output_dim, T_chart)

        if self.loss is not None:
            loss, log_dict = self.loss(notes, recon, valid_flag)
        else:
            # Simple MSE fallback
            loss = F.mse_loss(recon, notes)
            log_dict = {'mse_loss': loss.item()}

        return loss, log_dict

    def training_step(self, batch, batch_idx):
        loss, log_dict = self._compute_loss(batch)
        self.log('train/loss', loss, prog_bar=True)
        log_dict_train = {f'train/{k}': v for k, v in log_dict.items()}
        self.log_dict(log_dict_train, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self._compute_loss(batch)
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        log_dict_val = {f'val/{k}': v for k, v in log_dict.items()}
        self.log_dict(log_dict_val, prog_bar=False, logger=True, sync_dist=True)
        return log_dict_val

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)

        def lr_foo(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return 1.0

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
    def generate(self, mel):
        """
        Generate chart matrix from audio.
        mel: (B, n_mels, T)
        returns: (B, input_channels, T_chart) — sigmoid-activated binary + raw continuous
        """
        self.eval()
        raw_output = self(mel)  # (B, decoder_output_dim, T_chart)
        # For simple usage, just return raw output
        # Post-processing (sigmoid for binary channels, argmax for categorical)
        # should be handled by the caller based on feature_config
        return raw_output
