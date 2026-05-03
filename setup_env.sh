#!/bin/bash
# ============================================================
# Mai-Gen Environment Setup Script
# 
# Tested on: Python 3.11, Linux x86_64
# Driver: 12040 (CUDA 12.4 compatible)
# System nvcc: 12.0  |  torch target: cu124
#
# NOTE: torch 2.6+ only WARNS on CUDA version mismatch
#       (no longer raises RuntimeError), so no patch needed.
# ============================================================
set -e

echo "=========================================="
echo " Mai-Gen Environment Setup"
echo "=========================================="

# ── Step 1: Install PyTorch + basic deps ──
echo ""
echo "[1/5] Installing PyTorch 2.6.0+cu124 and basic dependencies..."
pip install torch==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
pip install ninja numpy scikit-learn pytorch-lightning librosa soundfile audioread \
    "jsonargparse[signatures]" google-api-python-client google-auth-httplib2 \
    google-auth-oauthlib optuna PyYAML ncps packaging

# ── Step 2: Verify torch ──
echo ""
echo "[2/5] Verifying PyTorch installation..."
python3 -c "
import torch
print(f'  torch version : {torch.__version__}')
print(f'  CUDA version  : {torch.version.cuda}')
print(f'  CXX11 ABI     : {torch._C._GLIBCXX_USE_CXX11_ABI}')
print(f'  GPU available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU name      : {torch.cuda.get_device_name(0)}')
"

# ── Step 3: Build causal-conv1d from source ──
echo ""
echo "[3/5] Building causal-conv1d from source (may take a few minutes)..."
echo "  Note: CUDA 12.0 vs 12.4 mismatch warning is expected and harmless."
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install causal-conv1d --no-build-isolation 2>&1 | \
    grep -v "CUDA_MISMATCH" || true

# ── Step 4: Build mamba-ssm from source ──
echo ""
echo "[4/5] Building mamba-ssm from source (may take several minutes)..."
# Clean any stale .so files from previous attempts
rm -f /usr/local/lib/python3.11/dist-packages/selective_scan_cuda*.so 2>/dev/null || true
rm -f /usr/local/lib/python3.11/dist-packages/mamba_ssm_cuda*.so 2>/dev/null || true

MAMBA_FORCE_BUILD=TRUE pip install mamba-ssm --no-build-isolation 2>&1 | \
    grep -v "CUDA_MISMATCH" || true

# ── Step 5: Final verification ──
echo ""
echo "=========================================="
echo "[5/5] Verification"
echo "=========================================="
python3 << 'VERIFY_SCRIPT'
errors = []

try:
    import torch
    print(f"✅ torch {torch.__version__} (CUDA {torch.version.cuda})")
except Exception as e:
    errors.append(f"❌ torch: {e}")

try:
    import causal_conv1d
    print(f"✅ causal-conv1d imported")
except Exception as e:
    errors.append(f"❌ causal-conv1d: {e}")

try:
    from mamba_ssm import Mamba
    print(f"✅ mamba-ssm imported (Mamba class available)")
except Exception as e:
    errors.append(f"❌ mamba-ssm: {e}")

try:
    import pytorch_lightning as pl
    print(f"✅ pytorch-lightning {pl.__version__}")
except Exception as e:
    errors.append(f"❌ pytorch-lightning: {e}")

try:
    import ncps
    print(f"✅ ncps imported")
except Exception as e:
    errors.append(f"❌ ncps: {e}")

if errors:
    print("\n⚠️  Some packages failed:")
    for err in errors:
        print(f"  {err}")
else:
    print("\n🎉 All packages installed and verified successfully!")
VERIFY_SCRIPT
