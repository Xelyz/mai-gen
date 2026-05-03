#!/bin/bash
# ============================================================
# Mai-Gen Environment Setup Script
# 
# Tested on: Python 3.11, Linux x86_64
# Driver: 12040 (CUDA 12.4 compatible)
# System nvcc: 12.0  |  torch target: cu124
# ============================================================
set -e

echo "=========================================="
echo " Mai-Gen Environment Setup"
echo "=========================================="

# ── Step 1: Install PyTorch + basic deps ──
echo ""
echo "[1/6] Installing PyTorch 2.6.0+cu124 and basic dependencies..."
pip install torch==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
pip install ninja numpy scikit-learn pytorch-lightning librosa soundfile audioread \
    "jsonargparse[signatures]" google-api-python-client google-auth-httplib2 \
    google-auth-oauthlib optuna PyYAML ncps packaging

# ── Step 2: Verify torch ──
echo ""
echo "[2/6] Verifying PyTorch installation..."
python3 -c "
import torch
print(f'  torch version : {torch.__version__}')
print(f'  CUDA version  : {torch.version.cuda}')
print(f'  CXX11 ABI     : {torch._C._GLIBCXX_USE_CXX11_ABI}')
print(f'  GPU available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU name      : {torch.cuda.get_device_name(0)}')
"

# ── Step 3: Patch torch CUDA version check ──
# System nvcc is 12.0 but torch expects 12.4
# CUDA 12.x is API-compatible across minor versions
echo ""
echo "[3/6] Temporarily patching torch CUDA version check..."
TORCH_EXT_FILE=$(python3 -c "import torch.utils.cpp_extension as e; print(e.__file__)")
echo "  Target file: $TORCH_EXT_FILE"

# Backup original
cp "$TORCH_EXT_FILE" "${TORCH_EXT_FILE}.bak"

# Patch: replace the version check raise with a warning
python3 << 'PATCH_SCRIPT'
import sys

ext_file = sys.argv[1] if len(sys.argv) > 1 else None
if not ext_file:
    import torch.utils.cpp_extension as e
    ext_file = e.__file__

with open(ext_file, 'r') as f:
    content = f.read()

# Find and neutralize the CUDA version mismatch check
old = "raise RuntimeError(CUDA_MISMATCH_MESSAGE"
new = "import warnings; warnings.warn('CUDA version mismatch patched for build compatibility')  # raise RuntimeError(CUDA_MISMATCH_MESSAGE"

if old in content:
    content = content.replace(old, new, 1)  # Only replace first occurrence
    with open(ext_file, 'w') as f:
        f.write(content)
    print("  ✅ Patch applied successfully")
else:
    print("  ⚠️  Pattern not found (may already be patched)")
PATCH_SCRIPT

# ── Step 4: Build causal-conv1d from source ──
echo ""
echo "[4/6] Building causal-conv1d from source (this may take a few minutes)..."
pip install causal-conv1d --no-build-isolation

# ── Step 5: Build mamba-ssm from source ──
echo ""
echo "[5/6] Building mamba-ssm from source (this may take several minutes)..."
# Clean any stale .so files first
rm -f /usr/local/lib/python3.11/dist-packages/selective_scan_cuda*.so 2>/dev/null || true
rm -f /usr/local/lib/python3.11/dist-packages/mamba_ssm_cuda*.so 2>/dev/null || true

FORCE_BUILD=TRUE pip install mamba-ssm --no-build-isolation

# ── Step 6: Restore torch patch ──
echo ""
echo "[6/6] Restoring original torch file..."
if [ -f "${TORCH_EXT_FILE}.bak" ]; then
    cp "${TORCH_EXT_FILE}.bak" "$TORCH_EXT_FILE"
    rm "${TORCH_EXT_FILE}.bak"
    echo "  ✅ Original file restored"
else
    echo "  ⚠️  Backup not found, skipping restore"
fi

# ── Final verification ──
echo ""
echo "=========================================="
echo " Verification"
echo "=========================================="
python3 << 'VERIFY_SCRIPT'
errors = []

# Check torch
try:
    import torch
    print(f"✅ torch {torch.__version__} (CUDA {torch.version.cuda})")
except Exception as e:
    errors.append(f"❌ torch: {e}")

# Check causal-conv1d
try:
    import causal_conv1d
    print(f"✅ causal-conv1d imported")
except Exception as e:
    errors.append(f"❌ causal-conv1d: {e}")

# Check mamba-ssm
try:
    from mamba_ssm import Mamba
    print(f"✅ mamba-ssm imported (Mamba class available)")
except Exception as e:
    errors.append(f"❌ mamba-ssm: {e}")

# Check pytorch-lightning
try:
    import pytorch_lightning as pl
    print(f"✅ pytorch-lightning {pl.__version__}")
except Exception as e:
    errors.append(f"❌ pytorch-lightning: {e}")

# Check ncps
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
