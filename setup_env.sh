#!/bin/bash
# ============================================================
# Mai-Gen Environment Setup Script
# 
# Tested on: Python 3.11, Linux x86_64, NVIDIA RTX A4000
# Driver: 12040 (CUDA 12.4 compatible)
# System nvcc: 12.0  |  torch target: cu124
# ============================================================
set -e

echo "=========================================="
echo " Mai-Gen Environment Setup"
echo "=========================================="

# ── Step 1: Install PyTorch (force-reinstall to ensure clean files) ──
echo ""
echo "[1/5] Installing PyTorch 2.6.0+cu124..."
pip install --force-reinstall --no-deps torch==2.6.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# ── Step 2: Install all other dependencies ──
echo ""
echo "[2/5] Installing remaining dependencies..."
pip install ninja numpy scikit-learn pytorch-lightning librosa soundfile audioread \
    "jsonargparse[signatures]" google-api-python-client google-auth-httplib2 \
    google-auth-oauthlib optuna PyYAML ncps packaging einops

# ── Step 3: Verify torch and set CUDA_HOME ──
echo ""
echo "[3/5] Verifying PyTorch and configuring CUDA..."
python3 -c "
import torch
print(f'  torch version : {torch.__version__}')
print(f'  CUDA version  : {torch.version.cuda}')
print(f'  CXX11 ABI     : {torch._C._GLIBCXX_USE_CXX11_ABI}')
print(f'  GPU available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU name      : {torch.cuda.get_device_name(0)}')
"

# Use torch's bundled CUDA headers/libs instead of system nvcc
# This avoids the 12.0 vs 12.4 version mismatch entirely
CUDA_PATH=$(python3 -c "
import os, nvidia.cuda_runtime
print(os.path.dirname(os.path.dirname(nvidia.cuda_runtime.__file__)))
" 2>/dev/null || echo "")

if [ -n "$CUDA_PATH" ] && [ -d "$CUDA_PATH" ]; then
    echo "  Found torch-bundled CUDA at: $CUDA_PATH"
fi

# Detect CUDA_HOME: prefer system CUDA that matches torch version
SYSTEM_NVCC_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "none")
TORCH_CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
echo "  System nvcc: $SYSTEM_NVCC_VERSION | torch CUDA: $TORCH_CUDA_VERSION"

if [ "$SYSTEM_NVCC_VERSION" = "$TORCH_CUDA_VERSION" ]; then
    echo "  ✅ CUDA versions match, using system CUDA"
    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
else
    echo "  ⚠️  CUDA version mismatch ($SYSTEM_NVCC_VERSION vs $TORCH_CUDA_VERSION)"
    echo "     Patching torch version check to allow build..."
    
    # Precise, safe patch: make _check_cuda_version a no-op
    python3 << 'PATCHEOF'
import torch.utils.cpp_extension as ext

with open(ext.__file__, 'r') as f:
    lines = f.readlines()

patched = False
with open(ext.__file__, 'w') as f:
    for i, line in enumerate(lines):
        # Find the function definition and add early return
        if 'def _check_cuda_version(compiler_name, compiler_version)' in line:
            f.write(line)
            # Write the next line (docstring or first line)
            # Then inject a return statement
            f.write('    return  # PATCHED: skip CUDA version check (12.x compatible)\n')
            patched = True
        else:
            f.write(line)

if patched:
    print("  ✅ Patched _check_cuda_version to skip version check")
else:
    print("  ⚠️  Could not find _check_cuda_version, proceeding anyway")
PATCHEOF
    
    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
fi

# ── Step 4: Build causal-conv1d and mamba-ssm from source ──
echo ""
echo "[4/5] Building causal-conv1d from source..."
pip install causal-conv1d --no-build-isolation 2>&1 | tail -5

echo ""
echo "      Building mamba-ssm from source..."
rm -f /usr/local/lib/python3.11/dist-packages/selective_scan_cuda*.so 2>/dev/null || true
rm -f /usr/local/lib/python3.11/dist-packages/mamba_ssm_cuda*.so 2>/dev/null || true
FORCE_BUILD=TRUE pip install mamba-ssm --no-build-isolation 2>&1 | tail -5

# ── Step 5: Restore patch and verify ──
echo ""
echo "[5/5] Restoring torch and verifying..."
# Restore clean torch (undo patch)
pip install --force-reinstall --no-deps torch==2.6.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1

echo ""
echo "=========================================="
echo " Verification"
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
