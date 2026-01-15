# PyTorch 2.9.1 Quick Start Guide

## TL;DR - Build Your First Variant

```bash
cd /home/daedalus/dev/builds/build-pytorch-2.9.1
flox activate

# For RTX 5090
flox build --stability=unstable pytorch-python313-cuda13_0-sm120-avx512

# For CPU-only
flox build pytorch-python313-cpu-avx2

# Test it
./test-build.sh
```

## Choose Your Variant

### Have NVIDIA GPU?

```bash
# Check your GPU
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

| Your GPU | Build Command |
|----------|---------------|
| RTX 5090 | `flox build --stability=unstable pytorch-python313-cuda13_0-sm120-avx512` |
| H100, L40S | `flox build --stability=unstable pytorch-python313-cuda13_0-sm90-avx512` |
| RTX 3090, A40 | `flox build --stability=unstable pytorch-python313-cuda13_0-sm86-avx2` |
| A100, A30 | `flox build --stability=unstable pytorch-python313-cuda13_0-sm80-avx2` |

### CPU-Only?

```bash
# Check CPU features
lscpu | grep avx
```

- Has `avx512`? → `flox build pytorch-python313-cpu-avx512`
- Only `avx2`? → `flox build pytorch-python313-cpu-avx2`

## Key Differences from PyTorch 2.8 Builds

1. **PyTorch Version**: 2.9.1 (vs 2.8.0)
2. **CUDA Version**: 13.0 (vs 12.8)
3. **Build Flag**: Requires `--stability=unstable` for GPU builds
4. **Features**: Full CUDA 13.0 support, better SM120 (RTX 5090) optimization

## Build Times

- CPU variants: ~1-2 hours
- GPU variants: ~2-3 hours

## After Building

```bash
# Test your build
./test-build.sh pytorch-python313-cuda13_0-sm120-avx512

# Use the built package
./result-pytorch-python313-cuda13_0-sm120-avx512/bin/python -c "import torch; print(torch.__version__)"
# Output: 2.9.1

# Check CUDA version
./result-pytorch-python313-cuda13_0-sm120-avx512/bin/python -c "import torch; print(torch.version.cuda)"
# Output: 13.0
```

## Publishing

```bash
# Initialize git repo
git init
git add .
git commit -m "PyTorch 2.9.1 builds with CUDA 13.0"
git remote add origin <your-repo-url>
git push origin main

# Publish to Flox
flox publish -o <your-org> pytorch-python313-cuda13_0-sm120-avx512
```

## Common Issues

### "cudaPackages_13 not found"
→ Add `--stability=unstable` flag

### "No such file or directory"
→ Build first: `flox build --stability=unstable <variant>`

### Build fails
→ Check you have 20GB free disk space

## Available Variants

**GPU (CUDA 13.0)**:
- `pytorch-python313-cuda13_0-sm120-avx512` - RTX 5090
- `pytorch-python313-cuda13_0-sm90-avx512` - H100/L40S
- `pytorch-python313-cuda13_0-sm86-avx2` - RTX 3090/A40
- `pytorch-python313-cuda13_0-sm80-avx2` - A100/A30

**CPU-Only**:
- `pytorch-python313-cpu-avx2` - Broad compatibility
- `pytorch-python313-cpu-avx512` - Modern CPUs

## Next Steps

- Read [README.md](./README.md) for detailed documentation
- Add more variants by copying and modifying `.flox/pkgs/*.nix` files
- Test thoroughly before publishing to production