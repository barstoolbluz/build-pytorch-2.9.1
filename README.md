# PyTorch 2.9.1 Custom Build Environment

This Flox environment builds custom PyTorch 2.9.1 variants with **CUDA 13.0** support and targeted optimizations for specific GPU architectures and CPU instruction sets.

## Key Differences from PyTorch 2.8.0 Builds

- **PyTorch Version**: 2.9.1 (latest stable with CUDA 13.0 support)
- **CUDA Version**: 13.0 (requires `--stability=unstable` flag)
- **GPU Support**: Full support for latest architectures including SM120 (RTX 5090)
- **Build Method**: Uses upstream nixpkgs PyTorch with overrides (not built from scratch)

## Quick Start

```bash
# Enter the build environment
cd /home/daedalus/dev/builds/build-pytorch-2.9.1
flox activate

# Build a specific variant (CUDA 13.0 requires unstable flag)
flox build --stability=unstable pytorch-python313-cuda13_0-sm120-avx512

# The result will be in ./result-pytorch-python313-cuda13_0-sm120-avx512/
ls -lh result-pytorch-python313-cuda13_0-sm120-avx512/lib/python3.13/site-packages/torch/
```

## Available Variants

### GPU Builds (CUDA 13.0)

| Package Name | GPU Architecture | CPU ISA | Hardware |
|--------------|-----------------|---------|----------|
| `pytorch-python313-cuda13_0-sm120-avx512` | SM120 | AVX-512 | RTX 5090 + modern CPUs |
| `pytorch-python313-cuda13_0-sm90-avx512` | SM90 | AVX-512 | H100/L40S + modern CPUs |
| `pytorch-python313-cuda13_0-sm86-avx2` | SM86 | AVX2 | RTX 3090/A40 + broad compatibility |
| `pytorch-python313-cuda13_0-sm80-avx2` | SM80 | AVX2 | A100/A30 + broad compatibility |

### CPU-Only Builds

| Package Name | CPU ISA | Hardware Support |
|--------------|---------|------------------|
| `pytorch-python313-cpu-avx2` | AVX2 | Intel Haswell+ (2013), AMD Zen 1+ (2017) |
| `pytorch-python313-cpu-avx512` | AVX-512 | Intel Skylake-X+ (2017), AMD Zen 4+ (2022) |

## Building Variants

### GPU Builds (Require CUDA 13.0)

```bash
# IMPORTANT: Use --stability=unstable for CUDA 13.0 support
flox build --stability=unstable pytorch-python313-cuda13_0-sm120-avx512
flox build --stability=unstable pytorch-python313-cuda13_0-sm90-avx512
flox build --stability=unstable pytorch-python313-cuda13_0-sm86-avx2
flox build --stability=unstable pytorch-python313-cuda13_0-sm80-avx2
```

### CPU-Only Builds

```bash
# CPU builds don't need unstable flag
flox build pytorch-python313-cpu-avx2
flox build pytorch-python313-cpu-avx512
```

## Hardware Selection Guide

### Check Your GPU

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

| Your GPU | Compute Cap | Use Build |
|----------|-------------|-----------|
| RTX 5090 | 12.0 | `pytorch-python313-cuda13_0-sm120-avx512` |
| H100, H200, L40S | 9.0 | `pytorch-python313-cuda13_0-sm90-avx512` |
| RTX 3090, A40 | 8.6 | `pytorch-python313-cuda13_0-sm86-avx2` |
| A100, A30 | 8.0 | `pytorch-python313-cuda13_0-sm80-avx2` |
| No GPU | N/A | `pytorch-python313-cpu-avx2` or `avx512` |

### Check Your CPU

```bash
lscpu | grep -E 'avx512|avx2'
```

- See `avx512f`? → Use AVX-512 variants
- See only `avx2`? → Use AVX2 variants

## Testing Your Build

```bash
# Test the built package
./test-build.sh pytorch-python313-cuda13_0-sm120-avx512

# Or test CPU build
./test-build.sh pytorch-python313-cpu-avx2
```

## Publishing to Flox Catalog

After successful builds:

```bash
# Ensure git remote is configured
git init
git add .
git commit -m "PyTorch 2.9.1 custom builds with CUDA 13.0"
git remote add origin <your-repo-url>
git push origin main

# Publish to your organization
flox publish -o <your-org> pytorch-python313-cuda13_0-sm120-avx512
flox publish -o <your-org> pytorch-python313-cpu-avx2

# Users install with:
flox install <your-org>/pytorch-python313-cuda13_0-sm120-avx512
```

## Build Times & Requirements

- **Time**: 1-3 hours per variant
- **Disk**: ~20GB per build
- **Memory**: 8GB+ RAM recommended
- **CUDA 13.0**: Requires `--stability=unstable` flag

## Technical Details

### CUDA 13.0 Support

CUDA 13.0 is available in unstable nixpkgs. Key features:
- Support for all architectures from Turing through Blackwell
- Removed legacy SM 5.x-7.0 support
- Enhanced performance for modern GPUs
- Required driver: 580+ (Linux)

### PyTorch 2.9.1 Features

- Full CUDA 13.0 support (prototype/experimental)
- Python 3.13 support
- Improved performance on modern hardware
- Better memory management

### BLAS Backends

- **GPU builds**: cuBLAS (primary) + OpenBLAS (host-side fallback)
- **CPU builds**: OpenBLAS with dynamic architecture detection

## Extending the Build Matrix

To add new variants:

1. Copy an existing `.nix` file in `.flox/pkgs/`
2. Modify the architecture and CPU flags
3. Update the package name and description
4. Build with appropriate flags

Example for RTX 4090 (SM89):
```bash
cp .flox/pkgs/pytorch-python313-cuda13_0-sm90-avx512.nix \
   .flox/pkgs/pytorch-python313-cuda13_0-sm89-avx512.nix

# Edit to change:
# - gpuArch = "sm_89"
# - pname = "pytorch-python313-cuda13_0-sm89-avx512"
# - Update descriptions

flox build --stability=unstable pytorch-python313-cuda13_0-sm89-avx512
```

## Troubleshooting

### "cudaPackages_13 not found"

Make sure you're using the `--stability=unstable` flag:
```bash
flox build --stability=unstable <package-name>
```

### Build fails with CUDA error

Verify CUDA 13.0 is being used - check build logs for "cuda13.0" in package names.

### Architecture mismatch

Ensure you're building for the correct GPU architecture. Use `nvidia-smi` to check your hardware.

## Related Documentation

- [PyTorch 2.9 Release Notes](https://pytorch.org/blog/pytorch-2-9/)
- [CUDA 13.0 Documentation](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [Flox Documentation](https://flox.dev/docs/)

## License

This build environment configuration is MIT licensed. PyTorch itself is BSD-3-Clause licensed.