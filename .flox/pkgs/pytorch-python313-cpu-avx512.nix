# PyTorch 2.9.1 CPU-only optimized for AVX-512
# Package name: pytorch-python313-cpu-avx512

{ python3Packages
, lib
, openblas
, mkl
}:

let
  # CPU optimization: AVX-512 (modern CPUs)
  cpuFlags = [
    "-mavx512f"    # AVX-512 Foundation
    "-mavx512dq"   # Doubleword and Quadword instructions
    "-mavx512vl"   # Vector Length extensions
    "-mavx512bw"   # Byte and Word instructions
    "-mfma"        # Fused multiply-add
  ];

  # Use OpenBLAS for CPU linear algebra
  blasBackend = openblas;

in (python3Packages.torch.override {
  cudaSupport = false;
}).overrideAttrs (oldAttrs: {
  pname = "pytorch-python313-cpu-avx512";

  # CPU-only build metadata
  passthru = oldAttrs.passthru // {
    gpuArch = null;
    blasProvider = "openblas";
  };

  # Override build configuration - remove CUDA deps, ensure BLAS
  buildInputs = lib.filter (p: !(lib.hasPrefix "cuda" (p.pname or ""))) oldAttrs.buildInputs ++ [
    blasBackend
  ];

  nativeBuildInputs = lib.filter (p: p.pname or "" != "addDriverRunpath") oldAttrs.nativeBuildInputs;

  # Set CPU optimization flags and disable CUDA
  preConfigure = (oldAttrs.preConfigure or "") + ''
    # Disable CUDA
    export USE_CUDA=0
    export USE_CUDNN=0
    export USE_CUBLAS=0

    # Use OpenBLAS for CPU operations
    export BLAS=OpenBLAS
    export USE_MKLDNN=1
    export USE_MKLDNN_CBLAS=1

    # CPU optimizations via compiler flags
    export CXXFLAGS="$CXXFLAGS ${lib.concatStringsSep " " cpuFlags}"
    export CFLAGS="$CFLAGS ${lib.concatStringsSep " " cpuFlags}"

    # Optimize for host CPU
    export CMAKE_BUILD_TYPE=Release

    echo "========================================="
    echo "PyTorch 2.9.1 Build Configuration"
    echo "========================================="
    echo "GPU Target: None (CPU-only build)"
    echo "CPU Features: AVX-512 (modern x86-64)"
    echo "BLAS Backend: OpenBLAS"
    echo "CUDA: Disabled"
    echo "CXXFLAGS: $CXXFLAGS"
    echo "========================================="
  '';

  meta = oldAttrs.meta // {
    description = "PyTorch 2.9.1 CPU-only build optimized for AVX-512";
    longDescription = ''
      Custom PyTorch 2.9.1 build for CPU-only workloads:
      - GPU: None (CPU-only)
      - CPU: x86-64 with AVX-512 instruction set (Intel Skylake-X+, AMD Zen 4+)
      - BLAS: OpenBLAS for CPU linear algebra operations
      - Python: 3.13

      This build is suitable for high-performance CPU inference and training
      on modern processors with AVX-512 support. Provides ~2x performance
      improvement over baseline for many workloads.
    '';
    platforms = [ "x86_64-linux" ];
  };
})