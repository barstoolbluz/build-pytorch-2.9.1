# PyTorch 2.9.1 CPU-only optimized for AVX2
# Package name: pytorch-python313-cpu-avx2

{ python3Packages
, lib
, openblas
, mkl
, fetchFromGitHub
}:

let
  # CPU optimization: AVX2 (broad compatibility)
  cpuFlags = [
    "-mavx2"       # Advanced Vector Extensions 2
    "-mfma"        # Fused multiply-add
    "-mf16c"       # 16-bit float conversion
  ];

  # Use OpenBLAS for CPU linear algebra
  blasBackend = openblas;

in (python3Packages.pytorch.override {
  cudaSupport = false;
}).overrideAttrs (oldAttrs: {
  pname = "pytorch-python313-cpu-avx2";
  version = "2.9.1";

  src = fetchFromGitHub {
    owner = "pytorch";
    repo = "pytorch";
    rev = "v2.9.1";
    hash = "sha256-MYzzceoQh01jzQU9tyAl47PU4M+QbuKwHXQAE8yt1Hg=";
    fetchSubmodules = true;
  };

  # Override patches - PyTorch 2.9.1 doesn't need 2.8.0 patches

  # Override postPatch - skip the setuptools replacement that doesn't apply to 2.9.1
  postPatch = ''
    # Add necessary postPatch commands for PyTorch 2.9.1 if needed
  '';
  patches = [];

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
    echo "CPU Features: AVX2 (broad compatibility)"
    echo "BLAS Backend: OpenBLAS"
    echo "CUDA: Disabled"
    echo "CXXFLAGS: $CXXFLAGS"
    echo "========================================="
  '';

  meta = oldAttrs.meta // {
    description = "PyTorch 2.9.1 CPU-only build optimized for AVX2";
    longDescription = ''
      Custom PyTorch 2.9.1 build for CPU-only workloads:
      - GPU: None (CPU-only)
      - CPU: x86-64 with AVX2 instruction set (Intel Haswell+, AMD Zen 1+)
      - BLAS: OpenBLAS for CPU linear algebra operations
      - Python: 3.13

      This build is suitable for inference, development, and workloads
      that don't require GPU acceleration. Compatible with most x86-64
      processors from 2013 onwards.
    '';
    platforms = [ "x86_64-linux" "aarch64-linux" ];
  };
})