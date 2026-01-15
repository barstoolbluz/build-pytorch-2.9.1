# PyTorch 2.9.1 optimized for NVIDIA Hopper (H100, L40S) + AVX-512
# Package name: pytorch-python313-cuda13_0-sm90-avx512

{ python3Packages
, lib
, config
, cudaPackages_13
, addDriverRunpath
, openblas
}:

let
  # GPU target: SM90 (NVIDIA Hopper - H100, H200, L40S)
  gpuArchSM = "sm_90";  # For TORCH_CUDA_ARCH_LIST
  gpuArchNum = "9.0";   # For gpuTargets override (numeric format)

  # CPU optimization: AVX-512
  cpuFlags = [
    "-mavx512f"    # AVX-512 Foundation
    "-mavx512dq"   # Doubleword and Quadword instructions
    "-mavx512vl"   # Vector Length extensions
    "-mavx512bw"   # Byte and Word instructions
    "-mfma"        # Fused multiply-add
  ];

in (python3Packages.pytorch.override {
  cudaSupport = true;
  cudaPackages = cudaPackages_13;
  gpuTargets = [ gpuArchNum ];
}).overrideAttrs (oldAttrs: {
  pname = "pytorch-python313-cuda13_0-sm90-avx512";
  version = "2.9.1";

  # Override build configuration
  buildInputs = oldAttrs.buildInputs ++ [
    cudaPackages_13.cuda_cudart
    cudaPackages_13.libcublas
    cudaPackages_13.libcufft
    cudaPackages_13.libcurand
    cudaPackages_13.libcusolver
    cudaPackages_13.libcusparse
    cudaPackages_13.cudnn
    # Explicitly add dynamic OpenBLAS for host-side operations
    (openblas.override {
      blas64 = false;
      singleThreaded = false;
    })
  ];

  nativeBuildInputs = oldAttrs.nativeBuildInputs ++ [
    addDriverRunpath
  ];

  # Set CUDA architecture and CPU optimization flags
  preConfigure = (oldAttrs.preConfigure or "") + ''
    export TORCH_CUDA_ARCH_LIST="${gpuArchSM}"
    export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

    # CPU optimizations via compiler flags
    export CXXFLAGS="$CXXFLAGS ${lib.concatStringsSep " " cpuFlags}"
    export CFLAGS="$CFLAGS ${lib.concatStringsSep " " cpuFlags}"

    # Enable cuBLAS
    export USE_CUBLAS=1
    export USE_CUDA=1

    # Optimize for target architecture
    export CMAKE_CUDA_ARCHITECTURES="${lib.removePrefix "sm_" gpuArchSM}"

    echo "========================================="
    echo "PyTorch 2.9.1 Build Configuration"
    echo "========================================="
    echo "GPU Target: ${gpuArchSM} (NVIDIA Hopper - H100, L40S)"
    echo "CPU Features: AVX-512"
    echo "CUDA: 13.0 with cuBLAS"
    echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
    echo "CXXFLAGS: $CXXFLAGS"
    echo "========================================="
  '';

  meta = oldAttrs.meta // {
    description = "PyTorch 2.9.1 optimized for NVIDIA Hopper (H100, L40S) with AVX-512";
    longDescription = ''
      Custom PyTorch 2.9.1 build with targeted optimizations:
      - GPU: NVIDIA Hopper (H100, H200, L40S) - SM90
      - CPU: x86-64 with AVX-512 instruction set
      - CUDA: 13.0
      - BLAS: cuBLAS for GPU operations, OpenBLAS for host-side
      - Python: 3.13
    '';
    platforms = [ "x86_64-linux" ];
  };
})