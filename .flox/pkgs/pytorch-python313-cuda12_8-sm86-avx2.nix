# PyTorch 2.9.1 optimized for NVIDIA Ampere (RTX 3090, A40) + AVX2
# Package name: pytorch-python313-cuda12_8-sm86-avx2

{ python3Packages
, lib
, config
, cudaPackages_12
, addDriverRunpath
, openblas
}:

let
  # GPU target: SM86 (NVIDIA Ampere - RTX 3090, RTX 3090 Ti, A40, A5000)
  gpuArchSM = "sm_86";  # For TORCH_CUDA_ARCH_LIST
  gpuArchNum = "8.6";   # For gpuTargets override (numeric format)

  # CPU optimization: AVX2 (broad compatibility)
  cpuFlags = [
    "-mavx2"       # Advanced Vector Extensions 2
    "-mfma"        # Fused multiply-add
    "-mf16c"       # 16-bit float conversion
  ];

in (python3Packages.torch.override {
  cudaSupport = true;
  cudaPackages = cudaPackages_12;
  gpuTargets = [ gpuArchNum ];
}).overrideAttrs (oldAttrs: {
  pname = "pytorch-python313-cuda12_8-sm86-avx2";

  # Override build configuration
  buildInputs = oldAttrs.buildInputs ++ [
    cudaPackages_12.cuda_cudart
    cudaPackages_12.libcublas
    cudaPackages_12.libcufft
    cudaPackages_12.libcurand
    cudaPackages_12.libcusolver
    cudaPackages_12.libcusparse
    cudaPackages_12.cudnn
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
    echo "GPU Target: ${gpuArchSM} (NVIDIA Ampere - RTX 3090, A40)"
    echo "CPU Features: AVX2 (broad compatibility)"
    echo "CUDA: 12.8 with cuBLAS"
    echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
    echo "CXXFLAGS: $CXXFLAGS"
    echo "========================================="
  '';

  meta = oldAttrs.meta // {
    description = "PyTorch 2.9.1 optimized for NVIDIA Ampere (RTX 3090, A40) with AVX2";
    longDescription = ''
      Custom PyTorch 2.9.1 build with targeted optimizations:
      - GPU: NVIDIA Ampere (RTX 3090, RTX 3090 Ti, A40, A5000) - SM86
      - CPU: x86-64 with AVX2 instruction set (broad compatibility)
      - CUDA: 12.8
      - BLAS: cuBLAS for GPU operations, OpenBLAS for host-side
      - Python: 3.13
    '';
    platforms = [ "x86_64-linux" ];
  };
})