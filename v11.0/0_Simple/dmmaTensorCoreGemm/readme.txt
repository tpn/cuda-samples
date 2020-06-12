Sample: dmmaTensorCoreGemm
Minimum spec: SM 8.0

CUDA sample demonstrates double precision GEMM computation using the Double precision Warp Matrix Multiply and Accumulate (WMMA) API introduced with CUDA 11 in Ampere chip family tensor cores for faster matrix operations. This sample also uses async copy provided by cuda pipeline interface for gmem to shmem async loads which improves kernel performance and reduces register presssure.

Key concepts:
Matrix Multiply
WMMA
Tensor Cores
