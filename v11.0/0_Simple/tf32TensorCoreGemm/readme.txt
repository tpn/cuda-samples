Sample: tf32TensorCoreGemm
Minimum spec: SM 8.0

A CUDA sample demonstrating tf32 (e8m10) GEMM computation using the Warp Matrix Multiply and Accumulate (WMMA) API introduced with CUDA 11 in Ampere chip family tensor cores for faster matrix operations. This sample also uses async copy provided by cuda pipeline interface for gmem to shmem async loads which improves kernel performance and reduces register presssure.

Key concepts:
Matrix Multiply
WMMA
Tensor Cores
