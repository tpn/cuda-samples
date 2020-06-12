Sample: simpleD3D12
Minimum spec: SM 3.5

A program which demonstrates Direct3D12 interoperability with CUDA.  The program creates a sinewave in DX12 vertex buffer which is created using CUDA kernels. DX12 and CUDA synchronizes using DirectX12 Fences. Direct3D then renders the results on the screen.  A DirectX12 Capable NVIDIA GPU is required on Windows10 or higher OS.

Key concepts:
Graphics Interop
CUDA DX12 Interop
Image Processing
