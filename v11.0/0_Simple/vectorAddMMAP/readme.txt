Sample: vectorAddMMAP
Minimum spec: SM 3.5

This sample replaces the device allocation in the vectorAddDrv sample with cuMemMap-ed allocations.  This sample demonstrates that the cuMemMap api allows the user to specify the physical properties of their memory while retaining the contiguos nature of their access, thus not requiring a change in their program structure.

Key concepts:
CUDA Driver API
Vector Addition
MMAP
