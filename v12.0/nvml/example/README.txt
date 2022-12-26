The NVIDIA GDK provides a simple example program that shows how to build an 
NVML client. When running an NVML client while the GDK is installed, be
sure your library path first includes the actual NVML library (installed 
with the driver), not the stub library that exists solely for 
compilation on systems without an NVIDIA driver available.

If you have installed this example code via your packaging system, 
you should first copy it to a user directory before compilation.
The packaging system uninstall feature will not remove this directory if it 
contains new files beyond what were installed as part of the GDK.
