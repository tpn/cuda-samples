
/**
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  pragma warning(disable:4819)
#endif

#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_string.h>
#include <helper_cuda.h>

bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer   = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);

    return true;
}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    try
    {
        std::string sFilename;
        char *filePath;

        int dev = findCudaDevice(argc, (const char **)argv);

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

        if (printfNPPinfo(argc, argv) == false)
        {
            exit(EXIT_SUCCESS);
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "file"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "file", &filePath);
        }
        else
        {
            filePath = sdkFindFilePath("Lena.pgm", argv[0]);
        }

        if (filePath)
        {
            sFilename = filePath;
        }
        else
        {
            sFilename = "Lena.pgm";
        }

        // if we specify the filename at the command line, then we only test sFilename[0].
        int file_errors = 0;
        std::ifstream infile(sFilename.data(), std::ifstream::in);

        if (infile.good())
        {
            std::cout << "boundSegmentsNPP opened: <" << sFilename.data() << "> successfully!" << std::endl;
            file_errors = 0;
            infile.close();
        }
        else
        {
            std::cout << "boundSegmentsNPP unable to open: <" << sFilename.data() << ">" << std::endl;
            file_errors++;
            infile.close();
        }

        if (file_errors > 0)
        {
            exit(EXIT_FAILURE);
        }

        std::string sResultFilename = sFilename;

        std::string::size_type dot = sResultFilename.rfind('.');

        if (dot != std::string::npos)
        {
            sResultFilename = sResultFilename.substr(0, dot);
        }

        sResultFilename += "_boundSegments.pgm";

        if (checkCmdLineFlag(argc, (const char **)argv, "output"))
        {
            char *outputFilePath;
            getCmdLineArgumentString(argc, (const char **)argv, "output", &outputFilePath);
            sResultFilename = outputFilePath;
        }

        // declare a host image object for an 8-bit grayscale image
        npp::ImageCPU_8u_C1 oHostSrc;
        // load gray-scale image from disk
        npp::loadImage(sFilename, oHostSrc);
        // declare a device image and copy construct from the host image,
        // i.e. upload host to device
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

        NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
        NppiPoint oSrcOffset = {0, 0};

        // create struct with ROI size
        NppiSize oSizeROI = {(int)oDeviceSrc.width() , (int)oDeviceSrc.height() };
        // allocate device image of appropriately reduced size
        npp::ImageNPP_8u_C1 oDeviceDst8u(oSizeROI.width, oSizeROI.height);
        npp::ImageNPP_32s_C1 oDeviceDst32u(oSizeROI.width, oSizeROI.height); // sample code limitation

        int nBufferSize = 0;
        Npp8u * pScratchBufferNPP = 0;

        // Clear values below threshold to 0 and above threshold to 255
        // The threshold is used here to help limit the maximum number of labels generated resulting in a more viewable output image
        NPP_CHECK_NPP (
                           nppiThreshold_LTValGTVal_8u_C1IR(oDeviceSrc.data(), oDeviceSrc.pitch(), oSizeROI, 166, 0, 165, 255) );


        // Get necessary scratch buffer size and allocate that much device memory
        NPP_CHECK_NPP (
                           nppiLabelMarkersUFGetBufferSize_32u_C1R(oSizeROI, &nBufferSize) );

        cudaMalloc((void **)&pScratchBufferNPP, nBufferSize);

        // Now generate label markers using 8 way search mode (nppiNormInf).  The union-find versions of nppiLabelMarkersUF() functions will
        // tend to generate large label values so output is always in 32 bits.
        if ((nBufferSize > 0) && (pScratchBufferNPP != 0))
        {
            NPP_CHECK_NPP (
                               nppiLabelMarkersUF_8u32u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(), 
                                                            reinterpret_cast<Npp32u *>(oDeviceDst32u.data()), oDeviceDst32u.pitch(), oSizeROI,
                                                            nppiNormInf, pScratchBufferNPP) );
        }


        // free scratch buffer memory
        cudaFree(pScratchBufferNPP);

        // The generated list of labels is likely to be very sparsely populated so it might be possible to compress them into a label list that
        // will fit into 8 bits.  In this particular sample, even after compression, the total number of labels is still way over 8 bits.
        // 
        // Get necessary scratch buffer size and allocate that much device memory.  When using nppiLabelMarkersUF functions to generate the labels,
        // the value of the startingNumber parameter MUST be set to oSizeROI.width * oSizeROI.height.
        // 
        // Get necessary scratch buffer size and allocate that much device memory
        NPP_CHECK_NPP (
                           nppiCompressMarkerLabelsGetBufferSize_32u8u_C1R(oSizeROI.width * oSizeROI.height, &nBufferSize) );

        cudaMalloc((void **)&pScratchBufferNPP, nBufferSize);

        int maxLabel = 0;

        if ((nBufferSize > 0) && (pScratchBufferNPP != 0))
        {

            NPP_CHECK_NPP (
                               nppiCompressMarkerLabels_32u8u_C1R(reinterpret_cast<Npp32u *>(oDeviceDst32u.data()), oDeviceDst32u.pitch(),
                                                                  oDeviceDst8u.data(), oDeviceDst8u.pitch(),
                                                                  oSizeROI, oSizeROI.width * oSizeROI.height,
                                                                  &maxLabel, pScratchBufferNPP) );
        }



        // free scratch buffer memory
        cudaFree(pScratchBufferNPP);

        // Since the maximum label value in maxlabel after label compression in this particular sample case is still way over 256 the output image
        // will contain some incorrect labels due to rollover resulting in incorrect intensity values for some segments.  Unfortunately freeImage does
        // not support either 32-bit or 16-bit single channel grayscale image output.
        //   
        // use a value of 255 for the segment boundary value.

        NPP_CHECK_NPP (
                           nppiBoundSegments_8u_C1IR(oDeviceDst8u.data(), oDeviceDst8u.pitch(), oSizeROI, 255 ) );

        // Declare a host image for the result
        npp::ImageCPU_8u_C1 oHostDst8u(oDeviceDst8u.size());
        // and copy the device result data into it
        oDeviceDst8u.copyTo(oHostDst8u.data(), oHostDst8u.pitch());

        saveImage(sResultFilename, oHostDst8u);
        std::cout << "Saved image: " << sResultFilename << std::endl;

        nppiFree(oDeviceSrc.data());
        nppiFree(oDeviceDst32u.data());
        nppiFree(oDeviceDst8u.data());

        exit(EXIT_SUCCESS);
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;
}
