/*
 * Copyright 2018-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
  
// This sample needs at least CUDA 10.1. It demonstrates usages of the nvJPEG
// library nvJPEG encoder supports single and multiple image encode.

#include <cuda_runtime_api.h>
#include "helper_nvJPEG.hxx"


int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }
int dev_free(void *p) { return (int)cudaFree(p); }

StopWatchInterface *timer = NULL;

bool is_interleaved(nvjpegOutputFormat_t format)
{
    if (format == NVJPEG_OUTPUT_RGBI || format == NVJPEG_OUTPUT_BGRI)
        return true;
    else
        return false;
}

struct encode_params_t {
  std::string input_dir;
  std::string output_dir;
  std::string format;
  std::string subsampling;
  int quality;
  int dev;
};

nvjpegEncoderParams_t encode_params;
nvjpegHandle_t nvjpeg_handle;
nvjpegJpegState_t jpeg_state;
nvjpegEncoderState_t encoder_state;

int decodeEncodeOneImage(std::string sImagePath, std::string sOutputPath, double &time, nvjpegOutputFormat_t output_format, nvjpegInputFormat_t input_format)
{
    time = 0.;
    // Get the file name, without extension.
    // This will be used to rename the output file.    
    size_t position = sImagePath.rfind("/");
    std::string sFileName = (std::string::npos == position)? sImagePath : sImagePath.substr(position + 1, sImagePath.size());
    position = sFileName.rfind(".");
    sFileName = (std::string::npos == position)? sFileName : sFileName.substr(0, position);
    position = sFileName.rfind("/");
    sFileName = (std::string::npos == position) ? sFileName : sFileName.substr(position + 1, sFileName.length());
    position = sFileName.rfind("\\");
    sFileName = (std::string::npos == position) ? sFileName : sFileName.substr(position+1, sFileName.length());

    // Read an image from disk.
    std::ifstream oInputStream(sImagePath.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    if(!(oInputStream.is_open()))
    {
        std::cerr << "Cannot open image: " << sImagePath << std::endl;
        return 1;
    }
    
    // Get the size.
    std::streamsize nSize = oInputStream.tellg();
    oInputStream.seekg(0, std::ios::beg);

    // Image buffers. 
    unsigned char * pBuffer = NULL; 
    double decode_time = 0.;
    
    std::vector<char> vBuffer(nSize);
    
    if (oInputStream.read(vBuffer.data(), nSize))
    {            
        unsigned char * dpImage = (unsigned char *)vBuffer.data();
        
        // Retrieve the componenet and size info.
        int nComponent = 0;
        nvjpegChromaSubsampling_t subsampling;
        int widths[NVJPEG_MAX_COMPONENT];
        int heights[NVJPEG_MAX_COMPONENT];
        if (NVJPEG_STATUS_SUCCESS != nvjpegGetImageInfo(nvjpeg_handle, dpImage, nSize, &nComponent, &subsampling, widths, heights))
        {
            std::cerr << "Error decoding JPEG header: " << sImagePath << std::endl;
            return 1;
        }

        // image information
        std::cout << "Image is " << nComponent << " channels." << std::endl;
        for (int i = 0; i < nComponent; i++)
        {
            std::cout << "Channel #" << i << " size: "  << widths[i]  << " x " << heights[i] << std::endl;    
        }
        
        switch (subsampling)
        {
            case NVJPEG_CSS_444:
                std::cout << "YUV 4:4:4 chroma subsampling" << std::endl;
                break;
            case NVJPEG_CSS_440:
                std::cout << "YUV 4:4:0 chroma subsampling" << std::endl;
                break;
            case NVJPEG_CSS_422:
                std::cout << "YUV 4:2:2 chroma subsampling" << std::endl;
                break;
            case NVJPEG_CSS_420:
                std::cout << "YUV 4:2:0 chroma subsampling" << std::endl;
                break;
            case NVJPEG_CSS_411:
                std::cout << "YUV 4:1:1 chroma subsampling" << std::endl;
                break;
            case NVJPEG_CSS_410:
                std::cout << "YUV 4:1:0 chroma subsampling" << std::endl;
                break;
            case NVJPEG_CSS_GRAY:
                std::cout << "Grayscale JPEG " << std::endl;
                break;
            case NVJPEG_CSS_UNKNOWN: 
                std::cout << "Unknown chroma subsampling" << std::endl;
                return 1;
        }

        {

            cudaError_t eCopy = cudaMalloc(&pBuffer, widths[0] * heights[0] * NVJPEG_MAX_COMPONENT);
            if(cudaSuccess != eCopy) 
            {
                std::cerr << "cudaMalloc failed for component Y: " << cudaGetErrorString(eCopy) << std::endl;
                return 1;
            }

            nvjpegImage_t imgdesc = 
            {
                {
                    pBuffer,
                    pBuffer + widths[0]*heights[0],
                    pBuffer + widths[0]*heights[0]*2,
                    pBuffer + widths[0]*heights[0]*3
                },
                {
                    (unsigned int)(is_interleaved(output_format) ? widths[0] * 3 : widths[0]),
                    (unsigned int)widths[0],
                    (unsigned int)widths[0],
                    (unsigned int)widths[0]
                }
            };
           
            int nReturnCode = 0;

            cudaDeviceSynchronize();
            // Create the CUTIL timer
            sdkCreateTimer(&timer);
            sdkStartTimer(&timer);
            nReturnCode = nvjpegDecode(nvjpeg_handle, jpeg_state, dpImage, nSize, output_format, &imgdesc, NULL);

            // alternatively decode by stages
            /*int nReturnCode = nvjpegDecodeCPU(nvjpeg_handle, dpImage, nSize, output_format, &imgdesc, NULL);
            nReturnCode = nvjpegDecodeMixed(nvjpeg_handle, NULL);
            nReturnCode = nvjpegDecodeGPU(nvjpeg_handle, NULL);*/
            cudaDeviceSynchronize();
            sdkStopTimer(&timer);
            decode_time =sdkGetTimerValue(&timer);
            if(nReturnCode != 0)
            {
                std::cerr << "Error in nvjpegDecode." << std::endl;
                return 1;
            }

            /////////////////////// encode ////////////////////
            if (NVJPEG_OUTPUT_YUV == output_format)
            {
                checkCudaErrors(nvjpegEncodeYUV(nvjpeg_handle,
                    encoder_state,
                    encode_params,
                    &imgdesc,
                    subsampling,
                    widths[0],
                    heights[0],
                    NULL));
            }
            else
            {
                checkCudaErrors(nvjpegEncodeImage(nvjpeg_handle,
                    encoder_state,
                    encode_params,
                    &imgdesc,
                    input_format,
                    widths[0],
                    heights[0],
                    NULL));
            }

            std::vector<unsigned char> obuffer;
            size_t length;
            checkCudaErrors(nvjpegEncodeRetrieveBitstream(
                nvjpeg_handle,
                encoder_state,
                NULL,
                &length,
                NULL));
            obuffer.resize(length);
            checkCudaErrors(nvjpegEncodeRetrieveBitstream(
                nvjpeg_handle,
                encoder_state,
                obuffer.data(),
                &length,
                NULL));
            std::string output_filename = sOutputPath + "/" + sFileName + ".jpg";
            char directory[120];
            char mkdir_cmd[256];
            std::string folder = sOutputPath;
            output_filename = folder + "/"+ sFileName +".jpg";
#if !defined(_WIN32)
            sprintf(directory, "%s", folder.c_str());
            sprintf(mkdir_cmd, "mkdir -p %s 2> /dev/null", directory);
#else
            sprintf(directory, "%s", folder.c_str());
            sprintf(mkdir_cmd, "mkdir %s 2> nul", directory);
#endif

            int ret = system(mkdir_cmd);

            std::cout << "Writing JPEG file: " << output_filename << std::endl;
            std::ofstream outputFile(output_filename.c_str(), std::ios::out | std::ios::binary);
            outputFile.write(reinterpret_cast<const char *>(obuffer.data()), static_cast<int>(length));
            
            // Free memory
            checkCudaErrors(cudaFree(pBuffer));
        }
    }

    time = decode_time;

    return 0;
}

int processArgs(encode_params_t param)
{
    std::string sInputPath(param.input_dir);
    std::string sOutputPath(param.output_dir);
    std::string sFormat(param.format);
    std::string sSubsampling(param.subsampling);
    nvjpegOutputFormat_t oformat = NVJPEG_OUTPUT_RGB;
    nvjpegInputFormat_t iformat = NVJPEG_INPUT_RGB;

    int error_code = 1;

    if (sFormat == "yuv")
    {
        oformat = NVJPEG_OUTPUT_YUV;
    } 
    else if (sFormat == "rgb")
    {
        oformat = NVJPEG_OUTPUT_RGB;
        iformat = NVJPEG_INPUT_RGB;
    }
    else if (sFormat == "bgr")
    {
        oformat = NVJPEG_OUTPUT_BGR;
        iformat = NVJPEG_INPUT_BGR;
    }
    else if (sFormat == "rgbi")
    {
        oformat = NVJPEG_OUTPUT_RGBI;
        iformat = NVJPEG_INPUT_RGBI;
    }
    else if (sFormat == "bgri")
    {
        oformat = NVJPEG_OUTPUT_BGRI;
        iformat = NVJPEG_INPUT_BGRI;
    }
    else 
    {
        std::cerr << "Unknown or unsupported output format: " << sFormat << std::endl;
        return error_code;
    }

    if (sSubsampling == "444")
    {
        checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_444, NULL));
    }
    else if (sSubsampling == "422")
    {
        checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_422, NULL));
    }
    else if (sSubsampling == "420")
    {
        checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_420, NULL));
    }
    else if (sSubsampling == "440")
    {
        checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_440, NULL));
    }
    else if (sSubsampling == "411")
    {
        checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_411, NULL));
    }
    else if (sSubsampling == "410")
    {
        checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_410, NULL));
    }
    else if (sSubsampling == "400")
    {
        checkCudaErrors(nvjpegEncoderParamsSetSamplingFactors(encode_params, NVJPEG_CSS_GRAY, NULL));
    }
    else 
    {
        std::cerr << "Unknown or unsupported subsampling: " << sSubsampling << std::endl;
        return error_code;
    }
    /*if( stat(sOutputPath.c_str(), &s) == 0 )
    {
        if( !(s.st_mode & S_IFDIR) )
        {
            std::cout << "Output path already exist as non-directory: " << sOutputPath << std::endl;
            return error_code;
        }
    }
    else
    {
        if (mkdir(sOutputPath.c_str(), 0775))
        {
            std::cout << "Cannot create output directory: " << sOutputPath << std::endl;
            return error_code;
        }
    }*/

    std::vector<std::string> inputFiles;
    if (readInput(sInputPath, inputFiles))
    {
        return error_code;
    }
    
    double total_time = 0., decode_time = 0.;
    int total_images = 0;

    for (unsigned int i = 0; i < inputFiles.size(); i++)
    {
        std::string &sFileName = inputFiles[i];
        std::cout << "Processing file: " << sFileName << std::endl;
        int image_error_code = decodeEncodeOneImage(sFileName, sOutputPath, decode_time, oformat, iformat);
        if (image_error_code)
        {
            std::cerr << "Error processing file: " << sFileName << std::endl;
            //return image_error_code;
        }
        else
        {
            total_images++;
            total_time += decode_time;
        }                      
    }
    std::cout << "Total images processed: " << total_images << std::endl;
    std::cout << "Total time spent on decoding: " << total_time << std::endl;
    std::cout << "Avg time/image: " << total_time/total_images << std::endl;

    return 0;
}

// parse parameters
int findParamIndex(const char **argv, int argc, const char *parm) {
  int count = 0;
  int index = -1;

  for (int i = 0; i < argc; i++) {
    if (strncmp(argv[i], parm, 100) == 0) {
      index = i;
      count++;
    }
  }

  if (count == 0 || count == 1) {
    return index;
  } else {
    std::cout << "Error, parameter " << parm
              << " has been specified more than once, exiting\n"
              << std::endl;
    return -1;
  }

  return -1;
}


int main(int argc, const char *argv[]) 
{
  int pidx;

  if ((pidx = findParamIndex(argv, argc, "-h")) != -1 ||
      (pidx = findParamIndex(argv, argc, "--help")) != -1) {
    std::cout << "Usage: " << argv[0]
              << " -i images_dir  [-o output_dir] [-device=device_id]"                 
                 "[-q quality][-s 420/444] [-fmt output_format]\n";
    std::cout << "Parameters: " << std::endl;
    std::cout << "\timages_dir\t:\tPath to single image or directory of images" << std::endl;
    std::cout << "\toutput_dir\t:\tWrite encoded images as jpeg to this directory" << std::endl;
    std::cout << "\tdevice_id\t:\tWhich device to use for encoding" << std::endl;
    std::cout << "\tQuality\t:\tUse image quality [default 70]" << std::endl;
    std::cout << "\tsubsampling\t:\tUse Subsampling [420, 444]" << std::endl;
    std::cout << "\toutput_format\t:\tnvJPEG output format for encoding. One "
                 "of [rgb, rgbi, bgr, bgri, yuv, y, unchanged]"
              << std::endl;
    return EXIT_SUCCESS;
  }

  encode_params_t params;

  params.input_dir = "./";
  if ((pidx = findParamIndex(argv, argc, "-i")) != -1) {
    params.input_dir = argv[pidx + 1];
  } else {
    // Search in default paths for input images.
    int found = getInputDir(params.input_dir, argv[0]);
    if (!found)
    {
      std::cout << "Please specify input directory with encoded images"<< std::endl;
      return EXIT_WAIVED;
    }
  }
  if ((pidx = findParamIndex(argv, argc, "-o")) != -1) {
    params.output_dir = argv[pidx + 1];
  } else {
      // by-default write the folder named "output" in cwd
      params.output_dir = "encode_output";
  }
  params.dev = 0;
  params.dev = findCudaDevice(argc, argv);

  params.quality = 70;
  if ((pidx = findParamIndex(argv, argc, "-q")) != -1) {
    params.quality = std::atoi(argv[pidx + 1]);
  }

  if ((pidx = findParamIndex(argv, argc, "-s")) != -1) {
    params.subsampling = argv[pidx + 1];
  } else {
      // by-default use subsampling as 420
      params.subsampling = "420";
  }
  if ((pidx = findParamIndex(argv, argc, "-fmt")) != -1) {
    params.format = argv[pidx + 1];
  } else {
   // by-default use output format yuv
    params.format = "yuv";
  }

    cudaDeviceProp props;
    checkCudaErrors(cudaGetDeviceProperties(&props, params.dev));

    printf("Using GPU %d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
         params.dev, props.name, props.multiProcessorCount,
         props.maxThreadsPerMultiProcessor, props.major, props.minor,
         props.ECCEnabled ? "on" : "off");

    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    checkCudaErrors(nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &nvjpeg_handle));
    checkCudaErrors(nvjpegJpegStateCreate(nvjpeg_handle, &jpeg_state));
    checkCudaErrors(nvjpegEncoderStateCreate(nvjpeg_handle, &encoder_state, NULL));
    checkCudaErrors(nvjpegEncoderParamsCreate(nvjpeg_handle, &encode_params, NULL));
    
    // sample input parameters
    checkCudaErrors(nvjpegEncoderParamsSetQuality(encode_params, params.quality, NULL));
    checkCudaErrors(nvjpegEncoderParamsSetOptimizedHuffman(encode_params, 1, NULL));

    pidx = processArgs(params);

    checkCudaErrors(nvjpegEncoderParamsDestroy(encode_params));
    checkCudaErrors(nvjpegEncoderStateDestroy(encoder_state));
    checkCudaErrors(nvjpegJpegStateDestroy(jpeg_state));
    checkCudaErrors(nvjpegDestroy(nvjpeg_handle));

    return pidx;
}
