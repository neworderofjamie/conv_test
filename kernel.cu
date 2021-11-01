// Standard C++ includes
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

// Standard C includes
#include <cassert>
#include <cmath>

// CUDA includes
#include <cuda_runtime.h>

// OpenCV includes
#include <opencv2/opencv.hpp>

//------------------------------------------------------------------------
// Macros
//------------------------------------------------------------------------
#define SEED 124

#define CHECK_CUDA_ERRORS(call) {                                                                   \
    cudaError_t error = call;                                                                       \
    if (error != cudaSuccess) {                                                                     \
            std::ostringstream errorMessageStream;                                                  \
            errorMessageStream << "cuda error:" __FILE__ << ": " << __LINE__ << " ";                \
            errorMessageStream << cudaGetErrorString(error) << "(" << error << ")" << std::endl;    \
            throw std::runtime_error(errorMessageStream.str());                                     \
        }                                                                                           \
    }

template<typename T>
using HostDeviceArray = std::pair < T*, T* > ;

enum Mode
{
    ModeProcedural,
    ModeToeplitz,
    ModeMax,
};

// Convolution kernels
const float s_KernelTemplates[9][4] = {
    {1.0f,  1.0f,   -1.0f,  0.1111f},
    {0.0f,  2.0f,   -1.0f,  0.1111f},
    {-1.0f, 1.0f,   -1.0f,  0.1111f},
    {2.0f,  0.0f,   -1.0f,  0.1111f},
    {0.0f,  0.0f,    8.0f,  0.1111f},
    {-2.0f, 0.0f,   -1.0f,  0.1111f},
    {1.0f,  -1.0f,  -1.0f,  0.1111f},
    {0.0f,  -2.0f,  -1.0f,  0.1111f},
    {-1.0f, -1.0f,  -1.0f,  0.1111f}};
/*
const float s_Kernel[9][3] = {
    { 1.    ,  1.    , -1},
    { 0.    ,  2.    , -1},
    {-1.    ,  1.    , -1},
    { 2.    ,  0.    , -1},
    { 0.    ,  0.    ,  8},
    {-2.    ,  0.    , -1},
    { 1.    , -1.    , -1},
    { 0.    , -2.    , -1},
    {-1.    , -1.    , -1}};*/
const char *const s_ModeNames[] = {
    "Procedural",
    "Toeplitz"};

//-----------------------------------------------------------------------------
// Kernels
//-----------------------------------------------------------------------------
template<int ConvKH, int ConvKW,
         int ConvIW, int ConvIC,
         int ConvOH, int ConvOW, int ConvOC>
__global__ void procedural(unsigned int numInSpikes, const unsigned int *d_inSpikes,
                           const float *d_kernel, float *d_outCurrents)
{
    const unsigned int spike = threadIdx.x + (blockIdx.x * blockDim.x);

    if (spike < numInSpikes) {
        const unsigned int preInd = d_inSpikes[spike];
        const int inRow = (preInd / ConvIC) / ConvIW;
        const int inCol = (preInd / ConvIC) % ConvIW;
        const int inChan = preInd % ConvIC;
        const int minOutRow = min(ConvOH, max(0, 1 + (inRow - ConvKH)));
        const int maxOutRow = min(ConvOH, max(0, 1 + inRow));
        const int minOutCol = min(ConvOW, max(0, 1 + (inCol - ConvKW)));
        const int maxOutCol = min(ConvOW, max(0, 1 + inCol));
        for(int outRow = minOutRow; outRow < maxOutRow; outRow++) {
            const int kernRow = inRow - outRow;
            for(int outCol = minOutCol; outCol < maxOutCol; outCol++) {
                const int kernCol = inCol - outCol;
                for(int outChan = 0; outChan < ConvOC; outChan++) {
                    const int idPost = ((outRow * ConvOW * ConvOC) +
                                        (outCol * ConvOC) +
                                        outChan);
                    const unsigned int kernelInd = (kernRow * ConvKW * ConvIC * ConvOC) + (kernCol * ConvIC * ConvOC) + (inChan * ConvOC) + outChan;
                    atomicAdd(&d_outCurrents[idPost], d_kernel[kernelInd]);
                }
            }
        }
    }
}

template<int ConvK, int ConvI, int ConvIC, int ConvO, int ConvOC>
__global__ void toeplitz(unsigned int numInSpikes, const unsigned int *d_inSpikes,
                         const float *d_kernel, float *d_outCurrents)
{
    extern __shared__ unsigned int s_buffer[];
    unsigned int *s_spike = &s_buffer[0];

    const int id = threadIdx.x + (blockIdx.x * blockDim.x);

    // **BEGIN COLUMN STATE VARIABLES**
    // Split id into kernel row, column and output channel
    const int kernRow = (id / ConvOC) / ConvK;
    const int kernCol = (id / ConvOC) % ConvK;
    const int kernOutChan = id % ConvOC;
    
    // From these, calculate partial (without input channel) kernel index
    const int kernelInd = (kernRow * ConvK * ConvIC * ConvOC) + (kernCol * ConvIC * ConvOC) + kernOutChan;

    // **END COLUMN STATE VARIABLES**

    // Calculate number of blocks (dictated by shared memory) spikes need to be processed in
    const unsigned int numSpikeBlocks = (numInSpikes + blockDim.x - 1) / blockDim.x;

    // Loop through spikes blocks
    for (unsigned int b = 0; b < numSpikeBlocks; b++) {
        // Determine how many spikes are in this block
        const unsigned int numSpikesInBlock = (b == (numSpikeBlocks - 1))
            ? ((numInSpikes - 1) % blockDim.x) + 1 : blockDim.x;
     
        __syncthreads();
            
        // Use first threads in block to read spikes and row lengths into shared memory
        if (threadIdx.x < numSpikesInBlock) {
            s_spike[threadIdx.x] = d_inSpikes[(b * blockDim.x) + threadIdx.x];
        }

        __syncthreads();

        // If there is a kernel entry for this thread to process
        // **NOTE** maxRowLength = ConvO * ConvO * ConvOC
        if(id < (ConvO * ConvO * ConvOC)) {
            // Loop through spikes in block
            for(unsigned int s = 0; s < numSpikesInBlock; s++) {
                // **BEGIN ROW STATE VARIABLES**
                // Split pre into row, column and channel
                // **NOTE** this COULD be done once and written to shared memory
                const int preRow = (s_spike[s] / ConvIC) / ConvI;
                const int preCol = (s_spike[s] / ConvIC) % ConvI;
                const int preChan = s_spike[s] % ConvIC;
                
                // **END ROW STATE VARIABLES**
                
                // **BEGIN DIAGONAL GENERATE CODE**
                // If we haven't gone off edge of output
                const int postRow = preRow + kernRow;
                const int postCol = preCol + kernCol;
                if(postRow < ConvO && kernCol < (ConvO - preCol)) {                    
                    // Read kernel value
                    // **NOTE** if we were only processing a single input channel this could be lifted right out
                    const float kernelVal = -d_kernel[kernelInd + (preChan * ConvOC)];
        
                    // Calculate postsynaptic index
                    const int postInd = ((postRow * ConvO * ConvOC) +
                                         (postCol * ConvOC) +
                                         kernOutChan);
                    
                    // Update output (coalesced reading of filter row and no collisions on atomic add)
                    atomicAdd(&d_outCurrents[postInd], kernelVal);
                }

                // **END DIAGONAL GENERATE CODE**
            }
        }
        
    }
}


//-----------------------------------------------------------------------------
// Host functions
//-----------------------------------------------------------------------------
//! Divide two integers, rounding up i.e. effectively taking ceil
template<typename T>
constexpr T ceilDivide(T numerator, T denominator)
{
    return ((numerator + denominator - 1) / denominator);
}
//-----------------------------------------------------------------------------
template<typename T>
HostDeviceArray<T> allocateHostDevice(unsigned int count)
{
    T *array = nullptr;
    T *d_array = nullptr;
    CHECK_CUDA_ERRORS(cudaMallocHost(&array, count * sizeof(T)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_array, count * sizeof(T)));

    return std::make_pair(array, d_array);
}
//-----------------------------------------------------------------------------
template<typename T>
void hostToDeviceCopy(HostDeviceArray<T> &array, unsigned int count, bool deleteHost=false)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(array.second, array.first, sizeof(T) * count, cudaMemcpyHostToDevice));
    if (deleteHost) {
        CHECK_CUDA_ERRORS(cudaFreeHost(array.first));
        array.first = nullptr;
    }
}
//-----------------------------------------------------------------------------
template<typename T>
void deviceToHostCopy(HostDeviceArray<T> &array, unsigned int count)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(array.first, array.second, count * sizeof(T), cudaMemcpyDeviceToHost));
}
//-----------------------------------------------------------------------------
std::vector<unsigned int> generateSpikes(int numChannels, int width, int height)
{
    auto img = cv::imread("test.png", cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Could not load input image");
    }
    
    // Resize image
    cv::resize(img, img, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    
    // Split image into channels
    cv::Mat channels[3];
    cv::split(img, channels);
    
    // Count spikes that will be required to represent each channel
    const int numChannelSpikes[3] = {cv::countNonZero(channels[0]),
                                     cv::countNonZero(channels[1]),
                                     cv::countNonZero(channels[2])};
    
    // Sum up spike counts until all desired input channels are populated
    int numSpikes = 0;
    for(int c = 0; c < numChannels; c++) {
        numSpikes += numChannelSpikes[c % 3];
    }
    
    // Reserve spike vector
    std::vector<unsigned int> spikes;
    spikes.reserve(numSpikes);
    
    for(int c = 0; c < numChannels; c++) {
        const cv::Mat &img = channels[c % 3];
        for(int i = 0; i < img.rows; i++) {
            for(int j = 0; j < img.cols; j++) {
                if(img.at<uint8_t>(i, j) != 0) {
                    spikes.push_back((i * img.cols * numChannels) +
                                     (j * numChannels) +
                                      c);
                }
            }
        }
    }
    
    return spikes;
}
//-----------------------------------------------------------------------------
template<int convKW, int convKH>
std::vector<float> generateKernels(int numInChannels, int numOutChannels)
{
    // Check kernel sizes are correct
    constexpr int kernelSize = convKW * convKH;
    constexpr int templateKernelSize = sizeof(s_KernelTemplates) / sizeof(s_KernelTemplates[0]);
    assert(kernelSize == templateKernelSize);

    // Determine how many times kernels should be repeated
    constexpr int numTemplates = sizeof(s_KernelTemplates[0]) / sizeof(s_KernelTemplates[0][0]);
    const int numChannels = numInChannels * numOutChannels;
    const int numRepeats = ceilDivide(numChannels, numTemplates);

    std::vector<float> kernels;
    kernels.reserve(kernelSize * numChannels);
    
    // Loop through kernel size (rows of s_KernelTemplates)
    for(int i = 0; i < kernelSize; i++) {
        // Loop through number of repeats we need to make
        for(int j = 0; j < numRepeats; j++) {
            // Calculate number of channels to copy
            const unsigned int numChannelsToCopy = (j == (numRepeats - 1))
                ? ((numChannels - 1) % numTemplates) + 1 : numTemplates;
            
            // Copy from template into vector
            std::copy_n(s_KernelTemplates[i], numChannelsToCopy, std::back_inserter(kernels));
        }
    }

    return kernels;
}
//-----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    try
    {
        constexpr int blockSize = 128;
        constexpr int convKH = 3;
        constexpr int convKW = 3;
        constexpr int convIH = 64;
        constexpr int convIW = 64;
        constexpr int convIC = 3;
        constexpr int convOH = 62;
        constexpr int convOW = 62;
        constexpr int convOC = 100;
        constexpr unsigned int numSpikesPerTimestep = 100;

        // Calculate sizes of kernels and neuron populations
        constexpr int numPre = convIH * convIW * convIC;
        constexpr int numPost = convOH * convOW * convOC;

        // Read mode from command line
        Mode mode;
        if(argc < 2) {
            std::cerr << "Expected parameters specifying:" << std::endl;
            std::cerr << "\t Mode (";
            for(int m = 0; m < ModeMax; m++) {
                std::cerr << m << " = " << s_ModeNames[m];
                if(m != (ModeMax - 1)) {
                    std::cerr << ", ";
                }
            }
            std::cerr << ")" << std::endl;
            return EXIT_FAILURE;
        }
        else {
            mode = (Mode)std::stoul(argv[1]);
        }


        // Generate spikes and kernels
        const auto spikes = generateSpikes(convIC, convIW, convIH);        
        const auto kernels = generateKernels<convKW, convKH>(convIC, convOC);
        
        // Calculate required timesteps
        const unsigned int numTimesteps =  ceilDivide((unsigned int)spikes.size(), numSpikesPerTimestep);

        // Calculate remaining spikes to process in last timestep
        const unsigned int lastTimestepSpikes = spikes.size() - ((numTimesteps - 1) * numSpikesPerTimestep);
        
        std::cout << "Mode:" << s_ModeNames[mode] << ", Num timesteps:" << numTimesteps << std::endl;
    
        CHECK_CUDA_ERRORS(cudaSetDevice(0));

        //------------------------------------------------------------------------
        // Create timing events
        //------------------------------------------------------------------------
        cudaEvent_t kernelStartEvent;
        cudaEvent_t kernelEndEvent;
        double kernelTime = 0.0;

        CHECK_CUDA_ERRORS(cudaEventCreate(&kernelStartEvent));
        CHECK_CUDA_ERRORS(cudaEventCreate(&kernelEndEvent));

        //------------------------------------------------------------------------
        // Configure kernels and spikes
        //------------------------------------------------------------------------
        // Create arrays to hold post-synaptic currents
        auto outCurrents = allocateHostDevice<float>(numPost);
        std::fill_n(&outCurrents.first[0], numPost, 0.0f);
        hostToDeviceCopy(outCurrents, numPost);

        // Create device array for kernels and copy in global data
        float *d_kernel = nullptr;
        CHECK_CUDA_ERRORS(cudaMalloc(&d_kernel, kernels.size() * sizeof(float)));
        CHECK_CUDA_ERRORS(cudaMemcpy(d_kernel, kernels.data(), kernels.size() * sizeof(float), cudaMemcpyHostToDevice));
        //CHECK_CUDA_ERRORS(cudaMalloc(&d_kernel, kernelSize * sizeof(float)));
        //CHECK_CUDA_ERRORS(cudaMemcpy(d_kernel, s_Kernel, kernelSize * sizeof(float), cudaMemcpyHostToDevice));

        // Create device array for spikes and copy in global data
        unsigned int *d_spikes = nullptr;
        CHECK_CUDA_ERRORS(cudaMalloc(&d_spikes, spikes.size() * sizeof(unsigned int)));
        CHECK_CUDA_ERRORS(cudaMemcpy(d_spikes, spikes.data(), spikes.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));

        {
            // Loop through time
            for(unsigned int t = 0; t < numTimesteps; t++) {
                const unsigned int numTimestepSpikes = (t == (numTimesteps - 1)) ? lastTimestepSpikes : numSpikesPerTimestep;

                CHECK_CUDA_ERRORS(cudaEventRecord(kernelStartEvent));

                if(mode == ModeProcedural) {
                    // Calculate number of presynaptically parallelised blocks are required to handle poisson spikes
                    constexpr unsigned int numPreSynapseBlocks = ceilDivide(numPre, blockSize);

                    dim3 threads(blockSize, 1);
                    dim3 grid(numPreSynapseBlocks, 1);

                    procedural<convKH, convKW, convIW, convIC, convOH, convOW, convOC><<<grid, threads>>> (
                        numTimestepSpikes, &d_spikes[t * numSpikesPerTimestep],
                        d_kernel, outCurrents.second);
                }
                else if(mode == ModeToeplitz) {
                    assert(convKH == convKW);
                    assert(convIW == convIH);
                    assert(convOW == convOH);

                    constexpr unsigned int numPostSynapseBlocks = ceilDivide(convKW * convKH, blockSize);
                    constexpr unsigned int sharedBytes = blockSize * sizeof(unsigned int);
                    
                    dim3 threads(blockSize, 1);
                    dim3 grid(numPostSynapseBlocks, 1);
                    toeplitz<convKH, convIH, convIC, convOH, convOC><<<grid, threads, sharedBytes>>>(
                        numTimestepSpikes, &d_spikes[t * numSpikesPerTimestep],
                        d_kernel, outCurrents.second);
                }
                CHECK_CUDA_ERRORS(cudaEventRecord(kernelEndEvent));
                CHECK_CUDA_ERRORS(cudaEventSynchronize(kernelEndEvent));

                float tmp;
                CHECK_CUDA_ERRORS(cudaEventElapsedTime(&tmp, kernelStartEvent, kernelEndEvent));
                kernelTime += tmp;
            }
        }
        deviceToHostCopy(outCurrents, numPost);

        std::cout << "Kernel time:" << kernelTime << " ms" << std::endl;
        std::ofstream outCurrentsFile("outCurrents" + std::string(s_ModeNames[mode]) + ".bin", std::ios_base::binary);
        outCurrentsFile.write(reinterpret_cast<const char*>(outCurrents.first), sizeof(float) * numPost);
    }
    catch(std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

