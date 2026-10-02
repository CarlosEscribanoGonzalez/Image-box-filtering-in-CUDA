#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

#pragma region Variables and defines
constexpr auto BLOCK_SIZE = 16; //Being dim3, it will be 16*16 = 256
constexpr auto SOFT_THRESHOLD = 0.025;
constexpr auto STRONG_THRESHOLD = 0.07;
constexpr auto SOFT_VALUE = 25; //Value of "soft" pixels
constexpr auto STRONG_VALUE = 50; //Value of "strong" pixels
constexpr auto OUTLINE_VALUE = 255; //Outline value

//#define _CONSTANT_MEMORY 
//#define _SHARED_MEMORY_CONV //Only for convolutions
//#define _SHARED_MEMORY_HYST //Only for hysteresis and outline
#define _CANNY_EDGE
#define _OUTLINE

int FILTERSIZE = 5;
unsigned char* d_red, * d_green, * d_blue, * d_gray, * d_grayFiltered, * d_classifiedValues, * d_hysteresis, * d_outline;
float* d_sobel_x, * d_sobel_y, * d_magnitude, * d_nms_output, * d_dir, * d_input_max, * d_output_max;
//Filter in constants memory:
__constant__ float d_filter_constant[81];
#pragma endregion

#pragma region Kernels
template<typename T>
__global__
void convolution(const unsigned char* const inputChannel,
    T* const outputChannel,
    int numRows, int numCols,
    const float* const filter, const int filterWidth)
{
    extern __shared__ float tile[];
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = blockIdx.y * blockDim.y + threadIdx.y;
    int filterRadius = filterWidth / 2;
#ifdef _SHARED_MEMORY_CONV
    //Center
    int image_x = min((int)id_x, numCols - 1);
    int image_y = min((int)id_y, numRows - 1);
    int tileIdx = (threadIdx.x + filterRadius) + (threadIdx.y + filterRadius) * (BLOCK_SIZE + filterWidth - 1);
    tile[tileIdx] = inputChannel[image_x + image_y * numCols];
    //Left edge
    if (threadIdx.x < filterRadius) {
        tileIdx = threadIdx.x + (threadIdx.y + filterRadius) * (BLOCK_SIZE + filterWidth - 1);
        tile[tileIdx] = inputChannel[max(image_x - filterRadius, 0) + image_y * numCols];
    }
    //Right edge
    if (threadIdx.x >= BLOCK_SIZE - filterRadius) {
        tileIdx = (threadIdx.x + 2 * filterRadius) + (threadIdx.y + filterRadius) * (BLOCK_SIZE + filterWidth - 1);
        tile[tileIdx] = inputChannel[min(image_x + filterRadius, numCols - 1) + image_y * numCols];
    }
    //Upper edge
    if (threadIdx.y < filterRadius) {
        tileIdx = (threadIdx.x + filterRadius) + threadIdx.y * (BLOCK_SIZE + filterWidth - 1);
        tile[tileIdx] = inputChannel[image_x + max(image_y - filterRadius, 0) * numCols];
    }
    //Lower edge
    if (threadIdx.y >= BLOCK_SIZE - filterRadius) {
        tileIdx = (threadIdx.x + filterRadius) + (threadIdx.y + 2 * filterRadius) * (BLOCK_SIZE + filterWidth - 1);
        tile[tileIdx] = inputChannel[image_x + min(image_y + filterRadius, numRows - 1) * numCols];
    }
    //Top-left corner
    if (threadIdx.x < filterRadius && threadIdx.y < filterRadius) {
        tileIdx = threadIdx.x + threadIdx.y * (BLOCK_SIZE + filterWidth - 1);
        tile[tileIdx] = inputChannel[max(image_x - filterRadius, 0) + max(image_y - filterRadius, 0) * numCols];
    }
    //Top-right corner
    if (threadIdx.x >= BLOCK_SIZE - filterRadius && threadIdx.y < filterRadius) {
        tileIdx = (threadIdx.x + 2 * filterRadius) + threadIdx.y * (BLOCK_SIZE + filterWidth - 1);
        tile[tileIdx] = inputChannel[min(image_x + filterRadius, numCols - 1) + max(image_y - filterRadius, 0) * numCols];
    }
    //Bottom-left corner
    if (threadIdx.x < filterRadius && threadIdx.y >= BLOCK_SIZE - filterRadius) {
        tileIdx = threadIdx.x + (threadIdx.y + 2 * filterRadius) * (BLOCK_SIZE + filterWidth - 1);
        tile[tileIdx] = inputChannel[max(image_x - filterRadius, 0) + min(image_y + filterRadius, numRows - 1) * numCols];
    }
    //Bottom-right corner
    if (threadIdx.x >= BLOCK_SIZE - filterRadius && threadIdx.y >= BLOCK_SIZE - filterRadius) {
        tileIdx = (threadIdx.x + 2 * filterRadius) + (threadIdx.y + 2 * filterRadius) * (BLOCK_SIZE + filterWidth - 1);
        tile[tileIdx] = inputChannel[min(image_x + filterRadius, numCols - 1) + min(image_y + filterRadius, numRows - 1) * numCols];
    }
    __syncthreads();
#endif
    if (id_x >= numCols || id_y >= numRows) return;
    float result = 0.0;
    for (int offset_y = -filterRadius; offset_y <= filterRadius; offset_y++) {
        for (int offset_x = -filterRadius; offset_x <= filterRadius; offset_x++) {
            float image_value;
            float filter_value;
#ifdef _SHARED_MEMORY_CONV
            int tile_x = threadIdx.x + filterRadius + offset_x;
            int tile_y = threadIdx.y + filterRadius + offset_y;
            image_value = tile[tile_x + tile_y * (BLOCK_SIZE + filterWidth - 1)];
#else
            int image_x = id_x + offset_x;
            int image_y = id_y + offset_y;
            image_x = min(max(image_x, 0), numCols - 1);
            image_y = min(max(image_y, 0), numRows - 1);
            image_value = inputChannel[image_x + numCols * image_y];
#endif
#ifdef _CONSTANT_MEMORY
            filter_value = d_filter_constant[(offset_x + filterRadius) + (offset_y + filterRadius) * filterWidth];
#else
            filter_value = filter[(offset_x + filterRadius) + (offset_y + filterRadius) * filterWidth];
#endif
            result += image_value * filter_value;
        }
    }
    if constexpr (std::is_same<T, unsigned char>::value) {
        result = fminf(fmaxf(result, 0), 255);
        outputChannel[id_x + id_y * numCols] = (unsigned char)result;
    }
    else {
        outputChannel[id_x + id_y * numCols] = result;
    }
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
    int numRows,
    int numCols,
    unsigned char* const redChannel,
    unsigned char* const greenChannel,
    unsigned char* const blueChannel)
{
    int id_x = threadIdx.x + blockIdx.x * blockDim.x;
    int id_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (id_x >= numCols || id_y >= numRows) return;
    int global_idx = id_x + id_y * numCols;
    redChannel[global_idx] = inputImageRGBA[global_idx].x;
    greenChannel[global_idx] = inputImageRGBA[global_idx].y;
    blueChannel[global_idx] = inputImageRGBA[global_idx].z;
}

//This kernel takes in three color channels and recombines them
//into one image. The alpha channel is set to 255 to represent
//that this image has no transparency.
template<typename T>
__global__
void recombineChannels(T* const redChannel,
    T* const greenChannel,
    T* const blueChannel,
    uchar4* const outputImageRGBA,
    int numRows,
    int numCols)
{
    const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);

    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
        return;

    unsigned char red = redChannel[thread_1D_pos];
    unsigned char green = greenChannel[thread_1D_pos];
    unsigned char blue = blueChannel[thread_1D_pos];

    //Alpha should be 255 for no transparency
    uchar4 outputPixel = make_uchar4(red, green, blue, 255);

    outputImageRGBA[thread_1D_pos] = outputPixel;
}

__global__
void toGrayScale(uchar4* const d_inputImageRGBA, unsigned char* const output, int numRows, int numCols) 
{
    int id_x = threadIdx.x + blockDim.x * blockIdx.x;
    int id_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (id_x >= numCols || id_y >= numRows) return;
    int global_idx = id_x + id_y * numCols;
    output[global_idx] = 0.299f * d_inputImageRGBA[global_idx].x + 
        0.587f * d_inputImageRGBA[global_idx].y + 0.114f * d_inputImageRGBA[global_idx].z;
}

__global__
void computeMagAndDir(const float* const d_sobel_x, const float* const d_sobel_y,
    float* const d_magnitude, float* const d_dir, int numRows, int numCols) 
{
    int id_x = threadIdx.x + blockDim.x * blockIdx.x;
    int id_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (id_x >= numCols || id_y >= numRows) return;
    int global_idx = id_x + id_y * numCols;
    float gx = d_sobel_x[global_idx];
    float gy = d_sobel_y[global_idx];
    d_magnitude[global_idx] = sqrtf(gx * gx + gy * gy);
    d_dir[global_idx] = atan2f(gy, gx);
}

__global__
void suppressNonMax(float* const d_magnitude, float* const d_nms_output, const float* const d_dir, int numRows, int numCols) {
    int id_x = threadIdx.x + blockDim.x * blockIdx.x;
    int id_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (id_x >= numCols || id_y >= numRows) return;
    int global_idx = id_x + id_y * numCols;
    float local_mag = d_magnitude[global_idx];
    //Calculation of offsets for neighbors 
    float angle = d_dir[global_idx] * 180.0f / CUDART_PI_F; //Angle in degrees
    if (angle < 0) angle += 180.0f; //Angle in [0, 180]
    int offset_x = angle <= 67.5f ? 1 : 0;
    offset_x = angle >= 112.5f ? -1 : offset_x;
    int offset_y = (angle >= 22.5f && angle <= 157.5f) ? 1 : 0;
    //First neighbor:
    int n_x = min(max(id_x + offset_x, 0), numCols - 1);
    int n_y = min(max(id_y + offset_y, 0), numRows - 1);
    if (d_magnitude[n_x + n_y * numCols] > local_mag) {
        d_nms_output[global_idx] = 0;
        return;
    }
    //Second neighbor:
    offset_x *= -1;
    offset_y *= -1;
    n_x = min(max(id_x + offset_x, 0), numCols - 1);
    n_y = min(max(id_y + offset_y, 0), numRows - 1);
    if (d_magnitude[n_x + n_y * numCols] > local_mag) {
        d_nms_output[global_idx] = 0;
        return;
    }
    d_nms_output[global_idx] = local_mag;
}

__global__ 
void findMax(const float* const input, float* output, const size_t N) {
    __shared__ float values[BLOCK_SIZE * BLOCK_SIZE];
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= N) values[threadIdx.x] = -INFINITY;
    else values[threadIdx.x] = input[id];
    __syncthreads();
    //blockDim must be multiple of 2
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s)
            values[threadIdx.x] = max(values[threadIdx.x], values[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) output[blockIdx.x] = values[0];
}

__global__ 
void applyThreshold(const float* const d_magnitude, unsigned char* const d_classifiedValues,
    float strongThreshold, float weakThreshold, int numRows, int numCols)
{
    int id_x = threadIdx.x + blockDim.x * blockIdx.x;
    int id_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (id_x >= numCols || id_y >= numRows) return;
    int global_idx = id_x + id_y * numCols;
    unsigned char output = (d_magnitude[global_idx] >= weakThreshold) ? SOFT_VALUE : 0;
    output = (d_magnitude[global_idx] >= strongThreshold) ? STRONG_VALUE : output;
    d_classifiedValues[global_idx] = output;
}

__global__
void hysteresis(const unsigned char* const d_classifiedValues, unsigned char* const d_hysteresis,
    int numRows, int numCols)
{
    extern __shared__ unsigned char hystTile[];
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = blockIdx.y * blockDim.y + threadIdx.y;
#ifdef _SHARED_MEMORY_HYST
    //Center
    int image_x = min((int)id_x, numCols - 1);
    int image_y = min((int)id_y, numRows - 1);
    int tileIdx = (threadIdx.x + 1) + (threadIdx.y + 1) * (BLOCK_SIZE + 2);
    hystTile[tileIdx] = d_classifiedValues[image_x + image_y * numCols];
    //Left edge
    if (threadIdx.x == 0) {
        tileIdx = (threadIdx.y + 1) * (BLOCK_SIZE + 2);
        hystTile[tileIdx] = d_classifiedValues[max(image_x - 1, 0) + image_y * numCols];
    }
    //Right edge
    if (threadIdx.x == BLOCK_SIZE - 1) {
        tileIdx = (BLOCK_SIZE + 1) + (threadIdx.y + 1) * (BLOCK_SIZE + 2);
        hystTile[tileIdx] = d_classifiedValues[min(image_x + 1, numCols - 1) + image_y * numCols];
    }
    //Upper edge
    if (threadIdx.y == 0) {
        tileIdx = (threadIdx.x + 1);
        hystTile[tileIdx] = d_classifiedValues[image_x + max(image_y - 1, 0) * numCols];
    }
    //Lower edge
    if (threadIdx.y == BLOCK_SIZE - 1) {
        tileIdx = (threadIdx.x + 1) + (BLOCK_SIZE + 1) * (BLOCK_SIZE + 2);
        hystTile[tileIdx] = d_classifiedValues[image_x + min(image_y + 1, numRows - 1) * numCols];
    }
    //Top-left corner
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        tileIdx = 0;
        hystTile[tileIdx] = d_classifiedValues[max(image_x - 1, 0) + max(image_y - 1, 0) * numCols];
    }
    //Top-right corner
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0) {
        tileIdx = (threadIdx.x + 2);
        hystTile[tileIdx] = d_classifiedValues[min(image_x + 1, numCols - 1) + max(image_y - 1, 0) * numCols];
    }
    //Bottom-left corner
    if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1) {
        tileIdx = (BLOCK_SIZE + 1) * (BLOCK_SIZE + 2);
        hystTile[tileIdx] = d_classifiedValues[max(image_x - 1, 0) + min(image_y + 1, numRows - 1) * numCols];
    }
    //Bottom-right corner
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1) {
        tileIdx = (BLOCK_SIZE + 1) + (BLOCK_SIZE + 1) * (BLOCK_SIZE + 2);
        hystTile[tileIdx] = d_classifiedValues[min(image_x + 1, numCols - 1) + min(image_y + 1, numRows - 1) * numCols];
    }
    __syncthreads();
#endif
    if (id_x >= numCols || id_y >= numRows) return;
    int global_idx = id_x + id_y * numCols;
    unsigned char localValue;
#ifdef _SHARED_MEMORY_HYST
    localValue = hystTile[(threadIdx.x + 1) + (threadIdx.y + 1) * (BLOCK_SIZE + 2)];
#else
    localValue = d_classifiedValues[id_x + id_y * numCols];
#endif
    if (localValue != SOFT_VALUE) { //Only "soft" pixels are evaluated
        d_hysteresis[global_idx] = localValue;
        return;
    }
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            unsigned char value;
#ifdef _SHARED_MEMORY_HYST
            int tile_x = min(max(threadIdx.x + 1 + x, 0), BLOCK_SIZE + 1);
            int tile_y = min(max(threadIdx.y + 1 + y, 0), BLOCK_SIZE + 1);
            value = hystTile[tile_x + tile_y * (BLOCK_SIZE + 2)];
#else
            int n_x = min(max(id_x + x, 0), numCols - 1);
            int n_y = min(max(id_y + y, 0), numRows - 1);
            value = d_classifiedValues[n_x + n_y * numCols];
#endif
            if (value == STRONG_VALUE) { //The pixel turns into "strong" if a neighbor is
                d_hysteresis[global_idx] = STRONG_VALUE;
                return;
            }
        }
    }
    d_hysteresis[global_idx] = 0; //The pixel is discarded
}

__global__
void outline(const unsigned char* const d_hysteresis, unsigned char* const d_outline,
    int numRows, int numCols)
{
    extern __shared__ unsigned char outlineTile[];
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = blockIdx.y * blockDim.y + threadIdx.y;
#ifdef _SHARED_MEMORY_HYST
    //Center
    int image_x = min((int)id_x, numCols - 1);
    int image_y = min((int)id_y, numRows - 1);
    int tileIdx = (threadIdx.x + 1) + (threadIdx.y + 1) * (BLOCK_SIZE + 2);
    outlineTile[tileIdx] = d_hysteresis[image_x + image_y * numCols];
    //Left edge
    if (threadIdx.x == 0) {
        tileIdx = (threadIdx.y + 1) * (BLOCK_SIZE + 2);
        outlineTile[tileIdx] = d_hysteresis[max(image_x - 1, 0) + image_y * numCols];
    }
    //Right edge
    if (threadIdx.x == BLOCK_SIZE - 1) {
        tileIdx = (BLOCK_SIZE + 1) + (threadIdx.y + 1) * (BLOCK_SIZE + 2);
        outlineTile[tileIdx] = d_hysteresis[min(image_x + 1, numCols - 1) + image_y * numCols];
    }
    //Upper edge
    if (threadIdx.y == 0) {
        tileIdx = (threadIdx.x + 1);
        outlineTile[tileIdx] = d_hysteresis[image_x + max(image_y - 1, 0) * numCols];
    }
    //Lower edge
    if (threadIdx.y == BLOCK_SIZE - 1) {
        tileIdx = (threadIdx.x + 1) + (BLOCK_SIZE + 1) * (BLOCK_SIZE + 2);
        outlineTile[tileIdx] = d_hysteresis[image_x + min(image_y + 1, numRows - 1) * numCols];
    }
    //Top-left corner
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        tileIdx = 0;
        outlineTile[tileIdx] = d_hysteresis[max(image_x - 1, 0) + max(image_y - 1, 0) * numCols];
    }
    //Top-right corner
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0) {
        tileIdx = (threadIdx.x + 2);
        outlineTile[tileIdx] = d_hysteresis[min(image_x + 1, numCols - 1) + max(image_y - 1, 0) * numCols];
    }
    //Bottom-left corner
    if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1) {
        tileIdx = (BLOCK_SIZE + 1) * (BLOCK_SIZE + 2);
        outlineTile[tileIdx] = d_hysteresis[max(image_x - 1, 0) + min(image_y + 1, numRows - 1) * numCols];
    }
    //Bottom-right corner
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1) {
        tileIdx = (BLOCK_SIZE + 1) + (BLOCK_SIZE + 1) * (BLOCK_SIZE + 2);
        outlineTile[tileIdx] = d_hysteresis[min(image_x + 1, numCols - 1) + min(image_y + 1, numRows - 1) * numCols];
    }
    __syncthreads();
#endif
    if (id_x >= numCols || id_y >= numRows) return;
    int global_idx = id_x + id_y * numCols;
    unsigned char localValue;
#ifdef _SHARED_MEMORY_HYST
    localValue = outlineTile[(threadIdx.x + 1) + (threadIdx.y + 1) * (BLOCK_SIZE + 2)];
#else
    localValue = d_hysteresis[id_x + id_y * numCols];
#endif
    if (localValue != 0) { //In this case outline only works with "irrelevant" pixels
        d_outline[global_idx] = localValue;
        return;
    }
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            unsigned char value;
#ifdef _SHARED_MEMORY_HYST
            int tile_x = min(max(threadIdx.x + 1 + x, 0), BLOCK_SIZE + 1);
            int tile_y = min(max(threadIdx.y + 1 + y, 0), BLOCK_SIZE + 1);
            value = outlineTile[tile_x + tile_y * (BLOCK_SIZE + 2)];
#else
            int n_x = min(max(id_x + x, 0), numCols - 1);
            int n_y = min(max(id_y + y, 0), numRows - 1);
            value = d_hysteresis[n_x + n_y * numCols];
#endif
            if (value == STRONG_VALUE) {
                d_outline[global_idx] = OUTLINE_VALUE;
                return;
            }
        }
    }
    d_outline[global_idx] = 0;
}
#pragma endregion

void allocateMemoryGPU(const size_t numRowsImage, const size_t numColsImage)
{
#ifdef _CANNY_EDGE
    checkCudaErrors(cudaMalloc(&d_gray, sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_grayFiltered, sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_classifiedValues, sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_hysteresis, sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_sobel_x, sizeof(float) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_sobel_y, sizeof(float) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_magnitude, sizeof(float) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_nms_output, sizeof(float) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_dir, sizeof(float) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_input_max, sizeof(float) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_output_max, sizeof(float) * numRowsImage * numColsImage));
#ifdef _OUTLINE
    checkCudaErrors(cudaMalloc(&d_outline, sizeof(unsigned char) * numRowsImage * numColsImage));
#endif
#else
    checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_blue, sizeof(unsigned char) * numRowsImage * numColsImage));
#endif
}

void allocateFilterAndCopyToGPU(const float* h_filter, const size_t filterWidth, float** d_filter, bool justCopy = false)
{
    size_t count = sizeof(float) * filterWidth * filterWidth;
#ifdef _CONSTANT_MEMORY
    checkCudaErrors(cudaMemcpyToSymbol(d_filter_constant, h_filter, count));
#else
    if(!justCopy) checkCudaErrors(cudaMalloc(d_filter, count));
    checkCudaErrors(cudaMemcpy(*d_filter, h_filter, count, cudaMemcpyHostToDevice));
#endif
}

//Free all the memory that we allocated
void cleanupGPU() {
#ifdef _CANNY_EDGE
    checkCudaErrors(cudaFree(d_gray));
    checkCudaErrors(cudaFree(d_grayFiltered));
    checkCudaErrors(cudaFree(d_classifiedValues));
    checkCudaErrors(cudaFree(d_hysteresis));
    checkCudaErrors(cudaFree(d_sobel_x));
    checkCudaErrors(cudaFree(d_sobel_y));
    checkCudaErrors(cudaFree(d_magnitude));
    checkCudaErrors(cudaFree(d_nms_output));
    checkCudaErrors(cudaFree(d_dir));
    checkCudaErrors(cudaFree(d_input_max));
    checkCudaErrors(cudaFree(d_output_max));
    #ifdef _OUTLINE
    checkCudaErrors(cudaFree(&d_outline));
    #endif
#else
    checkCudaErrors(cudaFree(d_red));
    checkCudaErrors(cudaFree(d_green));
    checkCudaErrors(cudaFree(d_blue));
#endif
}

void create_filter(float** h_filter, int* filterWidth, int id_filter) {

    const int KernelWidth = FILTERSIZE;
    *filterWidth = KernelWidth;

    //create and fill the filter we will convolve with
    *h_filter = new float[KernelWidth * KernelWidth];

    switch (id_filter)
    {

    case 0: //Gaussian filter: blur
    {
        const float KernelSigma = 2.;

        float filterSum = 0.f; //for normalization

        for (int r = -KernelWidth / 2; r <= KernelWidth / 2; ++r) {
            for (int c = -KernelWidth / 2; c <= KernelWidth / 2; ++c) {
                float filterValue = expf(-(float)(c * c + r * r) / (2.f * KernelSigma * KernelSigma));
                (*h_filter)[(r + KernelWidth / 2) * KernelWidth + c + KernelWidth / 2] = filterValue;
                filterSum += filterValue;
            }
        }

        float normalizationFactor = 1.f / filterSum;

        for (int r = -KernelWidth / 2; r <= KernelWidth / 2; ++r) {
            for (int c = -KernelWidth / 2; c <= KernelWidth / 2; ++c) {
                (*h_filter)[(r + KernelWidth / 2) * KernelWidth + c + KernelWidth / 2] *= normalizationFactor;
            }
        }
    }
    break;

    case 1: //Laplacian 5x5
    {
        (*h_filter)[0] = 0;   (*h_filter)[1] = 0;    (*h_filter)[2] = -1.;  (*h_filter)[3] = 0;    (*h_filter)[4] = 0;
        (*h_filter)[5] = 0;  (*h_filter)[6] = -1.;  (*h_filter)[7] = -2.;  (*h_filter)[8] = -1.;  (*h_filter)[9] = 0;
        (*h_filter)[10] = -1.; (*h_filter)[11] = -2.; (*h_filter)[12] = 17.; (*h_filter)[13] = -2.; (*h_filter)[14] = -1.;
        (*h_filter)[15] = 0; (*h_filter)[16] = -1.; (*h_filter)[17] = -2.; (*h_filter)[18] = -1.; (*h_filter)[19] = 0;
        (*h_filter)[20] = 0;  (*h_filter)[21] = 0;   (*h_filter)[22] = -1.; (*h_filter)[23] = 0;   (*h_filter)[24] = 0;
    }
    break;


    case 2: //Sharpening 3x3
    {
        (*h_filter)[0] = 0.; (*h_filter)[1] = -1.; (*h_filter)[2] = 0.;
        (*h_filter)[3] = -1.; (*h_filter)[4] = 5.; (*h_filter)[5] = -1.;
        (*h_filter)[6] = 0.; (*h_filter)[7] = -1.; (*h_filter)[8] = 0.;
    }
    break;

    case 3: //Horizontal sobel 3x3
    {
        (*h_filter)[0] = -1.; (*h_filter)[1] = 0.; (*h_filter)[2] = 1.;
        (*h_filter)[3] = -2.; (*h_filter)[4] = 0.; (*h_filter)[5] = 2.;
        (*h_filter)[6] = -1.; (*h_filter)[7] = 0.; (*h_filter)[8] = 1.;
    }
    break;

    case 4: //Vertical sobel 3x3
    {
        (*h_filter)[0] = -1.; (*h_filter)[1] = -2.; (*h_filter)[2] = -1.;
        (*h_filter)[3] = 0.; (*h_filter)[4] = 0.; (*h_filter)[5] = 0.;
        (*h_filter)[6] = 1.; (*h_filter)[7] = 2.; (*h_filter)[8] = 1.;
    }
    break;

    case 5: //Edge detection 3x3
    {
        (*h_filter)[0] = -1.; (*h_filter)[1] = -1.; (*h_filter)[2] = -1.;
        (*h_filter)[3] = -1.; (*h_filter)[4] = 8.; (*h_filter)[5] = -1.;
        (*h_filter)[6] = -1.; (*h_filter)[7] = -1.; (*h_filter)[8] = -1.;
    }
    break;

    case 6: //Sharpness 5x5
    {
        (*h_filter)[0] = 0.; (*h_filter)[1] = 0.; (*h_filter)[2] = -1.; (*h_filter)[3] = 0.; (*h_filter)[4] = 0.;
        (*h_filter)[5] = 0.; (*h_filter)[6] = -1.; (*h_filter)[7] = -1.; (*h_filter)[8] = -1.; (*h_filter)[9] = 0.;
        (*h_filter)[10] = -1.; (*h_filter)[11] = -1.; (*h_filter)[12] = 13.; (*h_filter)[13] = -1.; (*h_filter)[14] = -1.;
        (*h_filter)[15] = 0.; (*h_filter)[16] = -1.; (*h_filter)[17] = -1.; (*h_filter)[18] = -1.; (*h_filter)[19] = 0.;
        (*h_filter)[20] = 0.; (*h_filter)[21] = 0.; (*h_filter)[22] = -1.; (*h_filter)[23] = 0.; (*h_filter)[24] = 0.;
    }
    break;

    default:
        printf("Filter not defined\n");
        exit(1);
    }
}

void box_filter(uchar4* const d_inputImageRGBA,
    uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
    unsigned char* d_redFiltered,
    unsigned char* d_greenFiltered,
    unsigned char* d_blueFiltered,
    int id_filter)
{
    float* h_filter;
    float* d_filter;
    int filterWidth;

    //Creates d_red, d_green and d_blue in GPU
    allocateMemoryGPU(numRows, numCols);

#ifndef _CANNY_EDGE
    //Creates the filter in CPU and uploads it to the GPU 
    create_filter(&h_filter, &filterWidth, id_filter);
    allocateFilterAndCopyToGPU(h_filter, filterWidth, &d_filter);
#endif

    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x,
        (numRows + blockSize.y - 1) / blockSize.y);

#ifdef _CANNY_EDGE
    //Gray scale:
    toGrayScale << <gridSize, blockSize >> > (d_inputImageRGBA, d_gray, numRows, numCols);
    //Blur 5x5:
    FILTERSIZE = 5;
    create_filter(&h_filter, &filterWidth, 0);
    allocateFilterAndCopyToGPU(h_filter, filterWidth, &d_filter);
    size_t tileSize = (BLOCK_SIZE + filterWidth - 1) * (BLOCK_SIZE + filterWidth - 1) * sizeof(float);
    convolution << <gridSize, blockSize, tileSize >> > (d_gray, d_grayFiltered, numRows, numCols, d_filter, filterWidth);
    //Sobels and magnitude and direction gradient:
    FILTERSIZE = 3;
    create_filter(&h_filter, &filterWidth, 3);
    allocateFilterAndCopyToGPU(h_filter, filterWidth, &d_filter, true);
    tileSize = (BLOCK_SIZE + filterWidth - 1) * (BLOCK_SIZE + filterWidth - 1) * sizeof(float);
    convolution << <gridSize, blockSize, tileSize >> > (d_grayFiltered, d_sobel_x, numRows, numCols, d_filter, filterWidth);
    create_filter(&h_filter, &filterWidth, 4);
    allocateFilterAndCopyToGPU(h_filter, filterWidth, &d_filter, true);
    tileSize = (BLOCK_SIZE + filterWidth - 1) * (BLOCK_SIZE + filterWidth - 1) * sizeof(float);
    convolution << <gridSize, blockSize, tileSize >> > (d_grayFiltered, d_sobel_y, numRows, numCols, d_filter, filterWidth);
    computeMagAndDir << <gridSize, blockSize >> > (d_sobel_x, d_sobel_y, d_magnitude, d_dir, numRows, numCols);
    //Non max suppression in dradient's direction:
    suppressNonMax<<<gridSize, blockSize>>>(d_magnitude, d_nms_output, d_dir, numRows, numCols);
    //Double value threshold:
    checkCudaErrors(cudaMemcpy(d_input_max, d_nms_output, numRows * numCols * sizeof(float), cudaMemcpyDeviceToDevice));
    int totalBlockSize = BLOCK_SIZE * BLOCK_SIZE;
    int currentGridSize = (numRows * numCols + totalBlockSize - 1) / totalBlockSize;
    int currentSize = numRows * numCols;
    while (currentSize > 1) {
        findMax << <currentGridSize, totalBlockSize >> > (d_input_max, d_output_max, currentSize);
        std::swap(d_input_max, d_output_max);
        currentSize = currentGridSize;
        currentGridSize = (currentGridSize + totalBlockSize - 1) / totalBlockSize;
    }
    float maxValue;
    checkCudaErrors(cudaMemcpy(&maxValue, d_input_max, sizeof(float), cudaMemcpyDeviceToHost));
    float strongThreshold = maxValue * STRONG_THRESHOLD;
    float weakThreshold = maxValue * SOFT_THRESHOLD;
    applyThreshold << <gridSize, blockSize >> > (d_nms_output, d_classifiedValues, strongThreshold,
        weakThreshold, numRows, numCols);
    //Hysteresis thresholding:
    tileSize = (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) * sizeof(unsigned char);
    hysteresis << <gridSize, blockSize, tileSize >> > (d_classifiedValues, d_hysteresis, numRows, numCols);
    //Ouline (optional):
    #ifdef _OUTLINE
    outline << <gridSize, blockSize, tileSize >> > (d_hysteresis, d_outline, numRows, numCols);
    //Final image:
    recombineChannels << <gridSize, blockSize >> > (d_outline, d_outline, d_outline,
        d_outputImageRGBA, numRows, numCols);
    #else
    //Final image:
    recombineChannels << <gridSize, blockSize >> > (d_hysteresis, d_hysteresis, d_hysteresis,
        d_outputImageRGBA, numRows, numCols);
    #endif

#else
    size_t tileSize = (BLOCK_SIZE + filterWidth - 1) * (BLOCK_SIZE + filterWidth - 1) * sizeof(float);
    separateChannels << <gridSize, blockSize >> > (d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);
    convolution << <gridSize, blockSize, tileSize >> > (d_red, d_redFiltered, numRows, numCols, d_filter, filterWidth);
    convolution << <gridSize, blockSize, tileSize >> > (d_green, d_greenFiltered, numRows, numCols, d_filter, filterWidth);
    convolution << <gridSize, blockSize, tileSize >> > (d_blue, d_blueFiltered, numRows, numCols, d_filter, filterWidth);
    recombineChannels << <gridSize, blockSize >> > (d_redFiltered,
        d_greenFiltered,
        d_blueFiltered,
        d_outputImageRGBA,
        numRows,
        numCols);
#endif
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}