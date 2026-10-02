## Overview
GPU image box-filtering project written in CUDA. It applies convolution filters to images and chains several kernels to build a complete **Canny Edge Detector**, which is the main goal of the project. It's also used to measure how techniques such as constant memory, shared memory and block size affect performance.

## Features

**Canny Edge Detector**
* Full pipeline running on the GPU:
  * Grayscale conversion
  * Gaussian blur (5x5) to remove noise
  * Sobel filters and gradient magnitude and direction
  * Non-maximum suppression along the gradient direction
  * Double threshold, classifying pixels as strong, weak or irrelevant
  * Hysteresis, promoting weak pixels connected to strong ones
  * Optional outline step for thicker, more defined edges
* Maximum value of the image obtained with a parallel reduction, used to set the thresholds
<p align = "center">
 <img width="300" height="240" alt="raw" src="https://github.com/user-attachments/assets/b133d587-0690-4ae0-a426-a803bf1fa0af" />
 <img width="300" height="240" alt="canny edge" src="https://github.com/user-attachments/assets/50b12f06-c118-4f12-bc8d-9cf7d697f52f" />
</p>

**Single filters**
* Generic convolution kernel with border handling by clamping (edge values are repeated)
* Per-channel processing: the image is split into its RGB channels, filtered and recombined
* Included filters: Gaussian blur, Laplacian, sharpening, horizontal and vertical Sobel, edge detection and 5x5 sharpness
<p align = "center">
 <img width="300" height="240" alt="laplacian" src="https://github.com/user-attachments/assets/73d9931f-d2b9-4f49-b2b2-27a41de32773" />
 <img width="300" height="240" alt="edge detection" src="https://github.com/user-attachments/assets/fd4b908c-f098-4881-b48e-1467e20f7622" />
</p>

**Configurable behavior**
* Switch between Canny Edge Detector and single filters by changing `#define`s in the source code
* Constant memory can be enabled or disabled
* Shared memory can be enabled or disabled independently for convolutions and for hysteresis/outline

## Performance Study
The project was used to compare optimization techniques on different image sizes and filters:

* **Constant memory:** slower than regular global memory on the test GPU, since modern caches already handle filter accesses well
* **Shared memory in convolutions:** big gains with large filters and images (up to ~23% faster with a 9x9 blur), but not worth it for small 3x3 filters, where loading the halo costs more than it saves
* **Shared memory in hysteresis and outline:** slower, for the same reason
* **Block size:** 16x16 gave the best results compared to 8x8 and 32x32
* **Scalability:** sublinear time growth, with larger images making better use of the GPU

## Usage
* Open the project in Visual Studio with the CUDA Toolkit installed
* Run it with the input image as a command line argument
* To use a single filter instead of Canny, switch the corresponding `#define` in the source and pass the filter as an additional argument. Make sure that the variable FILTERSIZE matches the chosen filter.
* The filtered image is saved as output

## Technologies
* C++
* CUDA

## Requirements
* NVIDIA GPU with CUDA support
* CUDA Toolkit
* Visual Studio
