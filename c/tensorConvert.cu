/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#include "tensorConvert.h"

#include "cudaMappedMemory.h"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#define MIN(x, y)  ( ((x) < (y)) ? (x) : (y) )
#define ROUND(x) ((int)(x+0.5))

// gpuTensorMean
template<typename T, bool isBGR>
__global__ void gpuTensorMean( float2 scale, T* input, int iWidth, float* output, int oWidth, int oHeight, float3 mean_value )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int n = oWidth * oHeight;
	const int m = y * oWidth + x;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const T px = input[ dy * iWidth + dx ];

	const float3 rgb = isBGR ? make_float3(px.z, px.y, px.x)
						: make_float3(px.x, px.y, px.z);
	
	output[n * 0 + m] = rgb.x - mean_value.x;
	output[n * 1 + m] = rgb.y - mean_value.y;
	output[n * 2 + m] = rgb.z - mean_value.z;
}

template<bool isBGR>
cudaError_t launchTensorMean( void* input, imageFormat format, size_t inputWidth, size_t inputHeight,
						float* output, size_t outputWidth, size_t outputHeight, 
						const float3& mean_value, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	if( format == IMAGE_RGB8 )
		gpuTensorMean<uchar3, isBGR><<<gridDim, blockDim, 0, stream>>>(scale, (uchar3*)input, inputWidth, output, outputWidth, outputHeight, mean_value);
	else if( format == IMAGE_RGBA8 )
		gpuTensorMean<uchar4, isBGR><<<gridDim, blockDim, 0, stream>>>(scale, (uchar4*)input, inputWidth, output, outputWidth, outputHeight, mean_value);
	else if( format == IMAGE_RGB32F )
		gpuTensorMean<float3, isBGR><<<gridDim, blockDim, 0, stream>>>(scale, (float3*)input, inputWidth, output, outputWidth, outputHeight, mean_value);
	else if( format == IMAGE_RGBA32F )
		gpuTensorMean<float4, isBGR><<<gridDim, blockDim, 0, stream>>>(scale, (float4*)input, inputWidth, output, outputWidth, outputHeight, mean_value);
	else
		return cudaErrorInvalidValue;

	return CUDA(cudaGetLastError());
}

// cudaTensorMeanRGB
cudaError_t cudaTensorMeanRGB( void* input, imageFormat format, size_t inputWidth, size_t inputHeight,
				           float* output, size_t outputWidth, size_t outputHeight, 
						 const float3& mean_value, cudaStream_t stream )
{
	return launchTensorMean<false>(input, format, inputWidth, inputHeight, output, outputWidth, outputHeight, mean_value, stream);
}

// cudaTensorMeanBGR
cudaError_t cudaTensorMeanBGR( void* input, imageFormat format, size_t inputWidth, size_t inputHeight,
				           float* output, size_t outputWidth, size_t outputHeight, 
						 const float3& mean_value, cudaStream_t stream )
{
	return launchTensorMean<true>(input, format, inputWidth, inputHeight, output, outputWidth, outputHeight, mean_value, stream);
}


// gpuTensorNorm
template<typename T, bool isBGR>
__global__ void gpuTensorNorm( float2 scale, T* input, int iWidth, float* output, int oWidth, int oHeight, float multiplier, float min_value )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int n = oWidth * oHeight;
	const int m = y * oWidth + x;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const T px = input[ dy * iWidth + dx ];

	const float3 rgb = isBGR ? make_float3(px.z, px.y, px.x)
						: make_float3(px.x, px.y, px.z);
	
	output[n * 0 + m] = rgb.x * multiplier + min_value;
	output[n * 1 + m] = rgb.y * multiplier + min_value;
	output[n * 2 + m] = rgb.z * multiplier + min_value;
}

template<bool isBGR>
cudaError_t launchTensorNorm( void* input, imageFormat format, size_t inputWidth, size_t inputHeight,
						float* output, size_t outputWidth, size_t outputHeight, 
						const float2& range, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	const float multiplier = (range.y - range.x) / 255.0f;
	
	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	if( format == IMAGE_RGB8 )
		gpuTensorNorm<uchar3, isBGR><<<gridDim, blockDim, 0, stream>>>(scale, (uchar3*)input, inputWidth, output, outputWidth, outputHeight, multiplier, range.x);
	else if( format == IMAGE_RGBA8 )
		gpuTensorNorm<uchar4, isBGR><<<gridDim, blockDim, 0, stream>>>(scale, (uchar4*)input, inputWidth, output, outputWidth, outputHeight, multiplier, range.x);
	else if( format == IMAGE_RGB32F )
		gpuTensorNorm<float3, isBGR><<<gridDim, blockDim, 0, stream>>>(scale, (float3*)input, inputWidth, output, outputWidth, outputHeight, multiplier, range.x);
	else if( format == IMAGE_RGBA32F )
		gpuTensorNorm<float4, isBGR><<<gridDim, blockDim, 0, stream>>>(scale, (float4*)input, inputWidth, output, outputWidth, outputHeight, multiplier, range.x);
	else
		return cudaErrorInvalidValue;

	return CUDA(cudaGetLastError());
}

// cudaTensorNormRGB
cudaError_t cudaTensorNormRGB( void* input, imageFormat format, size_t inputWidth, size_t inputHeight,
						 float* output, size_t outputWidth, size_t outputHeight,
						 const float2& range, cudaStream_t stream )
{
	return launchTensorNorm<false>(input, format, inputWidth, inputHeight, output, outputWidth, outputHeight, range, stream);
}

// cudaTensorNormBGR
cudaError_t cudaTensorNormBGR( void* input, imageFormat format, size_t inputWidth, size_t inputHeight,
						 float* output, size_t outputWidth, size_t outputHeight,
						 const float2& range, cudaStream_t stream )
{
	return launchTensorNorm<true>(input, format, inputWidth, inputHeight, output, outputWidth, outputHeight, range, stream);
}


// gpuTensorNormMean
template<typename T, bool isBGR>
__global__ void gpuTensorNormMean( T* input, int iWidth, float* output, int oWidth, int oHeight, float2 scale, float multiplier, float min_value, const float3 mean, const float3 stdDev )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int n = oWidth * oHeight;
	const int m = y * oWidth + x;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const T px = input[ dy * iWidth + dx ];

	const float3 rgb = isBGR ? make_float3(px.z, px.y, px.x)
						: make_float3(px.x, px.y, px.z);
	
	output[n * 0 + m] = ((rgb.x * multiplier + min_value) - mean.x) / stdDev.x;
	output[n * 1 + m] = ((rgb.y * multiplier + min_value) - mean.y) / stdDev.y;
	output[n * 2 + m] = ((rgb.z * multiplier + min_value) - mean.z) / stdDev.z;
}

template<bool isBGR>
cudaError_t launchTensorNormMean( void* input, imageFormat format, size_t inputWidth, size_t inputHeight,
						    float* output, size_t outputWidth, size_t outputHeight, 
						    const float2& range, const float3& mean, const float3& stdDev,
						    cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	const float multiplier = (range.y - range.x) / 255.0f;
	
	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	if( format == IMAGE_RGB8 )
		gpuTensorNormMean<uchar3, isBGR><<<gridDim, blockDim, 0, stream>>>((uchar3*)input, inputWidth, output, outputWidth, outputHeight, scale, multiplier, range.x, mean, stdDev);
	else if( format == IMAGE_RGBA8 )
		gpuTensorNormMean<uchar4, isBGR><<<gridDim, blockDim, 0, stream>>>((uchar4*)input, inputWidth, output, outputWidth, outputHeight, scale, multiplier, range.x, mean, stdDev);
	else if( format == IMAGE_RGB32F )
		gpuTensorNormMean<float3, isBGR><<<gridDim, blockDim, 0, stream>>>((float3*)input, inputWidth, output, outputWidth, outputHeight, scale, multiplier, range.x, mean, stdDev);
	else if( format == IMAGE_RGBA32F )
		gpuTensorNormMean<float4, isBGR><<<gridDim, blockDim, 0, stream>>>((float4*)input, inputWidth, output, outputWidth, outputHeight, scale, multiplier, range.x, mean, stdDev);
	else
		return cudaErrorInvalidValue;

	return CUDA(cudaGetLastError());
}

// cudaTensorNormMeanRGB
cudaError_t cudaTensorNormMeanRGB( void* input, imageFormat format, size_t inputWidth, size_t inputHeight,
						     float* output, size_t outputWidth, size_t outputHeight, 
						     const float2& range, const float3& mean, const float3& stdDev,
						     cudaStream_t stream )
{
	return launchTensorNormMean<false>(input, format, inputWidth, inputHeight, output, outputWidth, outputHeight, range, mean, stdDev, stream);
}

// cudaTensorNormMeanRGB
cudaError_t cudaTensorNormMeanBGR( void* input, imageFormat format, size_t inputWidth, size_t inputHeight,
						     float* output, size_t outputWidth, size_t outputHeight, 
						     const float2& range, const float3& mean, const float3& stdDev,
						     cudaStream_t stream )
{
	return launchTensorNormMean<true>(input, format, inputWidth, inputHeight, output, outputWidth, outputHeight, range, mean, stdDev, stream);
}


template<> void cudaMemset<float>(float* input, const uint32_t size, const float value)
{
	thrust::device_ptr<float> devPtr(input);
	thrust::fill(devPtr, devPtr + size, value);
}

template<> void cudaMemset<int>(int* input, const uint32_t size, const int value)
{
	thrust::device_ptr<int> devPtr(input);
	thrust::fill(devPtr, devPtr + size, value);
}

template<typename T, bool isBGR>
__global__ void gpuTensorNormMeanInterNearest( T* input, int iWidth, int top, int bottom, int left, int right, float* output, int oWidth, int oHeight, float2 scale, float multiplier, float min_value, const float3 mean, const float3 stdDev )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int n = (oWidth+left+right) * (oHeight+top+bottom);
	const int m = top * (oWidth+left+right) + y * (oWidth+left+right) + (x+left);

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const T px = input[ dy * iWidth + dx ];

	const float3 rgb = isBGR ? make_float3(px.z, px.y, px.x)
						: make_float3(px.x, px.y, px.z);
	
	output[n * 0 + m] = ((rgb.x * multiplier + min_value) - mean.x) / stdDev.x;
	output[n * 1 + m] = ((rgb.y * multiplier + min_value) - mean.y) / stdDev.y;
	output[n * 2 + m] = ((rgb.z * multiplier + min_value) - mean.z) / stdDev.z;
}

template<typename T, bool isBGR>
__global__ void gpuTensorNormMeanInterLinear( T* input, int iWidth, int top, int bottom, int left, int right, float* output, int oWidth, int oHeight, float2 scale, float multiplier, float min_value, const float3 mean, const float3 stdDev )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int n = (oWidth+left+right) * (oHeight+top+bottom);
	const int m = top * (oWidth+left+right) + y * (oWidth+left+right) + (x+left);

	const int dx = (int)((x + 0.5f) * scale.x - 0.5f);
	const int dy = (int)((y + 0.5f) * scale.y - 0.5f);
	const float x_diff = ((x + 0.5f) * scale.x - 0.5f) - dx;
	const float y_diff = ((y + 0.5f) * scale.y - 0.5f) - dy;
	const int32_t index = dy * iWidth + dx;
	const T a = input[index];
	const T b = input[index + 1];
	const T c = input[index + iWidth];
	const T d = input[index + iWidth + 1];

	const float3 rgb_a = isBGR ? make_float3(a.z, a.y, a.x) : make_float3(a.x, a.y, a.z);
	const float3 rgb_b = isBGR ? make_float3(b.z, b.y, b.x) : make_float3(b.x, b.y, b.z);
	const float3 rgb_c = isBGR ? make_float3(c.z, c.y, c.x) : make_float3(c.x, c.y, c.z);
	const float3 rgb_d = isBGR ? make_float3(d.z, d.y, d.x) : make_float3(d.x, d.y, d.z);

	const float R = rgb_a.x*(1 - x_diff)*(1 - y_diff) + rgb_b.x*(x_diff)*(1 - y_diff) + rgb_c.x*(1 - x_diff)*(y_diff) + rgb_d.x*(x_diff*y_diff);
	const float G = rgb_a.y*(1 - x_diff)*(1 - y_diff) + rgb_b.y*(x_diff)*(1 - y_diff) + rgb_c.y*(1 - x_diff)*(y_diff) + rgb_d.y*(x_diff*y_diff);
	const float B = rgb_a.z*(1 - x_diff)*(1 - y_diff) + rgb_b.z*(x_diff)*(1 - y_diff) + rgb_c.z*(1 - x_diff)*(y_diff) + rgb_d.z*(x_diff*y_diff);

	output[n * 0 + m] = ((R * multiplier + min_value) - mean.x) / stdDev.x;
	output[n * 1 + m] = ((G * multiplier + min_value) - mean.y) / stdDev.y;
	output[n * 2 + m] = ((B * multiplier + min_value) - mean.z) / stdDev.z;
}

template<bool isBGR>
cudaError_t launchTensorNormMean( void* input, imageFormat format, size_t inputWidth, size_t inputHeight,
							int top, int bottom, int left, int right,
						    float* output, size_t outputWidth, size_t outputHeight, 
						    const float2& range, const float3& mean, const float3& stdDev,
						    imageResizeType resizeType, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || inputHeight == 0 || outputWidth == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	const float multiplier = (range.y - range.x) / 255.0f;
	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	if( format == IMAGE_RGB8 ) {
		if ( resizeType == INTER_LINEAR )
			gpuTensorNormMeanInterLinear<uchar3, isBGR><<<gridDim, blockDim, 0, stream>>>((uchar3*)input, inputWidth, top, bottom, left, right,
																				output, outputWidth, outputHeight, scale, multiplier, 0.0f, mean, stdDev);
		else
			gpuTensorNormMeanInterNearest<uchar3, isBGR><<<gridDim, blockDim, 0, stream>>>((uchar3*)input, inputWidth, top, bottom, left, right,
																				output, outputWidth, outputHeight, scale, multiplier, 0.0f, mean, stdDev);
	}
	else if( format == IMAGE_RGBA8 ) {
		if ( resizeType == INTER_LINEAR )
			gpuTensorNormMeanInterLinear<uchar4, isBGR><<<gridDim, blockDim, 0, stream>>>((uchar4*)input, inputWidth, top, bottom, left, right,
																				output, outputWidth, outputHeight, scale, multiplier, 0.0f, mean, stdDev);
		else
			gpuTensorNormMeanInterNearest<uchar4, isBGR><<<gridDim, blockDim, 0, stream>>>((uchar4*)input, inputWidth, top, bottom, left, right,
																				output, outputWidth, outputHeight, scale, multiplier, 0.0f, mean, stdDev);
	}
	else if( format == IMAGE_RGB32F ) {
		if ( resizeType == INTER_LINEAR )
			gpuTensorNormMeanInterLinear<float3, isBGR><<<gridDim, blockDim, 0, stream>>>((float3*)input, inputWidth, top, bottom, left, right,
																				output, outputWidth, outputHeight, scale, multiplier, 0.0f, mean, stdDev);
		else
			gpuTensorNormMeanInterNearest<float3, isBGR><<<gridDim, blockDim, 0, stream>>>((float3*)input, inputWidth, top, bottom, left, right,
																				output, outputWidth, outputHeight, scale, multiplier, 0.0f, mean, stdDev);
	}
	else if( format == IMAGE_RGBA32F ) {
		if ( resizeType == INTER_LINEAR )
			gpuTensorNormMeanInterLinear<float4, isBGR><<<gridDim, blockDim, 0, stream>>>((float4*)input, inputWidth, top, bottom, left, right,
																				output, outputWidth, outputHeight, scale, multiplier, 0.0f, mean, stdDev);
		else
			gpuTensorNormMeanInterNearest<float4, isBGR><<<gridDim, blockDim, 0, stream>>>((float4*)input, inputWidth, top, bottom, left, right,
																				output, outputWidth, outputHeight, scale, multiplier, 0.0f, mean, stdDev);
	}
	else
		return cudaErrorInvalidValue;

	return CUDA(cudaGetLastError());
}

cudaError_t launchTensorNormMeanBGR(void* input, imageFormat format, size_t inputWidth, size_t inputHeight,
							int top, int bottom, int left, int right,
						    float* output, size_t outputWidth, size_t outputHeight, 
						    const float2& range, const float3& mean, const float3& stdDev,
						    imageResizeType resizeType, cudaStream_t stream)
{
	return launchTensorNormMean<true>(input, format, inputWidth, inputHeight, top, bottom, left, right, 
	 								output, outputWidth, outputHeight, range, mean, stdDev, resizeType, stream);
}

cudaError_t launchTensorNormMeanRGB(void* input, imageFormat format, size_t inputWidth, size_t inputHeight,
							int top, int bottom, int left, int right,
						    float* output, size_t outputWidth, size_t outputHeight, 
						    const float2& range, const float3& mean, const float3& stdDev,
						    imageResizeType resizeType, cudaStream_t stream)
{
	return launchTensorNormMean<false>(input, format, inputWidth, inputHeight, top, bottom, left, right, 
	 								output, outputWidth, outputHeight, range, mean, stdDev, resizeType, stream);
}

cudaError_t cudaLetterbox(void* input, imageFormat format, float* output, const uint32_t height, const uint32_t width, const uint32_t newHeight, const uint32_t newWidth, bool scaleUp)
{
	float r = MIN(newHeight*1.0f / height, newHeight*1.0f / width);
	
	if ( !scaleUp ) {
		r = MIN(r, 1.0f);
	}
	uint32_t newUnpadWidth  = ROUND(width*r);
	uint32_t newUnpadHeight = ROUND(height*r);
	uint32_t dw = (newWidth - newUnpadWidth) % 32;
	uint32_t dh = (newHeight - newUnpadHeight) % 32;
	dw /= 2;
	dh /= 2;
	uint32_t top = dh, bottom = dh;
	uint32_t left = dw, right = dw;
	if (height != newUnpadHeight && width != newUnpadWidth) {
		float2 range = {-0.5, 0.5};
		float3 mean = {0.0, 0.0, 0.0};
		float3 stdDev = {1.0, 1.0, 1.0};
		launchTensorNormMeanRGB(input, format, width, height, top, bottom, left, right, 
	 								output, newUnpadWidth, newUnpadHeight, range, mean, stdDev);
	}
	
	return CUDA(cudaGetLastError());
}

cudaError_t cudaImagePreprocess(const uint32_t height, const uint32_t width, const uint32_t newHeight, const uint32_t newWidth, float** output, 
	uint32_t* outputHeight, uint32_t* outputWidth, const float padValue, bool scaleUp)
{
	if( width == 0 || height == 0 || newWidth == 0 || newHeight == 0) {
		return cudaErrorInvalidValue;
	}

	float r = MIN(newHeight*1.0f / height, newHeight*1.0f / width);
	
	if ( !scaleUp ) {
		r = MIN(r, 1.0f);
	}

	uint32_t newUnpadWidth  = ROUND(width*r);
	uint32_t newUnpadHeight = ROUND(height*r);
	uint32_t dw = (newWidth - newUnpadWidth) % 32;
	uint32_t dh = (newHeight - newUnpadHeight) % 32;
	dw /= 2;
	dh /= 2;

	uint32_t top = dh;
	uint32_t bottom = dh;
	uint32_t left = dw; 
	uint32_t right = dw;

	size_t imgSizePad = (*outputHeight) * (*outputWidth) * 3;

	if ( *outputHeight != (newUnpadHeight + top + bottom) || 
		 *outputWidth != (newUnpadWidth + left + right) || 
		 *output == NULL) {
		*outputHeight = (newUnpadHeight + top + bottom);
		*outputWidth = (newUnpadWidth + left + right);
		imgSizePad = (*outputHeight) * (*outputWidth) * 3;
		cudaAllocMapped((void**)output, imgSizePad * sizeof(float));
	}

	cudaMemset(*output, imgSizePad, padValue);

	return CUDA(cudaGetLastError());
}