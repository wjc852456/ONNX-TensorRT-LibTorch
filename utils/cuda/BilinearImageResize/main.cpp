#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>

#include "resizeGPU.cuh"
#include "resizeCPU.hpp"
#include "converter.hpp"

#define RESIZE_CALLS_NUM 1

using namespace cv;
int main(int argc, char **argv)
{
	cv::Mat image;
	cv::Mat image_resized_gpu;
	cv::Mat image_resized_cpu;
	cv::Mat image_resized_cv2;
	int32_t *argb = NULL;
	int32_t *argb_res_gpu = NULL;
	int32_t *argb_res_cpu = NULL;
	clock_t cpu_startTime, cpu_endTime;
	double cpu_ElapseTime = 0;
    cv::Size newSz(640, 360);

	//int32_t *argb_pinned = NULL;
	//int32_t *argb_res_gpu_pinned = NULL;

	if (argc < 2)
	{
		printf("Usage:\n\t %s path_to_image\n", argv[0]);
		//exit(0);
	}
	const char fname[] = "/home/jcwang/camera_detection/data/images/G312.jpg";
	image = cv::imread(fname, 1);
	if (image.empty())
	{
		printf("Can't load image %s\n", fname);
	}

	argb = cvtMat2Int32(image);

	reAllocPinned(image.cols, image.rows, newSz.width, newSz.height, argb); //allocate pinned host memory for fast cuda memcpy 

	//gpu block start
	initGPU(4096, 4096);
	argb_res_gpu = resizeBilinear_gpu(image.cols, image.rows, newSz.width, newSz.height); //init device
	cpu_startTime = clock();
	for (int i = 0; i < RESIZE_CALLS_NUM; i++)
	{
		argb_res_gpu = resizeBilinear_gpu(image.cols, image.rows, newSz.width, newSz.height);
	}
	cpu_endTime = clock();
	cpu_ElapseTime = ((double)(cpu_endTime - cpu_startTime) / (double)CLOCKS_PER_SEC);
	printf("Time GPU: %fms\n", cpu_ElapseTime*1000);
	deinitGPU();
	//gpu block end

	//cpu (no OpenMP) block start
	cpu_startTime = clock();
	for (int i = 0; i < RESIZE_CALLS_NUM; i++)
	{
		delete[] argb_res_cpu;
		argb_res_cpu = resizeBilinear_cpu(argb, image.cols, image.rows, newSz.width, newSz.height);
	}
	cpu_endTime = clock();
	cpu_ElapseTime = ((double)(cpu_endTime - cpu_startTime) / (double)CLOCKS_PER_SEC);
	printf("Time CPU: %fms\n", cpu_ElapseTime*1000);
	//cpu (no OpenMP) block end

	// use cv::resize
	cpu_startTime = clock();
	for (int i = 0; i < RESIZE_CALLS_NUM; i++)
	{
		//resize(image, image_resized_cv2, newSz, cv::INTER_LINEAR);
		resize(image, image_resized_cv2, newSz, 0, 0);
	}
	cpu_endTime = clock();
	cpu_ElapseTime = ((double)(cpu_endTime - cpu_startTime) / (double)CLOCKS_PER_SEC);
	printf("Time CV Resize: %fms\n", cpu_ElapseTime*1000);

	//show result images of each module
	image_resized_gpu = cv::Mat(newSz, CV_8UC3);
	image_resized_cpu = cv::Mat(newSz, CV_8UC3);
	cvtInt322Mat(argb_res_gpu, image_resized_gpu);
	cvtInt322Mat(argb_res_cpu, image_resized_cpu);
	imshow("Original", image);
	imshow("Resized_GPU", image_resized_gpu);
	imwrite("Risized_gpu.jpg",image_resized_gpu);
	imshow("Resized_CPU", image_resized_cpu);
	imwrite("Resized_CPU.jpg",image_resized_cpu);
	imwrite("Resized_CV.jpg",image_resized_cv2);
	waitKey(0);

	//free memory
	freePinned();
	delete[] argb_res_cpu;
	delete[] argb;

	return 0;
}
