#ifndef __YOLOV5_BACKBONE_H__
#define __YOLOV5_BACKBONE_H__

#include "tensorNet.h"

#define YOLOV5_DEFAULT_INPUT  DEFAULT_NO_INPUT

#define YOLOV5_DEFAULT_OUTPUT  "output"

#define YOLOV5_USAGE_STRING  "YOLOV5 arguments: \n" \
		  "  --network=NETWORK    pre-trained model to load\n" \
		  "  --model=MODEL        path to custom model to load (caffemodel, uff, or onnx)\n" \
		  "  --prototxt=PROTOTXT  path to custom prototxt to load (for .caffemodel only)\n" \
		  "  --labels=LABELS      path to text file containing the labels for each class\n" \
		  "  --input-blob=INPUT   name of the input layer (default is '" YOLOV5_DEFAULT_INPUT "')\n" \
		  "  --output-blob=OUTPUT name of the output layer (default is '" YOLOV5_DEFAULT_OUTPUT "')\n" \
		  "  --batch-size=BATCH   maximum batch size (default is 1)\n" \
		  "  --profile            enable layer profiling in TensorRT\n\n"


/**
 * profile network, using TensorRT.
 * @ingroup YOLOV5
 */
class YOLOV5 : public tensorNet
{
public:
	/**
	 * Network choice enumeration.
	 */
	enum NetworkType
	{
		CUSTOM,        /**< Custom model provided by the user */
	};

	static NetworkType NetworkTypeFromStr( const char* model_name );
	
	static YOLOV5* Create( const char* prototxt_path, const char* model_path, const char* mean_binary, 
		const char* input=YOLOV5_DEFAULT_INPUT, const char* output=YOLOV5_DEFAULT_OUTPUT, uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
		bool dynamicInput=false, const std::vector<int32_t>& shapeRange={0},
		precisionType precision=TYPE_FASTEST, deviceType device=DEVICE_GPU, bool allowGPUFallback=true);
	/**
	 * Load a new network instance by parsing the command line.
	 */
	static YOLOV5* Create( int argc, char** argv );

	/**
	 * Load a new network instance by parsing the command line.
	 */
	static YOLOV5* Create( const commandLine& cmdLine );

	/**
	 * Usage string for command line arguments to Create()
	 */
	static inline const char* Usage() 		{ return YOLOV5_USAGE_STRING; }

	/**
	 * Destroy
	 */
	~YOLOV5() final;

	bool cudaLetterbox(void* input, imageFormat format, const uint32_t height, const uint32_t width);

	template<typename T>
	bool cudaLetterbox(T* input, const uint32_t height, const uint32_t width)
	{
		return cudaLetterbox((void*)input, imageFormatFromType<T>(), height, width);
	}

	bool ImagePreprocess(const uint32_t height, const uint32_t width, const uint32_t newHeight, const uint32_t newWidth,
    	const float padValue=114.0, bool scaleUp=true);

	bool ImagePreprocess(const uint32_t height, const uint32_t width, const uint32_t newHeight,
		const float padValue=114.0/255, bool scaleUp=true)
	{
		return ImagePreprocess(height, width, newHeight, newHeight, padValue, scaleUp);
	}

	bool ImagePreprocess(void* input, imageFormat format, const uint32_t height, const uint32_t width, const uint32_t newHeight,
		const float padValue=114.0/255, bool scaleUp=true);

	template<typename T>
	bool ImagePreprocess(T* input, const uint32_t height, const uint32_t width, const uint32_t newHeight,
		const float padValue=114.0/255, bool scaleUp=true)
	{
		return ImagePreprocess((void*)input, imageFormatFromType<T>(), height, width, newHeight, padValue, scaleUp);
	}

	bool ProcessProbs(deviceType device=DEVICE_GPU, cudaStream_t stream=NULL);

	inline NetworkType GetNetworkType() const { return mNetworkType; }

	int GetImageProcesOWidth() { return mImageProcessOWidth; };

	int GetImageProcesOHeight() { return mImageProcessOHeight; };

	void SetNumDetectLayers(int nl) { mNl = nl; }

private:
	YOLOV5(int nl = 3);
	
    const int mBatchSize = 1;
    const int mInputChannel = 3;
	
	bool init(const char* prototxt_path, const char* model_path, const char* mean_binary, const char* input, const char* output, 
		uint32_t maxBatchSize, bool dynamicInput, const std::vector<int32_t>& shapeRange, 
		precisionType precision, deviceType device, bool allowGPUFallback );
	
	NetworkType mNetworkType;

	float* mImageProcess;
	int mPadTop;
	int mPadBottom;
	int mPadLeft; 
	int mPadRight;
	int mImageProcessOWidth;
	int mImageProcessOHeight;

	int mNl;

	std::vector<Dims> mProbInputDims;
	std::vector<Dims> mProbOutputDims;
};


#endif
