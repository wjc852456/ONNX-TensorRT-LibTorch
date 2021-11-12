#ifndef __YOLOV5_REGRESSION_H__
#define __YOLOV5_REGRESSION_H__

#include "tensorNet.h"

#define YOLOV5_REGRESSION_DEFAULT_INPUT  DEFAULT_NO_INPUT

#define YOLOV5_REGRESSION_DEFAULT_OUTPUT  "output"

#define YOLOV5_REGRESSION_USAGE_STRING  "YOLOV5_REGRESSION arguments: \n" \
          "  --network=NETWORK    pre-trained model to load\n" \
          "  --model=MODEL        path to custom model to load (caffemodel, uff, or onnx)\n" \
          "  --prototxt=PROTOTXT  path to custom prototxt to load (for .caffemodel only)\n" \
          "  --labels=LABELS      path to text file containing the labels for each class\n" \
          "  --input-blob=INPUT   name of the input layer (default is '" YOLOV5_REGRESSION_DEFAULT_INPUT "')\n" \
          "  --output-blob=OUTPUT name of the output layer (default is '" YOLOV5_REGRESSION_DEFAULT_OUTPUT "')\n" \
          "  --batch-size=BATCH   maximum batch size (default is 1)\n" \
          "  --profile            enable layer profiling in TensorRT\n\n"


/**
 * profile network, using TensorRT.
 * @ingroup YOLOV5_REGRESSION
 */
class YOLOV5Regression : public tensorNet
{
public:
    /**
     * Network choice enumeration.
     */
    enum NetworkType {
        CUSTOM,        /**< Custom model provided by the user */
    };

    static YOLOV5Regression* Create( const char* prototxt_path, const char* model_path, const char* mean_binary,
                        const char* input = YOLOV5_REGRESSION_DEFAULT_INPUT, const char* output = YOLOV5_REGRESSION_DEFAULT_OUTPUT, uint32_t maxBatchSize = DEFAULT_MAX_BATCH_SIZE,
                        bool dynamicInput = false, const std::vector<int32_t>& shapeRange = {0},
                        precisionType precision = TYPE_FASTEST, deviceType device = DEVICE_GPU, bool allowGPUFallback = true);
    /**
     * Load a new network instance by parsing the command line.
     */
    static YOLOV5Regression* Create( int argc, char** argv );

    /**
     * Load a new network instance by parsing the command line.
     */
    static YOLOV5Regression* Create( const commandLine& cmdLine );

    /**
     * Usage string for command line arguments to Create()
     */
    static inline const char* Usage()
    {
        return YOLOV5_REGRESSION_USAGE_STRING;
    }

    /**
     * Destroy
     */
    ~YOLOV5Regression() final;

    bool ProcessRegression(const std::vector<float*>& inputs, const std::vector<Dims>& regressionInputDims,
                    deviceType device = DEVICE_GPU, cudaStream_t stream = NULL);

    inline NetworkType GetNetworkType() const
    {
        return mNetworkType;
    }

private:
    YOLOV5Regression();

    bool init(const char* prototxt_path, const char* model_path, const char* mean_binary, const char* input, const char* output,
              uint32_t maxBatchSize, bool dynamicInput, const std::vector<int32_t>& shapeRange,
              precisionType precision, deviceType device, bool allowGPUFallback );

    const int mBatchSize = 1;

    NetworkType mNetworkType;
};


#endif
