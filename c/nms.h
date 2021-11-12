#ifndef __NMS_H__
#define __NMS_H__

#include "tensorNet.h"

#define NMS_DEFAULT_INPUT  DEFAULT_NO_INPUT

#define NMS_DEFAULT_OUTPUT  "output"

#define NMS_USAGE_STRING  "NMS arguments: \n" \
          "  --network=NETWORK    pre-trained model to load\n" \
          "  --model=MODEL        path to custom model to load (caffemodel, uff, or onnx)\n" \
          "  --prototxt=PROTOTXT  path to custom prototxt to load (for .caffemodel only)\n" \
          "  --labels=LABELS      path to text file containing the labels for each class\n" \
          "  --input-blob=INPUT   name of the input layer (default is '" NMS_DEFAULT_INPUT "')\n" \
          "  --output-blob=OUTPUT name of the output layer (default is '" NMS_DEFAULT_OUTPUT "')\n" \
          "  --batch-size=BATCH   maximum batch size (default is 1)\n" \
          "  --profile            enable layer profiling in TensorRT\n\n"


/**
 * profile network, using TensorRT.
 * @ingroup NMS
 */
class NMS : public tensorNet
{
public:
    /**
     * Network choice enumeration.
     */
    enum NetworkType {
        CUSTOM,        /**< Custom model provided by the user */
    };

    static NMS* Create( const char* prototxt_path, const char* model_path, const char* mean_binary,
                        const char* input = NMS_DEFAULT_INPUT, const char* output = NMS_DEFAULT_OUTPUT, uint32_t maxBatchSize = DEFAULT_MAX_BATCH_SIZE,
                        bool dynamicInput = false, const std::vector<int32_t>& shapeRange = {0},
                        precisionType precision = TYPE_FASTEST, deviceType device = DEVICE_GPU, bool allowGPUFallback = true);
    /**
     * Load a new network instance by parsing the command line.
     */
    static NMS* Create( int argc, char** argv );

    /**
     * Load a new network instance by parsing the command line.
     */
    static NMS* Create( const commandLine& cmdLine );

    /**
     * Usage string for command line arguments to Create()
     */
    static inline const char* Usage()
    {
        return NMS_USAGE_STRING;
    }

    /**
     * Destroy
     */
    ~NMS() final;

    bool ProcessNMS(const std::vector<float*>& inputs, const std::vector<Dims>& nmsInputDims,
                    deviceType device = DEVICE_GPU, cudaStream_t stream = NULL);

    inline NetworkType GetNetworkType() const
    {
        return mNetworkType;
    }

private:
    NMS();

    bool init(const char* prototxt_path, const char* model_path, const char* mean_binary, const char* input, const char* output,
              uint32_t maxBatchSize, bool dynamicInput, const std::vector<int32_t>& shapeRange,
              precisionType precision, deviceType device, bool allowGPUFallback );

    const int mBatchSize = 1;

    NetworkType mNetworkType;
};


#endif
