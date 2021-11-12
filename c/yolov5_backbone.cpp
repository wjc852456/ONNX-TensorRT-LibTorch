
#include <fstream>

#include "yolov5_backbone.h"
#include "tensorConvert.h"

#include "cudaMappedMemory.h"
#include "cudaResize.h"

#include "commandLine.h"
#include "filesystem.h"
#include "logging.h"


#define ROUND(x) ( (int)(x + 0.5) )
#define MIN(x, y)  ( ((x) < (y)) ? (x) : (y) )

// constructor
YOLOV5::YOLOV5(int nl) : tensorNet(), mNl(nl)
{
    mNetworkType = CUSTOM;

    mImageProcess = NULL;
    mImageProcessOWidth = 0;
    mImageProcessOHeight = 0;

    mPadTop = 0;
    mPadBottom = 0;
    mPadLeft = 0;
    mPadRight = 0;
}

// destructor
YOLOV5::~YOLOV5()
{
    CUDA_FREE_HOST(mImageProcess);
}

// Create
YOLOV5* YOLOV5::Create( const char* prototxt_path, const char* model_path, const char* mean_binary,
                        const char* input, const char* output, uint32_t maxBatchSize,
                        bool dynamicInput, const std::vector<int32_t>& shapeRange,
                        precisionType precision, deviceType device, bool allowGPUFallback)
{
    YOLOV5* net = new YOLOV5();

    if ( !net )
        return NULL;

    if ( !net->init(prototxt_path, model_path, mean_binary, input, output, maxBatchSize,
                    dynamicInput, shapeRange, precision, device, allowGPUFallback) ) {
        LogError(LOG_TRT "YOLOV5 -- failed to initialize.\n");
        return NULL;
    }

    return net;
}

bool YOLOV5::init(const char* prototxt_path, const char* model_path, const char* mean_binary,
                  const char* input, const char* output, uint32_t maxBatchSize,
                  bool dynamicInput, const std::vector<int32_t>& shapeRange,
                  precisionType precision, deviceType device, bool allowGPUFallback )
{
    if ( !model_path || !input || !output )
        return false;

    LogInfo("\n");
    LogInfo("YOLOV5 -- loading classification network model from:\n");
    LogInfo("         -- prototxt     %s\n", prototxt_path);
    LogInfo("         -- model        %s\n", model_path);
    LogInfo("         -- input_blob   '%s'\n", input);
    LogInfo("         -- output_blob  '%s'\n", output);
    LogInfo("         -- batch_size   %u\n\n", maxBatchSize);


    std::vector<std::string> inputs;
    inputs.push_back(input);

    std::vector<Dims3> input_dims;
    input_dims.push_back(Dims3(1, 1, 1));

    std::vector<std::string> outputs;
    outputs.push_back(output);

    if ( !tensorNet::LoadNetwork( prototxt_path, model_path, mean_binary,
                                  inputs, input_dims, outputs,
                                  dynamicInput, shapeRange,
                                  maxBatchSize, precision, device, allowGPUFallback ) ) {
        LogError(LOG_TRT "failed to load %s\n", model_path);
        return false;
    }

    LogSuccess(LOG_TRT "YOLOV5 -- %s initialized.\n", model_path);
    return true;
}

// NetworkTypeFromStr
YOLOV5::NetworkType YOLOV5::NetworkTypeFromStr( const char* modelName )
{
    return YOLOV5::CUSTOM;
}

// Create
YOLOV5* YOLOV5::Create( int argc, char** argv )
{
    return Create(commandLine(argc, argv));
}

// Create
YOLOV5* YOLOV5::Create( const commandLine& cmdLine )
{
    YOLOV5* net = NULL;

    // obtain the network name
    const char* modelName = cmdLine.GetString("network");

    if ( !modelName )
        modelName = cmdLine.GetString("model", "googlenet");

    // parse the network type
    const YOLOV5::NetworkType type = NetworkTypeFromStr(modelName);

    if ( type == YOLOV5::CUSTOM ) {
        const char* prototxt = cmdLine.GetString("prototxt");
        const char* labels   = cmdLine.GetString("labels");
        const char* input    = cmdLine.GetString("input_blob");
        const char* output   = cmdLine.GetString("output_blob");

        if ( !input )    input = YOLOV5_DEFAULT_INPUT;
        if ( !output )  output = YOLOV5_DEFAULT_OUTPUT;

        int maxBatchSize = cmdLine.GetInt("batch_size");

        if ( maxBatchSize < 1 )
            maxBatchSize = DEFAULT_MAX_BATCH_SIZE;

        bool dynamicInput = cmdLine.GetFlag("dynamic_input");
        const std::vector<int32_t> shapeRange = cmdLine.GetVectorInt("shape_range");

        net = YOLOV5::Create(prototxt, modelName, NULL, input, output, maxBatchSize, dynamicInput, shapeRange);
    }

    if ( !net )
        return NULL;

    // enable layer profiling if desired
    if ( cmdLine.GetFlag("profile") )
        net->EnableLayerProfiler();

    return net;
}

bool YOLOV5::cudaLetterbox(void* input, imageFormat format, const uint32_t height, const uint32_t width)
{
    float2 range = {-0.5, 0.5};
    float3 mean = {0.0, 0.0, 0.0};
    float3 stdDev = {1.0, 1.0, 1.0};
    uint32_t newUnpadWidth = mImageProcessOWidth - mPadLeft - mPadRight;
    uint32_t newUnpadHeight = mImageProcessOHeight - mPadTop - mPadBottom;
    if ( CUDA_FAILED(launchTensorNormMeanRGB(input, format, width, height, mPadTop, mPadBottom, mPadLeft, mPadRight,
                     mImageProcess, newUnpadWidth, newUnpadHeight, range, mean, stdDev)) )
        return false;

    return true;
}

bool YOLOV5::ImagePreprocess(const uint32_t height, const uint32_t width, const uint32_t newHeight, const uint32_t newWidth,
                             const float padValue, bool scaleUp)
{
    if ( width == 0 || height == 0 || newWidth == 0 || newHeight == 0) {
        return cudaErrorInvalidValue;
    }

    float r = MIN(newHeight * 1.0f / height, newHeight * 1.0f / width);

    if ( !scaleUp ) {
        r = MIN(r, 1.0f);
    }

    int newUnpadWidth  = ROUND(width * r);
    int newUnpadHeight = ROUND(height * r);
    int dw = (newWidth - newUnpadWidth) % 32;
    int dh = (newHeight - newUnpadHeight) % 32;

    mPadTop = dh / 2;
    mPadBottom = dh - mPadTop;
    mPadLeft = dw / 2;
    mPadRight = dw - mPadLeft;

    size_t imgSizePad = (mImageProcessOHeight) * (mImageProcessOWidth) * 3;

    if ( mImageProcessOHeight != (newUnpadHeight + mPadTop + mPadBottom) ||
            mImageProcessOWidth != (newUnpadWidth + mPadLeft + mPadRight) ||
            mImageProcess == NULL) { // if sizes match and memory allocated, do not allocate cuda memory
        CUDA_FREE_HOST(mImageProcess);
        float* output = NULL;
        mImageProcessOHeight = (newUnpadHeight + mPadTop + mPadBottom);
        mImageProcessOWidth = (newUnpadWidth + mPadLeft + mPadRight);
        imgSizePad = (mImageProcessOHeight) * (mImageProcessOWidth) * 3;
        if ( !cudaAllocMapped((void**)&output, imgSizePad * sizeof(float)) ) {
            return false;
        }

        mImageProcess = output;

        mProbInputDims.clear();
        mProbInputDims.push_back(Dims{4, {mBatchSize, mInputChannel, mImageProcessOHeight, mImageProcessOWidth}});

        mProbOutputDims.clear();
        for (int i = 0; i < mNl; i++) {
            mProbOutputDims.push_back(Dims{5, {mBatchSize, mInputChannel, mImageProcessOHeight >> (i + 3), mImageProcessOWidth >> (i + 3), 9}});
        }

    }

    cudaMemset(mImageProcess, imgSizePad, padValue);

    return true;
}

bool YOLOV5::ImagePreprocess(void* input, imageFormat format, const uint32_t height, const uint32_t width, const uint32_t newHeight,
                             const float padValue, bool scaleUp)
{
    if ( !ImagePreprocess(height, width, newHeight, newHeight, padValue, scaleUp) )
        return false;

    return cudaLetterbox(input, format, height, width);
}

// ProcessProbs
bool YOLOV5::ProcessProbs(deviceType device, cudaStream_t stream)
{
    PROFILER_BEGIN(PROFILER_NETWORK);

    std::vector<float*> inputBindings{mImageProcess};

    if ( !Run(inputBindings, mProbInputDims, mProbOutputDims, device, stream) ) {
        return false;
    }

    PROFILER_END(PROFILER_NETWORK);
    return true;
}