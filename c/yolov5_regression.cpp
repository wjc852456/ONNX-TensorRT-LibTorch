#include "yolov5_regression.h"

#include "cudaMappedMemory.h"
#include "cudaResize.h"

#include "commandLine.h"
#include "filesystem.h"
#include "logging.h"


// constructor
YOLOV5Regression::YOLOV5Regression() : tensorNet()
{
    mNetworkType = CUSTOM;
}

// destructor
YOLOV5Regression::~YOLOV5Regression()
{

}

// Create
YOLOV5Regression* YOLOV5Regression::Create( const char* prototxt_path, const char* model_path, const char* mean_binary,
                  const char* input, const char* output, uint32_t maxBatchSize,
                  bool dynamicInput, const std::vector<int32_t>& shapeRange,
                  precisionType precision, deviceType device, bool allowGPUFallback)
{
    YOLOV5Regression* net = new YOLOV5Regression();

    if ( !net )
        return NULL;

    if ( !net->init(prototxt_path, model_path, mean_binary, input, output, maxBatchSize,
                    dynamicInput, shapeRange, precision, device, allowGPUFallback) ) {
        LogError(LOG_TRT "YOLOV5Regression -- failed to initialize.\n");
        return NULL;
    }

    return net;
}

bool YOLOV5Regression::init(const char* prototxt_path, const char* model_path, const char* mean_binary,
               const char* input, const char* output, uint32_t maxBatchSize,
               bool dynamicInput, const std::vector<int32_t>& shapeRange,
               precisionType precision, deviceType device, bool allowGPUFallback )
{
    if ( !model_path || !input || !output )
        return false;

    LogInfo("\n");
    LogInfo("YOLOV5Regression -- loading classification network model from:\n");
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

    LogSuccess(LOG_TRT "YOLOV5Regression -- %s initialized.\n", model_path);
    return true;
}

// Create
YOLOV5Regression* YOLOV5Regression::Create( int argc, char** argv )
{
    return Create(commandLine(argc, argv));
}

// Create
YOLOV5Regression* YOLOV5Regression::Create( const commandLine& cmdLine )
{
    YOLOV5Regression* net = NULL;

    // obtain the network name
    const char* modelName = cmdLine.GetString("network");

    if ( !modelName )
        modelName = cmdLine.GetString("model", "googlenet");
    
    const char* prototxt = cmdLine.GetString("prototxt");
    const char* labels   = cmdLine.GetString("labels");
    const char* input    = cmdLine.GetString("input_blob");
    const char* output   = cmdLine.GetString("output_blob");

    if ( !input )    input = YOLOV5_REGRESSION_DEFAULT_INPUT;
    if ( !output )  output = YOLOV5_REGRESSION_DEFAULT_OUTPUT;

    int maxBatchSize = cmdLine.GetInt("batch_size");

    if ( maxBatchSize < 1 )
        maxBatchSize = DEFAULT_MAX_BATCH_SIZE;

    bool dynamicInput = cmdLine.GetFlag("dynamic_input");
    const std::vector<int32_t> shapeRange = cmdLine.GetVectorInt("shape_range");

    net = YOLOV5Regression::Create(prototxt, modelName, NULL, input, output, maxBatchSize, dynamicInput, shapeRange);

    if ( !net )
        return NULL;

    // enable layer profiling if desired
    if ( cmdLine.GetFlag("profile") )
        net->EnableLayerProfiler();

    return net;
}

// Process Regression
bool YOLOV5Regression::ProcessRegression(const std::vector<float*>& inputs, const std::vector<Dims>& regressionInputDims,
                     deviceType device, cudaStream_t stream)
{
    //PROFILER_BEGIN(PROFILER_NETWORK);

    if ( !Run(inputs, regressionInputDims, device, stream) ) {
        return false;
    }

    //PROFILER_END(PROFILER_NETWORK);
    return true;
}