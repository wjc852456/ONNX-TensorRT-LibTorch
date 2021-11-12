#include "nms.h"
#include "tensorConvert.h"

#include "cudaMappedMemory.h"
#include "cudaResize.h"

#include "commandLine.h"
#include "filesystem.h"
#include "logging.h"


// constructor
NMS::NMS() : tensorNet()
{
    mNetworkType = CUSTOM;
}

// destructor
NMS::~NMS()
{

}

// Create
NMS* NMS::Create( const char* prototxt_path, const char* model_path, const char* mean_binary,
                  const char* input, const char* output, uint32_t maxBatchSize,
                  bool dynamicInput, const std::vector<int32_t>& shapeRange,
                  precisionType precision, deviceType device, bool allowGPUFallback)
{
    NMS* net = new NMS();

    if ( !net )
        return NULL;

    if ( !net->init(prototxt_path, model_path, mean_binary, input, output, maxBatchSize,
                    dynamicInput, shapeRange, precision, device, allowGPUFallback) ) {
        LogError(LOG_TRT "NMS -- failed to initialize.\n");
        return NULL;
    }

    return net;
}

bool NMS::init(const char* prototxt_path, const char* model_path, const char* mean_binary,
               const char* input, const char* output, uint32_t maxBatchSize,
               bool dynamicInput, const std::vector<int32_t>& shapeRange,
               precisionType precision, deviceType device, bool allowGPUFallback )
{
    if ( !model_path || !input || !output )
        return false;

    LogInfo("\n");
    LogInfo("NMS -- loading classification network model from:\n");
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

    LogSuccess(LOG_TRT "NMS -- %s initialized.\n", model_path);
    return true;
}

// Create
NMS* NMS::Create( int argc, char** argv )
{
    return Create(commandLine(argc, argv));
}

// Create
NMS* NMS::Create( const commandLine& cmdLine )
{
    NMS* net = NULL;

    // obtain the network name
    const char* modelName = cmdLine.GetString("network");

    if ( !modelName )
        modelName = cmdLine.GetString("model", "googlenet");
    
    const char* prototxt = cmdLine.GetString("prototxt");
    const char* labels   = cmdLine.GetString("labels");
    const char* input    = cmdLine.GetString("input_blob");
    const char* output   = cmdLine.GetString("output_blob");

    if ( !input )    input = NMS_DEFAULT_INPUT;
    if ( !output )  output = NMS_DEFAULT_OUTPUT;

    int maxBatchSize = cmdLine.GetInt("batch_size");

    if ( maxBatchSize < 1 )
        maxBatchSize = DEFAULT_MAX_BATCH_SIZE;

    bool dynamicInput = cmdLine.GetFlag("dynamic_input");
    const std::vector<int32_t> shapeRange = cmdLine.GetVectorInt("shape_range");

    net = NMS::Create(prototxt, modelName, NULL, input, output, maxBatchSize, dynamicInput, shapeRange);

    if ( !net )
        return NULL;

    // enable layer profiling if desired
    if ( cmdLine.GetFlag("profile") )
        net->EnableLayerProfiler();

    return net;
}

// ProcessNMS
bool NMS::ProcessNMS(const std::vector<float*>& inputs, const std::vector<Dims>& nmsInputDims,
                     deviceType device, cudaStream_t stream)
{
    //PROFILER_BEGIN(PROFILER_NETWORK);

    if ( !Run(inputs, nmsInputDims, device, stream) ) {
        return false;
    }

    //PROFILER_END(PROFILER_NETWORK);
    return true;
}