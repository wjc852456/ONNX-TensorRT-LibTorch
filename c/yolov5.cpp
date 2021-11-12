
#include<torch/torch.h>
#include<torch/script.h>

#include <algorithm>

#include "yolov5.h"
#include "tensorConvert.h"

#include "cudaMappedMemory.h"
#include "cudaResize.h"
#include "cudaFont.h"

#include "commandLine.h"
#include "filesystem.h"
#include "logging.h"

using namespace torch::indexing;


float YOLOV5Detections::sAnchors[] = {
    10, 13, 16, 30, 33, 23,
    30, 61, 62, 45, 59, 119,
    116, 90, 156, 198, 373, 326
};

YOLOV5Detections::YOLOV5Detections()
{
    mYOLOV5BackBone = NULL;
    mYOLOV5Regression = NULL;
    mNMS = NULL;
    mDetections = NULL;

    mInputImageHeight = 0;
    mInputImageWidth = 0;

    mAnchorGrid = torch::from_blob(sAnchors, {mNl, 1, 3, 1, 1, 2}, torch::kFloat32).to(torch::kCUDA);
    mGrid = std::vector<torch::Tensor>(mNl, torch::ones({1, 1, 1, 1, 1}));

    mClassColors = NULL;
}

YOLOV5Detections::~YOLOV5Detections()
{
    SAFE_DELETE(mYOLOV5BackBone);
    SAFE_DELETE(mYOLOV5Regression);
    SAFE_DELETE(mNMS);

    CUDA_FREE_HOST(mDetections);

    CUDA_FREE_HOST(mClassColors);
}


cudaError_t cudaDetectionTransfer( float* input, Detection* output, const int numDets, const int entryLength );

// OverlayFlagsFromStr
uint32_t YOLOV5Detections::OverlayFlagsFromStr( const char* str_user )
{
    if ( !str_user )
        return OVERLAY_BOX;

    // copy the input string into a temporary array,
    // because strok modifies the string
    const size_t str_length = strlen(str_user);

    if ( str_length == 0 )
        return OVERLAY_BOX;

    char* str = (char*)malloc(str_length + 1);

    if ( !str )
        return OVERLAY_BOX;

    strcpy(str, str_user);

    // tokenize string by delimiters ',' and '|'
    const char* delimiters = ",|";
    char* token = strtok(str, delimiters);

    if ( !token ) {
        free(str);
        return OVERLAY_BOX;
    }

    // look for the tokens:  "box", "label", and "none"
    uint32_t flags = OVERLAY_NONE;

    while ( token != NULL ) {
        //printf("%s\n", token);

        if ( strcasecmp(token, "box") == 0 )
            flags |= OVERLAY_BOX;
        else if ( strcasecmp(token, "label") == 0 || strcasecmp(token, "labels") == 0 )
            flags |= OVERLAY_LABEL;
        else if ( strcasecmp(token, "conf") == 0 || strcasecmp(token, "confidence") == 0 )
            flags |= OVERLAY_CONFIDENCE;

        token = strtok(NULL, delimiters);
    }

    free(str);
    return flags;
}

// from detectNet.cu
cudaError_t cudaDetectionOverlay( void* input, void* output, uint32_t width, uint32_t height, imageFormat format, Detection* detections,
                                  int numDetections, float4* colors );

// Overlay
bool YOLOV5Detections::Overlay( void* input, void* output, uint32_t width, uint32_t height, imageFormat format, Detection* detections,
                                uint32_t numDetections, uint32_t flags )
{
    //PROFILER_BEGIN(PROFILER_VISUALIZE);

    if ( flags == OVERLAY_NONE ) {
        LogError(LOG_TRT "yolov5 -- Overlay() was called with OVERLAY_NONE, returning false\n");
        return false;
    }

    // if input and output are different images, copy the input to the output first
    // then overlay the bounding boxes, ect. on top of the output image
    if ( input != output ) {
        if ( CUDA_FAILED(cudaMemcpy(output, input, imageFormatSize(format, width, height), cudaMemcpyDeviceToDevice)) ) {
            LogError(LOG_TRT "yolov5 -- Overlay() failed to copy input image to output image\n");
            return false;
        }
    }

    // make sure there are actually detections
    if ( numDetections <= 0 ) {
        //PROFILER_END(PROFILER_VISUALIZE);
        return true;
    }

    // bounding box overlay
    if ( flags & OVERLAY_BOX ) {
        if ( CUDA_FAILED(cudaDetectionOverlay(input, output, width, height, format, detections, numDetections, (float4*)mClassColors)) )
            return false;
    }

    // class label overlay
    if ( (flags & OVERLAY_LABEL) || (flags & OVERLAY_CONFIDENCE) ) {
        static cudaFont* font = NULL;

        // make sure the font object is created
        if ( !font ) {
            font = cudaFont::Create(adaptFontSize(width));

            if ( !font ) {
                LogError(LOG_TRT "detectNet -- Overlay() was called with OVERLAY_FONT, but failed to create cudaFont()\n");
                return false;
            }
        }

        // draw each object's description
        std::vector< std::pair< std::string, int2 > > labels;

        for ( uint32_t n = 0; n < numDetections; n++ ) {
            const char* className  = GetClassDesc(detections[n].ClassID);
            const float confidence = detections[n].Confidence * 100.0f;
            const int2  position   = make_int2(detections[n].Right, detections[n].Top + 3);

            if ( flags & OVERLAY_CONFIDENCE ) {
                char str[256];

                if ( (flags & OVERLAY_LABEL) && (flags & OVERLAY_CONFIDENCE) )
                    sprintf(str, "%s %.1f%%", className, confidence);
                else
                    sprintf(str, "%.1f%%", confidence);

                labels.push_back(std::pair<std::string, int2>(str, position));
            } else {
                // overlay label only
                labels.push_back(std::pair<std::string, int2>(className, position));
            }
        }

        font->OverlayText(output, format, width, height, labels, make_float4(255, 255, 255, 255));
    }

    //PROFILER_END(PROFILER_VISUALIZE);
    return true;
}

YOLOV5Detections* YOLOV5Detections::Create( int argc, char** argv )
{
    return Create(commandLine(argc, argv));
}

YOLOV5Detections* YOLOV5Detections::Create( const commandLine& cmdLine  )
{
    YOLOV5Detections* detectNet = new YOLOV5Detections();

    if (!detectNet) return NULL;

    // create backbone network
    int backbone_argc = 3;
    char* backbone_argv[] = {
        "--network=./networks/yolov5/yolov5s_zhouwei_dynamic.onnx",
        "--dynamic_input",
        "--shape_range=300,800,2000"
    };
    detectNet->mYOLOV5BackBone = YOLOV5::Create(backbone_argc, backbone_argv);
    if ( !(detectNet->mYOLOV5BackBone) ) return NULL;
    detectNet->mYOLOV5BackBone->SetNumDetectLayers(detectNet->mNl);
    if ( !detectNet->AllocDetections() ) return NULL;

    // create regression
    int regression_argc = 3;
    char* regression_argv[] = {
        "--network=./networks/yolov5/regress_dynamic.onnx",
        "--dynamic_input",
        "--shape_range=10,200,500"
    };
    detectNet->mYOLOV5Regression = YOLOV5Regression::Create(regression_argc, regression_argv);
    if ( !(detectNet->mYOLOV5Regression) ) return NULL;

    // create nms
    int nms_argc = 3;
    char* nms_argv[] = {
        "--network=./networks/yolov5/nms_dynamicV2.onnx",
        "--dynamic_input",
        "--shape_range=0,1000,2000"
    };
    detectNet->mNMS = NMS::Create(nms_argc, nms_argv);
    if ( !(detectNet->mNMS) ) return NULL;

    if ( !detectNet->defaultColors() ) return NULL;

    return detectNet;
}

bool YOLOV5Detections::AllocDetections()
{
    LogVerbose(LOG_TRT "detectNet -- maximum bounding boxes:  %u\n", mMaxDetections);

    const size_t detSize = mMaxDetections * sizeof(Detection);

    if ( !cudaAllocMapped((void**)&mDetections, detSize) ) {
        return false;
    }
    memset(mDetections, 0, detSize);
    return true;
}

bool YOLOV5Detections::defaultColors()
{
    const uint32_t numClasses = GetNumClasses();

    if ( !cudaAllocMapped((void**)&mClassColors, numClasses * sizeof(float4)) )
        return false;

    // blue colors, except class 1 is green
    for ( uint32_t n = 0; n < numClasses; n++ ) {
        if ( n != 1 ) {
            mClassColors[n * 4 + 0] = 0.0f; // r
            mClassColors[n * 4 + 1] = 200.0f; // g
            mClassColors[n * 4 + 2] = 255.0f; // b
            mClassColors[n * 4 + 3] = DETECTNET_DEFAULT_ALPHA; // a
        } else {
            mClassColors[n * 4 + 0] = 0.0f; // r
            mClassColors[n * 4 + 1] = 255.0f; // g
            mClassColors[n * 4 + 2] = 175.0f; // b
            mClassColors[n * 4 + 3] = 75.0f; // a
        }
    }

    return true;
}

torch::Tensor YOLOV5Detections::MakeGrid(int nx, int ny)
{
    std::vector<torch::Tensor> grid = torch::meshgrid({torch::arange(ny), torch::arange(nx)});
    auto yv = grid[0];
    auto xv = grid[1];
    return torch::stack({xv, yv}, 2).view({1, 1, ny, nx, 2}).toType(torch::kFloat32);
}

bool YOLOV5Detections::RunBackBone()
{
    return this->mYOLOV5BackBone->ProcessProbs();
}

/***
torch::Tensor YOLOV5Detections::RunRegression()
{
    auto options = torch::TensorOptions().device(torch::kCUDA);
    std::vector<torch::Tensor> regressions;
    for (size_t i = 0; i < this->mYOLOV5BackBone->GetOutputLayers(); i++) {
        float* probOutputBinding = this->mYOLOV5BackBone->GetOutputBinding(i);
        Dims outputDim = this->mYOLOV5BackBone->GetOutputDimsAll(i);
        int64_t dim[outputDim.MAX_DIMS];
        for (int i = 0; i < outputDim.nbDims; i++) {
            dim[i] = outputDim.d[i];
        }
        torch::Tensor prob = torch::from_blob(probOutputBinding, torch::IntArrayRef(dim, outputDim.nbDims), options);
        auto ny = prob.size(2);
        auto nx = prob.size(3);
        if ( ny != mGrid[i].size(2) || nx != mGrid[i].size(3) ) {
            mGrid[i] = MakeGrid(nx, ny).to(prob.device());
        }
        auto xy = ( prob.index({"...", Slice(0, 2)}) * 2.0 - 0.5 + mGrid[i].to(prob.device()) )* mStride[i];
        prob.index_put_({"...", Slice(0, 2)}, xy);

        auto wh = ( prob.index({"...", Slice(2, 4)}) * 2.0 ).pow_(2) * mAnchorGrid[i].to(prob.device());
        prob.index_put_({"...", Slice(2, 4)}, wh);

        regressions.push_back(prob.view({-1, mNo}));
    }

    return torch::cat(regressions, 0);
}
***/


torch::Tensor YOLOV5Detections::RunRegression()
{
    int totalProbs = 0;
    std::vector<float*> regressionInputs;
    std::vector<Dims> inputDims;
    const uint32_t numBackBoneOutLayers = this->mYOLOV5BackBone->GetOutputLayers();

    for (size_t i = 0; i < numBackBoneOutLayers; i++) {
        float* probOutputBinding = this->mYOLOV5BackBone->GetOutputBinding(i);
        regressionInputs.push_back(probOutputBinding);
        inputDims.push_back(this->mYOLOV5BackBone->GetOutputDimsAll(i));

        totalProbs += 3 * (this->mYOLOV5BackBone->GetImageProcesOWidth() >> (i + 3)) * (this->mYOLOV5BackBone->GetImageProcesOHeight() >> (i + 3));
    }
    std::vector<Dims> outputDims{{3, {1, totalProbs, mNo}}};

    for (size_t i = 0; i < numBackBoneOutLayers; i++) {
        Dims outputDim = this->mYOLOV5BackBone->GetOutputDimsAll(i);

        const int32_t ny = outputDim.d[2];
        const int32_t nx = outputDim.d[3];
        if ( ny != mGrid[i].size(2) || nx != mGrid[i].size(3) ) {
            mGrid[i] = MakeGrid(nx, ny).to(torch::kCUDA);
        }
        regressionInputs.push_back((float*)mGrid[i].data_ptr());
        inputDims.push_back(Dims{5, {1, 1, ny, nx, 2}});
    }

    regressionInputs.push_back((float*)mStride.data_ptr());
    inputDims.push_back(Dims{1, {3}});

    regressionInputs.push_back((float*)mAnchorGrid.data_ptr());
    inputDims.push_back(Dims{6, {mNl, 1, 3, 1, 1, 2}});

    if ( !this->mYOLOV5Regression->Run(regressionInputs, inputDims, outputDims) ) {
        LogError(LOG_TRT "yolov5::Regression failed\n");
        return torch::Tensor();
    }

    auto options = torch::TensorOptions(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor regressions = torch::from_blob(this->mYOLOV5Regression->GetOutputBinding(0), torch::IntArrayRef({totalProbs, mNo}), options);

    return regressions;
}


torch::Tensor YOLOV5Detections::xywh2xyxy(const torch::Tensor& x)
{
    auto y = x.clone();
    y.index_put_({Slice(), 0}, x.select(1, 0) - x.select(1, 2) / 2 );
    y.index_put_({Slice(), 1}, x.select(1, 1) - x.select(1, 3) / 2 );
    y.index_put_({Slice(), 2}, x.select(1, 0) + x.select(1, 2) / 2 );
    y.index_put_({Slice(), 3}, x.select(1, 1) + x.select(1, 3) / 2 );
    return y;
}

bool YOLOV5Detections::ScaleCoords(const int ImgProcessHeight, const int ImgProcessWidth, const int ImgInputHeight, const int ImgInputWidth,
                                   torch::Tensor& dets)
{
    float gain = std::min(ImgProcessHeight * 1.0f / ImgInputHeight, ImgProcessWidth * 1.0f / ImgInputWidth);
    float padW = (ImgProcessWidth - ImgInputWidth * gain) / 2;
    float padH = (ImgProcessHeight - ImgInputHeight * gain) / 2;
    dets.select(1, 0) -= padW;
    dets.select(1, 2) -= padW;
    dets.select(1, 1) -= padH;
    dets.select(1, 3) -= padH;
    dets.slice(1, 0, 4) /= gain;

    // clip coords
    dets.select(1, 0).clip_(0, ImgInputWidth);
    dets.select(1, 1).clip_(0, ImgInputHeight);
    dets.select(1, 2).clip_(0, ImgInputWidth);
    dets.select(1, 3).clip_(0, ImgInputHeight);

    dets.slice(1, 0, 4).round_();

    return true;
}

torch::Tensor YOLOV5Detections::RunSelectAndNMS(const torch::Tensor& regressions)
{
    clock_t bgn, end;
    
    const auto probilities = (regressions.index({"...", 4}) > mConfThres);

    auto x = regressions.index({probilities});

    if (x.size(0) == 0) return x;

    auto x_slice = x.slice(1, 5);
    x_slice *= x.slice(1, 4, 5);

    auto boxes = xywh2xyxy(x.slice(1, 0, 4));

    const auto ij = torch::where(x.slice(1, 5) > mConfThres);
    const auto i = ij[0];
    const auto j = ij[1];

    auto nmsBoxes = boxes.index({i});
    auto nmsScores = x.index({i, j + 5}).unsqueeze(1);
    auto classes = j.clone().unsqueeze(1);
    auto dets = torch::cat({nmsBoxes, nmsScores, classes}, 1);

    auto c = classes * mMaxWh;
    nmsBoxes += c;
    auto boxNum = nmsBoxes.size(0);

    std::vector<float*> numsInputs{(float*)nmsBoxes.data_ptr(), (float*)nmsScores.data_ptr()};
    std::vector<Dims> nmsInputDims{Dims{4, {1, (int)boxNum, 1, 4}}, Dims{3, {1, (int)boxNum, 1}}};

    mNMS->ProcessNMS(numsInputs, nmsInputDims);

    int numDets = *(int*)mNMS->GetOutputBinding(0);
    int* nmsIndices = (int*)mNMS->GetOutputBinding(1);

    auto options = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
    auto nmsIndicesATen = torch::from_blob(nmsIndices, {numDets}, options).toType(torch::kLong);
    dets = dets.index({nmsIndicesATen});
    if ( !ScaleCoords(mYOLOV5BackBone->GetImageProcesOHeight(), mYOLOV5BackBone->GetImageProcesOWidth(),
                      mInputImageHeight, mInputImageWidth, dets) ) return torch::Tensor();

    return dets;
}

int YOLOV5Detections::RunDetection(void* input, imageFormat format, const uint32_t height, const uint32_t width, const uint32_t newHeight,
                                   Detection** detections, const uint32_t overlay, const float padValue, bool scaleUp)
{
    if (!input || width == 0 || height == 0) {
        LogError(LOG_TRT "yolov5::RunDetection( 0x%p, %u, %u ) -> invalid parameters\n", input, width, height);
        return -1;
    }

    if ( !imageFormatIsRGB(format) ) {
        LogError(LOG_TRT "yolov5::RunDetection() -- unsupported image format (%s)\n", imageFormatToStr(format));
        LogError(LOG_TRT "                       supported formats are:\n");
        LogError(LOG_TRT "                          * rgb8\n");
        LogError(LOG_TRT "                          * rgba8\n");
        LogError(LOG_TRT "                          * rgb32f\n");
        LogError(LOG_TRT "                          * rgba32f\n");

        return false;
    }

    mInputImageHeight = height;
    mInputImageWidth = width;

    if ( detections != NULL )
        *detections = mDetections;

    if ( !this->mYOLOV5BackBone->ImagePreprocess(input, format, height, width, newHeight) )
        return -1;

    if ( !RunBackBone() )
        return -1;

    auto preds = RunRegression();

    torch::Tensor dets = RunSelectAndNMS(preds);

    int numDets = dets.size(0);
    if ( numDets == 0 ) {
        LogWarning(LOG_TRT "yolov5::Detect() -- no objects detected\n");
        return 0;
    }

    if ( CUDA_FAILED(cudaDetectionTransfer((float*)dets.data_ptr(), mDetections, numDets, 6)) )
        return -1;

    CUDA(cudaDeviceSynchronize());

    if ( overlay != OverlayFlags::OVERLAY_NONE && numDets > 0 ) {
        if ( !Overlay(input, input, width, height, format, mDetections, numDets, overlay) )
            LogError(LOG_TRT "yolov5::Detect() -- failed to render overlay\n");
    }

    CUDA(cudaDeviceSynchronize());

    return numDets;
}