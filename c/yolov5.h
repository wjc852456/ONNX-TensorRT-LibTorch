#ifndef __YOLOV5_H__
#define __YOLOV5_H__

#include<torch/torch.h>
#include<torch/script.h>

#include "detect_utils.h"
#include "yolov5_backbone.h"
#include "nms.h"
#include "yolov5_regression.h"

#define DETECTNET_DEFAULT_ALPHA 120

class YOLOV5Detections
{
public:

    enum OverlayFlags {
        OVERLAY_NONE       = 0,         /**< No overlay. */
        OVERLAY_BOX        = (1 << 0),  /**< Overlay the object bounding boxes */
        OVERLAY_LABEL      = (1 << 1),  /**< Overlay the class description labels */
        OVERLAY_CONFIDENCE = (1 << 2),  /**< Overlay the detection confidence values */
    };

    static uint32_t OverlayFlagsFromStr( const char* flags );

    static YOLOV5Detections* Create( int argc, char** argv );

    static YOLOV5Detections* Create( const commandLine& cmdLine );

    bool AllocDetections();
    bool defaultColors();

    int RunDetection(void* input, imageFormat format, const uint32_t height, const uint32_t width, const uint32_t newHeight,
                     Detection** detections, const uint32_t overlay = OVERLAY_NONE, const float padValue = 114.0 / 255, bool scaleUp = true);

    template<typename T>
    int RunDetection(T* input, const uint32_t height, const uint32_t width, const uint32_t newHeight, Detection** detections,
                     const uint32_t overlay = OVERLAY_NONE, const float padValue = 114.0 / 255, bool scaleUp = true)
    {
        return RunDetection((void*)input, imageFormatFromType<T>(), height, width, newHeight, detections, overlay, padValue, scaleUp);
    }

    bool Overlay( void* input, void* output, uint32_t width, uint32_t height, imageFormat format, Detection* detections, uint32_t numDetections,
                  uint32_t flags = OVERLAY_BOX );

    inline const char* GetClassDesc( uint32_t index )   const
    {
        return mClassDesc[index].c_str();
    }

    inline uint32_t GetNumClasses() const { return mNc; }

    static torch::Tensor MakeGrid(int nx, int ny);

    static inline torch::Tensor xywh2xyxy(const torch::Tensor& x);

    ~YOLOV5Detections();

private:
    YOLOV5Detections();

    static float sAnchors[];

    inline bool RunBackBone();

    inline torch::Tensor RunRegression();

    torch::Tensor RunSelectAndNMS(const torch::Tensor& regressions);

    bool ScaleCoords(const int ImgProcessHeight, const int ImgProcessWidth, const int ImgInputHeight, const int ImgInputWidth,
                     torch::Tensor& dets);

    YOLOV5* mYOLOV5BackBone;
    YOLOV5Regression* mYOLOV5Regression;
    NMS* mNMS;

    int mInputImageHeight;
    int mInputImageWidth;


    Detection* mDetections;
    const int mNc = 4;
    const int mNl = 3;
    const int mNa = 3;
    const int mNo = mNc + 5;
    const torch::Tensor mStride = torch::tensor({8, 16, 32}, torch::kFloat32).to(torch::kCUDA);
    torch::Tensor mAnchorGrid;
    std::vector<torch::Tensor> mGrid;
    const int mMaxDetections = 300;
    const int mMinWh = 2;
    const int mMaxWh = 4096;
    const float mConfThres = 0.25;

    float* mClassColors;
    const std::vector<std::string> mClassDesc{
        "Pedestrians",
        "MotorVehicles",
        "NonMotorVehicles",
        "Masaic"
    };
};


#endif
