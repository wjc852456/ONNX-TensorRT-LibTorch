#include<torch/torch.h>
#include<torch/script.h>

#include "videoSource.h"
#include "videoOutput.h"

#include "yolov5.h"

#include <signal.h>

#include "tensorConvert.h"

#ifdef HEADLESS
#define IS_HEADLESS() "headless"    // run without display
#else
#define IS_HEADLESS() (const char*)NULL
#endif

#include<ctime>

bool signal_recieved = false;

void sig_handler(int signo)
{
    if ( signo == SIGINT ) {
        LogVerbose("received SIGINT\n");
        signal_recieved = true;
    }
}

int usage()
{
    printf("usage: testnet [--help] [--network=NETWORK] ...\n");
    printf("                input_URI [output_URI]\n\n");
    printf("Run a video/image stream using an custum DNN.\n");
    printf("See below for additional arguments that may not be shown above.\n\n");
    printf("positional arguments:\n");
    printf("    input_URI       resource URI of input stream  (see videoSource below)\n");
    printf("    output_URI      resource URI of output stream (see videoOutput below)\n\n");

    printf("%s", YOLOV5::Usage());
    printf("%s", videoSource::Usage());
    printf("%s", videoOutput::Usage());
    printf("%s", Log::Usage());

    return 0;
}

int main( int argc, char** argv )
{
    /*
     * parse command line
     */
    commandLine cmdLine(argc, argv, IS_HEADLESS());

    if ( cmdLine.GetFlag("help") )
        return usage();

    /*
     * attach signal handler
     */
    if ( signal(SIGINT, sig_handler) == SIG_ERR ) {
        LogError("can't catch SIGINT\n");
    }

    /*
     * create input stream
     */
    videoSource* input = videoSource::Create(cmdLine, ARG_POSITION(0));

    if ( !input ) {
        LogError("detectnet:  failed to create input stream\n");
        return 0;
    }

    /*
     * create output stream
     */
    videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));

    if ( !output ) {
        LogError("detectnet:  failed to create output stream\n");
    }

    /*
     * create detection network
     */
    YOLOV5Detections* yolov5 = YOLOV5Detections::Create(cmdLine);

    if ( !yolov5 ) {
        LogError("failed to load yolov5 model\n");
        return 0;
    }

    const uint32_t overlayFlags = YOLOV5Detections::OverlayFlagsFromStr(cmdLine.GetString("overlay", "box,labels,conf"));
    /*
     * processing loop
     */
    while ( !signal_recieved ) {
        // capture next image image
        uchar3* image = NULL;

        if ( !input->Capture(&image, 1000) ) { // RGB image 108ms
            // check for EOS
            if ( !input->IsStreaming() )
                break;

            LogError("failed to capture video frame\n");
            continue;
        }

        // detect objects in the frame
        Detection* detections = NULL;

        const uint32_t height = input->GetHeight();
        const uint32_t width = input->GetWidth();

        uint32_t imgSize = 640;
        
        int numDetections = yolov5->RunDetection(image, height, width, imgSize, &detections, overlayFlags);

        if ( numDetections == -1 ) {
            LogError("failed to Run Detection\n");
            return -1;
        }

        // render outputs
        if ( output != NULL ) {
            output->Render(image, input->GetWidth(), input->GetHeight());

            // check if the user quit
            if ( !output->IsStreaming() )
                signal_recieved = true;
        }
    }

    /*
     * destroy resources
     */
    LogVerbose("detectnet:  shutting down...\n");


    SAFE_DELETE(input);
    SAFE_DELETE(output);
    SAFE_DELETE(yolov5);

    LogVerbose("detectnet:  shutdown complete.\n");
    return 0;
}

