/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "videoSource.h"
#include "videoOutput.h"

#include "logging.h"
#include "commandLine.h"

#include <signal.h>


bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		LogInfo("received SIGINT\n");
		signal_recieved = true;
	}
}

int usage()
{
	printf("usage: video-viewer [--help] input_URI [output_URI]\n\n");
	printf("View/output a video or image stream.\n");
	printf("See below for additional arguments that may not be shown above.\n\n");
	printf("positional arguments:\n");
	printf("    input_URI       resource URI of input stream  (see videoSource below)\n");
	printf("    output_URI      resource URI of output stream (see videoOutput below)\n\n");

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
	commandLine cmdLine(argc, argv);

	if( cmdLine.GetFlag("help") )
		return usage();


	/*
	 * attach signal handler
	 */	
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		LogError("can't catch SIGINT\n");


	/*
	 * create input video stream
	 */
	videoSource* inputStream = videoSource::Create(cmdLine, ARG_POSITION(0));

	if( !inputStream )
	{
		LogError("video-viewer:  failed to create input stream\n");
		return 0;
	}


	/*
	 * create output video stream
	 */
	videoOutput* outputStream = videoOutput::Create(cmdLine, ARG_POSITION(1));
	
	if( !outputStream )
		LogError("video-viewer:  failed to create output stream\n");


	/*
	 * capture/display loop
	 */
	uint32_t numFrames = 0;

	while( !signal_recieved )
	{
		uchar3* nextFrame = NULL;

		if( !inputStream->Capture(&nextFrame, 1000) )
		{
			LogError("video-viewer:  failed to capture video frame\n");

			if( !inputStream->IsStreaming() )
				signal_recieved = true;

			continue;
		}

		LogInfo("video-viewer:  captured %u frames (%u x %u)\n", ++numFrames, inputStream->GetWidth(), inputStream->GetHeight());

		if( outputStream != NULL )
		{
			outputStream->Render(nextFrame, inputStream->GetWidth(), inputStream->GetHeight());

			// update status bar
			char str[256];
			sprintf(str, "Video Viewer (%ux%u) | %.1f FPS", inputStream->GetWidth(), inputStream->GetHeight(), outputStream->GetFrameRate());
			outputStream->SetStatus(str);	

			// check if the user quit
			if( !outputStream->IsStreaming() )
				signal_recieved = true;
		}
	}


	/*
	 * destroy resources
	 */
	printf("video-viewer:  shutting down...\n");
	
	SAFE_DELETE(inputStream);
	SAFE_DELETE(outputStream);

	printf("video-viewer:  shutdown complete\n");
}

