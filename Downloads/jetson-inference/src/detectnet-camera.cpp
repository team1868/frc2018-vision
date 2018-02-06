/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include "gstCamera.h"

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "cudaMappedMemory.h"
#include "cudaNormalize.h"
#include "cudaFont.h"

#include "detectNet.h"

#include "zhelpers.hpp"
#include <zmq.hpp>
#include <string>
#include <chrono>
#include <thread>
#include <string>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <fstream>
#include <ctime>
#include <stdio.h>

using namespace cv;
using namespace zmq;
using namespace std;

#define PI 3.14159265



//CAMERA SPECS
//http://vrguy.blogspot.com/2013/04/converting-diagonal-field-of-view-and.html
//logitech c270 specs; http://support.logitech.com/en_us/article/17556
//focal length calculation: http://answers.opencv.org/question/17076/conversion-focal-distance-from-mm-to-pixels/

const double DIAGONAL_FOV = 70; //degrees
const double HORIZONTAL_FOV = 60.0; // from logitech specs
const double VERTICAL_FOV = 37.9; //degrees
const double FOCAL_LENGTH = 1108.5;  // calculation: focal_pixel = (image_width_in_pixels * 0.5) / tan(FOV * 0.5 * PI/180)
const double SENSOR_WIDTH_mm = 4.0; //mm from logitech specs
const int PIXEL_WIDTH = 1280; // pixels
const int PIXEL_HEIGHT = 720; // pixels
const double CENTER_LINE = 639.5; //pixels
//CHECK BELOW
const double CAMERA_ANGLE = 0.0; //degrees
const double CAMERA_HEIGHT = 6.0; //inches, was 24
const double OFFSET_TO_FRONT = 0.0; //inches


#define DEFAULT_CAMERA -1	// -1 for onboard camera, or change to index of /dev/video V4L2 camera (>=0)




bool signal_recieved = false;

void sig_handler(int signo)
{
    if( signo == SIGINT )
    {
        printf("received SIGINT\n");
        signal_recieved = true;
    }
}


int main( int argc, char** argv )
{
    printf("detectnet-camera\n  args (%i):  ", argc);
    
    for( int i=0; i < argc; i++ )
    printf("%i [%s]  ", i, argv[i]);
    
    printf("\n\n");
    
    //zmq start
    
    context_t context(1);
    socket_t publisher(context, ZMQ_PUB);
    
    publisher.bind("tcp://*:5563");
    
    
    
    int confl = 1;
    publisher.setsockopt(ZMQ_CONFLATE, &confl, sizeof(confl));
    
    //zmq end
    
    
    if( signal(SIGINT, sig_handler) == SIG_ERR )
    printf("\ncan't catch SIGINT\n");
    
    
    /*
     * create the camera device
     */
    gstCamera* camera = gstCamera::Create(DEFAULT_CAMERA);
    
    if( !camera )
    {
        printf("\ndetectnet-camera:  failed to initialize video device\n");
        return 0;
    }
    
    printf("\ndetectnet-camera:  successfully initialized video device\n");
    printf("    width:  %u\n", camera->GetWidth());
    printf("   height:  %u\n", camera->GetHeight());
    printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
    
    
    /*
     * create detectNet
     */
    detectNet* net = detectNet::Create(argc, argv);
    
    if( !net )
    {
        printf("detectnet-camera:   failed to initialize imageNet\n");
        return 0;
    }
    
    
    /*
     * allocate memory for output bounding boxes and class confidence
     */
    const uint32_t maxBoxes = net->GetMaxBoundingBoxes();		printf("maximum bounding boxes:  %u\n", maxBoxes);
    const uint32_t classes  = net->GetNumClasses();
    
    float* bbCPU    = NULL;
    float* bbCUDA   = NULL;
    float* confCPU  = NULL;
    float* confCUDA = NULL;
    
    if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
       !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)) )
    {
        printf("detectnet-console:  failed to alloc output memory\n");
        return 0;
    }
    
    
    /*
     * create openGL window
     */
    glDisplay* display = glDisplay::Create();
    glTexture* texture = NULL;
    
    if( !display ) {
        printf("\ndetectnet-camera:  failed to create openGL display\n");
    }
    else
    {
        texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);
        
        if( !texture )
        printf("detectnet-camera:  failed to create openGL texture\n");
    }
    
    
    /*
     * create font
     */
    cudaFont* font = cudaFont::Create();
    
    
    /*
     * start streaming
     */
    if( !camera->Open() )
    {
        printf("\ndetectnet-camera:  failed to open camera for streaming\n");
        return 0;
    }
    
    printf("\ndetectnet-camera:  camera open for streaming\n");
    
    
    /*
     * processing loop
     */
    float confidence = 0.0f;
    
    while( !signal_recieved )
    {
        void* imgCPU  = NULL;
        void* imgCUDA = NULL;
        
        // get the latest frame
        if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
        printf("\ndetectnet-camera:  failed to capture frame\n");
        
        // convert from YUV to RGBA
        void* imgRGBA = NULL;
        
        if( !camera->ConvertRGBA(imgCUDA, &imgRGBA) )
        printf("detectnet-camera:  failed to convert from NV12 to RGBA\n");
        
        // classify image with detectNet
        int numBoundingBoxes = maxBoxes;
        
        if( net->Detect((float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), bbCPU, &numBoundingBoxes, confCPU))
        {
            printf("%i bounding boxes detected\n", numBoundingBoxes);
            
            int lastClass = 0;
            int lastStart = 0;
            
            for( int n=0; n < numBoundingBoxes; n++ )
            {
                const int nc = confCPU[n*2+1];
                float* bb = bbCPU + (n * 4);
                
                
                printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]);
                
                //calculations for angle
                
                double centerOfTargets = (bb[2]+bb[0])/2;
                double centerY =(bb[3]+bb[1])/2;
                
                
                //FINDING ANGLE
                double angleToMoveApprox;
                angleToMoveApprox = (centerOfTargets - CENTER_LINE) * HORIZONTAL_FOV / PIXEL_WIDTH;
                printf("\nangletomoveapprox: %d", angleToMoveApprox);
                
                double angleToMoveAgain;
                angleToMoveAgain = atan((centerOfTargets - CENTER_LINE) / FOCAL_LENGTH);
                angleToMoveAgain = angleToMoveAgain * 180 / PI;
                
                printf("\nangletomoveagain: %d", angleToMoveAgain);
                
                string giantString = "angle: "+ to_string(angleToMoveApprox) + "bb[0] " + to_string(bb[0])+ "bb[1] " + to_string(bb[1]) + "bb[2] " + to_string(bb[2]) +"bb[3] " + to_string(bb[3]) ;
                
                s_send(publisher, giantString);
                
                
                if( nc != lastClass || n == (numBoundingBoxes - 1) )
                {
                    if( !net->DrawBoxes((float*)imgRGBA, (float*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
                                        bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
                    printf("detectnet-console:  failed to draw boxes\n");
                    
                    lastClass = nc;
                    lastStart = n;
                    
                    CUDA(cudaDeviceSynchronize());
                }
            }
            
            /*if( font != NULL )
             {
             char str[256];
             sprintf(str, "%05.2f%% %s", confidence * 100.0f, net->GetClassDesc(img_class));
             
             font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
             str, 10, 10, make_float4(255.0f, 255.0f, 255.0f, 255.0f));
             }*/
            
            if( display != NULL )
            {
                char str[256];
                sprintf(str, "TensorRT build %x | %s | %04.1f FPS", NV_GIE_VERSION, net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
                //sprintf(str, "GIE build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
                display->SetTitle(str);
            }
        }
        
        
        // update display
        if( display != NULL )
        {
            display->UserEvents();
            display->BeginRender();
            
            if( texture != NULL )
            {
                // rescale image pixel intensities for display
                CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f),
                                       (float4*)imgRGBA, make_float2(0.0f, 1.0f),
                                       camera->GetWidth(), camera->GetHeight()));
                
                // map from CUDA to openGL using GL interop
                void* tex_map = texture->MapCUDA();
                
                if( tex_map != NULL )
                {
                    cudaMemcpy(tex_map, imgRGBA, texture->GetSize(), cudaMemcpyDeviceToDevice);
                    texture->Unmap();
                }
                
                // draw the texture
                texture->Render(100,100);
            }
            
            display->EndRender();
        }
    }
    
    printf("\ndetectnet-camera:  un-initializing video device\n");
    
    
    /*
     * shutdown the camera device
     */
    if( camera != NULL )
    {
        delete camera;
        camera = NULL;
    }
    
    if( display != NULL )
    {
        delete display;
        display = NULL;
    }
    
    printf("detectnet-camera:  video device has been un-initialized.\n");
    printf("detectnet-camera:  this concludes the test of the video device.\n");
    return 0;
}
