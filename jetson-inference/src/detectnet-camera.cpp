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


//CUDA includes
#include "cudaMappedMemory.h"
#include "cudaNormalize.h"
#include "cudaFont.h"

#include "detectNet.h"

//zmq includes
#include "zhelpers.hpp"
#include <zmq.hpp>

//misc includes
#include <string>
#include <chrono>
#include <thread>
#include <string>
#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <stdio.h>

//cscore
#include "llvm/StringRef.h"
#include <ntcore.h>
#include <iostream>
#include <llvm/raw_ostream.h>
#include <cscore.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace zmq;
using namespace std;

#define PI 3.14159265



//CAMERA SPECS
//http://vrguy.blogspot.com/2013/04/converting-diagonal-field-of-view-and.html
//logitech c270 specs; http://support.logitech.com/en_us/article/17556
//focal length calculation: http://answers.opencv.org/question/17076/conversion-focal-distance-from-mm-to-pixels/

//current is now logitech c615

//UNEEDED: const double DIAGONAL_FOV = 74;//70; //degrees
const double HORIZONTAL_FOV = 50.961435895579;//800x600:52.843265;//47.9248;//60.0; // from logitech specs //for 800x600
//52.8 inches in view from 58 inches away
//9.25 inches in view from 20 inches away
//divide by 2 to make equation for similar right triangles
//cotan(26.4/58+x) = cotan(9.25/20+x)
//x = 0.49562682
//28 inches in view 31 inches away
//cotan(14/(31+x))=23.9624 degrees
//times 2 = 47.9248
const double CUBE_HEIGHT = 11.0/12.0; //feet
const double CUBE_WIDTH = 13.0/12.0; //feet
const double HEIGHT_OFFSET = 0.0; //temporary value in feet
const double FORWARD_OFFSET = 0.0; //temporary value in feet
const double CAMERA_OFFSET = 0; //inch camera to center of mass
//camera values
//for logitech c270
/*const double VERTICAL_FOV = 37.9; //degrees
const double FOCAL_LENGTH = 1108.5;  // calculation: focal_pixel = (image_width_in_pixels * 0.5) / tan(FOV * 0.5 * PI/180)
const double SENSOR_WIDTH_mm = 4.0; //mm from logitech specs
const int PIXEL_WIDTH = 800; //1280; // pixels
const int PIXEL_HEIGHT = 600; //720; // pixels
const double CENTER_LINE = PIXEL_WIDTH*0.5; //pixels
*/
//UNEEDED: const double VERTICAL_FOV = 51.2133;//37.9; //degrees
const double FOCAL_LENGTH = 671.4754;//41226.34761500077;//1108.5;  // calculation: focal_pixel = (image_width_in_pixels * 0.5) / tan(horizontal_FOV * 0.5 * PI/180)
//UNEEDED: const double SENSOR_WIDTH_mm = 4.0; //mm from logitech specs
const int PIXEL_WIDTH = 640;//640;//640;//1280;//640;//1280;// pixels
const int PIXEL_HEIGHT = 480;//480;//480;//720;//480;//720;// pixels
//640x480 is 19.1 fps, 800x600 is 19.3 fps, 1280x720 is 13.13 fps; without display
const double CENTER_LINE = PIXEL_WIDTH*0.5; //pixels

//CHECK BELOW

const double CAMERA_ANGLE = 0.0; //degrees
const double CAMERA_HEIGHT = 6.0; //inches, was 24
const double OFFSET_TO_FRONT = 0.0; //inches

//IplImage * cvimage; //important

#define DEFAULT_CAMERA 0
//-1	// -1 for onboard camera, or change to index of /dev/video V4L2 camera (>=0)




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
    printf("*********************tcp socket connected******************");
    
    //zmq end
    
    
    if( signal(SIGINT, sig_handler) == SIG_ERR )
        printf("\ncan't catch SIGINT\n");
    
    
    /*
     * create the camera device
     */
    //system("v4l2-ctl -d /dev/video1 -c video=width=640 -c height = 480");
    //--set-fmt-video=width=640,height=480");
    gstCamera* camera = gstCamera::Create(PIXEL_WIDTH,PIXEL_HEIGHT,DEFAULT_CAMERA);//PIXEL_WIDTH, PIXEL_HEIGHT, DEFAULT_CAMERA);
    system("v4l2-ctl --set-fmt-video=width=1280,height=720,pixelformat=YUYV");
    //system("v4l2-ctl --set-fmt-video=pixelformat=YUYV");
    
    
    if( !camera )
    {
        printf("\ndetectnet-camera:  failed to initialize video device\n");
        return 0;
    }
    
    printf("\ndetectnet-camera:  successfully initialized video device\n");
    printf("    width:  %u\n", camera->GetWidth());
    printf("   height:  %u\n", camera->GetHeight());
    printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
    //camera->SetWidth(PIXEL_WIDTH);
    //camera->SetHeight(PIXEL_HEIGHT);
    
    /*
     * create detectNet
     */
    detectNet* net = detectNet::Create(argc, argv);
    
    if( !net )
    {
        printf("detectnet-camera:   failed to initialize detectNet\n");
        return 0;
    }
    int countangles = 0;
    
    /*
     * allocate memory for output bounding boxes and class confidence
     */
    const uint32_t maxBoxes = net->GetMaxBoundingBoxes();		//printf("maximum bounding boxes:  %u\n", maxBoxes);
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
    //glDisplay* display = NULL;
    glTexture* texture = NULL;
    
    if( !display ) {
        printf("\ndetectnet-camera:  failed to create openGL display\n");
    }
    else
    {
        texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB);
        
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
    
    //  printf("\ndetectnet-camera:  camera open for streaming\n");
    
    
    /*
     * processing loop
     */
    float confidence = 0.0f;

    //more streaming stuff initialization
    cs::CvSource cvsource{"cvsource", cs::VideoMode::kMJPEG, 640, 480, 15};
    cs::MjpegServer cvMjpegServer{"cvhttpserver", 8082};
    cvMjpegServer.SetSource(cvsource);


    while( !signal_recieved )
    {
        display = NULL;       //UNCOMMENT TO -----------DELETE DISPLAY----------------------------------------------
        
        void* imgCPU  = NULL;
        void* imgCUDA = NULL;
        
        // get the latest frame
        if( !camera->Capture(&imgCPU, &imgCUDA, 10000) )
            printf("\ndetectnet-camera:  failed to capture frame\n");
        
        // convert from YUV to RGBA
        void* imgRGBA = NULL;
        
        if( !camera->ConvertRGBA(imgCUDA, &imgRGBA) )
            printf("detectnet-camera:  failed to convert from NV12 to RGBA\n");
        
        // classify image with detectNet
        int numBoundingBoxes = maxBoxes;
        
        float* bb;
        double allbb[32];
        double robotAngleDegrees;
        double camDistance;
        if( net->Detect((float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), bbCPU, &numBoundingBoxes, confCPU))
        {
            
            int lastClass = 0;
            int lastStart = 0;
            countangles++;
            int index = -1; //index of widest and closest cube
            double maxcubewidth = -10;
            double cubeHeight2;
            double actualCenterOfTargets = 0;
            //printf("hi \n %d", maxcubewidth);
            printf("ANGLE COUNTS:\n %d", countangles);
           
            //if (countangles >= 500) break;
            
            /*if(numBoundingBoxes == 0){
                //printf("blanking last box values\n");
                bb[0] = 0;
                bb[1] = 0;
                bb[2] = 0;
                bb[3] = 0;
            }*/
            for( int n=0; n < numBoundingBoxes; n++ )
            {
                const int nc = confCPU[n*2+1];
                //float* bb = bbCPU + (n * 4);
                bb = bbCPU + (n * 4);
                allbb[n*4] = bb[0];
                allbb[n*4+1] = bb[1];
                allbb[n*4+2] = bb[2];
                allbb[n*4+3] = bb[3];
                
                //calc for distance
                double cubeHeight = bb[3]-bb[1];
                cubeHeight2 = cubeHeight;
                double cubeWidth = bb[2]-bb[0];
                printf("\n coordinates: %f; %f; %f; %f \n", bb[0],bb[1],bb[2],bb[3]);

		double centerOfTargets = (bb[2]+bb[0])/2.0; //center pixels X
                double centerY =(bb[3]+bb[1])/2.0; //center pixels Y

                if (n==0) {
                    maxcubewidth = cubeWidth;
		    actualCenterOfTargets = centerOfTargets;
                    index = n;
                }
                
                
                if (cubeWidth > maxcubewidth) {
                    maxcubewidth = cubeWidth;
                    actualCenterOfTargets = centerOfTargets;
                    index = n;
                }
                //printf("ready to stream\n");
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
            if (index >=0) {
                double cubeWidth = maxcubewidth;
                camDistance = (CUBE_HEIGHT*FOCAL_LENGTH)/cubeHeight2;
                //double camDistance = (CUBE_WIDTH*FOCAL_LENGTH)/(cubeWidth);
                double flatDistance = sqrt(pow(camDistance, 2) - pow(HEIGHT_OFFSET, 2));
                
                
                
                //TODO: test angletomoveapprox and angletomoveagain
                
                //find angle
                double angleToMoveApprox;
                angleToMoveApprox = (actualCenterOfTargets - CENTER_LINE) * HORIZONTAL_FOV / PIXEL_WIDTH;
                
                double angleToMoveAgain;
                angleToMoveAgain = atan((actualCenterOfTargets - CENTER_LINE) / FOCAL_LENGTH);
                double angleToMoveAgainDegrees = angleToMoveAgain * 180 / PI;
                double robotAngle = 0;
                if (angleToMoveAgainDegrees > 0) {
                    robotAngle = PI/2-atan((FORWARD_OFFSET+cos(angleToMoveAgain)*flatDistance)/flatDistance/sin(angleToMoveAgain));
                } else {
                    robotAngle = PI/2-atan((FORWARD_OFFSET+cos(angleToMoveAgain)*flatDistance)/flatDistance/sin(angleToMoveAgain));
                    if(robotAngle > PI/2) {
                        robotAngle = robotAngle - PI;
                    }
                    else if(robotAngle < -PI/2){
                        robotAngle = robotAngle + PI;
                    }
                }
                robotAngleDegrees = robotAngle * 180 / PI;
                
                
                double flatRobotDistance = camDistance*sin(angleToMoveAgain)/sin(robotAngle);
                
                string sendstring = to_string(robotAngleDegrees) + " " + to_string(angleToMoveAgainDegrees) +" " + to_string(flatRobotDistance) + " " + to_string(camDistance) + " " + to_string(flatDistance);
                //    printf("CONTENTS: %s \n", sendstring.c_str());
                printf("robot angle= %f ; camAngle= %f;  flatDistance= %f; camDistance= %f\n", robotAngleDegrees, angleToMoveAgainDegrees, flatDistance, camDistance);
                s_send(publisher, sendstring);
                
            }
            
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
//start streaming
//THIS IS WORKING
        cv::Mat m = cv::Mat::zeros(cv::Size(640,480),CV_8UC3);//CV_32FC4);
        IplImage * cvimage = cvCreateImage(cvSize(PIXEL_WIDTH, PIXEL_HEIGHT), IPL_DEPTH_8U, 3); 
        cvimage->imageData = ((char*)(imgCPU));
        m = cv::cvarrToMat(cvimage, false, false);
        cv::cvtColor(m,m,CV_BGR2RGB);
//TODO: Make this less a mess
        int largestbox[4] = {0,0,0,0};
        int largestboxindex = 0;
        for(int t = 0;t<numBoundingBoxes;t++){
            if(t == 0){
                largestbox[0] = allbb[0];
                largestbox[1] = allbb[1];
                largestbox[2] = allbb[2];
                largestbox[3] = allbb[3];
            }
            else{  //change below if robot going for greatest height instead of width
                if(largestbox[2]-largestbox[0] < allbb[4*t+2]-allbb[4*t]){
                    largestbox[0] = allbb[4*t];
		    largestbox[1] = allbb[4*t+1];
		    largestbox[2] = allbb[4*t+2];
		    largestbox[3] = allbb[4*t+3];
                    largestboxindex = t;
                }
            }
        }
        for(int t = 0;t<numBoundingBoxes;t++){
            double currentcamDistance;
            if(allbb[t*4+3]-allbb[t*4+1] > allbb[t*4+2]-allbb[t*4]){
                currentcamDistance = (CUBE_WIDTH*FOCAL_LENGTH)/(allbb[t*4+3]-allbb[t*4+1]);
            }
            else {
                currentcamDistance = (CUBE_HEIGHT*FOCAL_LENGTH)/(allbb[t*4+3]-allbb[t*4+1]);
            }
            int fontFace = cv::FONT_HERSHEY_PLAIN;
	    std::string mystring = "Distance: ";
	    mystring+=std::to_string(currentcamDistance);
            if(largestboxindex == t){
                cv::rectangle(m,cv::Point(allbb[t*4],allbb[t*4+1]),cv::Point(allbb[t*4+2],allbb[t*4+3]),cv::Scalar(255,255,0)); //colors are blue, green, red
		cv::putText(m,mystring.c_str(),cv::Point(allbb[t*4],allbb[t*4+1]),fontFace,2.0,cv::Scalar(255,255,0),2,4);
            }
	    else{
                cv::rectangle(m,cv::Point(allbb[t*4],allbb[t*4+1]),cv::Point(allbb[t*4+2],allbb[t*4+3]),cv::Scalar(0,0,255)); //colors are blue, green, red
		cv::putText(m,mystring.c_str(),cv::Point(allbb[t*4],allbb[t*4+1]),fontFace,2.0,cv::Scalar(0,0,255),2,4);
            }
        }
        //cv::resize(m, m, cvSize(320, 240), 0.5, 0.5, 3);
        cvsource.PutFrame(m);
        
//end streaming
         
    }
    
    
    
    printf("\ndetectnet-camera:  un-initializing video device\n");
    
    
    /*
     * shutdown the camera device
     */
    if( camera != NULL)
    {
        delete camera;
        camera = NULL;
    }
    
    if( display != NULL)
    {
        delete display;
        display = NULL;
    }
     
    //printf("ANGLE COUNT:\n %d", countangles);
    printf("detectnet-camera:  video device has been un-initialized.\n");
    printf("detectnet-camera:  this concludes the test of the video device.\n");
    return 0;
}





