/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: svlVidCapSrcSDI.h 3721 2012-07-10 00:45:02Z wliu25 $

  Author(s):  Wen P. Liu
  Created on: 2012

  (C) Copyright 2006-2012 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#ifndef _svlVidCapSrcSDI_h
#define _svlVidCapSrcSDI_h

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <GL/glxext.h>
#include <GL/glu.h>

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>

#include <list>
#include <iostream>
#include <sstream>

#include <X11/keysym.h>
#include <X11/Xlib.h>

#include <getopt.h>

#include "NVCtrlLib.h"
#include "NVCtrl.h"
#include "NvSDIin.h"
#include "NvSDIutils.h"
#include "NvSDIout.h"
#include "fbo.h"
#include "GraphicsN.h"

#include "ANCapi.h"
#include "commandline.h"
#include "fbo.h"
#include "audio.h"
#include "ringbuffer.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cisstStereoVision/svlFilterSourceVideoCapture.h>
#include <cisstStereoVision/svlRenderTargets.h>

#define MAX_VIDEO_OUT_STREAMS 2

class osaThread;
class svlBufferImage;
class svlVidCapSrcSDIThread;

// Render Target
/////////////////////////////////////////////////////////////////////////////////////////////
class svlVidCapSrcSDIRenderTarget : public svlRenderTargetBase
{
    friend class svlRenderTargets;
    friend class svlVidCapSrcSDIThread;

public:
    // Functions inherited from svlRenderTargetBase
    bool SetImage(unsigned char* buffer, int offsetx, int offsety, bool vflip, int index=0);
    unsigned int GetWidth();
    unsigned int GetHeight();

    // Capture and overlay function
    void* ThreadProc(void* CMN_UNUSED(param));
    // Capture call (OpenGL)
    GLenum  CaptureVideo(float runTime=0.0);
    // SDI output (OpenGL)
    GLboolean OutputVideo ();
    // Render output to OpenGL textures
    GLboolean DrawOutputScene(GLuint cudaOutTexture1=-1, GLuint cudaOutTexture2=-1, unsigned char* vtkPixelData = new unsigned char[0]);
    // Display captured video to screen as GL textures, in stacked windows
    GLenum DisplayVideo();

    void Shutdown();
    void MakeCurrentGLCtx(){glXMakeCurrent(dpy, win, ctx);}

protected:
    svlVidCapSrcSDIRenderTarget(unsigned int deviceID, unsigned int displayID=0);
    svlVidCapSrcSDIRenderTarget(Display * d, HGPUNV * g, unsigned int video_format, GLsizei num_streams, unsigned int deviceID=0, unsigned int displayID = 0);
    ~svlVidCapSrcSDIRenderTarget();

private:
    int SystemID;
    int DigitizerID;

    osaThread *Thread;
    osaThreadSignal NewFrameSignal;
    osaThreadSignal ThreadReadySignal;
    bool TransferSuccessful;
    bool KillThread;
    bool ThreadKilled;
    bool Running;

    // X windows
    Display * dpy;
    Window win;
    Window createWindow();
    void calcWindowSize();
    bool destroyWindow();

    HGPUNV *gpu;
    GLsizei m_num_streams;

    //output
    unsigned char *m_overlayBuf[MAX_VIDEO_STREAMS];   // System memory buffers
    CFBO m_FBO[MAX_VIDEO_STREAMS];			// Channel 1 FBO
    OutputOptions outputOptions;
    CNvSDIout m_SDIout;
    GLuint m_OutTexture[MAX_VIDEO_STREAMS];
    CFBO gFBO[MAX_VIDEO_OUT_STREAMS];  // FBOS, need two, one per channel
    bool m_SDIoutEnabled;

    //capture
    unsigned char *m_memBuf[MAX_VIDEO_STREAMS];   // System memory buffers
    CaptureOptions captureOptions;
    Colormap cmap;
    GLXContext ctx;        // OpenGL rendering context
    CNvSDIin m_SDIin;
    GLuint m_windowWidth;
    GLuint m_windowHeight;
    double m_inputFrameRate;
    GLuint gTexObjs[MAX_VIDEO_STREAMS];

    void drawUnsignedCharImage(GLuint gWidth, GLuint gHeight, unsigned char* imageData);
    void drawCircle(GLuint gWidth, GLuint gHeight);
    void drawOne();
    void drawTwo();
    void drawThree();
    void drawFour();

    bool setupSDIDevices(Display *d=NULL,HGPUNV *g=NULL);
    bool setupSDIinDevice(Display *d,HGPUNV *g);
    bool setupSDIoutDevice(Display * d, HGPUNV * g, unsigned int video_format);
    GLboolean setupSDIGL();
    GLboolean setupSDIinGL();
    GLboolean setupSDIoutGL();

    bool startSDIPipeline();
    bool stopSDIPipeline();

    bool cleanupSDIDevices ();
    bool cleanupSDIinDevices();
    bool cleanupSDIoutDevices ();
    GLboolean cleanupSDIGL ();
    GLboolean cleanupSDIinGL();
    GLboolean cleanupSDIoutGL();

    GLuint getTextureFromBuffer(unsigned int index);
    bool translateImage(unsigned char* src, unsigned char* dest, const int width, const int height, const int trhoriz, const int trvert, bool vflip);
};

// Video Capture
/////////////////////////////////////////////////////////////////////////////////////////////
class svlVidCapSrcSDI : public svlVidCapSrcBase
{
    friend class svlVidCapSrcSDIRenderTarget;
    friend class svlVidCapSrcSDIThread;

    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    static svlVidCapSrcSDI* GetInstance();
    svlFilterSourceVideoCapture::PlatformType GetPlatformType();
    int SetStreamCount(unsigned int numofstreams);
    int GetStreamCount(void){return NumOfStreams;}
    int GetDeviceList(svlFilterSourceVideoCapture::DeviceInfo **deviceinfo);
    int Open();
    void Close();
    int Start();
    svlImageRGB* GetLatestFrame(bool waitfornew, unsigned int videoch = 0);
    int Stop();
    bool IsRunning();
    int SetDevice(int devid, int inid, unsigned int videoch = 0);
    int GetWidth(unsigned int videoch = 0);
    int GetHeight(unsigned int videoch = 0);

    int GetFormatList(unsigned int deviceid, svlFilterSourceVideoCapture::ImageFormat **formatlist);
    int GetFormat(svlFilterSourceVideoCapture::ImageFormat& format, unsigned int videoch = 0);
    void Release();

    bool IsCaptureSupported(unsigned int sysid, unsigned int digid = 0);
    bool IsOverlaySupported(unsigned int sysid, unsigned int digid = 0);

    svlVidCapSrcSDIThread* GetCaptureProc(int i){return CaptureProc[i];}

private:

    svlVidCapSrcSDI();
    ~svlVidCapSrcSDI();

    unsigned int NumOfStreams;
    bool InitializedInput, InitializedOutput;
    bool Running;

    vctDynamicVector<int> SystemID;
    vctDynamicVector<int> DigitizerID;
    vctDynamicVector<svlBufferImage*> ImageBuffer;

    svlVidCapSrcSDIThread** CaptureProc;
    osaThread** CaptureThread;

};

// Video Capture Thread
// 20120710 - Does not render correctly wpliu
/////////////////////////////////////////////////////////////////////////////////////////////
class svlVidCapSrcSDIThread
{
public:
    svlVidCapSrcSDIThread(int streamid);
    ~svlVidCapSrcSDIThread() {Shutdown();XCloseDisplay(dpy);}
    void* Proc(svlVidCapSrcSDI* baseref);

    bool WaitForInit() { InitEvent.Wait(); return InitSuccess; }
    bool IsError() { return Error; }

    bool SetupSDIDevices(Display *d=NULL,HGPUNV *g=NULL);
    GLboolean  SetupGL();

    bool StartSDIPipeline();
    bool StopSDIPipeline();
    GLenum DisplayVideo();
    GLenum  CaptureVideo(float runTime = 0.0);
    Window CreateWindow();
    CNvSDIin GetSDIin(){return m_SDIin;}
    void Shutdown();
    void MakeCurrentGLCtx();
    Display * GetDisplay(){return dpy;}
    HGPUNV * GetGPU() {return gpu;}

private:
    int StreamID;
    bool Error;
    osaThreadSignal InitEvent;
    bool InitSuccess;
    IplImage *Frame;

    unsigned char *m_memBuf[MAX_VIDEO_STREAMS];   // System memory buffers
    unsigned char *comprBuffer[MAX_VIDEO_STREAMS];   // System memory buffers
    CaptureOptions captureOptions;
    //X stuff
    Display *dpy;
    Window win;
    Colormap cmap;
    GLXContext ctx;        // OpenGL rendering context
    HGPUNV *gpu;
    CNvSDIin m_SDIin;
    GLuint m_windowWidth;
    GLuint m_windowHeight;
    double m_inputFrameRate;
    GLuint gTexObjs[MAX_VIDEO_STREAMS];

    bool setupSDIinGL();
    bool setupSDIinDevice(Display *d,HGPUNV *g);

    void calcWindowSize();
    void drawOne();
    void drawTwo();
    void drawCircle(GLuint gWidth, GLuint gHeight);
    void drawThree();
    void drawFour();
    GLuint getTextureFromBuffer(unsigned int index);

    GLboolean cleanupGL();
    bool cleanupSDIin();
    bool cleanupSDIDevices();
    bool destroyWindow();

    void flip(unsigned char* image, int index);

};

CMN_DECLARE_SERVICES_INSTANTIATION(svlVidCapSrcSDI)

#endif // _svlVidCapSrcSDI_h
