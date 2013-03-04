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

#ifdef _WIN32
#include <windows.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glu.h>
#include <GL/wglext.h>

#ifndef USE_NVAPI
#define USE_NVAPI
#endif

#include "nvSDIin.h"
#include "nvSDIout.h"
#include "nvSDIutil.h"
#include "glExtensions.h"
#else

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <GL/glxext.h>
#include <GL/glu.h>

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
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <pthread.h>
#endif

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

#include <fcntl.h>
#include <errno.h>
#include <string.h>

#include <list>
#include <iostream>
#include <sstream>

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

    void MakeCurrentGLCtx(){
#ifdef _WIN32
		wglMakeCurrent(m_hDC, m_hRC); 
#else
		glXMakeCurrent(dpy, win, ctx);
#endif
	}

protected:
    svlVidCapSrcSDIRenderTarget(unsigned int deviceID, unsigned int displayID=0);
    ~svlVidCapSrcSDIRenderTarget();
#ifndef _WIN32
    svlVidCapSrcSDIRenderTarget(Display * d, HGPUNV * g, unsigned int video_format, GLsizei num_streams, unsigned int deviceID=0, unsigned int displayID = 0);
#endif

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

    void drawUnsignedCharImage(GLuint gWidth, GLuint gHeight, unsigned char* imageData);
    void drawCircle(GLuint gWidth, GLuint gHeight);
#ifdef _WIN32
    CNvSDIin m_SDIin;
    CNvSDIout m_SDIout;
    unsigned char *m_overlayBuf[MAX_VIDEO_STREAMS]; 
    CFBO m_FBO[MAX_VIDEO_STREAMS];
    GLuint m_OutTexture[MAX_VIDEO_STREAMS];

	CNvSDIoutGpu *m_gpu;
	Options options;
	bool m_bSDIout;
	GLuint m_num_streams;
	int m_videoWidth;
	int m_videoHeight;
	GLuint m_windowWidth;                   // Window width
	GLuint m_windowHeight;                  // Window height
	unsigned int m_videoBufferPitch;

	GLuint m_vidBufObj[MAX_VIDEO_STREAMS];

	HWND g_hWnd;
	HWND m_hWnd;							// Window handle
	HDC	m_hDC;								// Device context
	HGLRC m_hRC;							// OpenGL rendering context
	HDC m_hAffinityDC;
	GLuint m_videoTextures[MAX_VIDEO_OUT_STREAMS];

	HRESULT Configure(char *szCmdLine[]);
	HRESULT SetupSDIDevices();	
	HRESULT setupSDIinDevices();
	HRESULT setupSDIoutDevice();

	GLboolean SetupGL();
	HRESULT setupSDIinGL();
	HRESULT setupSDIoutGL();
	HRESULT StartSDIPipeline();
	HRESULT stopSDIPipeline();
    GLboolean cleanupSDIGL();
	HRESULT cleanupSDIinGL();
	HRESULT cleanupSDIoutGL();
    //bool cleanupSDIDevices ();

	void CalcWindowSize();
	HWND SetupWindow(HINSTANCE hInstance, int x, int y, char *title);

#else
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
#endif
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
#ifdef _WIN32
    ~svlVidCapSrcSDIThread() {Shutdown();}
#else
    ~svlVidCapSrcSDIThread() {Shutdown();XCloseDisplay(dpy);}
#endif
    void* Proc(svlVidCapSrcSDI* baseref);

    bool WaitForInit() { InitEvent.Wait(); return InitSuccess; }
    bool IsError() { return Error; }

    GLenum DisplayVideo();
    GLenum  CaptureVideo(float runTime = 0.0);

    CNvSDIin GetSDIin(){return m_SDIin;}

#ifdef _WIN32
    void Shutdown();
	HRESULT SetupSDIDevices();	
	HRESULT StartSDIPipeline();
#else
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
#endif

private:
    int StreamID;
    bool Error;
    osaThreadSignal InitEvent;
    bool InitSuccess;
    IplImage *Frame;
    unsigned char *m_memBuf[MAX_VIDEO_STREAMS];   // System memory buffers
    unsigned char *comprBuffer[MAX_VIDEO_STREAMS];   // System memory buffers

    void flip(unsigned char* image, int index);

#ifdef _WIN32
    CNvSDIin m_SDIin;
	CNvSDIoutGpu *m_gpu;
	Options options;	
	GLuint m_num_streams;
	int m_videoWidth;
	int m_videoHeight;
	GLuint m_windowWidth;                   // Window width
	GLuint m_windowHeight;                  // Window height
	HWND g_hWnd;
	HWND m_hWnd;							// Window handle
	HDC	m_hDC;								// Device context
	HGLRC m_hRC;							// OpenGL rendering context
	HDC m_hAffinityDC;
	GLuint m_videoTextures[MAX_VIDEO_STREAMS];

	HRESULT Configure(char *szCmdLine[]);
	HRESULT setupSDIinDevices();
	HRESULT setupSDIoutDevice();
	GLboolean SetupGL();
	HRESULT setupSDIinGL();
	HRESULT setupSDIoutGL();
	void CalcWindowSize();
	HWND SetupWindow(HINSTANCE hInstance, int x, int y, char *title);
	HRESULT stopSDIPipeline();
    GLboolean cleanupSDIGL();
	HRESULT cleanupSDIinGL();
    //bool cleanupSDIDevices ();
#else

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
#endif

};

CMN_DECLARE_SERVICES_INSTANTIATION(svlVidCapSrcSDI)

#endif // _svlVidCapSrcSDI_h
