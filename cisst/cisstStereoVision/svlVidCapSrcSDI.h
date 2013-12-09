#ifndef _svlVidCapSrcSDI_h
#define _svlVidCapSrcSDI_h

#include <windows.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glu.h>
#include <GL/wglext.h>

#include <cuda.h>
#include <cudaGL.h>

#ifndef USE_NVAPI
#define USE_NVAPI
#endif

#include "nvSDIin.h"
#include "nvSDIout.h"
#include "nvSDIutil.h"
#include "glExtensions.h"

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
#include <cisstStereoVision/svlImageProcessing.h>
#include <cisstStereoVision/svlConfig.h>

#if CISST_SVL_HAS_ISSI
#include "ISSI.h"
#include "conio.h"
#endif

#define MAX_VIDEO_OUT_STREAMS 2

class osaThread;
class svlBufferImage;
class svlVidCapSrcSDIThread;


// Video Capture & Target
/////////////////////////////////////////////////////////////////////////////////////////////
class svlVidCapSrcSDIRenderTargetCapture : public svlRenderTargetBase, public svlVidCapSrcBase
{
    friend class svlVidCapSrcSDIRenderTarget;
    friend class svlVidCapSrcSDIRenderTargetCaptureThread;
	friend class svlVidCapSrcSDIRenderTargetCaptureThreadSource;

    //CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    // Functions inherited from svlRenderTargetBase
    bool SetImage(unsigned char* buffer, int offsetx, int offsety, bool vflip, int index=0);
    unsigned int GetWidth(void){return m_videoWidth;};
    unsigned int GetHeight(void){return m_videoHeight;};

    svlVidCapSrcSDIRenderTargetCapture();
    svlVidCapSrcSDIRenderTargetCapture(unsigned int deviceID, unsigned int displayID=0);
    ~svlVidCapSrcSDIRenderTargetCapture();
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

    bool Running;
	vctDynamicVector<svlBufferImage*> ImageBuffer;
    unsigned char *m_overlayBuf[MAX_VIDEO_STREAMS];
    svlVidCapSrcSDIRenderTargetCaptureThread** CaptureProc;
	svlVidCapSrcSDIRenderTargetCaptureThreadSource** VideoSourceCaptureProc;

private:

    unsigned int NumOfStreams;
    bool InitializedInput, InitializedOutput;

    vctDynamicVector<int> SystemID;
    vctDynamicVector<int> DigitizerID;

    osaThread** CaptureThread;

	osaThreadSignal NewFrameSignal,VideoSourceNewFrameSignal;
    osaThreadSignal ThreadReadySignal,VideoSourceThreadReadySignal;
    bool TransferSuccessful;
    bool KillThread;
    bool ThreadKilled;

    osaThread **VideoSourceThread;

    unsigned int m_videoWidth;
    unsigned int m_videoHeight;
    unsigned int m_num_streams;
	double m_resizeRatio;
};

// Render Target + Capture
/////////////////////////////////////////////////////////////////////////////////////////////
class svlVidCapSrcSDIRenderTargetCaptureThreadSource
{

public:
    svlVidCapSrcSDIRenderTargetCaptureThreadSource(int streamid);
	~svlVidCapSrcSDIRenderTargetCaptureThreadSource();

	void* Proc(svlVidCapSrcSDIRenderTargetCapture* baseref);

    bool WaitForInit() { InitEvent.Wait(); return InitSuccess; }
    bool IsError() { return Error; }

private:
	void ResampleRGBA32toRGB24(unsigned char* overlay, unsigned char* src, unsigned char* dst,
															   const unsigned int srcwidth, const unsigned int srcheight,
															   const unsigned int dstwidth, const unsigned int dstheight,int index);
	void RGBA32toRGB24(unsigned char* overlay, unsigned char* input, unsigned char* output, const unsigned int pixelcount, int index);
	void ResampleRGBA24(unsigned char* src, const unsigned int srcwidth, const unsigned int srcheight,
										  unsigned char* dst, const unsigned int dstwidth, const unsigned int dstheight);
	void flip(unsigned char* image, int index);
#if CISST_SVL_HAS_ISSI
	void writeToISSI(unsigned char* image, int index, double resizeRatio = 1.0);
	unsigned char *issiBuffer[MAX_VIDEO_STREAMS];   // System memory buffers
#endif
    unsigned char *comprBuffer[MAX_VIDEO_STREAMS];   // System memory buffers

    unsigned int m_videoWidth;
    unsigned int m_videoHeight;
    unsigned int m_num_streams;
	double m_resizeRatio;
    osaThreadSignal InitEvent;
    bool InitSuccess;
    bool Error;
	int StreamID;
	unsigned int CaptureCount;

#if CISST_SVL_HAS_ISSI
	ISSI m_issi;
	ISSI_Endoscope_Image_MemoryMap_Entry *mem_left;
	ISSI_Endoscope_Image_MemoryMap_Entry *mem_right;
#endif

};

// Render Target + Capture
/////////////////////////////////////////////////////////////////////////////////////////////
class svlVidCapSrcSDIRenderTargetCaptureThread
{

public:
    svlVidCapSrcSDIRenderTargetCaptureThread(int streamid);
    ~svlVidCapSrcSDIRenderTargetCaptureThread() {Shutdown();}

	void* Proc(svlVidCapSrcSDIRenderTargetCapture* baseref);

    bool WaitForInit() { InitEvent.Wait(); return InitSuccess; }
    bool IsError() { return Error; }
    unsigned char *m_memBuf[MAX_VIDEO_STREAMS];   // System memory buffers
	unsigned int CaptureCount;
    CNvSDIin m_SDIin;

private:
    void Shutdown();
	double resizeRatio;

    // Capture call (OpenGL)
    GLenum  CaptureVideo(float runTime=0.0);
    // SDI output (OpenGL)
    GLboolean OutputVideo ();
    // Render output to OpenGL textures
    GLboolean DrawOutputScene();
    // Display captured video to screen as GL textures, in stacked windows
	void cudaCheckErrors(char *label);
	HRESULT cudaInitDevice();	
	HRESULT cudaRegisterBuffers();
	HRESULT cudaUnregisterBuffers();
	bool doCuda();
	bool ProcessVideo();
	HRESULT setupCUDA();

	CUdevice cuDevice;
	CUcontext cuContext;	
	cudaGraphicsResource_t cudaVideoGraphicsResource[MAX_VIDEO_STREAMS];
	cudaGraphicsResource_t cudaOutBufferGraphicsResource[MAX_VIDEO_STREAMS];
 
    osaThreadSignal InitEvent;
    bool InitSuccess;
    bool Error;
	int StreamID;

    unsigned int m_videoWidth;
    unsigned int m_videoHeight;
    unsigned int m_num_streams;

    void drawUnsignedCharImage(unsigned char* imageData);
    void drawCircle(GLuint gWidth, GLuint gHeight);

    CNvSDIout m_SDIout;

    CFBO m_FBO[MAX_VIDEO_STREAMS];
    GLuint m_OutTexture[MAX_VIDEO_STREAMS];

	CNvSDIoutGpu *m_gpu;
	Options options;
	bool m_bSDIout;

	GLuint m_windowWidth;                   // Window width
	GLuint m_windowHeight;                  // Window height
	unsigned int m_videoBufferPitch;

	HWND g_hWnd;
	HWND m_hWnd;							// Window handle
	HDC	m_hDC;								// Device context
	HGLRC m_hRC;							// OpenGL rendering context
	HDC m_hAffinityDC;
	GLuint m_videoTextures[MAX_VIDEO_OUT_STREAMS];
	GLuint m_videoFrameBuffer[MAX_VIDEO_OUT_STREAMS];
	GLuint m_vidBufObj[MAX_VIDEO_STREAMS];

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

	unsigned int m_streamID;
	svlVidCapSrcSDIRenderTargetCapture* m_baseref;

};

// Render Target
/////////////////////////////////////////////////////////////////////////////////////////////
class svlVidCapSrcSDIRenderTarget : public svlRenderTargetBase
{
    friend class svlRenderTargets;
    friend class svlVidCapSrcSDIThread;

public:
    // Functions inherited from svlRenderTargetBase
    bool SetImage(unsigned char* buffer, int offsetx, int offsety, bool vflip, int index=0);
    unsigned int GetWidth(){return m_videoWidth;}
    unsigned int GetHeight(){return m_videoHeight;}
    void Shutdown();

protected:
    svlVidCapSrcSDIRenderTarget(unsigned int deviceID, unsigned int displayID=0);
    ~svlVidCapSrcSDIRenderTarget();

private:
    // Capture and overlay function
    void* ThreadProc(void* CMN_UNUSED(param));
    // Capture call (OpenGL)
    GLenum  CaptureVideo(float runTime=0.0);
    // SDI output (OpenGL)
    GLboolean OutputVideo ();
    // Render output to OpenGL textures
    GLboolean DrawOutputScene();
    // Display captured video to screen as GL textures, in stacked windows
    //GLenum DisplayVideo();


    int SystemID;
    int DigitizerID;

    osaThread *Thread;
    osaThreadSignal NewFrameSignal;
    osaThreadSignal ThreadReadySignal;
    bool TransferSuccessful;
    bool KillThread;
    bool ThreadKilled;
    bool Running;


    unsigned int m_videoWidth;
    unsigned int m_videoHeight;
    unsigned int m_num_streams;

    void drawUnsignedCharImage(unsigned char* imageData);
    void drawCircle(GLuint gWidth, GLuint gHeight);

    CNvSDIin m_SDIin;
    CNvSDIout m_SDIout;
    unsigned char *m_overlayBuf[MAX_VIDEO_STREAMS]; 
    CFBO m_FBO[MAX_VIDEO_STREAMS];
    GLuint m_OutTexture[MAX_VIDEO_STREAMS];

	CNvSDIoutGpu *m_gpu;
	Options options;
	bool m_bSDIout;

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

// svlVidCapSrcSDIThread
/////////////////////////////////////////////////////////////////////////////////////////////
class svlVidCapSrcSDIThread
{
public:
    svlVidCapSrcSDIThread(int streamid);
    ~svlVidCapSrcSDIThread() {Shutdown();}

	void* Proc(svlVidCapSrcSDIRenderTargetCapture* baseref);
    void* Proc(svlVidCapSrcSDI* baseref);

    bool WaitForInit() { InitEvent.Wait(); return InitSuccess; }
    bool IsError() { return Error; }

private:
    int StreamID;
    bool Error;
    osaThreadSignal InitEvent;
    bool InitSuccess;
    IplImage *Frame;
    unsigned char *m_memBuf[MAX_VIDEO_STREAMS];   // System memory buffers
    unsigned char *comprBuffer[MAX_VIDEO_STREAMS];   // System memory buffers

    //GLenum DisplayVideo();
    void flip(unsigned char* image, int index);
    GLenum  CaptureVideo(float runTime = 0.0);
    CNvSDIin GetSDIin(){return m_SDIin;}
    void Shutdown();

	HRESULT SetupSDIDevices();
    HRESULT StartSDIPipeline();

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
};

CMN_DECLARE_SERVICES_INSTANTIATION(svlVidCapSrcSDI)
//CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlVidCapSrcSDIRenderTargetCapture)

#endif // _svlVidCapSrcSDI_h