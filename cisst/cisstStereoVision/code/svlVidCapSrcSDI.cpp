#include <cisstStereoVision/svlVidCapSrcSDI.h>
#include <cisstOSAbstraction/osaThread.h>
#include <cisstStereoVision/svlBufferImage.h>
#include <cisstOSAbstraction/osaSleep.h>
#include <cisstStereoVision/svlConverters.h>

#define __VERBOSE__  0
#define SDINUMDEVICES 1

#ifndef _WIN32
//-----------------------------------------------------------------------------
// Name: WaitForNotify
// Desc: Wait for notify event routine
//-----------------------------------------------------------------------------
static
Bool WaitForNotify(Display * d, XEvent * e, char *arg)
{
    return (e->type == MapNotify) && (e->xmap.window == (Window) arg);
}

//-----------------------------------------------------------------------------
// Name: calcScaledVideoDimensions
// Desc: Calculate scaled video dimensions that preserve aspect ratio
//-----------------------------------------------------------------------------
GLvoid
calcScaledVideoDimensions(GLuint ww, GLuint wh, GLuint vw, GLuint vh,
                          GLfloat *svw, GLfloat *svh)
{
    GLfloat fww = ww;
    GLfloat fwh = wh;
    GLfloat fvw = vw;
    GLfloat fvh = vh;

    // Set the scale video width to the window width.
    // Scale the video height by the aspect ratio.
    // If the resulting height is greater than the
    // window height, the set the video height to
    // the window height and scale the width by the
    // video aspect ratio.
    *svw = fww;
    *svh = (fvh / fvw) * *svw;
    if (*svh > wh) {
        *svh = fwh;
        *svw = (fvw / fvh) * *svh;
    }

    // Normalize
    *svh /= fwh;
    *svw /= fww;
}

#endif

/*************************************/
/* svlVidCapSrcSDIRenderTarget class */
/*************************************/
svlVidCapSrcSDIRenderTarget::svlVidCapSrcSDIRenderTarget(unsigned int deviceID, unsigned int displayID):
    svlRenderTargetBase(),
    Thread(0),
    TransferSuccessful(true),
    KillThread(false),
    ThreadKilled(true),
    Running(false)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDIRenderTarget::constructor()" << std::endl;
#endif

    SystemID = deviceID;
    DigitizerID = displayID;

#ifndef _WIN32
    // Fixes this error in linux:
    //[xcb] Unknown request in queue while dequeuing
    //[xcb] Most likely this is a multi-threaded client and XInitThreads has not been called
    //[xcb] Aborting, sorry about that.
    //../src/xcb_io.c:178: dequeue_pending_request: Assertion `!xcb_xlib_unknown_req_in_deq' failed.
    XInitThreads();
#endif
    // Start up overlay/capture thread
    Thread = new osaThread;
    Thread->Create<svlVidCapSrcSDIRenderTarget, void*>(this, &svlVidCapSrcSDIRenderTarget::ThreadProc, 0);
    if (ThreadReadySignal.Wait(2.0) && ThreadKilled == false) {
        ThreadReadySignal.Raise();
    }
    else {
        // If it takes longer than 2 sec, don't execute
        KillThread = true;
    }
    Running = true;
}

svlVidCapSrcSDIRenderTarget::~svlVidCapSrcSDIRenderTarget()
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDIRenderTarget::destructor()" << std::endl;
#endif

    KillThread = true;
    if (ThreadKilled == false) Thread->Wait();
    delete Thread;

    Shutdown();
}

#ifdef _WIN32
void* svlVidCapSrcSDIRenderTarget::ThreadProc(void* CMN_UNUSED(param))
{

    LPSTR lpCmdLine = 0;
    if(Configure(&lpCmdLine) == E_FAIL)
        return FALSE;

    if(SetupSDIDevices() == E_FAIL)
        return FALSE;

    // Calculate the window size based on the incoming and outgoing video signals
    CalcWindowSize();

    HINSTANCE hInstance = GetModuleHandle(NULL);	// Need a handle to this process instance
    // Create window.  Use video dimensions of video initialized above.
    g_hWnd = SetupWindow(hInstance, 0, 0, "NVIDIA Quadro SDI Capture to memory");

    // Exit on error.
    if (!g_hWnd)
        return FALSE;

    SetupGL();

    if(StartSDIPipeline() == E_FAIL)
        return FALSE;

    // debug
    std::cout << "svlVidCapSrcSDIRenderTarget::ThreadProc(), pitches: " << m_SDIin.GetBufferObjectPitch (0) << ", " << m_SDIin.GetBufferObjectPitch (1) << " width: " << m_SDIin.GetWidth() << " height: " << m_SDIin.GetHeight() << std::endl;

    // Allocate overlay buffers
    for (int i = 0; i < m_SDIin.GetNumStreams (); i++) {
        m_overlayBuf[i] = (unsigned char *) malloc (m_SDIin.GetWidth() * m_SDIin.GetHeight()*4);
    }

    // Main capture + overlay loop
    ThreadKilled = false;
    ThreadReadySignal.Raise();

    while (!KillThread) {
        if (CaptureVideo() != GL_FAILURE_NV)
        {
            //DisplayVideo();
            DrawOutputScene();
            OutputVideo();
            if(NewFrameSignal.Wait(0.0005))
            {
                ThreadReadySignal.Raise();
            }
        }
    }

    ThreadReadySignal.Raise();
    ThreadKilled = true;

    return this;
}
#else
void* svlVidCapSrcSDIRenderTarget::ThreadProc(void* CMN_UNUSED(param))
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDIRenderTarget::ThreadProc()" << std::endl;
#endif

    // GPU
    HGPUNV gpuList[MAX_GPUS];
    // Open X display
    dpy = XOpenDisplay(NULL);
    // Scan the systems for GPUs
    int	num_gpus = ScanHW(dpy,gpuList);
    if(num_gpus < 1)
        exit(1);
    // Grab the first GPU for now for DVP
    gpu = &gpuList[0];

    // Setup
    setupSDIDevices();
    win = createWindow();
    setupSDIGL();
    startSDIPipeline();

    // debug
    std::cout << "svlVidCapSrcSDIRenderTarget::ThreadProc(), pitches: " << m_SDIin.getBufferObjectPitch (0) << ", " << m_SDIin.getBufferObjectPitch (1) << " height: " << m_SDIin.getHeight() << std::endl;

    // Allocate overlay buffers
    for (int i = 0; i < m_SDIin.getNumStreams (); i++) {
        m_overlayBuf[i] = (unsigned char *) malloc (m_SDIin.getWidth() * m_SDIin.getHeight()*4);
    }

    // Main capture + overlay loop
    ThreadKilled = false;
    ThreadReadySignal.Raise();

    while (!KillThread) {
        if (CaptureVideo() != GL_FAILURE_NV)
        {
            //DisplayVideo();
            DrawOutputScene();
            OutputVideo();
            if(NewFrameSignal.Wait(0.0005))
            {
                ThreadReadySignal.Raise();
            }
        }
    }

    ThreadReadySignal.Raise();
    ThreadKilled = true;

    return this;
}
#endif

//This routine should be called after the capture has already been configured
//since it relies on the capture signal format configuration
bool svlVidCapSrcSDIRenderTarget::SetImage(unsigned char* buffer, int offsetx, int offsety, bool vflip, int index)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDIRenderTarget::SetImage(): " << index << " (" << offsetx<<"," <<offsety<< ")" <<std::endl;
#endif

    if (SystemID < 0) return false;

    // Wait for thread to finish previous transfer
    if (ThreadReadySignal.Wait(2.0) == false ||TransferSuccessful == false || KillThread || ThreadKilled) {
        //Something went terribly wrong on the thread
        return false;
    }

    // Copy image to overlay buffer
    if(index >= 0 && index < (int)m_num_streams)
    {
        TransferSuccessful = translateImage(buffer,
                                            m_overlayBuf[index],
                                            GetWidth() * 4,
                                            GetHeight(),
                                            offsetx*4,
                                            offsety,
                                            vflip);
    }else
    {
        return false;
    }
    //memcpy(m_overlayBuf[0],buffer,m_SDIout.getWidth()*m_SDIout.getHeight()*4);

    // Signal Thread that there is a new frame to transfer
    NewFrameSignal.Raise();

    // Frame successfully filed for transfer
    return true;
}

bool svlVidCapSrcSDIRenderTarget::translateImage(unsigned char* src, unsigned char* dest, const int width, const int height, const int trhoriz, const int trvert, bool vflip)
{
#if __VERBOSE__ == 1
    std::cerr << "svlVidCapSrcSDIRenderTarget::TranslateImage()" << std::endl;
#endif

    int abs_h = std::abs(trhoriz);
    int abs_v = std::abs(trvert);

    if (vflip) {
        if (width <= abs_h || height <= abs_v) {
            src += width * (height - 1);
            for (int j = 0; j < height; j ++) {
                memcpy(dest, src, width);
                src -= width;
                dest += width;
            }
            return true;
        }

        int linecopysize = width - abs_h;
        int xfrom = std::max(0, trhoriz);
        int yfrom = std::max(0, trvert);
        int yto = height + std::min(0, trvert);
        int copyxoffset = std::max(0, -trhoriz);
        int copyyoffset = std::max(0, -trvert);

        if (trhoriz == 0) {
            src += width * (height - copyyoffset - 1);
            dest += width * yfrom;
            for (int j = height - abs_v - 1; j >= 0; j --) {
                memcpy(dest, src, width);
                src -= width;
                dest += width;
            }
            return true;
        }

        src += width * (height - copyyoffset - 1) + copyxoffset;
        dest += width * yfrom + xfrom;
        for (int j = yfrom; j < yto; j ++) {
            memcpy(dest, src, linecopysize);
            src -= width;
            dest += width;
        }

        return true;
    }
    else {
        if (width <= abs_h || height <= abs_v) {
            memset(dest, 0, width * height);
            return false;
        }

        if (trhoriz == 0) {
            memcpy(dest + std::max(0, trvert) * width,
                   src + std::max(0, -trvert) * width,
                   width * (height - abs_v));
            return true;
        }

        int linecopysize = width - abs_h;
        int xfrom = std::max(0, trhoriz);
        int yfrom = std::max(0, trvert);
        int yto = height + std::min(0, trvert);
        int copyxoffset = std::max(0, -trhoriz);
        int copyyoffset = std::max(0, -trvert);

        src += width * copyyoffset + copyxoffset;
        dest += width * yfrom + xfrom;
        for (int j = yfrom; j < yto; j ++) {
            memcpy(dest, src, linecopysize);
            src += width;
            dest += width;
        }

        return true;
    }

    return false;
}
void svlVidCapSrcSDIRenderTarget::drawCircle(GLuint gWidth, GLuint gHeight)
{

    glViewport(0, 0, gWidth, gHeight);
    glClearColor(0.0, 0.0, 0.0, 0.0);
    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); /* clear window and z-buffer */

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat) gWidth / (GLfloat) gHeight, 1.0, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //                   eye point          center of view       up
    GLfloat distance = 10.0;
    gluLookAt( distance, distance, distance, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    glColor4f(1.0, 0.0, 0.0, 0.5);
    static GLUquadric * qobj = NULL;
    if (! qobj){
        qobj = gluNewQuadric();
        gluQuadricDrawStyle(qobj, GLU_FILL);
        gluQuadricNormals(qobj, GLU_SMOOTH);
        gluQuadricTexture(qobj, GL_TRUE);
    }
    gluSphere (qobj, 3, 50, 100);
}

#ifdef _WIN32
//
// Calculate the graphics window size.
//
void svlVidCapSrcSDIRenderTarget::CalcWindowSize()
{  
    switch(m_SDIin.GetSignalFormat()) {
    case NVVIOSIGNALFORMAT_487I_59_94_SMPTE259_NTSC:
    case NVVIOSIGNALFORMAT_576I_50_00_SMPTE259_PAL:
        if (m_SDIin.GetNumStreams() == 1) {
            m_windowWidth = m_videoWidth; m_windowHeight = m_videoHeight;
        } else if (m_SDIin.GetNumStreams() == 2) {
            m_windowWidth = m_videoWidth; m_windowHeight = m_videoHeight<<1;
        } else {
            m_windowWidth = m_videoWidth<<1; m_windowHeight = m_videoHeight<<1;
        }
        break;

    case NVVIOSIGNALFORMAT_720P_59_94_SMPTE296:
    case NVVIOSIGNALFORMAT_720P_60_00_SMPTE296:
    case NVVIOSIGNALFORMAT_720P_50_00_SMPTE296:
    case NVVIOSIGNALFORMAT_720P_30_00_SMPTE296:
    case NVVIOSIGNALFORMAT_720P_29_97_SMPTE296:
    case NVVIOSIGNALFORMAT_720P_25_00_SMPTE296:
    case NVVIOSIGNALFORMAT_720P_24_00_SMPTE296:
    case NVVIOSIGNALFORMAT_720P_23_98_SMPTE296:
        if (m_SDIin.GetNumStreams() == 1) {
            m_windowWidth = m_videoWidth>>2; m_windowHeight = m_videoHeight>>2;
        } else if (m_SDIin.GetNumStreams() == 2) {
            m_windowWidth = m_videoWidth>>2; m_windowHeight = m_videoHeight>>1;
        } else {
            m_windowWidth = m_videoWidth>>1; m_windowHeight = m_videoHeight>>1;
        }
        break;

    case NVVIOSIGNALFORMAT_1035I_59_94_SMPTE260:
    case NVVIOSIGNALFORMAT_1035I_60_00_SMPTE260:
    case NVVIOSIGNALFORMAT_1080I_50_00_SMPTE295:
    case NVVIOSIGNALFORMAT_1080I_50_00_SMPTE274:
    case NVVIOSIGNALFORMAT_1080I_59_94_SMPTE274:
    case NVVIOSIGNALFORMAT_1080I_60_00_SMPTE274:
    case NVVIOSIGNALFORMAT_1080P_23_976_SMPTE274:
    case NVVIOSIGNALFORMAT_1080P_24_00_SMPTE274:
    case NVVIOSIGNALFORMAT_1080P_25_00_SMPTE274:
    case NVVIOSIGNALFORMAT_1080P_29_97_SMPTE274:
    case NVVIOSIGNALFORMAT_1080P_30_00_SMPTE274:
    case NVVIOSIGNALFORMAT_1080I_48_00_SMPTE274:
    case NVVIOSIGNALFORMAT_1080I_47_96_SMPTE274:
    case NVVIOSIGNALFORMAT_1080PSF_25_00_SMPTE274:
    case NVVIOSIGNALFORMAT_1080PSF_29_97_SMPTE274:
    case NVVIOSIGNALFORMAT_1080PSF_30_00_SMPTE274:
    case NVVIOSIGNALFORMAT_1080PSF_24_00_SMPTE274:
    case NVVIOSIGNALFORMAT_1080PSF_23_98_SMPTE274:
        if (m_SDIin.GetNumStreams() == 1) {
            m_windowWidth = m_videoWidth>>2; m_windowHeight = m_videoHeight>>2;
        } else if (m_SDIin.GetNumStreams() == 2) {
            m_windowWidth = m_videoWidth>>2; m_windowHeight = m_videoHeight>>1;
        } else {
            m_windowWidth = m_videoWidth>>1; m_windowHeight = m_videoHeight>>1;
        }
        break;

    case NVVIOSIGNALFORMAT_2048P_30_00_SMPTE372:
    case NVVIOSIGNALFORMAT_2048P_29_97_SMPTE372:
    case NVVIOSIGNALFORMAT_2048I_60_00_SMPTE372:
    case NVVIOSIGNALFORMAT_2048I_59_94_SMPTE372:
    case NVVIOSIGNALFORMAT_2048P_25_00_SMPTE372:
    case NVVIOSIGNALFORMAT_2048I_50_00_SMPTE372:
    case NVVIOSIGNALFORMAT_2048P_24_00_SMPTE372:
    case NVVIOSIGNALFORMAT_2048P_23_98_SMPTE372:
    case NVVIOSIGNALFORMAT_2048I_48_00_SMPTE372:
    case NVVIOSIGNALFORMAT_2048I_47_96_SMPTE372:
        if (m_SDIin.GetNumStreams() == 1) {
            m_windowWidth = m_videoWidth>>2; m_windowHeight = m_videoHeight>>2;
        } else if (m_SDIin.GetNumStreams() == 2) {
            m_windowWidth = m_videoWidth>>2; m_windowHeight = m_videoHeight>>1;
        } else {
            m_windowWidth = m_videoWidth>>1; m_windowHeight = m_videoHeight>>1;
        }
        break;

    default:
        m_windowWidth = 500;
        m_windowHeight = 500;
    }
}

HWND svlVidCapSrcSDIRenderTarget::SetupWindow(HINSTANCE hInstance, int x, int y, 
                                              char *title)
{
    WNDCLASS   wndclass;
    HWND	   hWnd;
    HDC	  hDC;								// Device context

    BOOL bStatus;
    unsigned int uiNumFormats;
    CHAR szAppName[]="OpenGL SDI Demo";

    int pixelformat;

    // Register the frame class.
    wndclass.style         = 0;
    wndclass.lpfnWndProc   = DefWindowProc;//(WNDPROC) MainWndProc;
    wndclass.cbClsExtra    = 0;
    wndclass.cbWndExtra    = 0;
    wndclass.hInstance     = hInstance;
    wndclass.hIcon         = LoadIcon (hInstance, szAppName);
    wndclass.hCursor       = LoadCursor (NULL,IDC_ARROW);
    wndclass.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);
    wndclass.lpszMenuName  = szAppName;
    wndclass.lpszClassName = szAppName;

    if (!RegisterClass (&wndclass) )
        return NULL;

    // Create initial window frame.
    m_hWnd = CreateWindow ( szAppName, title,
                            WS_CAPTION | WS_BORDER |  WS_SIZEBOX | WS_SYSMENU | WS_MAXIMIZEBOX | WS_MINIMIZEBOX,
                            x,
                            y,
                            m_windowWidth,
                            m_windowHeight,
                            NULL,
                            NULL,
                            hInstance,
                            NULL );

    // Exit on error.
    if (!m_hWnd)
        return NULL;

    // Get device context for new window.
    hDC = GetDC(m_hWnd);

    PIXELFORMATDESCRIPTOR pfd =							// pfd Tells Windows How We Want Things To Be
    {
            sizeof (PIXELFORMATDESCRIPTOR),					// Size Of This Pixel Format Descriptor
            1,												// Version Number
            PFD_DRAW_TO_WINDOW |							// Format Must Support Window
            PFD_SUPPORT_OPENGL |							// Format Must Support OpenGL
            PFD_DOUBLEBUFFER,								// Must Support Double Buffering
            PFD_TYPE_RGBA,									// Request An RGBA Format
            24,												// Select Our Color Depth
            0, 0, 0, 0, 0, 0,								// Color Bits Ignored
            1,												// Alpha Buffer
            0,												// Shift Bit Ignored
            0,												// No Accumulation Buffer
            0, 0, 0, 0,										// Accumulation Bits Ignored
            24,												// 24 Bit Z-Buffer (Depth Buffer)
            8,												// 8 Bit Stencil Buffer
            0,												// No Auxiliary Buffer
            PFD_MAIN_PLANE,									// Main Drawing Layer
            0,												// Reserved
            0, 0, 0											// Layer Masks Ignored
};

// Choose pixel format.
if ( (pixelformat = ChoosePixelFormat(hDC, &pfd)) == 0 ) {
    MessageBox(NULL, "ChoosePixelFormat failed", "Error", MB_OK);
    return FALSE;
}

// Set pixel format.
if (SetPixelFormat(hDC, pixelformat, &pfd) == FALSE) {
    MessageBox(NULL, "SetPixelFormat failed", "Error", MB_OK);
    return FALSE;
}


// Release device context.
ReleaseDC(m_hWnd, hDC);

// Return window handle.
return(m_hWnd);

}

HRESULT svlVidCapSrcSDIRenderTarget::Configure(char *szCmdLine[])
{
    int numGPUs;
    // Note, this function enumerates GPUs which are both CUDA & GLAffinity capable (i.e. newer Quadros)
    numGPUs = CNvSDIoutGpuTopology::instance().getNumGpu();

    if(numGPUs <= 0)
    {
        MessageBox(NULL, "Unable to obtain system GPU topology", "Error", MB_OK);
        return E_FAIL;
    }

    int numCaptureDevices = CNvSDIinTopology::instance().getNumDevice();

    if(numCaptureDevices <= 0)
    {
        MessageBox(NULL, "Unable to obtain system Capture topology", "Error", MB_OK);
        return E_FAIL;
    }
    options.sampling = NVVIOCOMPONENTSAMPLING_422;
    options.dualLink = false;
    options.bitsPerComponent = 8;
    options.expansionEnable = true;
    options.captureDevice = 0;
    options.captureGPU = CNvSDIoutGpuTopology::instance().getPrimaryGpuIndex();

    ParseCommandLine(szCmdLine, &options);//get the user config

    if(options.captureDevice >= numCaptureDevices)
    {
        MessageBox(NULL, "Selected Capture Device is out of range", "Error", MB_OK);
        return E_FAIL;
    }
    if(options.captureGPU >= numGPUs)
    {
        MessageBox(NULL, "Selected Capture GPU is out of range", "Error", MB_OK);
        return E_FAIL;
    }
    m_gpu = CNvSDIoutGpuTopology::instance().getGpu(options.captureGPU);

    return S_OK;
}

HRESULT svlVidCapSrcSDIRenderTarget::SetupSDIDevices()
{
    if(setupSDIinDevices() != S_OK)
    {
        MessageBox(NULL, "Error setting up video capture.", "Error", MB_OK);
        return E_FAIL;
    }

    if (setupSDIoutDevice() == S_OK) {
        m_bSDIout = TRUE;
    } else {
        MessageBox(NULL, "SDI video output unavailable.", "Warning", MB_OK);
    }

    return S_OK;
}

HRESULT svlVidCapSrcSDIRenderTarget::setupSDIinDevices()
{
    m_SDIin.Init(&options);

    // Initialize the video capture device.
    if (m_SDIin.SetupDevice(true,options.captureDevice) != S_OK)
        return E_FAIL;
    m_videoWidth = m_SDIin.GetWidth();
    m_videoHeight = m_SDIin.GetHeight();
    m_num_streams = m_SDIin.GetNumStreams();
    return S_OK;
}

HRESULT svlVidCapSrcSDIRenderTarget::setupSDIoutDevice()
{
#ifdef _DEBUG
    options.console = TRUE;
#else
    options.console = FALSE;
#endif
    options.videoFormat = m_SDIin.GetSignalFormat();
    options.dataFormat = NVVIODATAFORMAT_R8G8B8_TO_YCRCB422;
    if(m_SDIin.GetNumStreams() == 2)
        options.dataFormat = NVVIODATAFORMAT_DUAL_R8G8B8_TO_DUAL_YCRCB422;
    options.syncType = NVVIOCOMPSYNCTYPE_AUTO;
    options.syncSource = NVVIOSYNCSOURCE_SDISYNC;
    options.testPattern = TEST_PATTERN_RGB_COLORBARS_100;
    options.syncEnable = 0;
    options.frameLock = 0;
    options.numFrames = 0;
    options.repeat = 0;
    options.gpu = 0;
    options.block = 0;
    options.videoInfo = 0;
    options.fps = 0;
    options.fsaa = 0;
    options.hDelay = 0;
    options.vDelay = 0;
    options.flipQueueLength = 5;
    options.field = 0;
    options.console = 0;
    options.log = 0;
    options.cscEnable = 0;

    options.yComp = 0;
    options.crComp = 0;
    options.cbComp = 0;

    options.x = 0;
    options.y = 0;
    options.width = 0;
    options.height = 0;
    //options.filename = 0;
    //options.audioFile = 0;
    options.audioChannels = 0;
    options.audioBits = 0;
    //Capture settings//
    options.captureGPU = 0; //capture GPU
    options.captureDevice = 0; //capture card number
    options.dualLink = 0;
    options.sampling = NVVIOCOMPONENTSAMPLING_422;
    options.bitsPerComponent = 8;
    options.expansionEnable = true;
    options.fullScreen = 0;

    return (m_SDIout.Init(&options, m_gpu));
}

GLboolean svlVidCapSrcSDIRenderTarget::SetupGL()
{
    // Create window device context and rendering context.
    m_hDC = GetDC(m_hWnd);

    HGPUNV  gpuMask[2];
    gpuMask[0] = m_gpu->getAffinityHandle();
    gpuMask[1] = NULL;
    if (!(m_hAffinityDC = wglCreateAffinityDCNV(gpuMask))) {
        printf("Unable to create GPU affinity DC\n");
    }

    PIXELFORMATDESCRIPTOR pfd =							// pfd Tells Windows How We Want Things To Be
    {
            sizeof (PIXELFORMATDESCRIPTOR),					// Size Of This Pixel Format Descriptor
            1,												// Version Number
            PFD_DRAW_TO_WINDOW |							// Format Must Support Window
            PFD_SUPPORT_OPENGL |							// Format Must Support OpenGL
            PFD_DOUBLEBUFFER,								// Must Support Double Buffering
            PFD_TYPE_RGBA,									// Request An RGBA Format
            24,												// Select Our Color Depth
            0, 0, 0, 0, 0, 0,								// Color Bits Ignored
            1,												// Alpha Buffer
            0,												// Shift Bit Ignored
            0,												// No Accumulation Buffer
            0, 0, 0, 0,										// Accumulation Bits Ignored
            24,												// 24 Bit Z-Buffer (Depth Buffer)
            8,												// 8 Bit Stencil Buffer
            0,												// No Auxiliary Buffer
            PFD_MAIN_PLANE,									// Main Drawing Layer
            0,												// Reserved
            0, 0, 0											// Layer Masks Ignored
};

GLuint pf = ChoosePixelFormat(m_hAffinityDC, &pfd);
HRESULT rslt = SetPixelFormat(m_hAffinityDC, pf, &pfd);
//		return NULL;
//Create affinity-rendering context from affinity-DC
if (!(m_hRC = wglCreateContext(m_hAffinityDC))) {
    printf("Unable to create GPU affinity RC\n");
}

//m_hRC = wglCreateContext(m_hDC);

// Make window rendering context current.
wglMakeCurrent(m_hDC, m_hRC);
//load the required OpenGL extensions:
if(!loadSwapIntervalExtension() || !loadTimerQueryExtension() || !loadAffinityExtension())
{
    printf("Could not load the required OpenGL extensions\n");
    return false;
}


// Unlock capture/draw loop from vsync of graphics display.
// This should lock the capture/draw loop to the vsync of
// input video.
if (wglSwapIntervalEXT) {
    wglSwapIntervalEXT(0);
}
glClearColor( 0.0, 0.0, 0.0, 0.0);
glClearDepth( 1.0 );

glDisable(GL_DEPTH_TEST);

glDisable(GL_TEXTURE_1D);
glDisable(GL_TEXTURE_2D);

setupSDIinGL();

setupSDIoutGL();

return GL_TRUE;
}

//
// Initialize OpenGL
//
HRESULT svlVidCapSrcSDIRenderTarget::setupSDIinGL()
{
    if(!loadCaptureVideoExtension() || !loadBufferObjectExtension() )
    {
        printf("Could not load the required OpenGL extensions\n");
        return false;
    }

    //setup the textures for capture
    glGenTextures(m_SDIin.GetNumStreams(), m_videoTextures);
    assert(glGetError() == GL_NO_ERROR);
    for(unsigned int i = 0; i < m_SDIin.GetNumStreams();i++)
    {
        glBindTexture(GL_TEXTURE_RECTANGLE_NV, m_videoTextures[i]);
        assert(glGetError() == GL_NO_ERROR);
        glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        assert(glGetError() == GL_NO_ERROR);

        glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA8, m_videoWidth, m_videoHeight,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        assert(glGetError() == GL_NO_ERROR);
    }

    // CSC parameters
    GLfloat mat[4][4];
    float scale = 1.0f;

    GLfloat max[] = {5000, 5000, 5000, 5000};;
    GLfloat min[] = {0, 0, 0, 0};

    GLfloat offset[] = {0, 0, 0, 0};

    if (1) {
        mat[0][0] = 1.164f *scale;
        mat[0][1] = 1.164f *scale;
        mat[0][2] = 1.164f *scale;
        mat[0][3] = 0;

        mat[1][0] = 0;
        mat[1][1] = -0.392f *scale;
        mat[1][2] = 2.017f *scale;
        mat[1][3] = 0;

        mat[2][0] = 1.596f *scale;
        mat[2][1] = -0.813f *scale;
        mat[2][2] = 0.f;
        mat[2][3] = 0;

        mat[3][0] = 0;
        mat[3][1] = 0;
        mat[3][2] = 0;
        mat[3][3] = 1;

        offset[0] =-0.87f;
        offset[1] = 0.53026f;
        offset[2] = -1.08f;
        offset[3] = 0;
    }

    GLuint gpuVideoSlot = 1;
    m_SDIin.SetCSCParams(&mat[0][0], offset, min, max);
    m_SDIin.BindDevice(gpuVideoSlot, m_hDC);
    for(unsigned int i = 0; i < m_SDIin.GetNumStreams(); i++)
        m_SDIin.BindVideoTexture(m_videoTextures[i],i);

    return S_OK;
}

HRESULT svlVidCapSrcSDIRenderTarget::setupSDIoutGL()
{
    //Setup the output after the capture is configured.
    glGenTextures(m_num_streams, m_OutTexture);
    for(unsigned int i=0;i<m_num_streams;i++)
    {
        glBindTexture(GL_TEXTURE_RECTANGLE_NV, m_OutTexture[i]);

        glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA8, m_SDIout.GetWidth(),m_SDIout.GetHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );

        if (!m_FBO[i].create(m_SDIout.GetWidth(),m_SDIout.GetHeight(), 8, 1, GL_TRUE, GL_TRUE, m_OutTexture[i])) {
            printf("Error creating FBO.\n");
        }
        m_FBO[i].unbind();
    }

    if(!loadPresentVideoExtension() ||  !loadFramebufferObjectExtension())
    {
        MessageBox(NULL, "Couldn't load required OpenGL extensions.",
                   "Error", MB_OK);

        exit(1);
    }

    // Enumerate the available video devices and
    // bind to the first one found
    HVIDEOOUTPUTDEVICENV *videoDevices;

    // Get list of available video devices.
    int numDevices = wglEnumerateVideoDevicesNV(m_hDC, NULL);

    if (numDevices <= 0) {
        MessageBox(NULL, "wglEnumerateVideoDevicesNV() did not return any devices.",
                   "Error", MB_OK);
        exit(1);
    }

    videoDevices = (HVIDEOOUTPUTDEVICENV *)malloc(numDevices *
                                                  sizeof(HVIDEOOUTPUTDEVICENV));

    if (!videoDevices) {
        fprintf(stderr, "malloc failed.  OOM?");
        exit(1);
    }

    if (numDevices != wglEnumerateVideoDevicesNV(m_hDC, videoDevices)) {
        free(videoDevices);
        MessageBox(NULL, "Invonsistent results from wglEnumerateVideoDevicesNV()",
                   "Error", MB_OK);
        exit(1);
    }

    //Bind the first device found.
    if (!wglBindVideoDeviceNV(m_hDC, 1, videoDevices[0], NULL)) {
        free(videoDevices);
        MessageBox(NULL, "Failed to bind a videoDevice to slot 0.\n",
                   "Error", MB_OK);
        exit(1);
    }

    // Free list of available video devices, don't need it anymore.
    free(videoDevices);

    // Start video transfers
    if ( m_SDIout.Start()!= S_OK ) {
        MessageBox(NULL, "Error starting video devices.", "Error", MB_OK);
    }
    return S_OK;
}

HRESULT svlVidCapSrcSDIRenderTarget::StartSDIPipeline()
{
    // Start video capture
    if(m_SDIin.StartCapture()!= S_OK)
    {
        MessageBox(NULL, "Error starting video capture.", "Error", MB_OK);
        return E_FAIL;
    }
    return S_OK;
}

GLenum svlVidCapSrcSDIRenderTarget::CaptureVideo(float runTime)
{
    static GLuint64EXT captureTime;
    GLuint sequenceNum;
    static GLuint prevSequenceNum = 0;
    GLenum ret;
    static int numFails = 0;
    static int numTries = 0;
    GLuint captureLatency = 0;
    unsigned int droppedFrames;

    if(numFails < 100) {
        // Capture the video to a buffer object
#ifdef _WIN32
        ret = m_SDIin.Capture(&sequenceNum, &captureTime);
#else
        ret = m_SDIin.capture(&sequenceNum, &captureTime);
#endif
        if(sequenceNum - prevSequenceNum > 1)
        {
            droppedFrames = sequenceNum - prevSequenceNum;
#if __VERBOSE__ == 1
            printf("glVideoCaptureNV: Dropped %d frames\n",sequenceNum - prevSequenceNum);
            printf("Frame:%d gpuTime:%f gviTime:%f\n", sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime);
#endif
            captureLatency = 1;
        }

#if __VERBOSE__ == 1
        if(m_SDIin.m_gviTime > 1.0/30)
        {
            printf("Frame:%d gpuTime:%f gviTime:%f\n", sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime);
            //*captureLatency = 1;
        }
#endif

        prevSequenceNum = sequenceNum;
        switch(ret) {
        case GL_SUCCESS_NV:
#if __VERBOSE__ == 1
            printf("Frame:%d gpuTime:%f gviTime:%f\n", sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime);
#endif
            numFails = 0;
            break;
        case GL_PARTIAL_SUCCESS_NV:
            printf("glVideoCaptureNV: GL_PARTIAL_SUCCESS_NV\n");
            numFails = 0;
            break;
        case GL_FAILURE_NV:
            printf("glVideoCaptureNV: GL_FAILURE_NV - Video capture failed.\n");
            numFails++;
            break;
        default:
            printf("glVideoCaptureNV: Unknown return value.\n");
            break;
        } // switch

    }
    // The incoming signal format or some other error occurred during
    // capture, shutdown and try to restart capture.
    else {
        if(numTries == 0) {
            stopSDIPipeline();
            //cleanupSDIDevices();

            cleanupSDIinGL();
        }

        // Initialize the video capture device.
#ifdef _WIN32
        if (setupSDIinDevices() != TRUE) {
#else
        if (setupSDIinDevice(dpy,gpu) != TRUE) {
#endif
            numTries++;
            return GL_FAILURE_NV;
        }

        // Reinitialize OpenGL.
        setupSDIinGL();

        StartSDIPipeline();
        numFails = 0;
        numTries = 0;
        return GL_FAILURE_NV;
    }

    // Rough workaround latency from capture queue
    if(captureLatency==1)
    {
        for(unsigned int i=0;i< droppedFrames+1;i++)
        {
#if __VERBOSE__ == 1
            printf("Call: %d of %d Frame:%d gpuTime:%f gviTime:%f goal:%f\n", i, droppedFrames+1, sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime,1.0/30);
#endif
            CaptureVideo();
        }
    }
    if(m_SDIin.m_gviTime + runTime > 1.0/30)
    {
#if __VERBOSE__ == 1
        printf("Call: %f decrease to %f Frame:%d gpuTime:%f gviTime:%f goal:%f\n", runTime,runTime-1.0/30,sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime,1.0/30);
#endif
        CaptureVideo(runTime-1.0/30);
        captureLatency = 1;
    }else if(m_SDIin.m_gviTime > 1.0/30)
    {
#if __VERBOSE__ == 1
        printf("Call: %f decrease to %f Frame:%d gpuTime:%f gviTime:%f goal:%f\n", runTime,runTime-1.0/30,sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime,1.0/30);
#endif
        CaptureVideo();
    }

    return ret;
}

GLboolean svlVidCapSrcSDIRenderTarget::DrawOutputScene()
{
    GLuint width;
    GLuint height;

    GLuint cudaOutTexture1 = m_videoTextures[0];
    GLuint cudaOutTexture2 = m_videoTextures[1];

    for(unsigned int i=0;i<m_num_streams;i++)
    {
        if(m_bSDIout)
        {

            m_FBO[i].bind(m_SDIout.GetWidth(), m_SDIout.GetHeight());

            width = m_SDIout.GetWidth();
            height = m_SDIout.GetHeight();
        }
        else
        {
            width = m_SDIin.GetWidth();
            height = m_SDIin.GetHeight();
        }
        glEnable(GL_TEXTURE_RECTANGLE_NV);
        glColor3f(1.0f, 1.0f, 1.0f);
        glClearColor( 1.0, 1.0, 1.0, 0.0);
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        glViewport( 0, 0, width, height );
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        gluOrtho2D(-1.0, 1.0, -1.0, 1.0);
        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();

        // Draw the scene here with the m_CudaOutTexture
        if(i == 0)
            glBindTexture(GL_TEXTURE_RECTANGLE_NV, cudaOutTexture1);
        else
            glBindTexture(GL_TEXTURE_RECTANGLE_NV, cudaOutTexture2);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0); glVertex2f(-1, -1);
        glTexCoord2f(0.0, (GLfloat)height); glVertex2f(-1, 1);
        glTexCoord2f((GLfloat)width, (GLfloat)height); glVertex2f(1, 1);
        glTexCoord2f((GLfloat)width, 0.0); glVertex2f(1, -1);
        glEnd();

#if 0
        // Simple overlay
        glEnable (GL_BLEND);
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        drawCircle(m_videoWidth,m_videoHeight;
                //usleep(1000*1000);
        #else
        // Enable GL alpha blending
        glEnable (GL_BLEND);
        //glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
        //glBlendFuncSeparate(GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE, GL_ZERO);

        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        //glBlendEquation(GL_FUNC_ADD);
        //glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,GL_ONE,GL_ZERO);

        // Draw overlay from unsigned char
        drawUnsignedCharImage(m_overlayBuf[i]);
#endif

        if(m_bSDIout)
        {
            m_FBO[i].unbind();
        }
    }
    return GL_TRUE;
}

void svlVidCapSrcSDIRenderTarget::drawUnsignedCharImage(unsigned char* imageData)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDIRenderTarget::drawUnsignedCharImage()" << std::endl;
#endif

    glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA8, m_SDIout.GetWidth(), m_SDIout.GetHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, imageData );

    // Draw textured quad in lower left quadrant of graphics window.
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex2f(-1, -1);
    glTexCoord2f(0.0, (GLfloat)m_SDIout.GetHeight()); glVertex2f(-1, 1);
    glTexCoord2f((GLfloat)m_SDIout.GetWidth(), (GLfloat)m_SDIout.GetHeight()); glVertex2f(1, 1);
    glTexCoord2f((GLfloat)m_SDIout.GetWidth(), 0.0); glVertex2f(1, -1);
    glEnd();

}

GLboolean svlVidCapSrcSDIRenderTarget::OutputVideo()
{
    if(!m_bSDIout)
        return GL_FALSE;

    glPresentFrameDualFillNV(1, 0, 0, 0, GL_FRAME_NV,
                             GL_TEXTURE_RECTANGLE_NV, m_OutTexture[0],//m_FBO[0].renderbufferIds[0],
                             GL_NONE, 0,
                             GL_TEXTURE_RECTANGLE_NV, m_OutTexture[1],//m_FBO[1].renderbufferIds[0],
                             GL_NONE, 0);


    glBindTexture(GL_TEXTURE_RECTANGLE_NV, 0);

    GLenum l_eVal = glGetError();
    if (l_eVal != GL_NO_ERROR) {
        fprintf(stderr, "glPresentFameKeyedNV returned error\n");
        return FALSE;
    }

    return GL_TRUE;
}


HRESULT svlVidCapSrcSDIRenderTarget::stopSDIPipeline()
{
    m_SDIin.EndCapture();
    return S_OK;
}

GLboolean svlVidCapSrcSDIRenderTarget::cleanupSDIGL()
{
    GLboolean val = GL_TRUE;
    cleanupSDIinGL();
    if(m_bSDIout)
        cleanupSDIoutGL();
    // Delete OpenGL rendering context.
    wglMakeCurrent(NULL,NULL) ;
    if (m_hRC)
    {
        wglDeleteContext(m_hRC) ;
        m_hRC = NULL ;
    }
    ReleaseDC(m_hWnd,m_hDC);

    wglDeleteDCNV(m_hAffinityDC);

    return val;
}

HRESULT svlVidCapSrcSDIRenderTarget::cleanupSDIinGL()
{
    for(unsigned int i = 0; i < m_SDIin.GetNumStreams(); i++)
        m_SDIin.UnbindVideoTexture(i);
    m_SDIin.UnbindDevice();
    glDeleteTextures(m_SDIin.GetNumStreams(),m_videoTextures);


    return S_OK;
}

HRESULT svlVidCapSrcSDIRenderTarget::cleanupSDIoutGL()
{
    glDeleteTextures(m_num_streams, m_OutTexture);
    return S_OK;
}

//-----------------------------------------------------------------------------
// Name: Shutdown
// Desc: Application teardown
//-----------------------------------------------------------------------------
void
svlVidCapSrcSDIRenderTarget::Shutdown()
{
    stopSDIPipeline();
    cleanupSDIGL();
    //CleanupGL();
    //cleanupSDIDevices();
}

#else
//-----------------------------------------------------------------------------
// Name: Capture
// Desc: Main SDI video capture function.
//-----------------------------------------------------------------------------
GLenum
svlVidCapSrcSDIRenderTarget::CaptureVideo(float runTime)
{
    static GLuint64EXT captureTime;
    GLuint sequenceNum;
    static GLuint prevSequenceNum = 0;
    GLenum ret;
    static int numFails = 0;
    static int numTries = 0;
    GLuint captureLatency = 0;
    unsigned int droppedFrames;

    if(numFails < 100) {
        // Capture the video to a buffer object
        ret = m_SDIin.capture(&sequenceNum, &captureTime);
        if(sequenceNum - prevSequenceNum > 1)
        {
            droppedFrames = sequenceNum - prevSequenceNum;
#if __VERBOSE__ == 1
            printf("glVideoCaptureNV: Dropped %d frames\n",sequenceNum - prevSequenceNum);
            printf("Frame:%d gpuTime:%f gviTime:%f\n", sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime);
#endif
            captureLatency = 1;
        }

#if __VERBOSE__ == 1
        if(m_SDIin.m_gviTime > 1.0/30)
        {
            printf("Frame:%d gpuTime:%f gviTime:%f\n", sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime);
            //*captureLatency = 1;
        }
#endif

        prevSequenceNum = sequenceNum;
        switch(ret) {
        case GL_SUCCESS_NV:
#if __VERBOSE__ == 1
            printf("Frame:%d gpuTime:%f gviTime:%f\n", sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime);
#endif
            numFails = 0;
            break;
        case GL_PARTIAL_SUCCESS_NV:
#if __VERBOSE__ == 1
            printf("glVideoCaptureNV: GL_PARTIAL_SUCCESS_NV\n");
#endif
            numFails = 0;
            break;
        case GL_FAILURE_NV:
            printf("glVideoCaptureNV: GL_FAILURE_NV - Video capture failed.\n");
            numFails++;
            break;
        default:
            printf("glVideoCaptureNV: Unknown return value.\n");
            break;
        } // switch

    }
    // The incoming signal format or some other error occurred during
    // capture, shutdown and try to restart capture.
    else {
        if(numTries == 0) {
            stopSDIPipeline();
            cleanupSDIinDevices();

            cleanupSDIinGL();
        }

        // Initialize the video capture device.
        if (setupSDIinDevice(dpy,gpu) != TRUE) {
            numTries++;
            return GL_FAILURE_NV;
        }

        // Reinitialize OpenGL.
        setupSDIinGL();

        startSDIPipeline();
        numFails = 0;
        numTries = 0;
        return GL_FAILURE_NV;
    }

    // Rough workaround latency from capture queue
    if(captureLatency==1)
    {
        for(unsigned int i=0;i< droppedFrames+1;i++)
        {
#if __VERBOSE__ == 1
            printf("Call: %d of %d Frame:%d gpuTime:%f gviTime:%f goal:%f\n", i, droppedFrames+1, sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime,1.0/30);
#endif
            CaptureVideo();
        }
    }
    if(m_SDIin.m_gviTime + runTime > 1.0/30)
    {
#if __VERBOSE__ == 1
        printf("Call: %f decrease to %f Frame:%d gpuTime:%f gviTime:%f goal:%f\n", runTime,runTime-1.0/30,sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime,1.0/30);
#endif
        CaptureVideo(runTime-1.0/30);
        captureLatency = 1;
    }else if(m_SDIin.m_gviTime > 1.0/30)
    {
#if __VERBOSE__ == 1
        printf("Call: %f decrease to %f Frame:%d gpuTime:%f gviTime:%f goal:%f\n", runTime,runTime-1.0/30,sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime,1.0/30);
#endif
        CaptureVideo();
    }

    return ret;
}

GLboolean svlVidCapSrcSDIRenderTarget::DrawOutputScene()
{
    GLuint width;
    GLuint height;

    GLuint cudaOutTexture1 = m_SDIin.getTextureObjectHandle(0);
    GLuint cudaOutTexture2 = m_SDIin.getTextureObjectHandle(1);

    for(unsigned int i=0;i<m_num_streams;i++)
    {
        if(m_SDIoutEnabled)
        {

            m_FBO[i].bind(m_SDIout.getWidth(), m_SDIout.getHeight());

            width = m_SDIout.getWidth();
            height = m_SDIout.getHeight();
        }

        glEnable(GL_TEXTURE_RECTANGLE_NV);
        glColor3f(1.0f, 1.0f, 1.0f);
        glClearColor( 1.0, 1.0, 1.0, 0.0);
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        glViewport( 0, 0, width, height );
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        gluOrtho2D(-1.0, 1.0, -1.0, 1.0);
        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();

        // Draw the scene here with the m_CudaOutTexture
        if(i == 0)
            glBindTexture(GL_TEXTURE_RECTANGLE_NV, cudaOutTexture1);
        else
            glBindTexture(GL_TEXTURE_RECTANGLE_NV, cudaOutTexture2);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0); glVertex2f(-1, -1);
        glTexCoord2f(0.0, (GLfloat)height); glVertex2f(-1, 1);
        glTexCoord2f((GLfloat)width, (GLfloat)height); glVertex2f(1, 1);
        glTexCoord2f((GLfloat)width, 0.0); glVertex2f(1, -1);
        glEnd();

#if 0
        // Simple overlay
        glEnable (GL_BLEND);
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        drawCircle(m_SDIout.getWidth(),m_SDIout.getHeight());
        //usleep(1000*1000);
#else
        // Enable GL alpha blending
        glEnable (GL_BLEND);
        //glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
        //glBlendFuncSeparate(GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE, GL_ZERO);

        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        //glBlendEquation(GL_FUNC_ADD);
        //glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,GL_ONE,GL_ZERO);

        // Draw overlay from unsigned char
        drawUnsignedCharImage(m_overlayBuf[i]);
#endif

        if(m_SDIoutEnabled)
        {
            m_FBO[i].unbind();
        }
    }
    return GL_TRUE;
}


/////////////////////////////////////
// Output
/////////////////////////////////////
GLboolean svlVidCapSrcSDIRenderTarget::OutputVideo()
{
    if(!m_SDIoutEnabled || m_num_streams < 1 || m_num_streams > 2)
        return GL_FALSE;

    if(m_num_streams == 2)
    {
        glPresentFrameDualFillNV(1, 0, 0, 0, GL_FRAME_NV,
                                 GL_TEXTURE_RECTANGLE_NV, m_OutTexture[0],
                                 GL_NONE, 0,
                                 GL_TEXTURE_RECTANGLE_NV, m_OutTexture[1],
                                 GL_NONE, 0);

    }else if(m_num_streams == 1)
    {
        glPresentFrameKeyedNV(1, 0,
                              0, 0,
                              GL_FRAME_NV,
                              GL_TEXTURE_RECTANGLE_NV, m_OutTexture[0], 0,
                              GL_NONE, 0, 0);
    }else
    {
        fprintf(stderr, "svlVidCapSrcSDIRenderTarget::OutputVideo() error, unrecongized stream count %i\n",m_num_streams);
        return false;
    }

    glBindTexture(GL_TEXTURE_RECTANGLE_NV, 0);

    GLenum l_eVal = glGetError();
    if (l_eVal != GL_NO_ERROR) {
        fprintf(stderr, "glPresentFameKeyedNV returned error\n");
        return FALSE;
    }

    return GL_TRUE;
}

bool svlVidCapSrcSDIRenderTarget::setupSDIDevices(Display * d, HGPUNV * g)
{
    if(d && g)
    {
        gpu = g;
        dpy = d;
    }
    if(!dpy || !gpu)
        return FALSE;

    if(setupSDIinDevice(dpy,gpu) != TRUE) {
        printf("Error setting up video capture.\n");
        return FALSE;
    }

    m_videoWidth = m_SDIin.getWidth();
    m_videoHeight = m_SDIin.getHeight();
    m_num_streams = m_SDIin.getNumStreams();
    if(setupSDIoutDevice(dpy,gpu,m_SDIin.getVideoFormat()) != TRUE)//NV_CTRL_GVO_VIDEO_FORMAT_487I_59_94_SMPTE259_NTSC) != TRUE)
    {
        printf("Error setting up video output.\n");
        m_SDIoutEnabled = FALSE;
    }
    else
        m_SDIoutEnabled = TRUE;

    return m_SDIoutEnabled;

}

bool svlVidCapSrcSDIRenderTarget::setupSDIoutDevice(Display *d,HGPUNV *g, unsigned int video_format)
{
    // Set the output frame rate
    outputOptions.video_format = video_format;//m_SDIin.getVideoFormat();
    outputOptions.xscreen = g->deviceXScreen;
    outputOptions.data_format = NV_CTRL_GVO_DATA_FORMAT_R8G8B8_TO_YCRCB422;

    // Max 2 stream output
    if(m_num_streams < 1 || m_num_streams > 2)
        return false;
    else if(m_num_streams == 2)
        outputOptions.data_format = NV_CTRL_GVO_DATA_FORMAT_DUAL_R8G8B8_TO_DUAL_YCRCB422;

    // SDI output frame buffer queue Length?
    outputOptions.fql = 5;
    outputOptions.sync_source = NV_CTRL_GVO_SYNC_SOURCE_SDI;
    outputOptions.sync_mode = NV_CTRL_GVO_SYNC_MODE_FREE_RUNNING;//NV_CTRL_GVO_SYNC_MODE_GENLOCK;

    m_SDIout.setOutputOptions(d,outputOptions);
    bool ret = m_SDIout.initOutputDeviceNVCtrl();
    return ret;
}

GLboolean svlVidCapSrcSDIRenderTarget::setupSDIoutGL()
{
    //Setup the output after the capture is configured.
    glGenTextures(m_num_streams, m_OutTexture);
    for(unsigned int i=0;i<m_num_streams;i++)
    {
        glBindTexture(GL_TEXTURE_RECTANGLE_NV, m_OutTexture[i]);

        glTexParameterf(GL_TEXTURE_RECTANGLE_NV, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA8, m_SDIout.getWidth(), m_SDIout.getHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
        //glBindTexture(GL_TEXTURE_RECTANGLE_NV, 0);

        // Initialize required FBOs

        //if (!m_FBO.create(m_SDIout.GetWidth(),m_SDIout.GetHeight(), 8, 1, GL_TRUE, GL_FALSE, m_OutTexture)) {
        if (!m_FBO[i].create(m_SDIout.getWidth(),m_SDIout.getHeight(), 8, 1, GL_TRUE, GL_TRUE, m_OutTexture[i])) {
            printf("Error creating FBO.\n");
        }
        m_FBO[i].unbind();
    }
    // Enumerate available video devices.
    unsigned int *videoDevices;
    int numDevices;
    videoDevices = glXEnumerateVideoDevicesNV(dpy, outputOptions.xscreen, &numDevices);
    if (!videoDevices || numDevices <= 0) {		XFree(videoDevices);		printf("Error: could not enumerate video devices\n");		return -1;	}

    // Bind first video device
    if (Success != glXBindVideoDeviceNV(dpy, 1, videoDevices[0], NULL)) {
        XFree(videoDevices);
        printf("Error: could not bind video device\n");
        return -1;
    }
    // Free list of available video devices, don't need it anymore.
    free(videoDevices);
    return GL_TRUE;
}

//-----------------------------------------------------------------------------
// Name: setupSDIinDevice
// Desc: Initialize SDI capture device state.
//-----------------------------------------------------------------------------
bool
svlVidCapSrcSDIRenderTarget::setupSDIinDevice(Display *d, HGPUNV *g)
{
    GLfloat mat[4][4];
    float scale = 1.0f;
    GLfloat max[] = {5000, 5000, 5000, 5000};
    //GLfloat max[] = {256, 256, 256, 256};
    GLfloat min[] = {0, 0, 0, 0};
    // Initialize matrix to the identity.
    mat[0][0] = scale; mat[0][1] = 0; mat[0][2] = 0; mat[0][3] = 0;
    mat[1][0] = 0; mat[1][1] = scale; mat[1][2] = 0; mat[1][3] = 0;
    mat[2][0] = 0; mat[2][1] = 0; mat[2][2] = scale; mat[2][3] = 0;
    mat[3][0] = 0; mat[3][1] = 0; mat[3][2] = 0; mat[3][3] = scale;
    GLfloat offset[] = {0, 0, 0, 0};
    mat[0][0] = 1.164f *scale;
    mat[0][1] = 1.164f *scale;
    mat[0][2] = 1.164f *scale;
    mat[0][3] = 0;

    mat[1][0] = 0;
    mat[1][1] = -0.392f *scale;
    mat[1][2] = 2.017f *scale;
    mat[1][3] = 0;

    mat[2][0] = 1.596f *scale;
    mat[2][1] = -0.813f *scale;
    mat[2][2] = 0.f;
    mat[2][3] = 0;

    mat[3][0] = 0;
    mat[3][1] = 0;
    mat[3][2] = 0;
    mat[3][3] = 1;

    offset[0] =-0.87f;
    offset[1] = 0.53026f;
    offset[2] = -1.08f;
    offset[3] = 0;


    captureOptions.cscMax = max;
    captureOptions.cscMin = min;
    captureOptions.cscMat = &mat[0][0];
    captureOptions.cscOffset = offset;
    captureOptions.captureType = TEXTURE_FRAME;
    captureOptions.textureInternalFormat =  GL_RGBA8;
    captureOptions.pixelFormat = GL_RGBA;
    captureOptions.bitsPerComponent = NV_CTRL_GVI_BITS_PER_COMPONENT_8;
    captureOptions.sampling = NV_CTRL_GVI_COMPONENT_SAMPLING_422;
    captureOptions.xscreen = g->deviceXScreen;
    captureOptions.bDualLink = false;
    captureOptions.bChromaExpansion = true;
    m_SDIin.setCaptureOptions(d,captureOptions);

    bool ret = m_SDIin.initCaptureDeviceNVCtrl();

    return ret;
}


//-----------------------------------------------------------------------------
// Name: setupSDIinGL
// Desc: Initialize OpenGL SDI capture state.
//-----------------------------------------------------------------------------
GLboolean
svlVidCapSrcSDIRenderTarget::setupSDIinGL()
{
    //Setup GL
    m_SDIin.initCaptureDeviceGL();
    return true;
}

/////////////////////////////////////
// Main Methods
/////////////////////////////////////


//-----------------------------------------------------------------------------
// Name: SetupGL
// Desc: Setup OpenGL capture.
//-----------------------------------------------------------------------------
GLboolean
svlVidCapSrcSDIRenderTarget::setupSDIGL()
{
    glClearColor( 0.0, 0.0, 0.0, 0.0);
    glClearDepth( 1.0 );

    glDisable(GL_DEPTH_TEST);

    glDisable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);

    setupSDIinGL();
    setupSDIoutGL();

    return GL_TRUE;
}

//-----------------------------------------------------------------------------
// Name: StartSDIPipeline
// Desc: Start SDI video capture.
//-----------------------------------------------------------------------------
bool
svlVidCapSrcSDIRenderTarget::startSDIPipeline()
{
    // Start video capture
    if(m_SDIin.startCapture()!= TRUE) {
        printf("Error starting video capture.\n");
        return FALSE;
    }
    //CaptureStarted = true;
    return TRUE;
}


//-----------------------------------------------------------------------------
// Name: StopSDIPipeline
// Desc: Stop SDI video capture.
//-----------------------------------------------------------------------------
bool
svlVidCapSrcSDIRenderTarget::stopSDIPipeline()
{
    if(m_SDIin.endCapture()!= TRUE) {
        printf("Error starting video capture.\n");
        return FALSE;
    }
    return TRUE;
}

void svlVidCapSrcSDIRenderTarget::Shutdown()
{
    KillThread = true;
    if (ThreadKilled == false) Thread->Wait();
    delete Thread;

    stopSDIPipeline();
    cleanupSDIGL();
    cleanupSDIDevices();
}

/////////////////////////////////////
// Cleanup
/////////////////////////////////////
//-----------------------------------------------------------------------------
// Name: cleanupSDIin()
// Desc: Destroy SDI capture device.
//-----------------------------------------------------------------------------
bool
svlVidCapSrcSDIRenderTarget::cleanupSDIinDevices()
{
    m_SDIin.endCapture();
    bool ret = m_SDIin.destroyCaptureDeviceNVCtrl();
    return ret;
}

GLboolean svlVidCapSrcSDIRenderTarget::cleanupSDIoutGL()
{
    // Destroy objects
    if(m_SDIoutEnabled)
    {
        for(unsigned int i=0;i<m_num_streams;i++)
        {
            m_FBO[i].destroy();
        }
        glDeleteTextures(m_num_streams, m_OutTexture);
    }

    return GL_TRUE;
}

bool svlVidCapSrcSDIRenderTarget::cleanupSDIoutDevices()
{
    bool ret = TRUE;
    if(m_SDIoutEnabled && (m_SDIout.destroyOutputDeviceNVCtrl() != TRUE))
        ret = FALSE;
    return ret;
}

//-----------------------------------------------------------------------------
// Name: CleanupSDIDevices
// Desc: Cleanup SDI capture devices.
//-----------------------------------------------------------------------------
bool
svlVidCapSrcSDIRenderTarget::cleanupSDIDevices()
{
    bool ret = TRUE;
    if(cleanupSDIinDevices() != TRUE)
        ret = FALSE;
    ret = cleanupSDIoutDevices();
    return ret;
}

//-----------------------------------------------------------------------------
// Name: CleanupGL
// Desc: OpenGL teardown.
//-----------------------------------------------------------------------------
GLboolean svlVidCapSrcSDIRenderTarget::cleanupSDIGL()
{
    cleanupSDIinGL();
    cleanupSDIoutGL();

    return GL_TRUE;
}

GLboolean svlVidCapSrcSDIRenderTarget::cleanupSDIinGL()
{
    m_SDIin.destroyCaptureDeviceGL();

    // Delete OpenGL rendering context.
    glXMakeCurrent(dpy,NULL,NULL) ;
    if (ctx) {
        glXDestroyContext(dpy,ctx) ;
        ctx = NULL;
    }

    return GL_TRUE;
}

/////////////////////////////////////
// Draw
/////////////////////////////////////
void svlVidCapSrcSDIRenderTarget::drawUnsignedCharImage(unsigned char* imageData)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDIRenderTarget::drawUnsignedCharImage()" << std::endl;
#endif

    glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA8, m_SDIout.getWidth(), m_SDIout.getHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, imageData );

    // Draw textured quad in lower left quadrant of graphics window.
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex2f(-1, -1);
    glTexCoord2f(0.0, (GLfloat)m_SDIout.getHeight()); glVertex2f(-1, 1);
    glTexCoord2f((GLfloat)m_SDIout.getWidth(), (GLfloat)m_SDIout.getHeight()); glVertex2f(1, 1);
    glTexCoord2f((GLfloat)m_SDIout.getWidth(), 0.0); glVertex2f(1, -1);
    glEnd();

}

//-----------------------------------------------------------------------------
// Name: CreateWindow
// Desc: Create window
//-----------------------------------------------------------------------------
Window
svlVidCapSrcSDIRenderTarget::createWindow()
{
    XVisualInfo *vi ;
    GLXFBConfig *configs, config;
    XEvent event;
    XSetWindowAttributes swa;

    unsigned long mask;
    int numConfigs;
    int config_list[] = { GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
                          GLX_DOUBLEBUFFER, GL_TRUE,
                          GLX_RENDER_TYPE, GLX_RGBA_BIT,
                          GLX_RED_SIZE, 8,
                          GLX_GREEN_SIZE, 8,
                          GLX_BLUE_SIZE, 8,
                          GLX_FLOAT_COMPONENTS_NV, GL_FALSE,
                          None };
    int i;
    // Find required framebuffer configuration
    configs = glXChooseFBConfig(dpy, captureOptions.xscreen, config_list, &numConfigs);
    if (!configs) {
        fprintf(stderr, "Unable to find a matching FBConfig.\n");
        exit(1);
    }

    // Find an FBconfig with the required number of color bits.
    for (i = 0; i < numConfigs; i++) {
        int attr;
        if (glXGetFBConfigAttrib(dpy, configs[i], GLX_RED_SIZE, &attr)) {
            printf("glXGetFBConfigAttrib(GLX_RED_SIZE) failed!\n");
            exit(1);
        }
        if (attr != 8)
            continue;

        if (glXGetFBConfigAttrib(dpy, configs[i], GLX_GREEN_SIZE, &attr)) {
            printf("glXGetFBConfigAttrib(GLX_GREEN_SIZE) failed!\n");
            exit(1);
        }
        if (attr != 8)
            continue;

        if (glXGetFBConfigAttrib(dpy, configs[i], GLX_BLUE_SIZE, &attr)) {
            printf("glXGetFBConfigAttrib(GLX_BLUE_SIZE) failed!\n");
            exit(1);
        }

        if (attr != 8)
            continue;

        if (glXGetFBConfigAttrib(dpy, configs[i], GLX_ALPHA_SIZE, &attr)) {
            printf("glXGetFBConfigAttrib(GLX_ALPHA_SIZE) failed\n");
            exit(1);
        }

        if (attr != 8)
            continue;

        break;
    }

    if (i == numConfigs) {
        printf("No FBConfigs found\n");
        exit(1);
    }

    config = configs[i];

    // Don't need the config list anymore so free it.
    XFree(configs);
    configs = NULL;

    // Create an OpenGL rendering context for the onscreen window.
    ctx = glXCreateNewContext(dpy, config, GLX_RGBA_TYPE, 0, GL_TRUE);

    // Get visual from FB config.
    if ((vi = glXGetVisualFromFBConfig(dpy, config)) != NULL) {
        printf("Using visual %0x\n", (int) vi->visualid);
        printf("Depth = %d\n", vi->depth);
    } else {
        printf("Couldn't find visual for onscreen window.\n");
        exit(1);
    }

    // Create color map.
    if (!(cmap = XCreateColormap(dpy, RootWindow(dpy, vi->screen),
                                 vi->visual, AllocNone))) {
        fprintf(stderr, "XCreateColormap failed!\n");
        exit(1);
    }


    // Calculate window width & height.
    calcWindowSize();

    // Create window.
    swa.colormap = cmap;
    swa.border_pixel = 0;
    swa.background_pixel = 1;
    swa.event_mask = ExposureMask | StructureNotifyMask | KeyPressMask |
            KeyReleaseMask | ButtonPressMask | ButtonReleaseMask |
            PointerMotionMask ;
    mask = CWBackPixel | CWBorderPixel | CWColormap | CWEventMask;
    win = XCreateWindow(dpy, RootWindow(dpy, vi->screen),
                        0, 0, m_windowWidth, m_windowHeight, 0,
                        vi->depth, InputOutput, vi->visual,
                        mask, &swa);

    // Map window.
    XMapWindow(dpy, win);
    XIfEvent(dpy, &event, WaitForNotify, (char *) win);

    // Set window colormap.
    XSetWMColormapWindows(dpy, win, &win, 1);

    // Make OpenGL rendering context current.
    if (!(glXMakeCurrent(dpy, win, ctx))) {
        fprintf(stderr, "glXMakeCurrent failed!\n");
        exit(1);
    }

    // Don't lock the capture/draw loop to the graphics vsync.
    glXSwapIntervalSGI(0);
    XFlush(dpy);

    return win;
}

//-----------------------------------------------------------------------------
// Name: DestroyWindow
// Desc: Destroy window
//-----------------------------------------------------------------------------
bool
svlVidCapSrcSDIRenderTarget::destroyWindow()
{
    XUnmapWindow(dpy,win);
    XDestroyWindow(dpy, win);
    XFreeColormap(dpy,cmap);

    return true;
}

//-----------------------------------------------------------------------------
// Name: calcWindowSize
// Desc: Calculate the graphics window size
//-----------------------------------------------------------------------------
void
svlVidCapSrcSDIRenderTarget::calcWindowSize()
{
    int numStreams = m_SDIin.getNumStreams();
    //TODO: fix frame rate
    m_inputFrameRate = 59.94;
    switch(m_SDIin.getVideoFormat()) {
    case NV_CTRL_GVIO_VIDEO_FORMAT_487I_59_94_SMPTE259_NTSC:
    case NV_CTRL_GVIO_VIDEO_FORMAT_576I_50_00_SMPTE259_PAL:
        if (numStreams == 1) {
            m_windowWidth = 360; m_windowHeight = 243;
        } else if (numStreams == 2) {
            m_windowWidth = 720; m_windowHeight = 496;
        } else {
            m_windowWidth = 720; m_windowHeight = 486;
        }
        break;

    case NV_CTRL_GVIO_VIDEO_FORMAT_720P_59_94_SMPTE296:
    case NV_CTRL_GVIO_VIDEO_FORMAT_720P_60_00_SMPTE296:
    case NV_CTRL_GVIO_VIDEO_FORMAT_720P_50_00_SMPTE296:
    case NV_CTRL_GVIO_VIDEO_FORMAT_720P_30_00_SMPTE296:
    case NV_CTRL_GVIO_VIDEO_FORMAT_720P_29_97_SMPTE296:
    case NV_CTRL_GVIO_VIDEO_FORMAT_720P_25_00_SMPTE296:
    case NV_CTRL_GVIO_VIDEO_FORMAT_720P_24_00_SMPTE296:
    case NV_CTRL_GVIO_VIDEO_FORMAT_720P_23_98_SMPTE296:
        if (numStreams == 1) {
            m_windowWidth = 320; m_windowHeight = 180;
        } else if (numStreams == 2) {
            m_windowWidth = 320; m_windowHeight = 360;
        } else {
            m_windowWidth = 640; m_windowHeight = 486;
        }
        break;
    case NV_CTRL_GVIO_VIDEO_FORMAT_1035I_59_94_SMPTE260:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1035I_60_00_SMPTE260:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_50_00_SMPTE295:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_50_00_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_59_94_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_60_00_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_23_976_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_24_00_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_25_00_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_29_97_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_30_00_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_48_00_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_47_96_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_25_00_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_29_97_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_30_00_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_24_00_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_23_98_SMPTE274:
        if (numStreams == 1) {
            m_windowWidth = 480; m_windowHeight = 270;
        } else if (numStreams == 2) {
            m_windowWidth = 480; m_windowHeight = 540;
        } else {
            m_windowWidth = 960; m_windowHeight = 540;
        }
        break;
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_30_00_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_29_97_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_60_00_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_59_94_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_25_00_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_50_00_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_24_00_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_23_98_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_48_00_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_47_96_SMPTE372:
        if (numStreams == 1) {
            m_windowWidth = 512; m_windowHeight = 270;
        } else if (numStreams == 2) {
            m_windowWidth = 512; m_windowHeight = 540;
        } else {
            m_windowWidth = 1024; m_windowHeight = 540;
        }
        break;
    default:
        m_windowWidth = 500;
        m_windowHeight = 500;
    }
}
#endif

/*************************************/
/*        svlVidCapSrcSDI class      */
/*************************************/
CMN_IMPLEMENT_SERVICES_DERIVED(svlVidCapSrcSDI, svlVidCapSrcBase)

////////////////////////////////////
// svlVidCapSrcSDI
////////////////////////////////////

svlVidCapSrcSDI::svlVidCapSrcSDI():
    svlVidCapSrcBase(),
    NumOfStreams(0),
    InitializedInput(false),
    InitializedOutput(false),
    Running(false),
    CaptureProc(0),
    CaptureThread(0)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::constructor()" << std::endl;
#endif


}

svlVidCapSrcSDI::~svlVidCapSrcSDI()
{

}

//TODO:GetInstance
svlVidCapSrcSDI* svlVidCapSrcSDI::GetInstance()
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::GetInstance()" << std::endl;
#endif

    static svlVidCapSrcSDI instance;
    return &instance;
}

svlFilterSourceVideoCapture::PlatformType svlVidCapSrcSDI::GetPlatformType()
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::GetPlatformType()" << std::endl;
#endif

    return svlFilterSourceVideoCapture::NVIDIAQuadroSDI;
}

int svlVidCapSrcSDI::SetStreamCount(unsigned int numofstreams)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::SetStreamCount(" << numofstreams << ")" << std::endl;
#endif

    if (numofstreams < 1) return SVL_FAIL;

    //Release();

    NumOfStreams = numofstreams;
    CaptureProc = new svlVidCapSrcSDIThread*[SDINUMDEVICES];
    CaptureThread = new osaThread*[SDINUMDEVICES];
    SystemID.SetSize(NumOfStreams);
    DigitizerID.SetSize(NumOfStreams);
    ImageBuffer.SetSize(NumOfStreams);

    for (unsigned int i = 0; i < SDINUMDEVICES; i ++) {
        CaptureProc[i] = 0;
        CaptureThread[i] = 0;
    }

    for (unsigned int i = 0; i < NumOfStreams; i ++) {
        SystemID[i] = -1;
        DigitizerID[i] = -1;
        ImageBuffer[i] = 0;
    }

    return SVL_OK;
}

int svlVidCapSrcSDI::GetDeviceList(svlFilterSourceVideoCapture::DeviceInfo **deviceinfo)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::GetDeviceList(...)" << std::endl;
#endif

    if (deviceinfo == 0 || (InitializedInput && InitializedOutput)) return SVL_FAIL;

    unsigned int sys;//, dig, digitizers;
    //bool cap, ovrl;

    // Allocate memory for device info array
    // CALLER HAS TO FREE UP THIS ARRAY!!!
    if (SDINUMDEVICES > 0) {

        deviceinfo[0] = new svlFilterSourceVideoCapture::DeviceInfo[SDINUMDEVICES];

        for (sys = 0; sys < SDINUMDEVICES; sys ++) {
            // platform
            deviceinfo[0][sys].platform = svlFilterSourceVideoCapture::NVIDIAQuadroSDI;

            // id
            deviceinfo[0][sys].ID = sys;

            // name
            std::stringstream dev_name;
            dev_name << "NVIDIA Quadro SDI (" << "SDI_DEV" << sys << ")";

            memset(deviceinfo[0][sys].name, 0, SVL_VCS_STRING_LENGTH);
            memcpy(deviceinfo[0][sys].name,
                   dev_name.str().c_str(),
                   std::min(SVL_VCS_STRING_LENGTH - 1, static_cast<int>(dev_name.str().length())));

            // test
            deviceinfo[0][sys].testok = true;
        }
    }
    else {
        deviceinfo[0] = 0;
    }

    return SDINUMDEVICES;
}

int svlVidCapSrcSDI::Open()
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::Open() - Number of video channels = " << NumOfStreams << std::endl;
#endif

    if (NumOfStreams <= 0) return SVL_FAIL;
    if (InitializedInput) return SVL_OK;

    Close();

    for (unsigned int i = 0; i < NumOfStreams; i ++) {
        //        // Allocate capture buffers
        const unsigned int width  = 1920;//MilWidth[SystemID[i]][DigitizerID[i]];
        const unsigned int height = 1080;//CaptureProc[0]->GetSDIin().getWidth();//MilHeight[SystemID[i]][DigitizerID[i]];
        //#if __VERBOSE__ == 1
        std::cout << "svlVidCapSrcSDI::Open - Allocate image buffer (" << width << ", " << height << ")" << std::endl;
        //#endif
        ImageBuffer[i] = new svlBufferImage(width, height);
    }

    InitializedInput = true;

    NumOfStreams = 2;//CaptureProc[0]->GetSDIin().getNumStreams();
    return SVL_OK;

labError:
    Close();
    return SVL_FAIL;
}

void svlVidCapSrcSDI::Close()
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::Close()" << std::endl;
#endif

    if (NumOfStreams == 0) return;

    Stop();

    InitializedInput = false;

    for (unsigned int i = 0; i < NumOfStreams; i ++) {
        delete ImageBuffer[i];
        ImageBuffer[i] = 0;
    }
}

int svlVidCapSrcSDI::Start()
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::Start()" << std::endl;
#endif

    if (!InitializedInput) return SVL_FAIL;
    Running = true;
    for (unsigned int i = 0; i < SDINUMDEVICES; i ++) {
        CaptureProc[i] = new svlVidCapSrcSDIThread(i);
        CaptureThread[i] = new osaThread;
        Running = true;
        CaptureThread[i]->Create<svlVidCapSrcSDIThread, svlVidCapSrcSDI*>(CaptureProc[i],
                                                                          &svlVidCapSrcSDIThread::Proc,
                                                                          this);
        if (CaptureProc[i]->WaitForInit() == false) return SVL_FAIL;
    }

    return SVL_OK;
}

svlImageRGB* svlVidCapSrcSDI::GetLatestFrame(bool waitfornew, unsigned int videoch)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::GetLatestFrame(" << waitfornew << ", " << videoch << ")" << std::endl;
#endif

    if (videoch >= NumOfStreams || !InitializedInput) return 0;
    return ImageBuffer[videoch]->Pull(waitfornew);
}

int svlVidCapSrcSDI::Stop()
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::Stop()" << std::endl;
#endif

    if (!Running) return SVL_FAIL;
    Running = false;
    for (unsigned int i = 0; i < SDINUMDEVICES; i ++) {
        if (CaptureThread[i]) {
            CaptureThread[i]->Wait();
            delete(CaptureThread[i]);
            CaptureThread[i] = 0;
        }
        if (CaptureProc[i]) {
            delete(CaptureProc[i]);
            CaptureProc[i] = 0;
        }
    }

    return SVL_OK;
}

bool svlVidCapSrcSDI::IsRunning()
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::IsRunning()" << std::endl;
#endif

    return Running;
}

int svlVidCapSrcSDI::SetDevice(int devid, int inid, unsigned int videoch)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::SetDevice(" << devid << ", " << inid << ", " << videoch << ")" << std::endl;
#endif

    if (videoch >= NumOfStreams) return SVL_FAIL;
    SystemID[videoch] = devid;
    DigitizerID[videoch] = inid;
    return SVL_OK;
}

int svlVidCapSrcSDI::GetWidth(unsigned int videoch)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::GetWidth(" << videoch << ")" << std::endl;
#endif

    if (videoch >= NumOfStreams) return SVL_FAIL;
    return ImageBuffer[videoch]->GetWidth();
}

int svlVidCapSrcSDI::GetHeight(unsigned int videoch)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::GetHeight(" << videoch << ")" << std::endl;
#endif

    if (videoch >= NumOfStreams) return SVL_FAIL;
    return ImageBuffer[videoch]->GetHeight();
}

int svlVidCapSrcSDI::GetFormatList(unsigned int deviceid, svlFilterSourceVideoCapture::ImageFormat **formatlist)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::GetFormatList(" << deviceid << ", ...)" << std::endl;
#endif

    if (static_cast<int>(deviceid) >= MAX_VIDEO_STREAMS || formatlist == 0) return SVL_FAIL;

    formatlist[0] = new svlFilterSourceVideoCapture::ImageFormat[1];
    formatlist[0][0].width = 1920;//CaptureProc[0]->GetSDIin().getWidth();
    formatlist[0][0].height = 1080;//CaptureProc[0]->GetSDIin().getHeight();
    formatlist[0][0].colorspace = svlFilterSourceVideoCapture::PixelRGB8;
    formatlist[0][0].rgb_order = true;
    formatlist[0][0].yuyv_order = false;
    formatlist[0][0].framerate = 59.94;
    formatlist[0][0].custom_mode = -1;

    return 1;
}

int svlVidCapSrcSDI::GetFormat(svlFilterSourceVideoCapture::ImageFormat& format, unsigned int videoch)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::GetFormat(..., " << videoch << ")" << std::endl;
#endif

    if (SystemID[videoch] > 0 && SystemID[videoch] >= static_cast<int>(MAX_VIDEO_STREAMS)) return SVL_FAIL;

    format.width = 1920;//CaptureProc[0]->GetSDIin().getWidth();
    format.height = 1080;//CaptureProc[0]->GetSDIin().getHeight();
    format.colorspace = svlFilterSourceVideoCapture::PixelRGB8;
    format.rgb_order = true;
    format.yuyv_order = false;
    format.framerate = 59.94;
    format.custom_mode = -1;

    return SVL_OK;
}

bool svlVidCapSrcSDI::IsCaptureSupported(unsigned int sysid, unsigned int digid)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::IsCaptureSupported(" << sysid << ", " << digid << ")" << std::endl;
#endif

    //if (sysid >= MILNumberOfSystems || digid >= MilNumberOfDigitizers[sysid]) return false;
    //return MilCaptureSupported[sysid][digid];
    //TODO
    return true;
}

bool svlVidCapSrcSDI::IsOverlaySupported(unsigned int sysid, unsigned int digid)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::IsOverlaySupported(" << sysid << ", " << digid << ")" << std::endl;
#endif

    //if (sysid >= MILNumberOfSystems || digid >= MilNumberOfDigitizers[sysid]) return false;
    //return MilOverlaySupported[sysid][digid];
    //TODO
    return true;
}


#ifdef _WIN32

svlVidCapSrcSDIThread::svlVidCapSrcSDIThread(int streamid)
{

    StreamID = streamid;
    InitSuccess = false;
}

void* svlVidCapSrcSDIThread::Proc(svlVidCapSrcSDI* baseref)
{
    // Signal success to main thread
    Error = false;
    InitSuccess = true;
    InitEvent.Raise();
    GLint inBuf;
    cudaError_t cerr;
    unsigned char* inDevicePtr;
    unsigned char* outDevicePtr;
    unsigned char* ptr;

    LPSTR lpCmdLine = 0;
    if(Configure(&lpCmdLine) == E_FAIL)
        return FALSE;

    if(SetupSDIDevices() == E_FAIL)
        return FALSE;

    // Calculate the window size based on the incoming and outgoing video signals
    CalcWindowSize();

    HINSTANCE hInstance = GetModuleHandle(NULL);	// Need a handle to this process instance
    // Create window.  Use video dimensions of video initialized above.
    g_hWnd = SetupWindow(hInstance, 0, 0, "NVIDIA Quadro SDI Capture to memory");

    // Exit on error.
    if (!g_hWnd)
        return FALSE;

    SetupGL();

    if(StartSDIPipeline() == E_FAIL)
        return FALSE;

    unsigned int pitch0 = m_SDIin.GetBufferObjectPitch (0);
    unsigned int pitch1 = m_SDIin.GetBufferObjectPitch (1);
    unsigned int height = m_SDIin.GetHeight();
    //unsigned int size = pitch0*height;
    std::cout << "svlVidCapSrcSDIThread::Proc(), pitches: " << pitch0 << ", " << pitch1 << " height: " << height << std::endl;
    for (int i = 0; i < m_SDIin.GetNumStreams (); i++) {
        m_memBuf[i] =
                (unsigned char *) malloc (m_SDIin.GetBufferObjectPitch (i) *
                                          m_SDIin.GetHeight ());

        //#if __VERBOSE__ == 1
        std::cout << "svlVidCapSrcSDIThread::Proc - Allocate image buffer (" << m_SDIin.GetWidth() << ", " << m_SDIin.GetHeight() << ")" << std::endl;
        //#endif
        baseref->ImageBuffer[i] = new svlBufferImage(m_SDIin.GetWidth(), m_SDIin.GetHeight());
        comprBuffer[i] =  (unsigned char *) malloc (m_SDIin.GetWidth() * 3 *
                                                    m_SDIin.GetHeight());
    }

    while (baseref->Running) {
        if (CaptureVideo() != GL_FAILURE_NV)
        {
            for (int i = 0; i < m_SDIin.GetNumStreams(); i++) {

                // Allocate required space in video capture buffer
                glBindBuffer(GL_VIDEO_BUFFER_NV, m_videoTextures[i]);
                assert(glGetError() == GL_NO_ERROR);
                glGetBufferSubData (GL_VIDEO_BUFFER_NV, 0,
                                    m_SDIin.GetBufferObjectPitch (i) *
                                    m_SDIin.GetHeight (), m_memBuf[i]);


                glBindBuffer (GL_VIDEO_BUFFER_NV, NULL);
                unsigned int size=0;
                ptr = baseref->ImageBuffer[i]->GetPushBuffer(size);
                if (!size || !ptr) { /* trouble */ }
                //memcpy(ptr, m_memBuf[i], size);
                svlConverter::RGBA32toRGB24(m_memBuf[i], ptr, m_SDIin.GetWidth()*m_SDIin.GetHeight());
                flip(ptr,i);
                baseref->ImageBuffer[i]->Push();
            }
        }
    }
    return this;
}

GLenum svlVidCapSrcSDIThread::CaptureVideo(float runTime)
{
    static GLuint64EXT captureTime;
    GLuint sequenceNum;
    static GLuint prevSequenceNum = 0;
    GLenum ret;
    static int numFails = 0;
    static int numTries = 0;
    GLuint captureLatency = 0;
    unsigned int droppedFrames;

    if(numFails < 100) {
        // Capture the video to a buffer object
#ifdef _WIN32
        ret = m_SDIin.Capture(&sequenceNum, &captureTime);
#else
        ret = m_SDIin.capture(&sequenceNum, &captureTime);
#endif
        if(sequenceNum - prevSequenceNum > 1)
        {
            droppedFrames = sequenceNum - prevSequenceNum;
#if __VERBOSE__ == 1
            printf("glVideoCaptureNV: Dropped %d frames\n",sequenceNum - prevSequenceNum);
            printf("Frame:%d gpuTime:%f gviTime:%f\n", sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime);
#endif
            captureLatency = 1;
        }

#if __VERBOSE__ == 1
        if(m_SDIin.m_gviTime > 1.0/30)
        {
            printf("Frame:%d gpuTime:%f gviTime:%f\n", sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime);
            //*captureLatency = 1;
        }
#endif

        prevSequenceNum = sequenceNum;
        switch(ret) {
        case GL_SUCCESS_NV:
#if __VERBOSE__ == 1
            printf("Frame:%d gpuTime:%f gviTime:%f\n", sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime);
#endif
            numFails = 0;
            break;
        case GL_PARTIAL_SUCCESS_NV:
            printf("glVideoCaptureNV: GL_PARTIAL_SUCCESS_NV\n");
            numFails = 0;
            break;
        case GL_FAILURE_NV:
            printf("glVideoCaptureNV: GL_FAILURE_NV - Video capture failed.\n");
            numFails++;
            break;
        default:
            printf("glVideoCaptureNV: Unknown return value.\n");
            break;
        } // switch

    }
    // The incoming signal format or some other error occurred during
    // capture, shutdown and try to restart capture.
    else {
        if(numTries == 0) {
            stopSDIPipeline();
            //cleanupSDIDevices();

            cleanupSDIinGL();
        }

        // Initialize the video capture device.
#ifdef _WIN32
        if (setupSDIinDevices() != TRUE) {
#else
        if (setupSDIinDevice(dpy,gpu) != TRUE) {
#endif
            numTries++;
            return GL_FAILURE_NV;
        }

        // Reinitialize OpenGL.
        setupSDIinGL();

        StartSDIPipeline();
        numFails = 0;
        numTries = 0;
        return GL_FAILURE_NV;
    }

    // Rough workaround latency from capture queue
    if(captureLatency==1)
    {
        for(unsigned int i=0;i< droppedFrames+1;i++)
        {
#if __VERBOSE__ == 1
            printf("Call: %d of %d Frame:%d gpuTime:%f gviTime:%f goal:%f\n", i, droppedFrames+1, sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime,1.0/30);
#endif
            CaptureVideo();
        }
    }
    if(m_SDIin.m_gviTime + runTime > 1.0/30)
    {
#if __VERBOSE__ == 1
        printf("Call: %f decrease to %f Frame:%d gpuTime:%f gviTime:%f goal:%f\n", runTime,runTime-1.0/30,sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime,1.0/30);
#endif
        CaptureVideo(runTime-1.0/30);
        captureLatency = 1;
    }else if(m_SDIin.m_gviTime > 1.0/30)
    {
#if __VERBOSE__ == 1
        printf("Call: %f decrease to %f Frame:%d gpuTime:%f gviTime:%f goal:%f\n", runTime,runTime-1.0/30,sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime,1.0/30);
#endif
        CaptureVideo();
    }

    return ret;
}

HRESULT svlVidCapSrcSDIThread::Configure(char *szCmdLine[])
{
    int numGPUs;
    // Note, this function enumerates GPUs which are both CUDA & GLAffinity capable (i.e. newer Quadros)
    numGPUs = CNvSDIoutGpuTopology::instance().getNumGpu();

    if(numGPUs <= 0)
    {
        MessageBox(NULL, "Unable to obtain system GPU topology", "Error", MB_OK);
        return E_FAIL;
    }

    int numCaptureDevices = CNvSDIinTopology::instance().getNumDevice();

    if(numCaptureDevices <= 0)
    {
        MessageBox(NULL, "Unable to obtain system Capture topology", "Error", MB_OK);
        return E_FAIL;
    }
    options.sampling = NVVIOCOMPONENTSAMPLING_422;
    options.dualLink = false;
    options.bitsPerComponent = 8;
    options.expansionEnable = true;
    options.captureDevice = 0;
    options.captureGPU = CNvSDIoutGpuTopology::instance().getPrimaryGpuIndex();

    ParseCommandLine(szCmdLine, &options);//get the user config

    if(options.captureDevice >= numCaptureDevices)
    {
        MessageBox(NULL, "Selected Capture Device is out of range", "Error", MB_OK);
        return E_FAIL;
    }
    if(options.captureGPU >= numGPUs)
    {
        MessageBox(NULL, "Selected Capture GPU is out of range", "Error", MB_OK);
        return E_FAIL;
    }
    m_gpu = CNvSDIoutGpuTopology::instance().getGpu(options.captureGPU);

    return S_OK;
}

HRESULT svlVidCapSrcSDIThread::SetupSDIDevices()
{
    if(setupSDIinDevices() != S_OK)
    {
        MessageBox(NULL, "Error setting up video capture.", "Error", MB_OK);
        return E_FAIL;
    }

    return S_OK;
}

HRESULT svlVidCapSrcSDIThread::setupSDIinDevices()
{
    m_SDIin.Init(&options);

    // Initialize the video capture device.
    if (m_SDIin.SetupDevice(true,options.captureDevice) != S_OK)
        return E_FAIL;
    m_videoWidth = m_SDIin.GetWidth();
    m_videoHeight = m_SDIin.GetHeight();
    m_num_streams = m_SDIin.GetNumStreams();
    return S_OK;
}

GLboolean svlVidCapSrcSDIThread::SetupGL()
{
    // Create window device context and rendering context.
    m_hDC = GetDC(m_hWnd);

    HGPUNV  gpuMask[2];
    gpuMask[0] = m_gpu->getAffinityHandle();
    gpuMask[1] = NULL;
    if (!(m_hAffinityDC = wglCreateAffinityDCNV(gpuMask))) {
        printf("Unable to create GPU affinity DC\n");
    }

    PIXELFORMATDESCRIPTOR pfd =							// pfd Tells Windows How We Want Things To Be
    {
            sizeof (PIXELFORMATDESCRIPTOR),					// Size Of This Pixel Format Descriptor
            1,												// Version Number
            PFD_DRAW_TO_WINDOW |							// Format Must Support Window
            PFD_SUPPORT_OPENGL |							// Format Must Support OpenGL
            PFD_DOUBLEBUFFER,								// Must Support Double Buffering
            PFD_TYPE_RGBA,									// Request An RGBA Format
            24,												// Select Our Color Depth
            0, 0, 0, 0, 0, 0,								// Color Bits Ignored
            1,												// Alpha Buffer
            0,												// Shift Bit Ignored
            0,												// No Accumulation Buffer
            0, 0, 0, 0,										// Accumulation Bits Ignored
            24,												// 24 Bit Z-Buffer (Depth Buffer)
            8,												// 8 Bit Stencil Buffer
            0,												// No Auxiliary Buffer
            PFD_MAIN_PLANE,									// Main Drawing Layer
            0,												// Reserved
            0, 0, 0											// Layer Masks Ignored
};

GLuint pf = ChoosePixelFormat(m_hAffinityDC, &pfd);
HRESULT rslt = SetPixelFormat(m_hAffinityDC, pf, &pfd);
//		return NULL;
//Create affinity-rendering context from affinity-DC
if (!(m_hRC = wglCreateContext(m_hAffinityDC))) {
    printf("Unable to create GPU affinity RC\n");
}

//m_hRC = wglCreateContext(m_hDC);

// Make window rendering context current.
wglMakeCurrent(m_hDC, m_hRC);
//load the required OpenGL extensions:
if(!loadSwapIntervalExtension() || !loadTimerQueryExtension() || !loadAffinityExtension())
{
    printf("Could not load the required OpenGL extensions\n");
    return false;
}


// Unlock capture/draw loop from vsync of graphics display.
// This should lock the capture/draw loop to the vsync of
// input video.
if (wglSwapIntervalEXT) {
    wglSwapIntervalEXT(0);
}
glClearColor( 0.0, 0.0, 0.0, 0.0);
glClearDepth( 1.0 );

glDisable(GL_DEPTH_TEST);

glDisable(GL_TEXTURE_1D);
glDisable(GL_TEXTURE_2D);

setupSDIinGL();

//setupSDIoutGL();

return GL_TRUE;
}

//
// Initialize OpenGL
//
HRESULT svlVidCapSrcSDIThread::setupSDIinGL()
{
    //load the required OpenGL extensions:
    if(!loadCaptureVideoExtension() || !loadBufferObjectExtension() )
    {
        printf("Could not load the required OpenGL extensions\n");
        return false;
    }
    // Setup CSC for each stream.
    GLfloat mat[4][4];
    float scale = 1.0f;

    GLfloat max[] = {5000, 5000, 5000, 5000};;
    GLfloat min[] = {0, 0, 0, 0};

    // Initialize matrix to the identity.
    mat[0][0] = scale; mat[0][1] = 0; mat[0][2] = 0; mat[0][3] = 0;
    mat[1][0] = 0; mat[1][1] = scale; mat[1][2] = 0; mat[1][3] = 0;
    mat[2][0] = 0; mat[2][1] = 0; mat[2][2] = scale; mat[2][3] = 0;
    mat[3][0] = 0; mat[3][1] = 0; mat[3][2] = 0; mat[3][3] = scale;

    GLfloat offset[] = {0, 0, 0, 0};

    if (1) {
        mat[0][0] = 1.164f *scale;
        mat[0][1] = 1.164f *scale;
        mat[0][2] = 1.164f *scale;
        mat[0][3] = 0;

        mat[1][0] = 0;
        mat[1][1] = -0.392f *scale;
        mat[1][2] = 2.017f *scale;
        mat[1][3] = 0;

        mat[2][0] = 1.596f *scale;
        mat[2][1] = -0.813f *scale;
        mat[2][2] = 0.f;
        mat[2][3] = 0;

        mat[3][0] = 0;
        mat[3][1] = 0;
        mat[3][2] = 0;
        mat[3][3] = 1;

        offset[0] =-0.87f;
        offset[1] = 0.53026f;
        offset[2] = -1.08f;
        offset[3] = 0;
    }
    m_SDIin.SetCSCParams(&mat[0][0], offset, min, max);

    GLuint gpuVideoSlot = 1;

    m_SDIin.BindDevice(gpuVideoSlot,m_hDC);

    glGenBuffers(m_SDIin.GetNumStreams(), m_videoTextures);

    m_videoHeight = m_SDIin.GetHeight();
    m_videoWidth = m_SDIin.GetWidth();
    //m_videoBufferFormat = GL_RGBA8;
    for(unsigned int i = 0; i < m_SDIin.GetNumStreams();i++)
    {
        m_SDIin.BindVideoFrameBuffer(m_videoTextures[i],GL_RGBA8, i);
        //m_videoBufferPitch = m_SDIin.GetBufferObjectPitch(i);

        // Allocate required space in video capture buffer
        glBindBuffer(GL_VIDEO_BUFFER_NV, m_videoTextures[i]);
        assert(glGetError() == GL_NO_ERROR);
        glBufferData(GL_VIDEO_BUFFER_NV, m_SDIin.GetBufferObjectPitch(i) * m_videoHeight,
                     NULL, GL_STREAM_COPY);
        assert(glGetError() == GL_NO_ERROR);

    }
}

void svlVidCapSrcSDIThread::CalcWindowSize()
{
    switch(m_SDIin.GetSignalFormat()) {
    case NVVIOSIGNALFORMAT_487I_59_94_SMPTE259_NTSC:
    case NVVIOSIGNALFORMAT_576I_50_00_SMPTE259_PAL:
        if (m_SDIin.GetNumStreams() == 1) {
            m_windowWidth = m_videoWidth; m_windowHeight = m_videoHeight;
        } else if (m_SDIin.GetNumStreams() == 2) {
            m_windowWidth = m_videoWidth; m_windowHeight = m_videoHeight<<1;
        } else {
            m_windowWidth = m_videoWidth<<1; m_windowHeight = m_videoHeight<<1;
        }
        break;

    case NVVIOSIGNALFORMAT_720P_59_94_SMPTE296:
    case NVVIOSIGNALFORMAT_720P_60_00_SMPTE296:
    case NVVIOSIGNALFORMAT_720P_50_00_SMPTE296:
    case NVVIOSIGNALFORMAT_720P_30_00_SMPTE296:
    case NVVIOSIGNALFORMAT_720P_29_97_SMPTE296:
    case NVVIOSIGNALFORMAT_720P_25_00_SMPTE296:
    case NVVIOSIGNALFORMAT_720P_24_00_SMPTE296:
    case NVVIOSIGNALFORMAT_720P_23_98_SMPTE296:
        if (m_SDIin.GetNumStreams() == 1) {
            m_windowWidth = m_videoWidth>>2; m_windowHeight = m_videoHeight>>2;
        } else if (m_SDIin.GetNumStreams() == 2) {
            m_windowWidth = m_videoWidth>>2; m_windowHeight = m_videoHeight>>1;
        } else {
            m_windowWidth = m_videoWidth>>1; m_windowHeight = m_videoHeight>>1;
        }
        break;

    case NVVIOSIGNALFORMAT_1035I_59_94_SMPTE260:
    case NVVIOSIGNALFORMAT_1035I_60_00_SMPTE260:
    case NVVIOSIGNALFORMAT_1080I_50_00_SMPTE295:
    case NVVIOSIGNALFORMAT_1080I_50_00_SMPTE274:
    case NVVIOSIGNALFORMAT_1080I_59_94_SMPTE274:
    case NVVIOSIGNALFORMAT_1080I_60_00_SMPTE274:
    case NVVIOSIGNALFORMAT_1080P_23_976_SMPTE274:
    case NVVIOSIGNALFORMAT_1080P_24_00_SMPTE274:
    case NVVIOSIGNALFORMAT_1080P_25_00_SMPTE274:
    case NVVIOSIGNALFORMAT_1080P_29_97_SMPTE274:
    case NVVIOSIGNALFORMAT_1080P_30_00_SMPTE274:
    case NVVIOSIGNALFORMAT_1080I_48_00_SMPTE274:
    case NVVIOSIGNALFORMAT_1080I_47_96_SMPTE274:
    case NVVIOSIGNALFORMAT_1080PSF_25_00_SMPTE274:
    case NVVIOSIGNALFORMAT_1080PSF_29_97_SMPTE274:
    case NVVIOSIGNALFORMAT_1080PSF_30_00_SMPTE274:
    case NVVIOSIGNALFORMAT_1080PSF_24_00_SMPTE274:
    case NVVIOSIGNALFORMAT_1080PSF_23_98_SMPTE274:
        if (m_SDIin.GetNumStreams() == 1) {
            m_windowWidth = m_videoWidth>>2; m_windowHeight = m_videoHeight>>2;
        } else if (m_SDIin.GetNumStreams() == 2) {
            m_windowWidth = m_videoWidth>>2; m_windowHeight = m_videoHeight>>1;
        } else {
            m_windowWidth = m_videoWidth>>1; m_windowHeight = m_videoHeight>>1;
        }
        break;

    case NVVIOSIGNALFORMAT_2048P_30_00_SMPTE372:
    case NVVIOSIGNALFORMAT_2048P_29_97_SMPTE372:
    case NVVIOSIGNALFORMAT_2048I_60_00_SMPTE372:
    case NVVIOSIGNALFORMAT_2048I_59_94_SMPTE372:
    case NVVIOSIGNALFORMAT_2048P_25_00_SMPTE372:
    case NVVIOSIGNALFORMAT_2048I_50_00_SMPTE372:
    case NVVIOSIGNALFORMAT_2048P_24_00_SMPTE372:
    case NVVIOSIGNALFORMAT_2048P_23_98_SMPTE372:
    case NVVIOSIGNALFORMAT_2048I_48_00_SMPTE372:
    case NVVIOSIGNALFORMAT_2048I_47_96_SMPTE372:
        if (m_SDIin.GetNumStreams() == 1) {
            m_windowWidth = m_videoWidth>>2; m_windowHeight = m_videoHeight>>2;
        } else if (m_SDIin.GetNumStreams() == 2) {
            m_windowWidth = m_videoWidth>>2; m_windowHeight = m_videoHeight>>1;
        } else {
            m_windowWidth = m_videoWidth>>1; m_windowHeight = m_videoHeight>>1;
        }
        break;

    default:
        m_windowWidth = 500;
        m_windowHeight = 500;
    }
}

HWND svlVidCapSrcSDIThread::SetupWindow(HINSTANCE hInstance, int x, int y,
                                        char *title)
{
    WNDCLASS   wndclass;
    HWND	   hWnd;
    HDC	  hDC;								// Device context

    BOOL bStatus;
    unsigned int uiNumFormats;
    CHAR szAppName[]="OpenGL SDI Demo";

    int pixelformat;

    // Register the frame class.
    wndclass.style         = 0;
    wndclass.lpfnWndProc   = DefWindowProc;//(WNDPROC) MainWndProc;
    wndclass.cbClsExtra    = 0;
    wndclass.cbWndExtra    = 0;
    wndclass.hInstance     = hInstance;
    wndclass.hIcon         = LoadIcon (hInstance, szAppName);
    wndclass.hCursor       = LoadCursor (NULL,IDC_ARROW);
    wndclass.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);
    wndclass.lpszMenuName  = szAppName;
    wndclass.lpszClassName = szAppName;

    if (!RegisterClass (&wndclass) )
        return NULL;

    // Create initial window frame.
    m_hWnd = CreateWindow ( szAppName, title,
                            WS_CAPTION | WS_BORDER |  WS_SIZEBOX | WS_SYSMENU | WS_MAXIMIZEBOX | WS_MINIMIZEBOX,
                            x,
                            y,
                            m_windowWidth,
                            m_windowHeight,
                            NULL,
                            NULL,
                            hInstance,
                            NULL );

    // Exit on error.
    if (!m_hWnd)
        return NULL;

    // Get device context for new window.
    hDC = GetDC(m_hWnd);

    PIXELFORMATDESCRIPTOR pfd =							// pfd Tells Windows How We Want Things To Be
    {
            sizeof (PIXELFORMATDESCRIPTOR),					// Size Of This Pixel Format Descriptor
            1,												// Version Number
            PFD_DRAW_TO_WINDOW |							// Format Must Support Window
            PFD_SUPPORT_OPENGL |							// Format Must Support OpenGL
            PFD_DOUBLEBUFFER,								// Must Support Double Buffering
            PFD_TYPE_RGBA,									// Request An RGBA Format
            24,												// Select Our Color Depth
            0, 0, 0, 0, 0, 0,								// Color Bits Ignored
            1,												// Alpha Buffer
            0,												// Shift Bit Ignored
            0,												// No Accumulation Buffer
            0, 0, 0, 0,										// Accumulation Bits Ignored
            24,												// 24 Bit Z-Buffer (Depth Buffer)
            8,												// 8 Bit Stencil Buffer
            0,												// No Auxiliary Buffer
            PFD_MAIN_PLANE,									// Main Drawing Layer
            0,												// Reserved
            0, 0, 0											// Layer Masks Ignored
};

// Choose pixel format.
if ( (pixelformat = ChoosePixelFormat(hDC, &pfd)) == 0 ) {
    MessageBox(NULL, "ChoosePixelFormat failed", "Error", MB_OK);
    return FALSE;
}

// Set pixel format.
if (SetPixelFormat(hDC, pixelformat, &pfd) == FALSE) {
    MessageBox(NULL, "SetPixelFormat failed", "Error", MB_OK);
    return FALSE;
}


// Release device context.
ReleaseDC(m_hWnd, hDC);

// Return window handle.
return(m_hWnd);

}

HRESULT svlVidCapSrcSDIThread::StartSDIPipeline()
{
    // Start video capture
    if(m_SDIin.StartCapture()!= S_OK)
    {
        MessageBox(NULL, "Error starting video capture.", "Error", MB_OK);
        return E_FAIL;
    }
    return S_OK;
}

HRESULT svlVidCapSrcSDIThread::stopSDIPipeline()
{
    m_SDIin.EndCapture();
    return S_OK;
}

GLboolean svlVidCapSrcSDIThread::cleanupSDIGL()
{
    GLboolean val = GL_TRUE;
    cleanupSDIinGL();
    //if(m_bSDIout)
    //	cleanupSDIoutGL();
    // Delete OpenGL rendering context.
    wglMakeCurrent(NULL,NULL) ;
    if (m_hRC)
    {
        wglDeleteContext(m_hRC) ;
        m_hRC = NULL ;
    }
    ReleaseDC(m_hWnd,m_hDC);

    wglDeleteDCNV(m_hAffinityDC);

    return val;
}

HRESULT svlVidCapSrcSDIThread::cleanupSDIinGL()
{
    for(unsigned int i = 0; i < m_SDIin.GetNumStreams(); i++)
        m_SDIin.UnbindVideoTexture(i);
    m_SDIin.UnbindDevice();
    glDeleteTextures(m_SDIin.GetNumStreams(),m_videoTextures);


    return S_OK;
}

void
svlVidCapSrcSDIThread::Shutdown()
{
    stopSDIPipeline();
    cleanupSDIGL();
    //CleanupGL();
    //cleanupSDIDevices();
}

void svlVidCapSrcSDIThread::flip(unsigned char* image, int index)
{
    const unsigned int stride =  m_SDIin.GetWidth() * 3;
    const unsigned int rows = m_SDIin.GetHeight() >> 1;
    unsigned char *down = image;
    unsigned char *up = image + (m_SDIin.GetHeight() - 1) * stride;

    for (unsigned int i = 0; i < rows; i ++) {
        memcpy(comprBuffer[index], down, stride);
        memcpy(down, up, stride);
        memcpy(up, comprBuffer[index], stride);

        down += stride; up -= stride;
    }
}

#else
/////////////////////////////////////
// Windows routines
/////////////////////////////////////


//-----------------------------------------------------------------------------
// Name: calcWindowSize
// Desc: Calculate the graphics window size
//-----------------------------------------------------------------------------
void
svlVidCapSrcSDIThread::calcWindowSize()
{
    int numStreams = m_SDIin.getNumStreams();
    //TODO: fix frame rate
    m_inputFrameRate = 59.94;
    switch(m_SDIin.getVideoFormat()) {
    case NV_CTRL_GVIO_VIDEO_FORMAT_487I_59_94_SMPTE259_NTSC:
    case NV_CTRL_GVIO_VIDEO_FORMAT_576I_50_00_SMPTE259_PAL:
        if (numStreams == 1) {
            m_windowWidth = 360; m_windowHeight = 243;
        } else if (numStreams == 2) {
            m_windowWidth = 720; m_windowHeight = 496;
        } else {
            m_windowWidth = 720; m_windowHeight = 486;
        }
        break;

    case NV_CTRL_GVIO_VIDEO_FORMAT_720P_59_94_SMPTE296:
    case NV_CTRL_GVIO_VIDEO_FORMAT_720P_60_00_SMPTE296:
    case NV_CTRL_GVIO_VIDEO_FORMAT_720P_50_00_SMPTE296:
    case NV_CTRL_GVIO_VIDEO_FORMAT_720P_30_00_SMPTE296:
    case NV_CTRL_GVIO_VIDEO_FORMAT_720P_29_97_SMPTE296:
    case NV_CTRL_GVIO_VIDEO_FORMAT_720P_25_00_SMPTE296:
    case NV_CTRL_GVIO_VIDEO_FORMAT_720P_24_00_SMPTE296:
    case NV_CTRL_GVIO_VIDEO_FORMAT_720P_23_98_SMPTE296:
        if (numStreams == 1) {
            m_windowWidth = 320; m_windowHeight = 180;
        } else if (numStreams == 2) {
            m_windowWidth = 320; m_windowHeight = 360;
        } else {
            m_windowWidth = 640; m_windowHeight = 486;
        }
        break;
    case NV_CTRL_GVIO_VIDEO_FORMAT_1035I_59_94_SMPTE260:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1035I_60_00_SMPTE260:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_50_00_SMPTE295:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_50_00_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_59_94_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_60_00_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_23_976_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_24_00_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_25_00_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_29_97_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080P_30_00_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_48_00_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080I_47_96_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_25_00_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_29_97_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_30_00_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_24_00_SMPTE274:
    case NV_CTRL_GVIO_VIDEO_FORMAT_1080PSF_23_98_SMPTE274:
        if (numStreams == 1) {
            m_windowWidth = 480; m_windowHeight = 270;
        } else if (numStreams == 2) {
            m_windowWidth = 480; m_windowHeight = 540;
        } else {
            m_windowWidth = 960; m_windowHeight = 540;
        }
        break;
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_30_00_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_29_97_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_60_00_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_59_94_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_25_00_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_50_00_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_24_00_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048P_23_98_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_48_00_SMPTE372:
    case NV_CTRL_GVIO_VIDEO_FORMAT_2048I_47_96_SMPTE372:
        if (numStreams == 1) {
            m_windowWidth = 512; m_windowHeight = 270;
        } else if (numStreams == 2) {
            m_windowWidth = 512; m_windowHeight = 540;
        } else {
            m_windowWidth = 1024; m_windowHeight = 540;
        }
        break;
    default:
        m_windowWidth = 500;
        m_windowHeight = 500;
    }
}

//-----------------------------------------------------------------------------
// Name: Shutdown
// Desc: Application teardown
//-----------------------------------------------------------------------------
void
svlVidCapSrcSDIThread::Shutdown()
{
    StopSDIPipeline();
    cleanupSDIDevices();
    cleanupGL();
    XCloseDisplay(dpy);
}

//-----------------------------------------------------------------------------
// Name: CreateWindow
// Desc: Create window
//-----------------------------------------------------------------------------
Window
svlVidCapSrcSDIThread::CreateWindow()
{
    XVisualInfo *vi ;
    GLXFBConfig *configs, config;
    XEvent event;
    XSetWindowAttributes swa;

    unsigned long mask;
    int numConfigs;
    int config_list[] = { GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
                          GLX_DOUBLEBUFFER, GL_TRUE,
                          GLX_RENDER_TYPE, GLX_RGBA_BIT,
                          GLX_RED_SIZE, 8,
                          GLX_GREEN_SIZE, 8,
                          GLX_BLUE_SIZE, 8,
                          GLX_FLOAT_COMPONENTS_NV, GL_FALSE,
                          None };
    int i;
    // Find required framebuffer configuration
    configs = glXChooseFBConfig(dpy, captureOptions.xscreen, config_list, &numConfigs);
    if (!configs) {
        fprintf(stderr, "Unable to find a matching FBConfig.\n");
        exit(1);
    }

    // Find an FBconfig with the required number of color bits.
    for (i = 0; i < numConfigs; i++) {
        int attr;
        if (glXGetFBConfigAttrib(dpy, configs[i], GLX_RED_SIZE, &attr)) {
            printf("glXGetFBConfigAttrib(GLX_RED_SIZE) failed!\n");
            exit(1);
        }
        if (attr != 8)
            continue;

        if (glXGetFBConfigAttrib(dpy, configs[i], GLX_GREEN_SIZE, &attr)) {
            printf("glXGetFBConfigAttrib(GLX_GREEN_SIZE) failed!\n");
            exit(1);
        }
        if (attr != 8)
            continue;

        if (glXGetFBConfigAttrib(dpy, configs[i], GLX_BLUE_SIZE, &attr)) {
            printf("glXGetFBConfigAttrib(GLX_BLUE_SIZE) failed!\n");
            exit(1);
        }

        if (attr != 8)
            continue;

        if (glXGetFBConfigAttrib(dpy, configs[i], GLX_ALPHA_SIZE, &attr)) {
            printf("glXGetFBConfigAttrib(GLX_ALPHA_SIZE) failed\n");
            exit(1);
        }

        if (attr != 8)
            continue;

        break;
    }

    if (i == numConfigs) {
        printf("No FBConfigs found\n");
        exit(1);
    }

    config = configs[i];

    // Don't need the config list anymore so free it.
    XFree(configs);
    configs = NULL;

    // Create an OpenGL rendering context for the onscreen window.
    ctx = glXCreateNewContext(dpy, config, GLX_RGBA_TYPE, 0, GL_TRUE);

    // Get visual from FB config.
    if ((vi = glXGetVisualFromFBConfig(dpy, config)) != NULL) {
        printf("Using visual %0x\n", (int) vi->visualid);
        printf("Depth = %d\n", vi->depth);
    } else {
        printf("Couldn't find visual for onscreen window.\n");
        exit(1);
    }

    // Create color map.
    if (!(cmap = XCreateColormap(dpy, RootWindow(dpy, vi->screen),
                                 vi->visual, AllocNone))) {
        fprintf(stderr, "XCreateColormap failed!\n");
        exit(1);
    }


    // Calculate window width & height.
    calcWindowSize();

    // Create window.
    swa.colormap = cmap;
    swa.border_pixel = 0;
    swa.background_pixel = 1;
    swa.event_mask = ExposureMask | StructureNotifyMask | KeyPressMask |
            KeyReleaseMask | ButtonPressMask | ButtonReleaseMask |
            PointerMotionMask ;
    mask = CWBackPixel | CWBorderPixel | CWColormap | CWEventMask;
    win = XCreateWindow(dpy, RootWindow(dpy, vi->screen),
                        0, 0, m_windowWidth, m_windowHeight, 0,
                        vi->depth, InputOutput, vi->visual,
                        mask, &swa);

    // Map window.
    XMapWindow(dpy, win);
    XIfEvent(dpy, &event, WaitForNotify, (char *) win);

    // Set window colormap.
    XSetWMColormapWindows(dpy, win, &win, 1);

    // Make OpenGL rendering context current.
    if (!(glXMakeCurrent(dpy, win, ctx))) {
        fprintf(stderr, "glXMakeCurrent failed!\n");
        exit(1);
    }

    // Don't lock the capture/draw loop to the graphics vsync.
    glXSwapIntervalSGI(0);
    XFlush(dpy);

    return win;
}


//-----------------------------------------------------------------------------
// Name: DestroyWindow
// Desc: Destroy window
//-----------------------------------------------------------------------------
bool
svlVidCapSrcSDIThread::destroyWindow()
{
    XUnmapWindow(dpy,win);
    XDestroyWindow(dpy, win);
    XFreeColormap(dpy,cmap);

    return true;
}


//-----------------------------------------------------------------------------
// Name: setupSDIinDevice
// Desc: Initialize SDI capture device state.
//-----------------------------------------------------------------------------
bool
svlVidCapSrcSDIThread::setupSDIinDevice(Display *d, HGPUNV *g)
{
    GLfloat mat[4][4];
    float scale = 1.0f;
    GLfloat max[] = {5000, 5000, 5000, 5000};
    //GLfloat max[] = {256, 256, 256, 256};
    GLfloat min[] = {0, 0, 0, 0};
    // Initialize matrix to the identity.
    mat[0][0] = scale; mat[0][1] = 0; mat[0][2] = 0; mat[0][3] = 0;
    mat[1][0] = 0; mat[1][1] = scale; mat[1][2] = 0; mat[1][3] = 0;
    mat[2][0] = 0; mat[2][1] = 0; mat[2][2] = scale; mat[2][3] = 0;
    mat[3][0] = 0; mat[3][1] = 0; mat[3][2] = 0; mat[3][3] = scale;
    GLfloat offset[] = {0, 0, 0, 0};
    mat[0][0] = 1.164f *scale;
    mat[0][1] = 1.164f *scale;
    mat[0][2] = 1.164f *scale;
    mat[0][3] = 0;

    mat[1][0] = 0;
    mat[1][1] = -0.392f *scale;
    mat[1][2] = 2.017f *scale;
    mat[1][3] = 0;

    mat[2][0] = 1.596f *scale;
    mat[2][1] = -0.813f *scale;
    mat[2][2] = 0.f;
    mat[2][3] = 0;

    mat[3][0] = 0;
    mat[3][1] = 0;
    mat[3][2] = 0;
    mat[3][3] = 1;

    offset[0] =-0.87f;
    offset[1] = 0.53026f;
    offset[2] = -1.08f;
    offset[3] = 0;


    captureOptions.cscMax = max;
    captureOptions.cscMin = min;
    captureOptions.cscMat = &mat[0][0];
    captureOptions.cscOffset = offset;
    captureOptions.captureType = BUFFER_FRAME;
    captureOptions.bufferInternalFormat = GL_RGBA8;
    captureOptions.bitsPerComponent = NV_CTRL_GVI_BITS_PER_COMPONENT_8;
    captureOptions.sampling = NV_CTRL_GVI_COMPONENT_SAMPLING_422;
    captureOptions.xscreen = g->deviceXScreen;
    captureOptions.bDualLink = false;
    captureOptions.bChromaExpansion = true;
    m_SDIin.setCaptureOptions(d,captureOptions);
    bool ret = m_SDIin.initCaptureDeviceNVCtrl();

    return ret;
}


//-----------------------------------------------------------------------------
// Name: setupSDIinGL
// Desc: Initialize OpenGL SDI capture state.
//-----------------------------------------------------------------------------
bool
svlVidCapSrcSDIThread::setupSDIinGL()
{
    //Setup GL
    m_SDIin.initCaptureDeviceGL();
    return true;
}


//-----------------------------------------------------------------------------
// Name: cleanupSDIin()
// Desc: Destroy SDI capture device.
//-----------------------------------------------------------------------------
bool
svlVidCapSrcSDIThread::cleanupSDIin()
{
    m_SDIin.endCapture();
    bool ret = m_SDIin.destroyCaptureDeviceNVCtrl();
    return ret;
}


/////////////////////////////////////
// Main Methods
/////////////////////////////////////


//-----------------------------------------------------------------------------
// Name: SetupGL
// Desc: Setup OpenGL capture.
//-----------------------------------------------------------------------------
GLboolean
svlVidCapSrcSDIThread::SetupGL()
{
    glClearColor( 0.0, 0.0, 0.0, 0.0);
    glClearDepth( 1.0 );

    glDisable(GL_DEPTH_TEST);

    glDisable(GL_TEXTURE_1D);
    glDisable(GL_TEXTURE_2D);

    setupSDIinGL();

    return GL_TRUE;
}

//-----------------------------------------------------------------------------
// Name: SetupSDIDevices
// Desc: Setup SDI capture devices
//-----------------------------------------------------------------------------
bool
svlVidCapSrcSDIThread::SetupSDIDevices(Display *d,HGPUNV *g)
{
    if(d && g)
    {
        gpu = g;
        dpy = d;
    }
    if(!dpy || !gpu)
        return FALSE;

    if(setupSDIinDevice(dpy,gpu) != TRUE) {
        printf("Error setting up video capture.\n");
        return FALSE;
    }

    return TRUE;
}


//-----------------------------------------------------------------------------
// Name: StartSDIPipeline
// Desc: Start SDI video capture.
//-----------------------------------------------------------------------------
bool
svlVidCapSrcSDIThread::StartSDIPipeline()
{
    // Start video capture
    if(m_SDIin.startCapture()!= TRUE) {
        printf("Error starting video capture.\n");
        return FALSE;
    }
    //CaptureStarted = true;
    return TRUE;
}


//-----------------------------------------------------------------------------
// Name: StopSDIPipeline
// Desc: Stop SDI video capture.
//-----------------------------------------------------------------------------
bool
svlVidCapSrcSDIThread::StopSDIPipeline()
{
    if(m_SDIin.endCapture()!= TRUE) {
        printf("Error starting video capture.\n");
        return FALSE;
    }
    return TRUE;
}


//-----------------------------------------------------------------------------
// Name: CleanupSDIDevices
// Desc: Cleanup SDI capture devices.
//-----------------------------------------------------------------------------
bool
svlVidCapSrcSDIThread::cleanupSDIDevices()
{
    bool ret = TRUE;
    if(cleanupSDIin() != TRUE)
        ret = FALSE;

    return ret;
}


//-----------------------------------------------------------------------------
// Name: CleanupGL
// Desc: OpenGL teardown.
//-----------------------------------------------------------------------------
GLboolean svlVidCapSrcSDIThread::cleanupGL()
{

    m_SDIin.destroyCaptureDeviceGL();

    // Delete OpenGL rendering context.
    glXMakeCurrent(dpy,NULL,NULL) ;
    if (ctx) {
        glXDestroyContext(dpy,ctx) ;
        ctx = NULL;
    }

    return GL_TRUE;
}

//-----------------------------------------------------------------------------
// Name: Capture
// Desc: Main SDI video capture function.
//-----------------------------------------------------------------------------
GLenum
svlVidCapSrcSDIThread::CaptureVideo(float runTime)
{
    static GLuint64EXT captureTime;
    GLuint sequenceNum;
    static GLuint prevSequenceNum = 0;
    GLenum ret;
    static int numFails = 0;
    static int numTries = 0;
    GLuint captureLatency = 0;
    unsigned int droppedFrames;

    if(numFails < 100) {

        // Capture the video to a buffer object
        ret = m_SDIin.capture(&sequenceNum, &captureTime);
        if(sequenceNum - prevSequenceNum > 1)
        {
            droppedFrames = sequenceNum - prevSequenceNum;
#if __VERBOSE__ == 1
            printf("glVideoCaptureNV: Dropped %d frames\n",sequenceNum - prevSequenceNum);
            printf("Frame:%d gpuTime:%f gviTime:%f\n", sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime);
#endif
            captureLatency = 1;
        }
        //    if(m_SDIin.m_gviTime > 1.0/30)
        //    {
        //      printf("Frame:%d gpuTime:%f gviTime:%f\n", sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime);
        //      *captureLatency = 1;
        //    }

        prevSequenceNum = sequenceNum;
        switch(ret) {
        case GL_SUCCESS_NV:
#if __VERBOSE__ == 1
            printf("Frame:%d gpuTime:%f gviTime:%f\n", sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime);
#endif
            numFails = 0;
            break;
        case GL_PARTIAL_SUCCESS_NV:
            printf("glVideoCaptureNV: GL_PARTIAL_SUCCESS_NV\n");
            numFails = 0;
            break;
        case GL_FAILURE_NV:
            printf("glVideoCaptureNV: GL_FAILURE_NV - Video capture failed.\n");
            numFails++;
            break;
        default:
            printf("glVideoCaptureNV: Unknown return value.\n");
            break;
        } // switch

    }
    // The incoming signal format or some other error occurred during
    // capture, shutdown and try to restart capture.
    else {
        if(numTries == 0) {
            StopSDIPipeline();
            cleanupSDIDevices();

            cleanupGL();
        }

        // Initialize the video capture device.
        if (SetupSDIDevices(dpy,gpu) != TRUE) {
            numTries++;
            return GL_FAILURE_NV;
        }

        // Reinitialize OpenGL.
        SetupGL();

        StartSDIPipeline();
        numFails = 0;
        numTries = 0;
        return GL_FAILURE_NV;
    }

    if(captureLatency==1)
    {
        for(unsigned int i=0;i< droppedFrames+1;i++)
        {
#if __VERBOSE__ == 1
            printf("Call: %d of %d Frame:%d gpuTime:%f gviTime:%f goal:%f\n", i, droppedFrames+1, sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime,1.0/30);
#endif
            CaptureVideo();
        }
    }
    if(m_SDIin.m_gviTime + runTime > 1.0/30)
    {
#if __VERBOSE__ == 1
        printf("Call: %f decrease to %f Frame:%d gpuTime:%f gviTime:%f goal:%f\n", runTime,runTime-1.0/30,sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime,1.0/30);
#endif
        CaptureVideo(runTime-1.0/30);
        captureLatency = 1;
    }else if(m_SDIin.m_gviTime > 1.0/30)
    {
#if __VERBOSE__ == 1
        printf("Call: %f decrease to %f Frame:%d gpuTime:%f gviTime:%f goal:%f\n", runTime,runTime-1.0/30,sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime,1.0/30);
#endif
        CaptureVideo();
    }

    return ret;
}

/****************************************/
/*     svlVidCapSrcSDIThread class      */
/****************************************/

svlVidCapSrcSDIThread::svlVidCapSrcSDIThread(int streamid)
{

    StreamID = streamid;
    InitSuccess = false;
}

void* svlVidCapSrcSDIThread::Proc(svlVidCapSrcSDI* baseref)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDIThread::Proc()" << std::endl;
#endif

    // signal success to main thread
    Error = false;
    InitSuccess = true;
    InitEvent.Raise();
    unsigned char* ptr;

    HGPUNV gpuList[MAX_GPUS];
    // Open X display
    dpy = XOpenDisplay(NULL);
    //scan the systems for GPUs
    int	num_gpus = ScanHW(dpy,gpuList);
    if(num_gpus < 1)
        exit(1);
    //grab the first GPU for now for DVP
    gpu = &gpuList[0];
    SetupSDIDevices();
    win = CreateWindow();
    SetupGL();
    StartSDIPipeline();
    unsigned int pitch0 = m_SDIin.getBufferObjectPitch (0);
    unsigned int pitch1 = m_SDIin.getBufferObjectPitch (1);
    unsigned int height = m_SDIin.getHeight();
    //unsigned int size = pitch0*height;
    std::cout << "svlVidCapSrcSDIThread::Proc(), pitches: " << pitch0 << ", " << pitch1 << " height: " << height << std::endl;
    for (int i = 0; i < m_SDIin.getNumStreams (); i++) {
        m_memBuf[i] =
                (unsigned char *) malloc (m_SDIin.getBufferObjectPitch (i) *
                                          m_SDIin.getHeight ());

        //#if __VERBOSE__ == 1
        std::cout << "svlVidCapSrcSDIThread::Proc - Allocate image buffer (" << m_SDIin.getWidth() << ", " << m_SDIin.getHeight() << ")" << std::endl;
        //#endif
        baseref->ImageBuffer[i] = new svlBufferImage(m_SDIin.getWidth(), m_SDIin.getHeight());
        comprBuffer[i] =  (unsigned char *) malloc (m_SDIin.getWidth() * 3 *
                                                    m_SDIin.getHeight());
    }

    while (baseref->Running) {
        if (CaptureVideo() != GL_FAILURE_NV)
        {
            for (int i = 0; i < m_SDIin.getNumStreams(); i++) {
                glBindBufferARB (GL_VIDEO_BUFFER_NV,
                                 m_SDIin.getBufferObjectHandle (i));
                glGetBufferSubDataARB (GL_VIDEO_BUFFER_NV, 0,
                                       m_SDIin.getBufferObjectPitch (i) *
                                       m_SDIin.getHeight (), m_memBuf[i]);
                glBindBufferARB (GL_VIDEO_BUFFER_NV, NULL);
                unsigned int size=0;
                ptr = baseref->ImageBuffer[i]->GetPushBuffer(size);
                if (!size || !ptr) { /* trouble */ }
                //memcpy(ptr, m_memBuf[i], size);
                svlConverter::RGBA32toRGB24(m_memBuf[i], ptr, m_SDIin.getWidth()*m_SDIin.getHeight());
                flip(ptr,i);
                baseref->ImageBuffer[i]->Push();
            }
        }
    }
    return this;
}

void svlVidCapSrcSDIThread::flip(unsigned char* image, int index)
{
    const unsigned int stride =  m_SDIin.getWidth() * 3;
    const unsigned int rows = m_SDIin.getHeight() >> 1;
    unsigned char *down = image;
    unsigned char *up = image + (m_SDIin.getHeight() - 1) * stride;

    for (unsigned int i = 0; i < rows; i ++) {
        memcpy(comprBuffer[index], down, stride);
        memcpy(down, up, stride);
        memcpy(up, comprBuffer[index], stride);

        down += stride; up -= stride;
    }
}

#endif

