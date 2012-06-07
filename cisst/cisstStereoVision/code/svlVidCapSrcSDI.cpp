#include <cisstStereoVision/svlVidCapSrcSDI.h>
#include <cisstOSAbstraction/osaThread.h>
#include <cisstStereoVision/svlBufferImage.h>
#include <cisstOSAbstraction/osaSleep.h>

#define __VERBOSE__  1

/*************************************/
/* svlVidCapSrcSDIRenderTarget class */
/*************************************/

svlVidCapSrcSDIRenderTarget::svlVidCapSrcSDIRenderTarget(Display *d,HGPUNV *g, unsigned int video_format, GLsizei num_streams,unsigned int deviceID, unsigned int displayID):
    svlRenderTargetBase(),
    Thread(0),
    TransferSuccessful(true),
    KillThread(false),
    ThreadKilled(true)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDIRenderTarget::constructor()" << std::endl;
#endif

    SystemID = deviceID;
    DigitizerID = displayID;

    // Try to initialize overlay module
    //svlVidCapSrcSDI *device = svlVidCapSrcSDI::GetInstance();
    //TODO::CAPTURE

    // Start up overlay thread
    Thread = new osaThread;
    Thread->Create<svlVidCapSrcSDIRenderTarget, void*>(this, &svlVidCapSrcSDIRenderTarget::ThreadProc, 0);
    if (ThreadReadySignal.Wait(1.0) && ThreadKilled == false) {
        ThreadReadySignal.Raise();
    }
    else {
        // If it takes longer than 1 sec, don't execute
        KillThread = true;
    }

    setupSDIDevices(d,g,video_format,num_streams);
    setupGL();
}
svlVidCapSrcSDIRenderTarget::~svlVidCapSrcSDIRenderTarget()
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcMILRenderTarget::destructor()" << std::endl;
#endif

    KillThread = true;
    if (ThreadKilled == false) Thread->Wait();
    delete Thread;

    cleanupGL();
    cleanupSDIDevices();
}



bool svlVidCapSrcSDIRenderTarget::setupSDIDevices(Display *d,HGPUNV *g, unsigned int video_format, GLsizei num_streams)
{
    gpu = g;
    dpy = d;
    m_num_streams = num_streams;
    if(setupSDIoutDevice(dpy,gpu,video_format) != TRUE)
    {
        printf("Error setting up video output.\n");
        m_SDIoutEnabled = FALSE;
    }
    else
        m_SDIoutEnabled = TRUE;

    return m_SDIoutEnabled;

}

//This routine should be called after the capture has already been configured
//since it relies on the capture signal format configuration
bool svlVidCapSrcSDIRenderTarget::SetImage(unsigned char* buffer, int offsetx, int offsety, bool vflip)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDIRenderTarget::SetImage()" << std::endl;
#endif

    if (SystemID < 0) return false;

    // Wait for thread to finish previous transfer
    if (ThreadReadySignal.Wait(2.0) == false || TransferSuccessful == false || KillThread || ThreadKilled) {
        // Something went terribly wrong on the thread
        return false;
    }

    // Copy image to the Matrox buffer with translation and flip...
    //TODO::CAPTURE?
    //svlVidCapSrcSDI *device = svlVidCapSrcSDI::GetInstance();
    //    TranslateImage(buffer,
    //                   device->MilOverlayBuffer[SystemID][DigitizerID],
    //                   device->MilWidth[SystemID][DigitizerID] * 3,
    //                   device->MilHeight[SystemID][DigitizerID],
    //                   offsetx * 3,
    //                   offsety,
    //                   vflip);

    // Signal Thread that there is a new frame to transfer
    NewFrameSignal.Raise();

    // Frame successfully filed for transfer
    return true;
}

unsigned int svlVidCapSrcSDIRenderTarget::GetWidth()
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcMILRenderTarget::GetWidth()" << std::endl;
#endif

    if (SystemID < 0) return 0;
    return m_SDIout.getWidth();
}

unsigned int svlVidCapSrcSDIRenderTarget::GetHeight()
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDIRenderTarget::GetHeight()" << std::endl;
#endif

    if (SystemID < 0) return 0;
    return m_SDIout.getHeight();
}

void* svlVidCapSrcSDIRenderTarget::ThreadProc(void* CMN_UNUSED(param))
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDIRenderTarget::ThreadProc()" << std::endl;
#endif

    ThreadKilled = false;
    ThreadReadySignal.Raise();

    while (!KillThread) {
        if (NewFrameSignal.Wait(0.5)) {
            //TODO::TRANSFER
            //TransferSuccessful = svlVidCapSrcMIL::GetInstance()->MILUploadOverlay(SystemID, DigitizerID);
            ThreadReadySignal.Raise();
        }
    }

    // Release waiting threads (if any)
    ThreadReadySignal.Raise();

    ThreadKilled = true;
    return this;
}

bool svlVidCapSrcSDIRenderTarget::setupSDIoutDevice(Display *d,HGPUNV *g, unsigned int video_format)
{
    m_captureOptions.xscreen = g->deviceXScreen;
    //m_captureOptions.numCaptureSurfaces = 2;

    //Set the output to be the same frame rate as the input here
    outputOptions.video_format = video_format;//m_SDIin.getVideoFormat();
    outputOptions.xscreen = g->deviceXScreen;
    outputOptions.data_format = NV_CTRL_GVO_DATA_FORMAT_R8G8B8_TO_YCRCB422;
    if(m_num_streams == 2)
        outputOptions.data_format = NV_CTRL_GVO_DATA_FORMAT_DUAL_R8G8B8_TO_DUAL_YCRCB422;
    outputOptions.fql = 5;
    outputOptions.sync_source = NV_CTRL_GVO_SYNC_SOURCE_SDI;
    outputOptions.sync_mode = NV_CTRL_GVO_SYNC_MODE_FREE_RUNNING;

    m_SDIout.setOutputOptions(d,outputOptions);
    bool ret = m_SDIout.initOutputDeviceNVCtrl();
    return ret;
}

GLboolean svlVidCapSrcSDIRenderTarget::setupGL()
{
    if(m_SDIoutEnabled)
        setupSDIoutGL();
    return GL_TRUE;
}

bool svlVidCapSrcSDIRenderTarget::setupSDIoutGL()
{
    //Setup the output after the capture is configured.
    glGenTextures(m_num_streams, m_OutTexture);
    for(int i=0;i<m_num_streams;i++)
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
    return 1;
}


////////////////////////
// SDI-OUT Cleanup
////////////////////////


bool svlVidCapSrcSDIRenderTarget::cleanupSDIout()
{
    bool ret = m_SDIout.destroyOutputDeviceNVCtrl();
    return ret;
}

/////////////////////////////////////
// Draw
/////////////////////////////////////
void svlVidCapSrcSDIRenderTarget::drawVTKPixels(GLuint gWidth, GLuint gHeight, unsigned char* vtkPixelData)
{
    glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGBA8, m_SDIout.getWidth(), m_SDIout.getHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, vtkPixelData );

    // Draw textured quad in lower left quadrant of graphics window.
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex2f(-1, -1);
    glTexCoord2f(0.0, (GLfloat)gHeight); glVertex2f(-1, 1);
    glTexCoord2f((GLfloat)gWidth, (GLfloat)gHeight); glVertex2f(1, 1);
    glTexCoord2f((GLfloat)gWidth, 0.0); glVertex2f(1, -1);
    glEnd();
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


/////////////////////////////////////
// Draw Scene
/////////////////////////////////////

GLboolean svlVidCapSrcSDIRenderTarget::DrawOutputScene(GLuint cudaOutTexture1, GLuint cudaOutTexture2, unsigned char* vtkPixelData)
{
    GLuint width;
    GLuint height;

    for(int i=0;i<m_num_streams;i++)
    {
        if(m_SDIoutEnabled)
        {

            m_FBO[i].bind(m_SDIout.getWidth(), m_SDIout.getHeight());

            width = m_SDIout.getWidth();
            height = m_SDIout.getHeight();
        }
        //else
        //{
        //    width = m_SDIin.getWidth();
        //    height = m_SDIin.getHeight();
        //}
        glEnable(GL_TEXTURE_RECTANGLE_NV);
        glColor3f(1.0f, 1.0f, 1.0f);
        glClearColor( 1.0, 1.0, 1.0, 0.0);
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        glViewport( 0, 0, width, height );
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        //gluOrtho2D( 0, width, 0, height );
        gluOrtho2D(-1.0, 1.0, -1.0, 1.0);
        glMatrixMode( GL_MODELVIEW );
        glLoadIdentity();
        //draw the scene here with the m_CudaOutTexture
        if(i==0)
            glBindTexture(GL_TEXTURE_RECTANGLE_NV, cudaOutTexture1);
        else
            glBindTexture(GL_TEXTURE_RECTANGLE_NV, cudaOutTexture2);
        // Draw textured quad in lower left quadrant of graphics window.
        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0); glVertex2f(-1, -1);
        glTexCoord2f(0.0, (GLfloat)height); glVertex2f(-1, 1);
        glTexCoord2f((GLfloat)width, (GLfloat)height); glVertex2f(1, 1);
        glTexCoord2f((GLfloat)width, 0.0); glVertex2f(1, -1);
        glEnd();

        if(1)
        {
            glEnable (GL_BLEND);
            glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            drawCircle(m_SDIout.getWidth(),m_SDIout.getHeight());
            //drawSpinningSphere(m_SDIout.getWidth(), m_SDIout.getHeight());
            //usleep(1000*1000);
        }else
        {
            glEnable (GL_BLEND);
            glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            drawVTKPixels(m_SDIout.getWidth(), m_SDIout.getHeight(),vtkPixelData);
        }

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
    if(!m_SDIoutEnabled)
        return GL_FALSE;
    //    for(int i=0;i<m_num_streams;i++)
    //    {
    //        //send the texture to SDI out.
    //        glBindTexture(GL_TEXTURE_RECTANGLE_NV, m_OutTexture[i]);
    //        glEnable(GL_TEXTURE_RECTANGLE_NV);
    //    }

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

/////////////////////////////////////
// Cleanup
/////////////////////////////////////

GLboolean svlVidCapSrcSDIRenderTarget::cleanupGL()
{
    // Destroy objects
    if(m_SDIoutEnabled)
    {
        for(int i=0;i<m_num_streams;i++)
        {
            m_FBO[i].destroy();
        }
        glDeleteTextures(m_num_streams, m_OutTexture);
    }

    return GL_TRUE;
}

bool svlVidCapSrcSDIRenderTarget::cleanupSDIDevices()
{
    bool ret = TRUE;
    if(m_SDIoutEnabled && (cleanupSDIout() != TRUE))
        ret = FALSE;
    return ret;
}

/*************************************/
/*        svlVidCapSrcSDI class      */
/*************************************/
CMN_IMPLEMENT_SERVICES_DERIVED(svlVidCapSrcSDI, svlVidCapSrcBase)

//
// vid2tex.cpp - SDI video capture to texture example application
//
// Alina Alt (aalt@nvidia.com)
// August 2010
//

//-----------------------------------------------------------------------------
// Name: drawOGLString
// Desc: Draw string using OpenGL
//-----------------------------------------------------------------------------
static void
drawOGLString(const std::string &str, float xpos=50, float ypos=50)
{
    // Enable blend
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glRasterPos2f(xpos, ypos);
    for (unsigned int i=0; i < str.length(); i++) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, str[i]);
    }

    // Disable blend.
    glDisable(GL_BLEND);
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


////////////////////////////////////
// svlVidCapSrcSDI
////////////////////////////////////

svlVidCapSrcSDI::svlVidCapSrcSDI():
    svlVidCapSrcBase(),
    NumOfStreams(0),
    Initialized(false),
    Running(false)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::constructor()" << std::endl;
#endif

    //TODO:??
//    HGPUNV gpuList[MAX_GPUS];
//    // Open X display
//    dpy = XOpenDisplay(NULL);
//    //scan the systems for GPUs
//    int	num_gpus = ScanHW(dpy,gpuList);
//    if(num_gpus < 1)
//        exit(1);
//    //grab the first GPU for now for DVP
//    gpu = &gpuList[0];

//    SetupSDIDevices(dpy,gpu);
//    CreateWindow();
//    SetupGL();

}


svlVidCapSrcSDI::~svlVidCapSrcSDI()
{
    Shutdown();
}

//TODO:GetInstance
//svlVidCapSrcSDI* svlVidCapSrcSDI::GetInstance()
//{
//#if __VERBOSE__ == 1
//    std::cout << "svlVidCapSrcSDI::GetInstance()" << std::endl;
//#endif

//    static svlVidCapSrcSDI instance;
//    return &instance;
//}

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

    SystemID.SetSize(NumOfStreams);
    DigitizerID.SetSize(NumOfStreams);
    ImageBuffer.SetSize(NumOfStreams);

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

    if (deviceinfo == 0 || Initialized) return SVL_FAIL;

    unsigned int sys, dig, digitizers;
    bool cap, ovrl;

    //MILNumberOfSystems = 0;
    for (sys = 0; sys < MAX_VIDEO_STREAMS; sys ++) {
        //TODO
//        if (!MILInitializeSystem(sys)) break;
//        MILNumberOfSystems ++;

//        for (dig = 0; dig < MilNumberOfDigitizers[sys]; dig ++) {
//            if (MILInitializeDigitizer(sys, dig)) {
//                if (MILInitializeCapture(sys, dig)) MILReleaseCapture(sys, dig);
//                if (MILInitializeOverlay(sys, dig)) MILReleaseOverlay(sys, dig);
//                MILReleaseDigitizer(sys, dig);
//            }
//        }
//        MILReleaseSystem(sys);
    }

    // Allocate memory for device info array
    // CALLER HAS TO FREE UP THIS ARRAY!!!
//    if (MILNumberOfSystems > 0) {

//        deviceinfo[0] = new svlFilterSourceVideoCapture::DeviceInfo[MILNumberOfSystems];

//        for (sys = 0; sys < MILNumberOfSystems; sys ++) {
//            // platform
//            deviceinfo[0][sys].platform = svlFilterSourceVideoCapture::MatroxImaging;

//            // id
//            deviceinfo[0][sys].ID = sys;

//            // name
//            std::stringstream dev_name;
//            dev_name << "Matrox Imaging Device (" << "M_DEV" << sys << ")";

//            memset(deviceinfo[0][sys].name, 0, SVL_VCS_STRING_LENGTH);
//            memcpy(deviceinfo[0][sys].name,
//                   dev_name.str().c_str(),
//                   std::min(SVL_VCS_STRING_LENGTH - 1, static_cast<int>(dev_name.str().length())));

//            // digitizers
//            digitizers = MilNumberOfDigitizers[sys];
//            if (digitizers > SVL_VCS_ARRAY_LENGTH) digitizers = SVL_VCS_ARRAY_LENGTH;
//            deviceinfo[0][sys].inputcount = digitizers;
//            deviceinfo[0][sys].activeinput = 0;

//            for (dig = 0; dig < digitizers; dig ++) {
//                std::stringstream in_name;
//                in_name << "Digitizer #" << dig << " (M_DEV" << dig << ": ";

//                cap = MilCaptureSupported[sys][dig];
//                ovrl = MilOverlaySupported[sys][dig];
//                if (cap && ovrl) in_name << "Capture+Overlay)";
//                else if (cap) in_name << "Capture only)";
//                else if (ovrl) in_name << "Overlay only)";
//                else in_name << "Not supported)";

//                memset(deviceinfo[0][sys].inputnames[dig], 0, SVL_VCS_STRING_LENGTH);
//                memcpy(deviceinfo[0][sys].inputnames[dig],
//                       in_name.str().c_str(),
//                       std::min(SVL_VCS_STRING_LENGTH - 1, static_cast<int>(in_name.str().length())));
//            }

//            // test
//            deviceinfo[0][sys].testok = true;
//        }
//    }
//    else {
//        deviceinfo[0] = 0;
//    }

    return 1;//m_SDIin.getNumStreams();//MILNumberOfSystems;
}

int svlVidCapSrcSDI::Open()
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::Open() - Number of video channels = " << NumOfStreams << std::endl;
#endif

    if (NumOfStreams <= 0) return SVL_FAIL;
    if (Initialized) return SVL_OK;

    Close();

    for (unsigned int i = 0; i < NumOfStreams; i ++) {
    //TODO:
//        if (MilSystem[SystemID[i]] == M_NULL) {
//            if (!MILInitializeSystem(SystemID[i])) {
//#if __VERBOSE__ == 1
//                std::cout << "svlVidCapSrcMIL::Open() - Failed to initialize system M_DEV" << SystemID[i] << std::endl;
//#endif
//                goto labError;
//            }
//        }

//        if (MilDigitizer[SystemID[i]][DigitizerID[i]] == M_NULL) {
//            if (!MILInitializeDigitizer(SystemID[i], DigitizerID[i])) {
//#if __VERBOSE__ == 1
//                std::cout << "svlVidCapSrcMIL::Open() - Failed to initialize digitizer M_DEV"
//                          << DigitizerID[i] << " on system M_DEV" << SystemID[i] << std::endl;
//#endif
//                goto labError;
//            }
//        }

//        if (!MILInitializeCapture(SystemID[i], DigitizerID[i])) {
//#if __VERBOSE__ == 1
//            std::cout << "svlVidCapSrcMIL::Open() - Failed to initialize capture on system M_DEV"
//                      << SystemID[i] << ", digitizer M_DEV" << DigitizerID[i] << std::endl;
//#endif
//            goto labError;
//        }

//        if (MilBands[SystemID[i]][DigitizerID[i]] != 1 &&
//            MilBands[SystemID[i]][DigitizerID[i]] != 3) goto labError;

//        // Allocate capture buffers
        const unsigned int width  = m_SDIin.getWidth();//MilWidth[SystemID[i]][DigitizerID[i]];
        const unsigned int height =  m_SDIin.getWidth();//MilHeight[SystemID[i]][DigitizerID[i]];
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcMIL::Open - Allocate image buffer (" << width << ", " << height << ")" << std::endl;
#endif
        ImageBuffer[i] = new svlBufferImage(width, height);
        // Set the pointer in the capture structure that will be accessed in the callback
//        MilCaptureParams[SystemID[i]][DigitizerID[i]].ImageBuffer = ImageBuffer[i];
//        MilCaptureParams[SystemID[i]][DigitizerID[i]].SystemID    = SystemID[i];
//        MilCaptureParams[SystemID[i]][DigitizerID[i]].DigitizerID = DigitizerID[i];
    }

    Initialized = true;
    return SVL_OK;

labError:
    Close();
    return SVL_FAIL;
}

void svlVidCapSrcSDI::Close()
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcMIL::Close()" << std::endl;
#endif

    if (NumOfStreams == 0) return;

    Stop();

    Initialized = false;

    Shutdown();

//    for (unsigned int sys = 0; sys < MIL_MAX_SYS; sys ++) {
//        if (MilSystem[sys] == M_NULL) continue;
//        bool no_release = false;

//        for (unsigned int dig = 0; dig < MIL_MAX_DIG; dig ++) {
//            if (MilDigitizer[dig] == M_NULL) continue;

//            // Release capture
//            MILReleaseCapture(sys, dig);

//            // Do not release system and digitizer if overlay is used
//            if (MilDisplay[sys][dig] == M_NULL) {
//                MILReleaseDigitizer(sys, dig);
//            }
//            else {
//                no_release = true;
//            }
//        }

//        if (!no_release) MILReleaseSystem(sys);
//    }

    for (unsigned int i = 0; i < NumOfStreams; i ++) {
        delete ImageBuffer[i];
        ImageBuffer[i] = 0;
    }
}

int svlVidCapSrcSDI::Start()
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcMIL::Start()" << std::endl;
#endif

    if (!Initialized) return SVL_FAIL;
    Running = true;
    return SVL_OK;
}

svlImageRGB* svlVidCapSrcSDI::GetLatestFrame(bool waitfornew, unsigned int videoch)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcMIL::GetLatestFrame(" << waitfornew << ", " << videoch << ")" << std::endl;
#endif

    if (videoch >= NumOfStreams || !Initialized) return 0;
    return ImageBuffer[videoch]->Pull(waitfornew);
}

int svlVidCapSrcSDI::Stop()
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::Stop()" << std::endl;
#endif

    if (!Running) return SVL_FAIL;
    Running = false;
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
    formatlist[0][0].width = m_SDIin.getWidth();
    formatlist[0][0].height = m_SDIin.getHeight();
    formatlist[0][0].colorspace = svlFilterSourceVideoCapture::PixelRGB8;
    formatlist[0][0].rgb_order = true;
    formatlist[0][0].yuyv_order = false;
    formatlist[0][0].framerate = -1.0;
    formatlist[0][0].custom_mode = -1;

    return 1;
}

int svlVidCapSrcSDI::GetFormat(svlFilterSourceVideoCapture::ImageFormat& format, unsigned int videoch)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcSDI::GetFormat(..., " << videoch << ")" << std::endl;
#endif

    if (SystemID[videoch] > 0 && SystemID[videoch] >= static_cast<int>(MAX_VIDEO_STREAMS)) return SVL_FAIL;

    format.width = m_SDIin.getWidth();
    format.height = m_SDIin.getHeight();
    format.colorspace = svlFilterSourceVideoCapture::PixelRGB8;
    format.rgb_order = true;
    format.yuyv_order = false;
    format.framerate = -1.0;
    format.custom_mode = -1;

    return SVL_OK;
}

bool svlVidCapSrcSDI::IsCaptureSupported(unsigned int sysid, unsigned int digid)
{
#if __VERBOSE__ == 1
    std::cout << "svlVidCapSrcMIL::IsCaptureSupported(" << sysid << ", " << digid << ")" << std::endl;
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

/////////////////////////////////////
// Windows routines
/////////////////////////////////////


//-----------------------------------------------------------------------------
// Name: calcWindowSize
// Desc: Calculate the graphics window size
//-----------------------------------------------------------------------------
void
svlVidCapSrcSDI::calcWindowSize()
{
    int numStreams = m_SDIin.getNumStreams();
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
svlVidCapSrcSDI::Shutdown()
{
    StopSDIPipeline();
    cleanupSDIDevices();

    cleanupGL();
}


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
// Name: CreateWindow
// Desc: Create window
//-----------------------------------------------------------------------------
Window
svlVidCapSrcSDI::CreateWindow()
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

void svlVidCapSrcSDI::MakeCurrentGLCtx()
{
    glXMakeCurrent(dpy, win, ctx);
}


//-----------------------------------------------------------------------------
// Name: DestroyWindow
// Desc: Destroy window
//-----------------------------------------------------------------------------
bool
svlVidCapSrcSDI::destroyWindow()
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
svlVidCapSrcSDI::setupSDIinDevice(Display *d, HGPUNV *g)
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
bool
svlVidCapSrcSDI::setupSDIinGL()
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
svlVidCapSrcSDI::cleanupSDIin()
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
svlVidCapSrcSDI::SetupGL()
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
svlVidCapSrcSDI::SetupSDIDevices(Display *d,HGPUNV *g)
{
    gpu = g;
    dpy = d;
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
svlVidCapSrcSDI::StartSDIPipeline()
{
    // Start video capture
    if(m_SDIin.startCapture()!= TRUE) {
        printf("Error starting video capture.\n");
        return FALSE;
    }

    return TRUE;
}


//-----------------------------------------------------------------------------
// Name: StopSDIPipeline
// Desc: Stop SDI video capture.
//-----------------------------------------------------------------------------
bool
svlVidCapSrcSDI::StopSDIPipeline()
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
svlVidCapSrcSDI::cleanupSDIDevices()
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
GLboolean svlVidCapSrcSDI::cleanupGL()
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
svlVidCapSrcSDI::CaptureVideo(GLuint *captureLatency,float runTime)
{
    static GLuint64EXT captureTime;
    GLuint sequenceNum;
    static GLuint prevSequenceNum = 0;
    GLenum ret;
    static int numFails = 0;
    static int numTries = 0;
    *captureLatency = 0;
    unsigned int droppedFrames;
    float longGVITime=1;

    if(numFails < 100) {

        // Capture the video to a buffer object
        ret = m_SDIin.capture(&sequenceNum, &captureTime);
        if(sequenceNum - prevSequenceNum > 1)
        {
            droppedFrames = sequenceNum - prevSequenceNum;
            printf("glVideoCaptureNV: Dropped %d frames\n",sequenceNum - prevSequenceNum);
            printf("Frame:%d gpuTime:%f gviTime:%f\n", sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime);
            *captureLatency = 1;
        }
        //    if(m_SDIin.m_gviTime > 1.0/30)
        //    {
        //      printf("Frame:%d gpuTime:%f gviTime:%f\n", sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime);
        //      *captureLatency = 1;
        //    }

        prevSequenceNum = sequenceNum;
        switch(ret) {
        case GL_SUCCESS_NV:
            //printf("Frame:%d gpuTime:%f gviTime:%f\n", sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime);
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

    if(*captureLatency)
    {
        for(unsigned int i=0;i< droppedFrames+1;i++)
        {
            printf("Call: %d of %d Frame:%d gpuTime:%f gviTime:%f goal:%f\n", i, droppedFrames+1, sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime,1.0/30);
            CaptureVideo(captureLatency);
        }
    }
    if(m_SDIin.m_gviTime + runTime > 1.0/30)
    {
        //      longGVITime = m_SDIin.m_gviTime;
        //      for(unsigned int i=0;i< longGVITime*30+2;i++)
        //      {
        printf("Call: %f decrease to %f Frame:%d gpuTime:%f gviTime:%f goal:%f\n", runTime,runTime-1.0/30,sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime,1.0/30);
        CaptureVideo(captureLatency,runTime-1.0/30);
        *captureLatency = 1;
    }else if(m_SDIin.m_gviTime > 1.0/30)
    {
        //      longGVITime = m_SDIin.m_gviTime;
        //      for(unsigned int i=0;i< longGVITime*30+2;i++)
        //      {
        //printf("Call: %f decrease to %f Frame:%d gpuTime:%f gviTime:%f goal:%f\n", runTime,runTime-1.0/30,sequenceNum, m_SDIin.m_gpuTime,m_SDIin.m_gviTime,1.0/30);
        CaptureVideo(captureLatency);
    }

    return ret;
}

//-----------------------------------------------------------------------------
// Name: drawOne
// Desc: Draw single SDI video stream in graphics window.
//-----------------------------------------------------------------------------
GLvoid
svlVidCapSrcSDI::drawOne()
{
    // Calculate scaled video dimensions.
    GLfloat scaledVideoWidth;
    GLfloat scaledVideoHeight;
    int videoWidth = m_SDIin.getWidth();
    int videoHeight = m_SDIin.getHeight();
    calcScaledVideoDimensions(m_windowWidth, m_windowHeight,
                              videoWidth, videoHeight,
                              &scaledVideoWidth,
                              &scaledVideoHeight);

    // Set draw color.
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

    // Enable texture mapping
    glEnable(GL_TEXTURE_RECTANGLE_NV);

    // Bind texture object for first video stream
    GLuint tex0 = m_SDIin.getTextureObjectHandle(0);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, tex0);

    // Set viewport to whole window area.
    glViewport(0, 0, m_windowWidth, m_windowHeight);

    // Draw textured quad in graphics window.
    glBegin(GL_QUADS);
    glTexCoord2i(0, 0);
    glVertex2f(-scaledVideoWidth, -scaledVideoHeight);
    glTexCoord2i(0, videoHeight);
    glVertex2f(-scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, videoHeight);
    glVertex2f(scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, 0);
    glVertex2f(scaledVideoWidth, -scaledVideoHeight);
    glEnd();
}

//-----------------------------------------------------------------------------
// Name: drawTwo
// Desc: Draw two SDI video stream in graphics window.
//       Video streams are stacked on atop the other.
//-----------------------------------------------------------------------------
GLvoid
svlVidCapSrcSDI::drawTwo()
{
    // Calculate scaled video dimensions.
    GLfloat scaledVideoWidth;
    GLfloat scaledVideoHeight;
    int videoWidth = m_SDIin.getWidth();
    int videoHeight = m_SDIin.getHeight();
    calcScaledVideoDimensions(m_windowWidth, m_windowHeight / 2,
                              videoWidth, videoHeight,
                              &scaledVideoWidth,
                              &scaledVideoHeight);

    // Set draw color.
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

    // Enable texture mapping
    glEnable(GL_TEXTURE_RECTANGLE_NV);

    // Bind texture object for first video stream
    GLuint tex0 = m_SDIin.getTextureObjectHandle(0);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, tex0);

    // Set viewport to lower half of window.
    glViewport(0, m_windowHeight / 2, m_windowWidth, m_windowHeight / 2);

    // Draw textured quad in lower half of graphics window.
    glBegin(GL_QUADS);
    glTexCoord2i(0, 0);
    glVertex2f(-scaledVideoWidth, -scaledVideoHeight);
    glTexCoord2i(0, videoHeight);
    glVertex2f(-scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, videoHeight);
    glVertex2f(scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, 0);
    glVertex2f(scaledVideoWidth, -scaledVideoHeight);
    glEnd();

    // Bind texture object for second video stream
    GLuint tex1 = m_SDIin.getTextureObjectHandle(1);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, tex1);

    // Set viewport to upper half of window.
    glViewport(0, 0, m_windowWidth, m_windowHeight / 2);

    // Draw textured quad in upper half of graphics window.
    glBegin(GL_QUADS);
    glTexCoord2i(0, 0);
    glVertex2f(-scaledVideoWidth, -scaledVideoHeight);
    glTexCoord2i(0, videoHeight);
    glVertex2f(-scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, videoHeight);
    glVertex2f(scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, 0);
    glVertex2f(scaledVideoWidth, -scaledVideoHeight);
    glEnd();
}

void svlVidCapSrcSDI::drawCircle(GLuint gWidth, GLuint gHeight)
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
    glColor4f(1.0, 0.0, 0.0, 0.1);
    static GLUquadric * qobj = NULL;
    if (! qobj){
        qobj = gluNewQuadric();
        gluQuadricDrawStyle(qobj, GLU_FILL);
        gluQuadricNormals(qobj, GLU_SMOOTH);
        gluQuadricTexture(qobj, GL_TRUE);
    }
    gluSphere (qobj, 3, 50, 100);
}

//-----------------------------------------------------------------------------
// Name: drawThree
// Desc: Draw three SDI video stream in graphics window.
//       Use 3 quadrants with the remaining quadrant black.
//-----------------------------------------------------------------------------
GLvoid
svlVidCapSrcSDI::drawThree()
{
    // Calculate scaled video dimensions.
    GLfloat scaledVideoWidth;
    GLfloat scaledVideoHeight;
    int videoWidth = m_SDIin.getWidth();
    int videoHeight = m_SDIin.getHeight();
    calcScaledVideoDimensions(m_windowWidth / 2.0f, m_windowHeight / 2.0f,
                              videoWidth, videoHeight,
                              &scaledVideoWidth,
                              &scaledVideoHeight);

    // Set draw color.
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

    // Enable texture mapping
    glEnable(GL_TEXTURE_RECTANGLE_NV);

    // Bind texture object for first video stream
    GLuint tex0 = m_SDIin.getTextureObjectHandle(0);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, tex0);

    // Set viewport to lower left quadrant
    glViewport(0, 0, m_windowWidth / 2, m_windowHeight / 2);

    // Draw textured quad in lower left quadrant of graphics window.
    glBegin(GL_QUADS);
    glTexCoord2i(0, 0);
    glVertex2f(-scaledVideoWidth, -scaledVideoHeight);
    glTexCoord2i(0, videoHeight);
    glVertex2f(-scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, videoHeight);
    glVertex2f(scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, 0);
    glVertex2f(scaledVideoWidth, -scaledVideoHeight);
    glEnd();

    // Bind texture object for second video stream
    GLuint tex1 = m_SDIin.getTextureObjectHandle(1);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, tex1);

    // Set viewport to upper left quadrant
    glViewport(0, m_windowHeight / 2, m_windowWidth / 2, m_windowHeight / 2);

    // Draw textured quad in upper left quadrant of graphics window.
    glBegin(GL_QUADS);
    glTexCoord2i(0, 0);
    glVertex2f(-scaledVideoWidth, -scaledVideoHeight);
    glTexCoord2i(0, videoHeight);
    glVertex2f(-scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, videoHeight);
    glVertex2f(scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, 0);
    glVertex2f(scaledVideoWidth, -scaledVideoHeight);
    glEnd();

    // Bind texture object for third video stream
    GLuint tex2 = m_SDIin.getTextureObjectHandle(2);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, tex2);

    // Set viewport to upper right quadrant.
    glViewport(m_windowWidth / 2, m_windowHeight / 2, m_windowWidth / 2, m_windowHeight / 2);

    // Draw textured quad in upper right quadrant of graphics window.
    glBegin(GL_QUADS);
    glTexCoord2i(0, 0);
    glVertex2f(-scaledVideoWidth, -scaledVideoHeight);
    glTexCoord2i(0, videoHeight);
    glVertex2f(-scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, videoHeight);
    glVertex2f(scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, 0);
    glVertex2f(scaledVideoWidth, -scaledVideoHeight);
    glEnd();

    glBindTexture(GL_TEXTURE_RECTANGLE_NV, NULL);
}


//-----------------------------------------------------------------------------
// Name: drawFour
// Desc: Draw four SDI video stream tiled in graphics window.
//       One stream is drawn in each quadrant.
//-----------------------------------------------------------------------------
GLvoid
svlVidCapSrcSDI::drawFour()
{
    // Calculate scaled video dimensions.
    GLfloat scaledVideoWidth;
    GLfloat scaledVideoHeight;
    int videoWidth = m_SDIin.getWidth();
    int videoHeight = m_SDIin.getHeight();
    calcScaledVideoDimensions(m_windowWidth / 2.0f, m_windowHeight / 2.0f,
                              videoWidth, videoHeight,
                              &scaledVideoWidth,
                              &scaledVideoHeight);

    // Set draw color.
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

    // Enable texture mapping
    glEnable(GL_TEXTURE_RECTANGLE_NV);

    // Bind texture object for first video stream
    GLuint tex0 = m_SDIin.getTextureObjectHandle(0);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, tex0);

    // Set viewport to lower left corner quadrant
    glViewport(0, 0, m_windowWidth / 2, m_windowHeight / 2);

    // Draw textured quad in lower left quadrant of graphics window.
    glBegin(GL_QUADS);
    glTexCoord2i(0, 0);
    glVertex2f(-scaledVideoWidth, -scaledVideoHeight);
    glTexCoord2i(0, videoHeight);
    glVertex2f(-scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, videoHeight);
    glVertex2f(scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, 0);
    glVertex2f(scaledVideoWidth, -scaledVideoHeight);
    glEnd();

    // Bind texture object for second video stream
    GLuint tex1 = m_SDIin.getTextureObjectHandle(1);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, tex1);

    // Set viewport to upper left quadrant.
    glViewport(0, m_windowHeight / 2, m_windowWidth / 2, m_windowHeight / 2);

    // Draw textured quad in lower right quadrant of graphics window.
    glBegin(GL_QUADS);
    glTexCoord2i(0, 0);
    glVertex2f(-scaledVideoWidth, -scaledVideoHeight);
    glTexCoord2i(0, videoHeight);
    glVertex2f(-scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, videoHeight);
    glVertex2f(scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, 0);
    glVertex2f(scaledVideoWidth, -scaledVideoHeight);
    glEnd();

    // Bind texture object for third video stream
    GLuint tex2 = m_SDIin.getTextureObjectHandle(2);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, tex2);

    // Set viewport to upper right quadrant.
    glViewport(m_windowWidth / 2, m_windowHeight / 2, m_windowWidth / 2, m_windowHeight / 2);

    // Draw textured quad in upper right quadrant of graphics window.
    glBegin(GL_QUADS);
    glTexCoord2i(0, 0);
    glVertex2f(-scaledVideoWidth, -scaledVideoHeight);
    glTexCoord2i(0, videoHeight);
    glVertex2f(-scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, videoHeight);
    glVertex2f(scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, 0);
    glVertex2f(scaledVideoWidth, -scaledVideoHeight);
    glEnd();

    // Bind texture object for fourth video stream
    GLuint tex3 = m_SDIin.getTextureObjectHandle(3);
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, tex3);

    // Set viewport to lower right quadrant.
    glViewport(m_windowWidth / 2, 0, m_windowWidth / 2, m_windowHeight / 2);

    // Draw textured quad in upper right quadrant of graphics window.
    glBegin(GL_QUADS);
    glTexCoord2i(0, 0);
    glVertex2f(-scaledVideoWidth, -scaledVideoHeight);
    glTexCoord2i(0, videoHeight);
    glVertex2f(-scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, videoHeight);
    glVertex2f(scaledVideoWidth, scaledVideoHeight);
    glTexCoord2i(videoWidth, 0);
    glVertex2f(scaledVideoWidth, -scaledVideoHeight);
    glEnd();

    glBindTexture(GL_TEXTURE_RECTANGLE_NV, NULL);
}


//-----------------------------------------------------------------------------
// Name: DisplayVideo
// Desc: Main drawing routine.
//-----------------------------------------------------------------------------
GLenum
svlVidCapSrcSDI::DisplayVideo(bool drawFrameRate)
{
    glClearColor(0.3f, 0.3f, 0.3f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Set view parameters.
    glViewport(0, 0, m_windowWidth, m_windowHeight);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    int numStreams =  m_SDIin.getNumStreams();
    switch(numStreams) {
    case 1:
        drawOne();
        break;
    case 2:
        drawTwo();
        break;
    case 3:
        drawThree();
        break;
    case 4:
        drawFour();
        break;
    default:
        drawOne();
    };

    // Disable texture mapping
    glDisable(GL_TEXTURE_RECTANGLE_NV);

    // Reset view parameters
    glViewport(0, 0, m_windowWidth, m_windowHeight);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, m_windowWidth, 0.0, m_windowHeight);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Set draw color
    glColor3f(1.0f, 1.0f, 0.0f);

    // Draw frames per second
    if(drawFrameRate)
    {
        std::stringstream ss;
        ss << CalcFPS() << " fps";
        drawOGLString(ss.str(), m_windowWidth - 80.0f, 0.0f);
    }
    glXSwapBuffers(dpy, win);
    glDisable(GL_TEXTURE_RECTANGLE_NV);
    return GL_TRUE;
}

