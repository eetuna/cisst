//
// svlVidCapSrcSDI - demonstrates SDI capture, interop with CUDA and SDI output
//
// Alina Alt (aalt@nvidia.com)
// August 2010
//

#include <cisstStereoVision.h>
#include <cisstStereoVision/svlRenderTargets.h>
#include <sys/time.h>

#include "error.h"
//vtkSphereSource *sphere;
//vtkPolyDataMapper *sphereDataMapper;
//// actor coordinates geometry, properties, transformation
//vtkActor *aSphere;
//// a renderer and render window
//vtkRenderer *ren1;
//vtkRenderWindow *renWin;
//vtkUnsignedCharArray * OffScreenBuffer;

//// an interactor
////vtkRenderWindowInteractor *iren;

//void setupVTK(int width, int height)
//{
//    // create sphere geometry
//    sphere = vtkSphereSource::New();
//    sphere->SetRadius(1.0);
//    sphere->SetThetaResolution(18);
//    sphere->SetPhiResolution(18);

//    // map to graphics library
//    sphereDataMapper = vtkPolyDataMapper::New();
//    sphereDataMapper->SetInput(sphere->GetOutput());

//    // actor coordinates geometry, properties, transformation
//    aSphere = vtkActor::New();
//    aSphere->SetMapper(sphereDataMapper);
//    aSphere->GetProperty()->SetColor(0,0,1); // sphere color blue
//    aSphere->GetProperty()->SetOpacity(0.5);

//    // a renderer and render window
//    ren1 = vtkRenderer::New();
//    ren1->SetLayer(0);
//    renWin = vtkRenderWindow::New();
//    renWin->AddRenderer(ren1);
//    renWin->SetSize(width,height);
//    renWin->SetAlphaBitPlanes(1);

//    // an interactor
//    //vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::New();
//    //iren->SetRenderWindow(renWin);

//    // add the actor to the scene
//    ren1->AddActor(aSphere);
//    //ren1->SetBackground(1,0,0); // Background color white

//    // render an image (lights and cameras are created automatically)
//    //renWin->Render();

//    //renWin->OffScreenRenderingOn();
//    renWin->DoubleBufferOff();
//    OffScreenBuffer = vtkUnsignedCharArray::New();
//    OffScreenBuffer->Resize(width * height * 3);

//}

//void getVTKData(int x2, int y2)
//{
//    renWin->GetRGBACharPixelData(0, 0, x2 - 1, y2 - 1, 0, OffScreenBuffer);
//}

//-----------------------------------------------------------------------------
// Name: main
// Desc: Main function.
//-----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    timeval setup, start, end;
    svlVidCapSrcSDI vidCapSrcSDI;
    // Required for drawOGLString.
    glutInit(&argc, argv);

    HGPUNV gpuList[MAX_GPUS];
    // Open X display
    Display *dpy = XOpenDisplay(NULL);
    Window win;

    //scan the systems for GPUs
    int	num_gpus = ScanHW(dpy,gpuList);

    if(num_gpus < 1)
        exit(1);

    //grab the first GPU for now for DVP
    HGPUNV *gpu = &gpuList[0];

    vidCapSrcSDI.SetupSDIDevices(dpy,gpu);
    //setupVTK(400,500);//vidCapSrcSDI.GetSDIin().getWidth(),vidCapSrcSDI.GetSDIin().getWidth());//

    // Calculate the window size based on the incoming and outgoing video signals
    win = vidCapSrcSDI.CreateWindow();
    vidCapSrcSDI.SetupGL();
    svlVidCapSrcSDIRenderTarget target(dpy,gpu,vidCapSrcSDI.GetSDIin().getVideoFormat(),vidCapSrcSDI.GetSDIin().getNumStreams());
    gettimeofday(&setup, 0);
    if(vidCapSrcSDI.StartSDIPipeline() != TRUE)
        return FALSE;
    double runtime;
    int count =0;
    //
    // Capture,draw and output
    //

    bool bNotDone = TRUE;
    bool drawCaptureFrameRate = 1;
    GLuint captureLatency = 1;

    while (bNotDone) {
        gettimeofday(&start, 0);
        //aSphere->SetPosition(count%2,count%3,0);
        //renWin->Render();
        //getVTKData(vidCapSrcSDI.GetSDIin().getWidth(),vidCapSrcSDI.GetSDIin().getHeight());
        //vidCapSrcSDI.MakeCurrentGLCtx();

        if(count == 0)
        {
            gettimeofday(&end, 0);
            runtime = end.tv_sec * 1000000 + end.tv_usec;
            runtime -= setup.tv_sec * 1000000 + setup.tv_usec;
            printf("Count:%d Loop Runtime: %f\n",count, runtime/1000000);
            vidCapSrcSDI.CaptureVideo(&captureLatency,runtime/1000000);
        }else
        {
            if(vidCapSrcSDI.CaptureVideo(&captureLatency) != GL_FAILURE_NV)
            {
                vidCapSrcSDI.DisplayVideo(drawCaptureFrameRate);
                target.DrawOutputScene(vidCapSrcSDI.GetSDIin().getTextureObjectHandle(0),vidCapSrcSDI.GetSDIin().getTextureObjectHandle(1));//,OffScreenBuffer->GetPointer(0));
                target.OutputVideo();
            }
            gettimeofday(&end, 0);
            if(runtime/1000000 > 1.0/30)// && !captureLatency)
            {
                printf("Count:%d Loop Runtime: %f\n",count, runtime/1000000);
                //vidCapSrcSDI.CaptureVideo(&captureLatency,runtime/1000000);
                //gettimeofday(&start, 0);
                runtime = end.tv_sec * 1000000 + end.tv_usec;
                runtime -= start.tv_sec * 1000000 + start.tv_usec;
            }
        }
        //bNotDone = false;
        count++;

    }
    vidCapSrcSDI.Shutdown();
    XCloseDisplay(dpy);

}


