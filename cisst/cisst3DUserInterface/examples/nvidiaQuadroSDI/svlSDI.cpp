//
// svlExNVIDIAQuadroSDI - demonstrates SDI capture, vtk overlay and SDI output
//
// Wen P. Liu
// June 2012
//

#include <cisstStereoVision.h>
#include <cisstStereoVision/svlRenderTargets.h>
#include <sys/time.h>
#include "vtkSphereSource.h"
#include "vtkPolyDataMapper.h"
#include "vtkProperty.h"
#include "vtkActor.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkRenderWindowInteractor.h"

vtkSphereSource *sphere,*sphere0;
vtkPolyDataMapper *sphereDataMapper,*sphereDataMapper0;
vtkActor *aSphere,*aSphere0;
vtkRenderer *ren,*ren0;
vtkRenderWindow *renWin,*renWin0;
vtkUnsignedCharArray * OffScreenBuffer,*OffScreenBuffer0;
////vtkRenderWindowInteractor *iren;

void setupVTK(int width, int height)
{
    // create sphere geometry
    sphere = vtkSphereSource::New();
    sphere->SetRadius(1.0);
    sphere->SetThetaResolution(18);
    sphere->SetPhiResolution(18);

    // map to graphics library
    sphereDataMapper = vtkPolyDataMapper::New();
    sphereDataMapper->SetInput(sphere->GetOutput());

    // actor coordinates geometry, properties, transformation
    aSphere = vtkActor::New();
    aSphere->SetMapper(sphereDataMapper);
    aSphere->GetProperty()->SetColor(0,0,1); // sphere color blue
    aSphere->GetProperty()->SetOpacity(0.5);

    // a renderer and render window
    ren = vtkRenderer::New();
    ren->SetLayer(0);
    renWin = vtkRenderWindow::New();
    renWin->AddRenderer(ren);
    renWin->SetSize(width,height);
    renWin->SetAlphaBitPlanes(1);

    // an interactor
    //vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::New();
    //iren->SetRenderWindow(renWin);

    // add the actor to the scene
    ren->AddActor(aSphere);

    renWin->DoubleBufferOff();
    OffScreenBuffer = vtkUnsignedCharArray::New();
    OffScreenBuffer->Resize(width * height * 3);
}

void setupVTK0(int width, int height)
{
    // create sphere geometry
    sphere0 = vtkSphereSource::New();
    sphere0->SetRadius(1.0);
    sphere0->SetThetaResolution(18);
    sphere0->SetPhiResolution(18);

    // map to graphics library
    sphereDataMapper0 = vtkPolyDataMapper::New();
    sphereDataMapper0->SetInput(sphere->GetOutput());

    // actor coordinates geometry, properties, transformation
    aSphere0 = vtkActor::New();
    aSphere0->SetMapper(sphereDataMapper0);
    aSphere0->GetProperty()->SetColor(1,0,0); // sphere color red
    aSphere0->GetProperty()->SetOpacity(0.5);

    // a renderer and render window
    ren0 = vtkRenderer::New();
    ren0->SetLayer(0);
    renWin0 = vtkRenderWindow::New();
    renWin0->AddRenderer(ren0);
    renWin0->SetSize(width,height);
    renWin0->SetAlphaBitPlanes(1);

    // an interactor
    //vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::New();
    //iren->SetRenderWindow(renWin);

    // add the actor to the scene
    ren0->AddActor(aSphere0);

    renWin0->DoubleBufferOff();
    OffScreenBuffer0 = vtkUnsignedCharArray::New();
    OffScreenBuffer0->Resize(width * height * 3);
}

void getVTKData(int x2, int y2)
{
    renWin->GetRGBACharPixelData(0, 0, x2 - 1, y2 - 1, 0, OffScreenBuffer);
    renWin0->GetRGBACharPixelData(0, 0, x2 - 1, y2 - 1, 0, OffScreenBuffer0);
}

//-----------------------------------------------------------------------------
// Name: main
// Desc: Main function.
//-----------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    svlVidCapSrcSDIRenderTarget* target = (svlVidCapSrcSDIRenderTarget*)svlRenderTargets::Get(0);
    svlRenderTargets::Get(0);
    setupVTK(400,500);//target->GetWidth(),target->GetHeight()
    setupVTK0(400,500);

    //initialization for svlVidCapSrcSDI
//    svlVidCapSrcSDI* vidCapSrcSDI = svlVidCapSrcSDI::GetInstance();
//    vidCapSrcSDI->SetStreamCount(2);
//    vidCapSrcSDI->SetDevice(0,0,0);

//    if(vidCapSrcSDI->Open() == SVL_OK)
//    {
//        if(vidCapSrcSDI->Open() != SVL_OK)
//            return 0;
//    }
//    if(vidCapSrcSDI->Start() != SVL_OK)
//        return 0;

    int count = 0;
    bool bNotDone = 1;

    while (bNotDone) {
        aSphere->SetPosition(count%2,count%3,0);
        aSphere0->SetPosition(count%3,count%2,0);
        renWin->Render();
        renWin0->Render();
        getVTKData(target->GetWidth(),target->GetHeight());
        target->MakeCurrentGLCtx();
        //if(count%1000==0)
        //{
            target->SetImage(OffScreenBuffer->GetPointer(0),0,0,false);
            target->SetImage(OffScreenBuffer0->GetPointer(0),0,0,false,1);
        //}
        count++;
    }

    if(target)
        target->Shutdown();

}


