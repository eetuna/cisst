//
// svlExNVIDIAQuadroSDI - demonstrates SDI capture, vtk overlay and SDI output
//
// Wen P. Liu
// June 2012
//

#include <cisstStereoVision.h>
#include <cisstStereoVision/svlRenderTargets.h>

#include "vtkSphereSource.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"
#include "vtkBoxWidget.h"
#include "vtkCamera.h"
#include "vtkCommand.h"
#include "vtkColorTransferFunction.h"
#include "vtkDICOMImageReader.h"
#include "vtkImageData.h"
#include "vtkImageResample.h"
#include "vtkMetaImageReader.h"
#include "vtkPiecewiseFunction.h"
#include "vtkPlanes.h"
#include "vtkProperty.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
//#include "vtkVolume.h"
//#include "vtkVolumeProperty.h"
//#include "vtkXMLImageDataReader.h"
//#include "vtkSmartVolumeMapper.h"

vtkSphereSource *sphere,*sphere0;
vtkPolyDataMapper *sphereDataMapper,*sphereDataMapper0;
vtkActor *aSphere,*aSphere0;
vtkRenderer *ren,*ren0;
vtkRenderWindow *renWin,*renWin0;
vtkUnsignedCharArray * OffScreenBuffer,*OffScreenBuffer0;
////vtkRenderWindowInteractor *iren;
//vtkVolume *volume;
//vtkSmartVolumeMapper *mapper;

void setupObjects()
{
    //DICOM
    // Read the data
    //vtkAlgorithm *reader=0;
    //vtkImageData *input=0;
    //double opacityWindow = 4096;
    //double opacityLevel = 1024;
    //vtkMetaImageReader *metaReader = vtkMetaImageReader::New();
    //metaReader->SetFileName("/home/wen/Images/20130201_Sphere/SiemensReconstruction/SPHERIC/20130201_Sphere_SiemensReconstructionCentered.mha");
    //metaReader->Update();
    //input=metaReader->GetOutput();
    //reader=metaReader;
    //// Verify that we actually have a volume
    //int dim[3];
    //input->GetDimensions(dim);
    //if ( dim[0] < 2 ||
    //     dim[1] < 2 ||
    //     dim[2] < 2 )
    //{
    //    cout << "Error loading data!" << endl;
    //    exit(EXIT_FAILURE);
    //}
    //// Create our volume and mapper
    //volume = vtkVolume::New();
    //mapper = vtkSmartVolumeMapper::New();

    //mapper->SetInputConnection( reader->GetOutputPort() );

    //// Create our transfer function
    //vtkColorTransferFunction *colorFun = vtkColorTransferFunction::New();
    //vtkPiecewiseFunction *opacityFun = vtkPiecewiseFunction::New();

    //// Create the property and attach the transfer functions
    //vtkVolumeProperty *property = vtkVolumeProperty::New();
    //property->SetIndependentComponents(true);
    //property->SetColor( colorFun );
    //property->SetScalarOpacity( opacityFun );
    //property->SetInterpolationTypeToLinear();

    //// connect up the volume to the property and the mapper
    //volume->SetProperty( property );
    //volume->SetMapper( mapper );

    ////MIP
    //colorFun->AddRGBSegment(0.0, 1.0, 1.0, 1.0, 255.0, 1.0, 1.0, 1.0 );
    //opacityFun->AddSegment( opacityLevel - 0.5*opacityWindow, 0.0,
    //                        opacityLevel + 0.5*opacityWindow, 1.0 );
    //mapper->SetBlendModeToMaximumIntensity();

    // create sphere geometry
    sphere = vtkSphereSource::New();
    sphere->SetRadius(0.025);
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

    // create sphere geometry
    sphere0 = vtkSphereSource::New();
    sphere0->SetRadius(0.025);
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
}

void setupVTK(int width, int height)
{

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

    // Add the volume to the scene
    //if(volume)
    //    ren->AddVolume( volume );

    renWin->DoubleBufferOff();
    OffScreenBuffer = vtkUnsignedCharArray::New();
    OffScreenBuffer->Resize(width * height * 4);
    renWin->OffScreenRenderingOn();
}

void setupVTK0(int width, int height)
{

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
    ren0->AddActor(aSphere);

    // Add the volume to the scene
    //if(volume)
    //    ren0->AddVolume( volume );
    //ren0->ResetCamera();

    renWin0->DoubleBufferOff();
    OffScreenBuffer0 = vtkUnsignedCharArray::New();
    OffScreenBuffer0->Resize(width * height * 4);
    renWin0->OffScreenRenderingOn();
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
    svlVidCapSrcSDIRenderTarget* target = (svlVidCapSrcSDIRenderTarget*)svlRenderTargets::Get(-1);
    //svlRenderTargets::Get(0);
    setupObjects();
    setupVTK(1920,1080);//target->GetWidth(),target->GetHeight()//400,500
    setupVTK0(1920,1080);

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
        //target->MakeCurrentGLCtx();
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


