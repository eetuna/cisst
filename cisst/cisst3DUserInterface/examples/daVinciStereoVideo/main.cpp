/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id$

Author(s):  Balazs Vagvolgyi, Simon DiMaio, Anton Deguet
Created on: 2008-05-23

(C) Copyright 2008-2009 Johns Hopkins University (JHU), All Rights
Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


#include <cisstCommon/cmnPath.h>
#include <cisstCommon/cmnGetChar.h>
#include <cisstOSAbstraction/osaThreadedLogFile.h>
#include <cisstOSAbstraction/osaSleep.h>
#include <cisstMultiTask/mtsTaskManager.h>
#include <cisstStereoVision.h>
#include <cisst3DUserInterface/ui3CursorSphere.h>
#include <sawOpenIGTLink/mtsOpenIGTLink.h>
//#include <sawIntuitiveDaVinci/mtsIntuitiveDaVinci.h>

#ifdef sawIntuitiveDaVinci
#include <sawIntuitiveDaVinci/mtsIntuitiveDaVinci.h>
#else
#ifdef cisstDaVinci_isi_bbapi
#include <cisstDaVinci/cdvReadWrite.h>
#endif
#endif

#include "BehaviorLUS.h"

#include <MeasurementBehavior.h>
#include <MapBehavior.h>
#include <ImageViewer.h>
#include <ImageViewerKidney.h>
#include <PNGViewer3D.h>
#include <ManualRegistration.h>

#define HAS_ULTRASOUDS 0
#define TORS 1
#define VOLUME_RENDERING 0
#define SVL_VID_CAPTURE 0
#define PROVIDED_INTERFACE 0
#define STREAM_ALL 0;

#include "ISSIEventHandler.h"
#include "svlWindow.h"

int DaVinciStereoVideo()
{
std::cout << "Demo started" << std::endl;
	// log configuration
	cmnLogger::SetMask(CMN_LOG_ALLOW_ALL);
	cmnLogger::AddChannel(std::cout, CMN_LOG_ALLOW_ERRORS_AND_WARNINGS);
	cmnLogger::SetMaskDefaultLog(CMN_LOG_ALLOW_ALL);
	// specify a higher, more verbose log level for these classes
	cmnLogger::SetMaskClassMatching("ui3", CMN_LOG_ALLOW_ALL);
	cmnLogger::SetMaskClassMatching("mts", CMN_LOG_ALLOW_ALL);

	mtsComponentManager * componentManager = mtsComponentManager::GetInstance();
	//mtsIntuitiveDaVinci * daVinci = new mtsIntuitiveDaVinci("daVinci", 50);
#ifdef sawIntuitiveDaVinci
	mtsIntuitiveDaVinci * daVinci = new mtsIntuitiveDaVinci("daVinci", 50);
#else
#ifdef cisstDaVinci_isi_bbapi
	cdvReadWrite * daVinci = new cdvReadWrite("daVinci", 60 /* Hz */);
#endif
#endif
	componentManager->AddComponent(daVinci);

	ui3Manager guiManager;
	MeasurementBehavior measurementBehavior("Measure");
	guiManager.AddBehavior(&measurementBehavior,
		1,
		"measure.png");

	MapBehavior mapBehavior("Map");
	guiManager.AddBehavior(&mapBehavior,
		2,
		"map.png");

	ImageViewer imageViewer("image");
	guiManager.AddBehavior(&imageViewer,
		3,
		"move.png");
#if TORS
	ManualRegistration manualRegistration("manualRegistration");
	guiManager.AddBehavior(&manualRegistration,
		4,
		"move.png");
#else	
	ImageViewerKidney imageViewerKidney("imageKidney");
	guiManager.AddBehavior(&imageViewerKidney,
		4,
		"move.png");
#endif

	// this is were the icons have been copied by CMake post build rule
	cmnPath path;
	path.AddRelativeToCisstShare("/cisst3DUserInterface/icons");
	std::string fileName = path.Find("move.png", cmnPath::READ);
	PNGViewer3D * pngViewer;
	if (fileName != "") {
		pngViewer = new PNGViewer3D("PGNViewer", fileName);
		guiManager.AddBehavior(pngViewer,
			5,
			"square.png");
	} else {
		std::cerr << "PNG viewer not added, can't find \"move.png\" in path: " << path << std::endl;
	}

	////////////////////////////////////////////////////////////////
	// setup renderers

	svlCameraGeometry camera_geometry;
	// Load Camera calibration results
	path.AddRelativeToCisstShare("models/cameras");
	std::string calibrationFile = path.Find("mock_or_calib_results.txt");
	if (calibrationFile == "") {
		std::cerr << "Unable to find camera calibration file in path: " << path << std::endl;
		exit(-1);
	}
	camera_geometry.LoadCalibration(calibrationFile);
	// Center world in between the two cameras (da Vinci specific)
	camera_geometry.SetWorldToCenter();
	// Rotate world by 180 degrees (VTK specific)
	camera_geometry.RotateWorldAboutY(180.0);

#if CISST_SVL_HAS_NVIDIA_QUADRO_SDI
	// *** Left view ***
	guiManager.AddRenderer(svlRenderTargets::Get(-1)->GetWidth(),  // render width
		svlRenderTargets::Get(-1)->GetHeight(), // render height
		1.0,                                   // virtual camera zoom
		false,                                 // borderless?
		0, 0,                                  // window position
		camera_geometry, SVL_LEFT,             // camera parameters
		"LeftEyeView");                        // name of renderer
#else
	guiManager.AddRenderer(svlRenderTargets::Get(1)->GetWidth(),  // render width
		svlRenderTargets::Get(1)->GetHeight(), // render height
		1.0,                                   // virtual camera zoom
		false,                                 // borderless?
		0, 0,                                  // window position
		camera_geometry, SVL_LEFT,             // camera parameters
		"LeftEyeView");                        // name of renderer
#endif
	// *** Right view ***

	guiManager.AddRenderer(svlRenderTargets::Get(-1)->GetWidth(),  // render width
		svlRenderTargets::Get(-1)->GetHeight(), // render height
		1.0,                                   // virtual camera zoom
		false,                                 // borderless?
		0, 0,                                  // window position
		camera_geometry, SVL_RIGHT,            // camera parameters
		"RightEyeView");                       // name of renderer

	// Sending renderer output to external render targets
#if CISST_SVL_HAS_NVIDIA_QUADRO_SDI
	guiManager.SetRenderTargetToRenderer("LeftEyeView",  svlRenderTargets::Get(-1));
#else
	guiManager.SetRenderTargetToRenderer("LeftEyeView",  svlRenderTargets::Get(1));
#endif
	guiManager.SetRenderTargetToRenderer("RightEyeView", svlRenderTargets::Get(-1));

	///////////////////////////////////////////////////////////////
	// start streaming

	vctFrm3 transform;
	transform.Translation().Assign(0.0, 0.0, 0.0);
	transform.Rotation().From(vctAxAnRot3(vctDouble3(0.0, 1.0, 0.0), cmnPI));

	// setup first arm
	ui3MasterArm * rightMaster = new ui3MasterArm("MTMR1");
	guiManager.AddMasterArm(rightMaster);
	rightMaster->SetInput(daVinci, "MTMR1",
		daVinci, "MTMR1Select",
		daVinci, "MTMR1Clutch",
		ui3MasterArm::PRIMARY);
	rightMaster->SetTransformation(transform, 0.8 /* scale factor */);
	ui3CursorBase * rightCursor = new ui3CursorSphere();
	rightCursor->SetAnchor(ui3CursorBase::CENTER_RIGHT);
	rightMaster->SetCursor(rightCursor);

	// setup second arm
	ui3MasterArm * leftMaster = new ui3MasterArm("MTML1");
	guiManager.AddMasterArm(leftMaster);
	leftMaster->SetInput(daVinci, "MTML1",
		daVinci, "MTML1Select",
		daVinci, "MTML1Clutch",
		ui3MasterArm::SECONDARY);
	leftMaster->SetTransformation(transform, 0.8 /* scale factor */);
	ui3CursorBase * leftCursor = new ui3CursorSphere();
	leftCursor->SetAnchor(ui3CursorBase::CENTER_LEFT);
	leftMaster->SetCursor(leftCursor);

	// first slave arm, i.e. PSM1
	ui3SlaveArm * slave1 = new ui3SlaveArm("Slave1");
	guiManager.AddSlaveArm(slave1);
	slave1->SetInput(daVinci, "PSM1");
	slave1->SetTransformation(transform, 1.0 /* scale factor */);

	//set up ECM as slave arm
	ui3SlaveArm * ecm1 = new ui3SlaveArm("ECM1");
	guiManager.AddSlaveArm(ecm1);
	ecm1->SetInput(daVinci, "ECM1");
	ecm1->SetTransformation(transform, 1.0);

	// setup event for MaM transitions
	guiManager.SetupMaM(daVinci, "MastersAsMice");
	guiManager.ConnectAll();

	// connect measurement behavior
	componentManager->Connect(measurementBehavior.GetName(), "StartStopMeasure", daVinci->GetName(), "Clutch");

	// following should be replaced by a utility function or method of ui3Manager
	std::cout << "Creating components" << std::endl;
	componentManager->CreateAll();
	componentManager->WaitForStateAll(mtsComponentState::READY);

	std::cout << "Starting components" << std::endl;
	componentManager->StartAll();
	componentManager->WaitForStateAll(mtsComponentState::ACTIVE);

	int ch;

	cerr << endl << "Keyboard commands:" << endl << endl;
	cerr << "  In command window:" << endl;
	cerr << "    'q'   - Quit" << endl << endl;
	do {
		ch = cmnGetChar();
		osaSleep(100.0 * cmn_ms);
	} while (ch != 'q');


	std::cout << "Stopping components" << std::endl;
	componentManager->KillAll();
	componentManager->WaitForStateAll(mtsComponentState::FINISHED, 30.0 * cmn_s);
	componentManager->Cleanup();

	cmnLogger::Kill();

	// Balazs: Clean-up before the destructor.
	svlRenderTargets::ReleaseAll();

	return 0;
}


int DaVinciStereoVideoVolumeRendering()
{
std::cout << "Demo started" << std::endl;
	unsigned int deviceID = -1;

#if SVL_VID_CAPTURE
	deviceID = 0;
#endif

    MyCallBacks callbacks;
	std::cout << "Demo started" << std::endl;
	// log configuration
	cmnLogger::SetMask(CMN_LOG_ALLOW_ALL);
	cmnLogger::AddChannel(std::cout, CMN_LOG_ALLOW_ERRORS_AND_WARNINGS);
	cmnLogger::SetMaskDefaultLog(CMN_LOG_ALLOW_ALL);
	// specify a higher, more verbose log level for these classes
	cmnLogger::SetMaskClassMatching("ui3", CMN_LOG_ALLOW_ALL);
	cmnLogger::SetMaskClassMatching("mts", CMN_LOG_ALLOW_ALL);

	mtsComponentManager * componentManager = mtsComponentManager::GetInstance();
#if TORS
	mtsIntuitiveDaVinci * daVinci = new mtsIntuitiveDaVinci("daVinci", 50);
#else
	cdvReadWrite * daVinci = new cdvReadWrite("daVinci", 60 /* Hz */);
#endif
	componentManager->AddComponent(daVinci);

	ui3Manager guiManager;

#if TORS
	ManualRegistration manualRegistration("manualRegistration");
	guiManager.AddBehavior(&manualRegistration,
		4,
		"move.png");
#else	
	MeasurementBehavior measurementBehavior("Measure");
	guiManager.AddBehavior(&measurementBehavior,
		1,
		"measure.png");

	MapBehavior mapBehavior("Map");
	guiManager.AddBehavior(&mapBehavior,
		2,
		"map.png");

	ImageViewer imageViewer("image");
	guiManager.AddBehavior(&imageViewer,
		3,
		"move.png");

	ImageViewerKidney imageViewerKidney("imageKidney");
	guiManager.AddBehavior(&imageViewerKidney,
		4,
		"move.png");
#endif

	// this is were the icons have been copied by CMake post build rule
	cmnPath path;;
	path.AddRelativeToCisstShare("/cisst3DUserInterface/icons");
	std::string fileName = path.Find("move.png", cmnPath::READ);
	PNGViewer3D * pngViewer;
	if (fileName != "") {
		pngViewer = new PNGViewer3D("PGNViewer", fileName);
		guiManager.AddBehavior(pngViewer,
			5,
			"square.png");
	} else {
		std::cerr << "PNG viewer not added, can't find \"move.png\" in path: " << path << std::endl;
	}

#if HAS_ULTRASOUDS
	svlInitialize();

	BehaviorLUS lus("BehaviorLUS");
	guiManager.AddBehavior(&lus,       // behavior reference
		5,             // position in the menu bar: default
		"LUS.png");            // icon file: no texture

	svlStreamManager vidUltrasoundStream(1);  // running on single thread

	svlFilterSourceVideoCapture vidUltrasoundSource(1); // mono source
	if (vidUltrasoundSource.LoadSettings("usvideo.dat") != SVL_OK) {
		cout << "Setup Ultrasound video input:" << endl;
		vidUltrasoundSource.DialogSetup();
		vidUltrasoundSource.SaveSettings("usvideo.dat");
	}

	vidUltrasoundStream.SetSourceFilter(&vidUltrasoundSource);

	// add image cropper
	svlFilterImageCropper vidUltrasoundCropper;
	vidUltrasoundCropper.SetRectangle(186, 27, 186 + 360 - 1, 27 + 332 - 1);
	vidUltrasoundSource.GetOutput()->Connect(vidUltrasoundCropper.GetInput());
	// add guiManager as a filter to the pipeline, so it will receive video frames
	// "StereoVideo" is defined in the UI Manager as a possible video interface
	vidUltrasoundCropper.GetOutput()->Connect(lus.GetStreamSamplerFilter("USVideo")->GetInput());

	// add debug window
	svlFilterImageWindow vidUltrasoundWindow;
	lus.GetStreamSamplerFilter("USVideo")->GetOutput()->Connect(vidUltrasoundWindow.GetInput());

	// save one frame
	// svlFilterImageFileWriter vidUltrasoundWriter;
	// vidUltrasoundWriter.SetFilePath("usimage", "bmp");
	// vidUltrasoundWriter.Record(1);
	// vidUltrasoundStream.Trunk().Append(&vidUltrasoundWriter);

	vidUltrasoundStream.Initialize();
#endif

	////////////////////////////////////////////////////////////////
	// setup renderers

	svlCameraGeometry camera_geometry;
	// Load Camera calibration results on TORS
#ifdef _WIN32
    // this is were the icons have been copied by CMake post build rule
    // Load Camera calibration results
    path.AddRelativeToCisstShare("models/cameras");
    std::string calibrationFile = path.Find("mock_or_calib_results.txt");
    if (calibrationFile == "") {
        std::cerr << "Unable to find camera calibration file in path: " << path << std::endl;
        exit(-1);
    }
    camera_geometry.LoadCalibration(calibrationFile);
#else
	camera_geometry.LoadCalibration("/home/wen/MyCommon/calib_results.txt");
#endif
	//Manubrium
	//camera_geometry.LoadCalibration("E:/Users/davinci_mock_or/calib_results.txt");

	// Center world in between the two cameras (da Vinci specific)
	camera_geometry.SetWorldToCenter();
	// Rotate world by 180 degrees (VTK specific)
	camera_geometry.RotateWorldAboutY(180.0);

#if CISST_SVL_HAS_NVIDIA_QUADRO_SDI
	// *** Left view ***
	guiManager.AddRenderer(svlRenderTargets::Get(deviceID)->GetWidth(),  // render width
		svlRenderTargets::Get(deviceID)->GetHeight(), // render height
		1.0,                                   // virtual camera zoom
		false,                                 // borderless?
		0, 0,                                  // window position
		camera_geometry, SVL_LEFT,             // camera parameters
		"LeftEyeView");                        // name of renderer
#else
	guiManager.AddRenderer(svlRenderTargets::Get(1)->GetWidth(),  // render width
		svlRenderTargets::Get(1)->GetHeight(), // render height
		1.0,                                   // virtual camera zoom
		false,                                 // borderless?
		0, 0,                                  // window position
		camera_geometry, SVL_LEFT,             // camera parameters
		"LeftEyeView");                        // name of renderer
#endif
	// *** Right view ***

	guiManager.AddRenderer(svlRenderTargets::Get(deviceID)->GetWidth(),  // render width
		svlRenderTargets::Get(deviceID)->GetHeight(), // render height
		1.0,                                   // virtual camera zoom
		false,                                 // borderless?
		0, 0,                                  // window position
		camera_geometry, SVL_RIGHT,            // camera parameters
		"RightEyeView");                       // name of renderer

	// Sending renderer output to external render targets
#if CISST_SVL_HAS_NVIDIA_QUADRO_SDI
	guiManager.SetRenderTargetToRenderer("LeftEyeView",  svlRenderTargets::Get(deviceID));
#else
	guiManager.SetRenderTargetToRenderer("LeftEyeView",  svlRenderTargets::Get(1));
#endif
	guiManager.SetRenderTargetToRenderer("RightEyeView", svlRenderTargets::Get(deviceID));

	////////////////////////////////////////////////////////////////
	// setup volume rendering stream
	int videoAugmentationVisibilityToggle = 1;
	int volumeVisibilityToggle = 1;
	bool save = false;
    svlFilterOutput *output;

    svlFilterImageFileWriter imagewriter;
    svlFilterVideoFileWriter videowriter;
	svlFilterImageRegistrationGUI registrationGUILeft(SVL_LEFT,             // background video channel
												true);                 // visible
	svlFilterImageRegistrationGUI registrationGUIRight(SVL_RIGHT,             // background video channel
												true);                 // visible
    CViewerEventHandler window_eh;

	unsigned char alphaTransparent = 0;
	unsigned char alpha = 200;
	bool noise = true;
	svlInitialize();
	svlStreamManager volumeRenderStream(2);  // running on multiple threads

#if SVL_VID_CAPTURE
	svlFilterSourceVideoCapture source(2);
#else
	svlFilterSourceDummy source;
#endif

	svlFilterStreamTypeConverter converter;
	svlFilterSplitter splitter, splitterSave;
	svlFilterImageResizer resizer, resizerCoronal, resizerSagittalPriori, resizerSagittal, resizerFluoroPriori, resizerFluoro, resizerHolder;
	svlFilterImageOverlay overlay, overlayAxial, overlayCoronal, overlaySagittal, overlayFluoro, overlayHolder;
	svlFilterImageWindow window, windowFluoro;

#ifdef _WIN32
	if(svlImageIO::Read(image, SVL_LEFT, "C:/Users/Wen/Images/StructuredLight/20120421/Phantom/L1.bmp")!= SVL_OK) return 0;
	if(svlImageIO::Read(image, SVL_RIGHT, "C:/Users/Wen/Images/StructuredLight/20120421/Phantom/R1.bmp")!= SVL_OK) return 0;
	//if(svlImageIO::Read(imageAlpha, SVL_LEFT, "C:/Users/Wen/Images/StructuredLight/20120421/Phantom/L1.bmp")!= SVL_OK) return 0;
	//if(svlImageIO::Read(imageAlpha, SVL_RIGHT, "C:/Users/Wen/Images/StructuredLight/20120421/Phantom/R1.bmp")!= SVL_OK) return 0;
	if(svlImageIO::Read(imageLeft, SVL_LEFT, "C:/Users/Wen/Images/20130122_FlouroFlourescentFiducials/left_1358879543.818.bmp") != SVL_OK) return 0;
	if(svlImageIO::Read(imageRight, SVL_LEFT, "C:/Users/Wen/Images/20130122_FlouroFlourescentFiducials/left_1358879543.818.bmp")!= SVL_OK) return 0;
	if(svlImageIO::Read(imageFluoroLeft, SVL_LEFT, "C:/Users/Wen/Images/20130122_FlouroFlourescentFiducials/left_1358879543.818.bmp")!= SVL_OK) return 0;
	if(svlImageIO::Read(imageFluoroRight, SVL_LEFT, "C:/Users/Wen/Images/20130122_FlouroFlourescentFiducials/left_1358879543.818.bmp")!= SVL_OK) return 0;
#endif
	//imagePtr = new svlSampleImageRGBStereo();//svlSample::GetNewFromType(image.GetType());
	//imagePtrAlpha = new svlSampleImageRGBAStereo();//svlSample::GetNewFromType(imageAlpha.GetType());
	//imagePtr->CopyOf(image);
	//imagePtrAlpha->CopyOf(imageAlpha);
	//svlConverter::ConvertImage(imagePtr,imagePtrAlpha,alphaTransparent);
	//imageAlpha.CopyOf(imagePtrAlpha);

	imageFluoroLeft.SetSize(512,512);
	imageFluoroRight.SetSize(512,512);

	image_axial_left.SetImage(imageLeft);
	image_axial_right.SetImage(imageRight);
	image_axial_left.SetAlpha(alphaTransparent);
	image_axial_right.SetAlpha(alphaTransparent);

	image_fluoro_left.SetImage(imageFluoroLeft);
	image_fluoro_right.SetImage(imageFluoroLeft);
	image_fluoro_left.SetAlpha(alphaTransparent);
	image_fluoro_right.SetAlpha(alphaTransparent);

	image_sagittal_left.SetImage(imageFluoroLeft);
	image_sagittal_right.SetImage(imageFluoroLeft);
	image_sagittal_left.SetAlpha(alphaTransparent);
	image_sagittal_right.SetAlpha(alphaTransparent);

#if STREAM_ALL

	image_coronal_left.SetImage(imageLeft);
	image_coronal_right.SetImage(imageRight);
	image_coronal_left.SetAlpha(alphaTransparent);
	image_coronal_right.SetAlpha(alphaTransparent);

	image_holder_left.SetImage(imageLeft);
	image_holder_right.SetImage(imageLeft);
	image_holder_left.SetAlpha(0);
	image_holder_right.SetAlpha(0);
#endif

	// source
#if SVL_VID_CAPTURE
	if (source.LoadSettings("issi.dat") != SVL_OK) {
		cout << endl;
		source.DialogSetup(SVL_LEFT);
		source.DialogSetup(SVL_RIGHT);
		source.SaveSettings("issi.dat");
	}
#else
	source.SetImage(image);
	//source.EnableNoiseImage(noise);
#endif

	// splitter
	
	splitter.AddOutput("fluoro");
	resizerFluoroPriori.SetOutputSize(imageFluoroLeft.GetWidth(),imageFluoroLeft.GetHeight(),SVL_LEFT);
	resizerFluoroPriori.SetOutputSize(imageFluoroRight.GetWidth(),imageFluoroRight.GetHeight(),SVL_RIGHT);
	resizerFluoro.SetOutputRatio(0.5, 0.5, SVL_LEFT);
	resizerFluoro.SetOutputRatio(0.5, 0.5, SVL_RIGHT);
	resizerFluoro.SetInterpolation(true);
	splitter.AddOutput("sagittal");
	resizerSagittalPriori.SetOutputSize(imageFluoroLeft.GetWidth(),imageFluoroLeft.GetHeight(),SVL_LEFT);
	resizerSagittalPriori.SetOutputSize(imageFluoroRight.GetWidth(),imageFluoroRight.GetHeight(),SVL_RIGHT);
	resizerSagittal.SetOutputRatio(0.5, 0.5, SVL_LEFT);
	resizerSagittal.SetOutputRatio(0.5, 0.5, SVL_RIGHT);
	resizerSagittal.SetInterpolation(true);

#if STREAM_ALL

	//splitter.AddOutput("holder");

	// resizer
	int scale = 1;
	resizer.SetOutputRatio(0.9,0.9,SVL_LEFT);
	resizer.SetOutputRatio(0.9,0.9,SVL_RIGHT);

	splitter.AddOutput("coronal");
	resizerCoronal.SetOutputRatio(0.25, 0.25, SVL_LEFT);
	resizerCoronal.SetOutputRatio(0.25, 0.25, SVL_RIGHT);
	resizerCoronal.SetInterpolation(true);

	resizerHolder.SetOutputRatio(0.25, 0.25, SVL_LEFT);
	resizerHolder.SetOutputRatio(0.25, 0.25, SVL_RIGHT);
	resizerHolder.SetInterpolation(true);
#endif

	// overlay
	overlay.AddInputImage("sagittal");
	overlay.AddInputImage("fluoro");
#if STREAM_ALL
	overlay.AddInputImage("coronal");
	overlay.AddInputImage("holder");
#endif

   svlOverlayImage image_overlay_fluoro_left(SVL_LEFT,        // background video channel
		true,            // visible
		"fluoro",         // image input name
		SVL_LEFT,        // image input channel
		vctInt2(image.GetWidth()*1/8, image.GetHeight()*3/4), // position
		200);            // alpha (transparency)
	svlOverlayImage image_overlay_fluoro_right(SVL_RIGHT,        // background video channel
		true,            // visible
		"fluoro",         // image input name
		SVL_RIGHT,        // image input channel
		vctInt2(image.GetWidth()*1/8, image.GetHeight()*3/4), // position
		200);            // alpha (transparency)
	svlOverlayImage image_overlay_sagittal_left(SVL_LEFT,        // background video channel
		true,            // visible
		"sagittal",         // image input name
		SVL_LEFT,        // image input channel
		vctInt2(image.GetWidth()*2/8, image.GetHeight()*3/4), // position
		200);            // alpha (transparency)
	svlOverlayImage image_overlay_sagittal_right(SVL_RIGHT,        // background video channel
		true,            // visible
		"sagittal",         // image input name
		SVL_RIGHT,        // image input channel
		vctInt2(image.GetWidth()*2/8, image.GetHeight()*3/4), // position
		200);            // alpha (transparency)
	svlOverlayImage image_overlay_coronal_left(SVL_LEFT,        // background video channel
		true,            // visible
		"coronal",         // image input name
		SVL_LEFT,        // image input channel
		vctInt2(image.GetWidth()*3/8, image.GetHeight()*3/4), // position
		200);            // alpha (transparency)
	svlOverlayImage image_overlay_coronal_right(SVL_RIGHT,        // background video channel
		true,            // visible
		"coronal",         // image input name
		SVL_RIGHT,        // image input channel
		vctInt2(image.GetWidth()*3/8, image.GetHeight()*3/4), // position
		200);            // alpha (transparency)
	svlOverlayImage image_overlay_holder_left(SVL_LEFT,        // background video channel
		true,            // visible
		"holder",         // image input name
		SVL_LEFT,        // image input channel
		vctInt2(image.GetWidth()*5/8, image.GetHeight()*3/4), // position
		200);            // alpha (transparency)
	svlOverlayImage image_overlay_holder_right(SVL_RIGHT,        // background video channel
		true,            // visible
		"holder",         // image input name
		SVL_RIGHT,        // image input channel
		vctInt2(image.GetWidth()*5/8, image.GetHeight()*3/4), // position
		200);            // alpha (transparency)

	overlay.AddOverlay(image_overlay_fluoro_left);
	overlay.AddOverlay(image_overlay_fluoro_right);
	overlay.AddOverlay(image_overlay_sagittal_left);
	overlay.AddOverlay(image_overlay_sagittal_right);


#if STREAM_ALL
	overlay.AddOverlay(image_overlay_coronal_left);
	overlay.AddOverlay(image_overlay_coronal_right);
	overlay.AddOverlay(image_overlay_holder_left);
	overlay.AddOverlay(image_overlay_holder_right);
#endif
	overlay.AddQueuedItems();

	overlayAxial.AddOverlay(image_axial_left);
	overlayAxial.AddOverlay(image_axial_right);
	overlayAxial.AddQueuedItems();
	overlaySagittal.AddOverlay(image_sagittal_left);
	overlaySagittal.AddOverlay(image_sagittal_right);
	overlaySagittal.AddQueuedItems();;
	overlayFluoro.AddOverlay(image_fluoro_left);
	overlayFluoro.AddOverlay(image_fluoro_right);
	overlayFluoro.AddQueuedItems();

#if STREAM_ALL
	overlayCoronal.AddOverlay(image_coronal_left);
	overlayCoronal.AddOverlay(image_coronal_right);
	overlayCoronal.AddQueuedItems();
	overlayHolder.AddOverlay(image_holder_left);
	overlayHolder.AddOverlay(image_holder_right);
	overlayHolder.AddQueuedItems();
#endif

	// converter
	converter.SetType(svlTypeImageRGBStereo,svlTypeImageRGBAStereo);
	converter.SetAlpha(alpha);

	// window
	window.SetTitle("Original Stream");
	windowFluoro.SetTitle("Fluoro Stream");

	volumeRenderStream.SetSourceFilter(&source);
	source.GetOutput()->Connect(overlayAxial.GetInput());
	overlayAxial.GetOutput()->Connect(splitter.GetInput());
	splitter.GetOutput()->Connect(overlay.GetInput());
	//overlay.GetOutput()->Connect(window.GetInput());
	//window.GetOutput()->Connect(converter.GetInput());
	overlay.GetOutput()->Connect(converter.GetInput());
	converter.GetOutput()->Connect(guiManager.GetStreamSamplerFilter("StereoVideoRGBA")->GetInput());

	splitter.GetOutput("sagittal")->Connect(resizerSagittalPriori.GetInput());
	resizerSagittalPriori.GetOutput()->Connect(overlaySagittal.GetInput());
	overlaySagittal.GetOutput()->Connect(resizerSagittal.GetInput());
	resizerSagittal.GetOutput()->Connect(overlay.GetInput("sagittal"));

	splitter.GetOutput("fluoro")->Connect(resizerFluoroPriori.GetInput());
	resizerFluoroPriori.GetOutput()->Connect(overlayFluoro.GetInput());
	overlayFluoro.GetOutput()->Connect(resizerFluoro.GetInput());
	//resizerFluoro.GetOutput()->Connect(windowFluoro.GetInput());
	//windowFluoro.GetOutput()->Connect(overlay.GetInput("fluoro"));
    resizerFluoro.GetOutput()->Connect(overlay.GetInput("fluoro"));

#if STREAM_ALL
	splitter.GetOutput("coronal")->Connect(overlayCoronal.GetInput());
	overlayCoronal.GetOutput()->Connect(resizerCoronal.GetInput());
	resizerCoronal.GetOutput()->Connect(overlay.GetInput("coronal"));

	splitter.GetOutput("holder")->Connect(overlayHolder.GetInput());
	overlayHolder.GetOutput()->Connect(resizerHolder.GetInput());
	resizerHolder.GetOutput()->Connect(overlay.GetInput("holder"));
#endif

	//volumeRenderStream.Play();
	volumeRenderStream.Initialize();
#ifdef _WIN32
	m_issi.SetRenderedImageCallback(CVolumeRendererEventHandlerUtilities::rendered_image_cb, (void*)&m_issi);
	m_issi.SetFluoroImageCallback(CVolumeRendererEventHandlerUtilities::fluoro_image_cb, (void*)&m_issi);
#if STREAM_ALL
	m_issi.SetEndoscopeImageCallback(CVolumeRendererEventHandlerUtilities::endoscope_image_cb, (void*)&m_issi);
#endif
#endif
	////////////////////////////////////////////////////////////////
	// Creating video background image planes
	guiManager.AddVideoBackgroundToRenderer("LeftEyeView",  "StereoVideoRGBA", SVL_LEFT);
	guiManager.AddVideoBackgroundToRenderer("RightEyeView", "StereoVideoRGBA", SVL_RIGHT);
	//guiManager.SetAlphaVideoBackgroundToRenderer("LeftEyeView",  "StereoVideoRGBA", SVL_LEFT,alpha,image.GetHeight()*3/4);
	//guiManager.SetAlphaVideoBackgroundToRenderer("RightEyeView", "StereoVideoRGBA", SVL_RIGHT,alpha,image.GetHeight()*3/4);

#if 0
	// Add third camera: simple perspective camera placed in the world center
	camera_geometry.SetPerspective(400.0, 2);

	guiManager.AddRenderer(384,                // render width
		216,                // render height
		1.0,                // virtual camera zoom
		false,              // borderless?
		0, 0,               // window position
		camera_geometry, 2, // camera parameters
		"ThirdEyeView");    // name of renderer
#endif

	///////////////////////////////////////////////////////////////
	// start streaming
#if HAS_ULTRASOUDS
	vidUltrasoundStream.Start();
#endif

	// not needed
	//#if VOLUME_RENDERING
	//volumeRenderStream.Start();
	//#endif

	vctFrm3 transform;
	transform.Translation().Assign(0.0, 0.0, 0.0);
	transform.Rotation().From(vctAxAnRot3(vctDouble3(0.0, 1.0, 0.0), cmnPI));

	// setup first arm
	ui3MasterArm * rightMaster = new ui3MasterArm("MTMR1");
	guiManager.AddMasterArm(rightMaster);
	rightMaster->SetInput(daVinci, "MTMR1",
		daVinci, "MTMR1Select",
		daVinci, "MTMR1Clutch",
		ui3MasterArm::PRIMARY);
	rightMaster->SetTransformation(transform, 0.8 /* scale factor */);
	ui3CursorBase * rightCursor = new ui3CursorSphere();
	rightCursor->SetAnchor(ui3CursorBase::CENTER_RIGHT);
	rightMaster->SetCursor(rightCursor);

	// setup second arm
	ui3MasterArm * leftMaster = new ui3MasterArm("MTML1");
	guiManager.AddMasterArm(leftMaster);
	leftMaster->SetInput(daVinci, "MTML1",
		daVinci, "MTML1Select",
		daVinci, "MTML1Clutch",
		ui3MasterArm::SECONDARY);
	leftMaster->SetTransformation(transform, 0.8 /* scale factor */);
	ui3CursorBase * leftCursor = new ui3CursorSphere();
	leftCursor->SetAnchor(ui3CursorBase::CENTER_LEFT);
	leftMaster->SetCursor(leftCursor);

	// first slave arm, i.e. PSM1
	ui3SlaveArm * slave1 = new ui3SlaveArm("Slave1");
	guiManager.AddSlaveArm(slave1);
	slave1->SetInput(daVinci, "PSM1");
	slave1->SetTransformation(transform, 1.0 /* scale factor */);

	//set up ECM as slave arm
	ui3SlaveArm * ecm1 = new ui3SlaveArm("ECM1");
	guiManager.AddSlaveArm(ecm1);
	ecm1->SetInput(daVinci, "ECM1");
	ecm1->SetTransformation(transform, 1.0);

	// setup event for MaM transitions
	guiManager.SetupMaM(daVinci, "MastersAsMice");
	guiManager.ConnectAll();

	// connect measurement behavior
#if !TORS 
	componentManager->Connect(measurementBehavior.GetName(), "StartStopMeasure", daVinci->GetName(), "Clutch");
#endif

#if PROVIDED_INTERFACE
	mtsOpenIGTLink * mtsOpenIGTLinkObj = new mtsOpenIGTLink("MyOpenIGTLink", 50.0 * cmn_ms);
	std::cout << "Running as OpenIGTLink server on port 18944" << std::endl;
	mtsOpenIGTLinkObj->Configure("18944");

	// add components to component manager
	componentManager->AddComponent(mtsOpenIGTLinkObj);

	// connect components
	componentManager->Connect(mtsOpenIGTLinkObj->GetName(), "SlaveArm1",
		manualRegistration.GetName(), "SlaveArm1");
	componentManager->Connect(mtsOpenIGTLinkObj->GetName(), "ECM_T_ECMRCM",
		manualRegistration.GetName(), "ECM_T_ECMRCM");
	componentManager->Connect(mtsOpenIGTLinkObj->GetName(), "ECMRCM_T_Virtual",
		manualRegistration.GetName(), "ECMRCM_T_Virtual");
	componentManager->Connect(mtsOpenIGTLinkObj->GetName(), "ECMRCM_T_VirtualGlobal",
		manualRegistration.GetName(), "ECMRCM_T_VirtualGlobal");

#endif

	// following should be replaced by a utility function or method of ui3Manager
	std::cout << "Creating components" << std::endl;
	componentManager->CreateAll();
	componentManager->WaitForStateAll(mtsComponentState::READY);

	std::cout << "Starting components" << std::endl;
	componentManager->StartAll();
	componentManager->WaitForStateAll(mtsComponentState::ACTIVE);

    if (!callbacks.AddHandlers(daVinci)) {
        return -1;
    }

	int ch;
	ManualRegistration::VisibleObjectType manualRegistrationFiducialToggle = manualRegistration.GetFiducialToggle();
	ManualRegistration::VisibleObjectType manualRegistrationVisibilityToggle;// = manualRegistration.SetVisibleToggle(ManualRegistration::VisibleObjectType::ALL);
	manualRegistration.m_registrationGUIRight = (&registrationGUIRight);
	manualRegistration.m_registrationGUILeft = (&registrationGUILeft);

	bool continuousRegister = manualRegistration.SetContinuousRegistration(false);
	videoAugmentationVisibilityToggle = manualRegistration.GetVideoAugmentationVisibility();
	bool wristTipOffsetToggle = true;
	double externalTransformationThreshold = manualRegistration.UpdateExternalTransformationThreshold(0.0);
	vctFrm3 position;
	std::map<int, svlFilterImageRegistrationGUI::PointInternal> points = registrationGUIRight.GetRegistrationPoints();
	svlFilterImageRegistrationGUI::PointInternal point;

	do {
		manualRegistrationFiducialToggle = manualRegistration.GetFiducialToggle();
		manualRegistrationVisibilityToggle = manualRegistration.GetVisibleToggle();
		videoAugmentationVisibilityToggle = manualRegistration.GetVideoAugmentationVisibility();

        cerr << endl << "Keyboard commands:" << endl << endl;
        cerr << "  In image window:" << endl;
        cerr << "    's'   - Take image snapshots" << endl;
#if SAVE
            cerr << "    SPACE - Video recorder control: Record/Pause" << endl;
#endif
        cerr << "  In command window:" << endl;
        cerr << "    'v'   - PiP visibility" << endl;
		cerr << "    'x'   - Overlay visibility: " << videoAugmentationVisibilityToggle << endl;
		cerr << "    't'   - Toggle overlay visibility: " << manualRegistrationVisibilityToggle << endl;
        cerr << "    'b'   - Position back" << endl;
        cerr << "    'h'   - Position home" << endl;
		cerr << "    'f'   - Toggle fiducial: " << manualRegistrationFiducialToggle << endl;
        cerr << "    'r'   - Register" << endl;
		cerr << "    'd'   - Delete fiducial" << endl;
		cerr << "    'a'   - add fiducial" << endl;
		cerr << "    'o'   - decrease opacity" << endl;
		cerr << "    'c'   - continuous register: " << continuousRegister << endl;
		cerr << "    'w'   - wrist to tip offset" << endl;
		cerr << "    'n,m' - external trans threshold: " << externalTransformationThreshold << endl;
		cerr << "    'e'   - compute TRE" << endl;
		cerr << "    'l'   - print & close TRE log" << endl;
		cerr << "    'q'   - Quit" << endl << endl;

        ch = cmnGetChar();

        switch (ch) 
		{
            case 'x':
                cerr << endl << endl;
				if(videoAugmentationVisibilityToggle == 0)
				{
					videoAugmentationVisibilityToggle = 1;
				}else
				{
					videoAugmentationVisibilityToggle = 0;
				}	
				manualRegistration.SetVideoAugmentationVisibility(videoAugmentationVisibilityToggle);
				cerr << endl;
				break;
            case 'v':
                cerr << endl << endl;
				if(volumeVisibilityToggle == 0)
				{
					converter.SetAlpha(alpha);
					volumeVisibilityToggle = 1;
				}else
				{
					converter.SetAlpha(alphaTransparent);
					volumeVisibilityToggle = 0;
				}	
				cerr << endl;
				break;
			case 't':
                cerr << endl << endl;
				manualRegistrationVisibilityToggle = manualRegistration.ToggleVisibility();
				cerr << endl;
				break;
			case 'b':
                cerr << endl << endl;
				manualRegistration.PositionBack();
				cerr << endl;
				break;
			case 'h':
                cerr << endl << endl;
				manualRegistration.PositionHome();
				cerr << endl;
				break;
			case 'f':
                cerr << endl << endl;
				manualRegistrationFiducialToggle = manualRegistration.ToggleFiducials();
				cerr << endl;
				break;
			case 'r':
                cerr << endl << endl;
				registrationGUIRight.ComputationalStereo(&registrationGUILeft);

				points = registrationGUIRight.GetRegistrationPoints();
				std::cout << " UpdateFiducialRegistration with " << points.size() << " points" << std::endl;
				for (int i=0;i<points.size();i++) 
				{
					point = points[i];
					position.Translation().Assign(vctDouble3(point.pointLocation.x,point.pointLocation.y,point.pointLocation.z));
					manualRegistration.SetFiducial(position,ManualRegistration::VisibleObjectType::FIDUCIALS_REAL,point.Valid,i);
				}

				points = registrationGUIRight.GetCalibrationPoints();
				for (int i=0;i<points.size();i++) 
				{
					point = points[i];
					position.Translation().Assign(vctDouble3(point.pointLocation.x,point.pointLocation.y,point.pointLocation.z));
					manualRegistration.SetFiducial(position,ManualRegistration::VisibleObjectType::FIDUCIALS_REAL,point.Valid,i);
				}
				manualRegistration.Register();
				cerr << endl;
				break;
			case 'd':
                cerr << endl << endl;
				cerr << "Enter index of fiducial to remove: " ;
				ch = cmnGetChar();
				cerr << endl;
				registrationGUIRight.SetValidity(false,(ch-48));
				registrationGUILeft.SetValidity(false,(ch-48));
				cerr << endl;
				break;
			case 'w':
				{
						//cerr << " Enter offset of wrist to tool tip: " ;
						//ch = cmnGetChar();
						double offset = 9.7;//ch - 48;
						offset = manualRegistration.SetWristTipOffset(wristTipOffsetToggle);
						cerr << " Offset now set to: " << offset << std::endl;
						wristTipOffsetToggle = !wristTipOffsetToggle;
						cerr << endl;
				}
			break;
			case 'a':
			{
                cerr << endl << endl;
				
				switch(manualRegistrationFiducialToggle)
				{
					case(ManualRegistration::TARGETS_REAL):
					{
						cerr << "Current # of fiducials is " << registrationGUIRight.GetTargetPoints().size() << " Enter index of fiducial to add: " ;
						ch = cmnGetChar();
						int index = ch - 48;
						if(index < registrationGUIRight.GetTargetPoints().size())
						{
							registrationGUIRight.SetValidity(true,index,svlFilterImageRegistrationGUI::TARGETS_REAL);
							registrationGUILeft.SetValidity(true,index,svlFilterImageRegistrationGUI::TARGETS_REAL);
							window_eh.SetTargetPointIndex(index);
						}else if(index == registrationGUIRight.GetTargetPoints().size())
						{
							registrationGUILeft.AddPoint(svlPoint2D(440, 327), // rectangle size and position
														5,                   // radius horizontally
														5,                   // radius vertically
														60.0,                 // angle
														svlRGB(0, 32, 32),  // color
														svlFilterImageRegistrationGUI::RED,
														1.0,
														4.7625,
														svlFilterImageRegistrationGUI::TARGETS_REAL,
														true,
														SVL_LEFT);                // filled
							registrationGUIRight.AddPoint(svlPoint2D(440, 327), // rectangle size and position
														5,                   // radius horizontally
														5,                   // radius vertically
														60.0,                 // angle
														svlRGB(0, 32, 32),  // color
														svlFilterImageRegistrationGUI::RED,
														4.7625,
														1.0,
														svlFilterImageRegistrationGUI::TARGETS_REAL,
														true,
														SVL_RIGHT);                // filled
							window_eh.SetTargetPointIndex(index);
						}
					}
					break;
					case(ManualRegistration::FIDUCIALS_REAL):
					{
						cerr << "Current # of fiducials is " << registrationGUIRight.GetPoints().size() << " Enter index of fiducial to add: " ;
						ch = cmnGetChar();
						int index = ch - 48;
						if(index < registrationGUIRight.GetPoints().size())
						{
							registrationGUIRight.SetValidity(true,index);
							registrationGUILeft.SetValidity(true,index);
							window_eh.SetPointIndex(index);
						}else if (index == registrationGUIRight.GetPoints().size())
						{
							registrationGUILeft.AddPoint(svlPoint2D(440, 327), // rectangle size and position
														5,                   // radius horizontally
														5,                   // radius vertically
														60.0,                 // angle
														svlRGB(0, 64, 0),  // color
														svlFilterImageRegistrationGUI::RED,
														4.7625,
														1.0,
														svlFilterImageRegistrationGUI::FIDUCIALS_REAL,
														true,
														SVL_LEFT);                // filled
							registrationGUIRight.AddPoint(svlPoint2D(440, 327), // rectangle size and position
														5,                   // radius horizontally
														5,                   // radius vertically
														60.0,                 // angle
														svlRGB(0, 64, 0),  // color
														svlFilterImageRegistrationGUI::RED,
														4.7625,
														1.0,
														svlFilterImageRegistrationGUI::FIDUCIALS_REAL,
														true,
														SVL_RIGHT);                // filled
							window_eh.SetPointIndex(index);
						}
					}
					break;
					default:
					break;
				}
				cerr << endl;
			}
			break;
			case 'o':
			{
                cerr << endl << endl;
				manualRegistration.UpdateOpacity(-0.1);
				cerr << endl;
				break;
			}
			case 'p':
			{
                cerr << endl << endl;
				manualRegistration.UpdateOpacity(0.1);
				cerr << endl;
				break;
			}
			case 'c':
			{
                cerr << endl << endl;
				continuousRegister = manualRegistration.SetContinuousRegistration(!continuousRegister);
				cerr << endl;
				break;
			}
			case 'e':
			{
                cerr << endl << endl;
				manualRegistration.ComputeTRE();
                cerr << endl << endl;
				break;
			}
			case 'l':
			{
                cerr << endl << endl;
				manualRegistration.ComputeTRE(false);
                cerr << endl << endl;
				break;
			}
			break;
			case 'n':
			{
                cerr << endl << endl;
				externalTransformationThreshold = manualRegistration.UpdateExternalTransformationThreshold(2.0);
                cerr << endl << endl;
				break;
			}
			case 'm':
			{
                cerr << endl << endl;
				externalTransformationThreshold = manualRegistration.UpdateExternalTransformationThreshold(-2.0);
                cerr << endl << endl;
				break;
			}

            default:
            break;
		}	
		osaSleep(100.0 * cmn_ms);
	} while (ch != 'q');
#if HAS_ULTRASOUDS
	vidUltrasoundStream.Release();
#endif

	volumeRenderStream.Stop();
	volumeRenderStream.Release();
	volumeRenderStream.DisconnectAll();

	std::cout << "Stopping components" << std::endl;
	componentManager->KillAll();
	componentManager->WaitForStateAll(mtsComponentState::FINISHED, 30.0 * cmn_s);
	componentManager->Cleanup();

	cmnLogger::Kill();

	// Balazs: Clean-up before the destructor.
	svlRenderTargets::ReleaseAll();

	return 0;
}

int main()
{
#if VOLUME_RENDERING
	DaVinciStereoVideoVolumeRendering();
#else
	DaVinciStereoVideo();
#endif
}
