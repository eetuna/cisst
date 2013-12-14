/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id$

Author(s):  Wen P. Liu, Anton Deguet
Created on: 2012-01-27

(C) Copyright 2012 Johns Hopkins University (JHU), All Rights
Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


#include <cisstParameterTypes/prmPositionCartesianSet.h>
#include <cisstOSAbstraction/osaThreadedLogFile.h>
#include <cisstOSAbstraction/osaSleep.h>
#include <cisstMultiTask/mtsTaskManager.h>
#include <cisstMultiTask/mtsInterfaceRequired.h>
#include <cisst3DUserInterface/ui3Widget3D.h>
#include <cisst3DUserInterface/ui3Manager.h>
#include <cisst3DUserInterface/ui3SlaveArm.h> // bad, ui3 should not have slave arm to start with (adeguet1)
#include <cisstVector/vctQuaternionRotation3Base.h>
#include <cisstVector/vctDeterminant.h>

#include "ManualRegistration.h"

#define CUBE_DEMO 1
#define SUPERFLAB_DEMO 0
#define IMPORT_FIDUCIALS 0
#define FIDUCIAL_COUNT_MAX 30
#define IMPORT_MULTIPLE_TARGETS 0
#define PROVIDED_INTERFACE 0
#define TOOL_TRACKING 0
#define MARGIN_RADIUS 0//10
#define ERROR_ANALYSIS 0

// z-axis translation between tool eye and tip (11.0 mm) for debakey forceps; (9.7mm) for Large Needle Driver; (20mm) for 5mm Monopolar Cautery with spatula
//#define WRIST_TIP_OFFSET (9.7)

class ManualRegistrationSurfaceVisibleStippleObject: public ui3VisibleObject
{
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);
public:
    enum GeometryType {NONE=0,CUBE,SPHERE,CYLINDER,TEXT};
    inline ManualRegistrationSurfaceVisibleStippleObject(const std::string & inputFile, const GeometryType & geometry=NONE, double size = 0):
        ui3VisibleObject(),
        Visible(true),
        Valid(true),
        InputFile(inputFile),
        SurfaceReader(0),
        SurfaceMapper(0),
        SurfaceActor(0),
        TextActor(0),
        Geometry(geometry),
        Size(size)
    {

    }

    inline ~ManualRegistrationSurfaceVisibleStippleObject()
    {
        if (this->SurfaceActor) {
            this->SurfaceActor->Delete();
            this->SurfaceActor = 0;
        }
        if (this->SurfaceMapper) {
            this->SurfaceMapper->Delete();
            this->SurfaceMapper = 0;
        }
        if (this->SurfaceReader) {
            this->SurfaceReader->Delete();
            this->SurfaceReader = 0;
        }
        if (this->TextActor) {
            this->TextActor->Delete();
            this->TextActor = 0;
        }
    }

    inline bool CreateVTKObjectCube()
    {
        vtkCubeSource *source = vtkCubeSource::New();
        source->SetBounds(-1*this->Size,this->Size,-1*this->Size,this->Size,-1*this->Size,this->Size);
        SurfaceMapper = vtkPolyDataMapper::New();
        CMN_ASSERT(SurfaceMapper);
        SurfaceMapper->SetInputConnection(source->GetOutputPort());
        SurfaceMapper->SetScalarRange(0,7);
        SurfaceMapper->ScalarVisibilityOff();
        SurfaceMapper->ImmediateModeRenderingOn();
        SurfaceActor = ui3VTKStippleActor::New();
        CMN_ASSERT(SurfaceActor);
        SurfaceActor->SetMapper(SurfaceMapper);
        TextActor = vtkOpenGLActor::New();

        // Add the actor
        this->AddPart(this->SurfaceActor);
        return true;
    }

    inline bool CreateVTKObjectCylinder()
    {
        vtkCylinderSource *source = vtkCylinderSource::New();
        source->SetHeight(5*this->Size);
        source->SetRadius(this->Size);
        SurfaceMapper = vtkPolyDataMapper::New();
        CMN_ASSERT(SurfaceMapper);
        SurfaceMapper->SetInputConnection(source->GetOutputPort());
        SurfaceMapper->SetScalarRange(0,7);
        SurfaceMapper->ScalarVisibilityOff();
        SurfaceMapper->ImmediateModeRenderingOn();
        SurfaceActor = ui3VTKStippleActor::New();
        CMN_ASSERT(SurfaceActor);
        SurfaceActor->SetMapper(SurfaceMapper);

        // Create a vector text
        vtkVectorText* vecText = vtkVectorText::New();
        vecText->SetText(InputFile.c_str());
		vtkLinearExtrusionFilter* extrude = vtkLinearExtrusionFilter::New();
		if(sizeof(InputFile.c_str()) > 0)
		{
			extrude->SetInputConnection( vecText->GetOutputPort());
			extrude->SetExtrusionTypeToNormalExtrusion();
			extrude->SetVector(0, 0, 1 );
			extrude->SetScaleFactor (0.5);
		}

        vtkPolyDataMapper* txtMapper = vtkPolyDataMapper::New();
        txtMapper->SetInputConnection( extrude->GetOutputPort());
        TextActor = vtkOpenGLActor::New();
        CMN_ASSERT(TextActor);
        TextActor->SetMapper(txtMapper);

        vtkSphereSource *sphereSource = vtkSphereSource::New();
        CMN_ASSERT(sphereSource);
        sphereSource->SetRadius(this->Size);

        vtkPolyDataMapper * sphereSurfaceMapper = vtkPolyDataMapper::New();
        CMN_ASSERT(sphereSurfaceMapper);
        sphereSurfaceMapper->SetInputConnection(sphereSource->GetOutputPort());
        sphereSurfaceMapper->ImmediateModeRenderingOn();

        ui3VTKStippleActor * sphereSurfaceActor = ui3VTKStippleActor::New();
        CMN_ASSERT(sphereSurfaceActor);
        sphereSurfaceActor->SetMapper(sphereSurfaceMapper);
        double sphereOffset[3] = {-9.7,0.0,0.0};//{-wristToTipOffset,0.0,0.0};
        sphereSurfaceActor->SetPosition(sphereOffset);
        sphereSurfaceActor->GetProperty()->SetOpacity(0.25);

        // Add the actor(s)
        this->AddPart(sphereSurfaceActor);
        this->AddPart(SurfaceActor);
        this->AddPart(TextActor);
        return true;
    }

    inline bool CreateVTKObjectSphere(void) {
        vtkSphereSource *source = vtkSphereSource::New();
        CMN_ASSERT(source);
        source->SetRadius(this->Size);

        SurfaceMapper = vtkPolyDataMapper::New();
        CMN_ASSERT(SurfaceMapper);
        SurfaceMapper->SetInputConnection(source->GetOutputPort());
        SurfaceMapper->ImmediateModeRenderingOn();

        SurfaceActor = ui3VTKStippleActor::New();
        CMN_ASSERT(SurfaceActor);
        SurfaceActor->SetMapper(SurfaceMapper);

        // Create a vector text
        vtkVectorText* vecText = vtkVectorText::New();
        vecText->SetText(InputFile.c_str());

        vtkLinearExtrusionFilter* extrude = vtkLinearExtrusionFilter::New();
        extrude->SetInputConnection( vecText->GetOutputPort());
        extrude->SetExtrusionTypeToNormalExtrusion();
        extrude->SetVector(0, 0, 1 );
        extrude->SetScaleFactor (0.2);

        vtkPolyDataMapper* txtMapper = vtkPolyDataMapper::New();
        txtMapper->SetInputConnection( extrude->GetOutputPort());
        TextActor = vtkOpenGLActor::New();
        CMN_ASSERT(TextActor);
        TextActor->SetMapper(txtMapper);

        // Add the actor(s)
        this->AddPart(SurfaceActor);
        this->AddPart(TextActor);


        return true;
    }

	inline bool CreateVTKObjectSphereSimple(void) {
		if(!SurfaceActor)
		{
			vtkSmartPointer<vtkSphereSource> source = vtkSmartPointer<vtkSphereSource>::New();
			CMN_ASSERT(source);
			source->SetRadius(1.0);

			SurfaceMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
			CMN_ASSERT(SurfaceMapper);
			SurfaceMapper->SetInputConnection(source->GetOutputPort());
			SurfaceMapper->ImmediateModeRenderingOn();

			SurfaceActor = vtkSmartPointer<vtkOpenGLActor>::New();
			CMN_ASSERT(SurfaceActor);
			SurfaceActor->SetMapper(SurfaceMapper);
			TextActor = vtkSmartPointer<ui3VTKStippleActor>::New();

			// Add the actor(s)
			this->AddPart(SurfaceActor);
		}

        return true;
    }

    inline bool CreateVTKObjectFromFile()
    {
		SurfaceReader = vtkSmartPointer<vtkPolyDataReader>::New();
		CMN_ASSERT(SurfaceReader);
		CMN_LOG_CLASS_INIT_VERBOSE << "Loading file \"" << InputFile << "\"" << std::endl;
		SurfaceReader->SetFileName(InputFile.c_str());
		CMN_LOG_CLASS_INIT_VERBOSE << "File \"" << InputFile << "\" loaded" << std::endl;
		SurfaceReader->Update();

		SurfaceMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
		CMN_ASSERT(SurfaceMapper);
		SurfaceMapper->SetInputConnection(SurfaceReader->GetOutputPort());
		SurfaceMapper->ScalarVisibilityOff();
		SurfaceMapper->ImmediateModeRenderingOn();
		SurfaceActor = vtkSmartPointer<vtkOpenGLActor>::New();
		CMN_ASSERT(SurfaceActor);
		SurfaceActor->SetMapper(SurfaceMapper);
		TextActor = vtkSmartPointer<vtkOpenGLActor>::New();
        // Add the actor
	    this->AddPart(this->SurfaceActor);
	

        return true;
    }

    inline bool CreateVTKObjectText(void) {
        SurfaceReader = vtkPolyDataReader::New();
        SurfaceMapper = vtkPolyDataMapper::New();
        SurfaceActor = vtkOpenGLActor::New();

        // Create a vector text
        vtkVectorText* vecText = vtkVectorText::New();
        vecText->SetText(InputFile.c_str());

        vtkLinearExtrusionFilter* extrude = vtkLinearExtrusionFilter::New();
        extrude->SetInputConnection( vecText->GetOutputPort());
        extrude->SetExtrusionTypeToNormalExtrusion();
        extrude->SetVector(0, 0, 1 );
        extrude->SetScaleFactor (0.0);

        vtkPolyDataMapper* txtMapper = vtkPolyDataMapper::New();
        txtMapper->SetInputConnection( extrude->GetOutputPort());
        
		TextActor = vtkOpenGLActor::New();
        CMN_ASSERT(TextActor);
        TextActor->SetMapper(txtMapper);

        // Add the actor(s)
        this->AddPart(TextActor);
        return true;
    }

    inline bool CreateVTKObjects(void) 
	{
        // Create surface actor/mapper
        switch (Geometry)
        {
        case CUBE:
            return CreateVTKObjectCube();
            break;
        case CYLINDER:
            return CreateVTKObjectCylinder();
            break;
        case SPHERE:
            return CreateVTKObjectSphere();
            break;
        case TEXT:
            return CreateVTKObjectText();
            break;
        default:
            return CreateVTKObjectFromFile();
            break;
        }

        return true;
    }

	inline bool CleanupVTKObjects(void) 
	{
        //if (this->SurfaceActor) {
        //    this->SurfaceActor->Delete();
        //    this->SurfaceActor = 0;
        //}
        //if (this->SurfaceMapper) {
        //    this->SurfaceMapper->Delete();
        //    this->SurfaceMapper = 0;
        //}
        //if (this->SurfaceReader) {
        //    this->SurfaceReader->Delete();
        //    this->SurfaceReader = 0;
        //}
        //if (this->TextActor) {
        //    this->TextActor->Delete();
        //    this->TextActor = 0;
        //}
		return true;
    }

    inline bool UpdateVTKObjects(void) {
        return true;
    }

    inline void SetColor(double r, double g, double b) {
        SurfaceActor->GetProperty()->SetDiffuseColor(r, g, b);
    }

	inline double GetOpacity()
	{
		return SurfaceActor->GetProperty()->GetOpacity();
	}

    inline void SetOpacity(double opacity) {
        SurfaceActor->GetProperty()->SetOpacity(opacity);
		if(TextActor)
	        TextActor->GetProperty()->SetOpacity(opacity);
    }

    inline void SetText(std::string text)
    {
        // Create a vector text
        vtkVectorText* vecText = vtkVectorText::New();
        vecText->SetText(text.c_str());
        vtkPolyDataMapper* txtMapper = vtkPolyDataMapper::New();

        if(text.size() > 0 && text.length() > 0)
        {
            vtkLinearExtrusionFilter* extrude = vtkLinearExtrusionFilter::New();
            extrude->SetInputConnection( vecText->GetOutputPort());
            extrude->SetExtrusionTypeToNormalExtrusion();
            extrude->SetVector(0, 0, 1 );
            extrude->SetScaleFactor (0.8);
			if(Geometry == TEXT)
				extrude->SetScaleFactor (0.0);

            txtMapper->SetInputConnection( extrude->GetOutputPort());
        }else
        {
            txtMapper->SetInputConnection( vecText->GetOutputPort());
        }
        TextActor->SetMapper(txtMapper);
    }

    bool Valid;
    bool Visible;
    vctFrm3 HomePositionUI3;
    typedef std::map<int,vctFrm3> vctFrm3MapType;
    int PreviousIndex;
    vctFrm3MapType PreviousPositions;

protected:
    std::string InputFile;
    vtkSmartPointer<vtkPolyDataReader> SurfaceReader;
    vtkSmartPointer<vtkPolyDataMapper> SurfaceMapper;
    vtkSmartPointer<vtkOpenGLActor> SurfaceActor;
    vtkSmartPointer<vtkOpenGLActor> TextActor;
    GeometryType Geometry;
    double Size;
};


CMN_DECLARE_SERVICES_INSTANTIATION(ManualRegistrationSurfaceVisibleStippleObject);
CMN_IMPLEMENT_SERVICES(ManualRegistrationSurfaceVisibleStippleObject);

ManualRegistration::ManualRegistration(const std::string & name):
    ui3BehaviorBase(std::string("ManualRegistration::") + name, 0),
    VisibleList(0),
    VisibleListECM(0),
    VisibleListECMRCM(0),
    VisibleListVirtual(0),
    VisibleListReal(0),
	UpdateExternalTransformation(0)
{
    // add video source interfaces
    //AddStream(svlTypeImageRGBA, "StereoVideo");

    VisibleList = new ui3VisibleList("ManualRegistration");
    VisibleListECM = new ui3VisibleList("ManualRegistrationECM");
    VisibleListECMRCM = new ui3VisibleList("ManualRegistrationECMRCM");
    VisibleListVirtual = new ui3VisibleList("ManualRegistrationVirtual");
	VisibleListVirtualGlobal = new ui3VisibleList("ManualRegistrationVirtualGlobal");
    VisibleListReal = new ui3VisibleList("ManualRegistrationReal");

    ManualRegistrationSurfaceVisibleStippleObject * model;
    ManualRegistrationSurfaceVisibleStippleObject * tumor;
    ManualRegistrationSurfaceVisibleStippleObject * text;
    ManualRegistrationSurfaceVisibleStippleObject * toolTip;
    ManualRegistrationSurfaceVisibleStippleObject * toolTop;
    this->Cursor = new ui3VisibleAxes;

#if CUBE_DEMO
    model = new ManualRegistrationSurfaceVisibleStippleObject("",ManualRegistrationSurfaceVisibleStippleObject::CUBE,25);
    tumor = new ManualRegistrationSurfaceVisibleStippleObject("",ManualRegistrationSurfaceVisibleStippleObject::SPHERE,5);

#else
#if SUPERFLAB_DEMO
	model = new ManualRegistrationSurfaceVisibleStippleObject("C:/Users/wenl/Projects/Data/20130701_Superflab_Zeego/Slicer/SuperflabPreopMesh.vtk");//"C:/Users/Wen/Images/20120307_TORS_Pig_Phantoms/T3/BoneModel.vtk");//"
	//tumor = new ManualRegistrationSurfaceVisibleStippleObject("C:/Users/wenl/Projects/Data/20130211_Superflab_preop/Slicer/SuperflabPreopMesh.vtk");
    //model = new ManualRegistrationSurfaceVisibleStippleObject("",ManualRegistrationSurfaceVisibleStippleObject::CUBE,25);
    tumor = new ManualRegistrationSurfaceVisibleStippleObject("C:/Users/wenl/Projects/Data/20130712_ISI_Richmon/20130711_Tongue2_ImageGuided_Segmentation_Tumor.vtk");
#else
	//model = new ManualRegistrationSurfaceVisibleStippleObject("C:/Users/wenl/Projects/Data/20130709_ISI_TORS_Porcine/run1/Lingual1.vtk");
    //tumor = new ManualRegistrationSurfaceVisibleStippleObject("C:/Users/wenl/Projects/Data/20130709_ISI_TORS_Porcine/run1/Tumor1_2.vtk");
	model = new ManualRegistrationSurfaceVisibleStippleObject("C:/Users/wenl/Projects/Data/dev/Experiment/Background.vtk");
    tumor = new ManualRegistrationSurfaceVisibleStippleObject("C:/Users/wenl/Projects/Data/dev/Experiment/Foreground.vtk");
#endif
	//model = new ManualRegistrationSurfaceVisibleStippleObject("/home/wen/Images/20121121_Maori/fixedSegmentation.vtk");
    //tumor = new ManualRegistrationSurfaceVisibleStippleObject("/home/wen/Images/20121017_T2/20121012_TORS_tongue2_intraop_targets.vtk");
    //tumor = new ManualRegistrationSurfaceVisibleStippleObject("C:/Users/Wen/Images/20130211_Superflab_preop/Slicer/SuperflabPreopMesh.vtk");
    text = new ManualRegistrationSurfaceVisibleStippleObject(".",ManualRegistrationSurfaceVisibleStippleObject::TEXT,5);
    //model = new ManualRegistrationSurfaceVisibleStippleObject("E:/Users/wliu25/MyCommon/data/TORS/TORS_tongue.vtk");
    //model = new ManualRegistrationSurfaceVisibleStippleObject("/home/wen/Images/BronchoBoy/Lychron20120809/Slicer/Tongue.vtk");
    //model = new ManualRegistrationSurfaceVisibleStippleObject("E:/Users/wliu25/MyCommon/data/RedSkull/Red_Skull_CT_TORS_ROI_Resample1.vtk");
    //model = new ManualRegistrationSurfaceVisibleStippleObject("E:/Users/wliu25/MyCommon/data/20120223_TORS_Pig_Phantoms/20120223_TORS_PigTongue_sc4_c191100_ROI_Resample0.6.vtk");
    //model = new ManualRegistrationSurfaceVisibleStippleObject("E:/Users/wliu25/MyCommon/data/20120307_TORS_Pig_Phantoms/T3Final/BoneModel.vtk");
    //tumor = new ManualRegistrationSurfaceVisibleStippleObject("/home/wen/Images/BronchoBoy/Lychron20120809/Slicer/BaseofTongueTumor00001.vtk");
    //tumor = new ManualRegistrationSurfaceVisibleStippleObject("E:/Users/wliu25/MyCommon/data/20120223_TORS_Pig_Phantoms/20120223_TORS_PigTongue_sc4_c191100_Targets.vtk");
    //tumor = new ManualRegistrationSurfaceVisibleStippleObject("E:/Users/wliu25/MyCommon/data/20120307_TORS_Pig_Phantoms/T3Final/TargetPlanning.vtk");
#endif

    VisibleObjects[MODEL] = model;
	VisibleObjects[TUMOR] = tumor;

#if IMPORT_MULTIPLE_TARGETS
    toolTip = new ManualRegistrationSurfaceVisibleStippleObject(".",ManualRegistrationSurfaceVisibleStippleObject::SPHERE,2);
    toolTop = new ManualRegistrationSurfaceVisibleStippleObject(".",ManualRegistrationSurfaceVisibleStippleObject::SPHERE,2);
#endif

    for (ManualRegistrationObjectType::iterator iter = VisibleObjects.begin();
         iter != VisibleObjects.end();
         iter++) {
		if(iter->first != MODEL)
		{
			VisibleListVirtual->Add(iter->second);
		}else
		{
			VisibleListVirtualGlobal->Add(iter->second);
		}
    }

	VisibleListECMRCM->Add(VisibleListVirtualGlobal);
    VisibleListECMRCM->Add(VisibleListVirtual);
    VisibleListECMRCM->Add(VisibleListReal);
    VisibleListECM->Add(VisibleListECMRCM);
    VisibleList->Add(VisibleListECM);


#if IMPORT_MULTIPLE_TARGETS
    VisibleList->Add(this->Cursor);
    VisibleObjectsVirtualFeedback[WRIST] = text;
    VisibleObjectsVirtualFeedback[TOOLTIP] = toolTip;
    VisibleObjectsVirtualFeedback[TOOLTOP] = toolTop;
    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsVirtualFeedback.begin();
         iter != VisibleObjectsVirtualFeedback.end();
         iter++) {
        VisibleList->Add(iter->second);
    }
#endif

    // Initialize boolean flags
    this->BooleanFlags[DEBUG] = false;
    this->BooleanFlags[VISIBLE] = true;
    this->BooleanFlags[PREVIOUS_MAM] = false;
    this->BooleanFlags[CAMERA_PRESSED] = false;
    this->BooleanFlags[CLUTCH_PRESSED] = false;
    this->BooleanFlags[UPDATE_FIDUCIALS] = false;
	this->BooleanFlags[TOOL_TRACKING_CORRECTION] = false;
    ResetButtonEvents();

    this->VisibleToggle = ALL;
    this->FiducialToggle = FIDUCIALS_REAL;
    this->MeanTRE = 0.0;
    this->MaxTRE = -1.0;
	this->MeanToolTRE = 0.0;
    this->MaxToolTRE = -1.0;
    this->MeanTREProjection = 0.0;
    this->MaxTREProjection = -1.0;
    this->TREFiducialCount = 0;
#if ERROR_ANALYSIS
	char file[50];
    sprintf(file,"tre%d.txt");
    TRE = fopen(file,"w");
    sprintf(file,"treProjection%d.txt");
    TREProjection = fopen(file,"w");
    sprintf(file,"treTool%d.txt");
    TRETool = fopen(file,"w");
#endif

    calibrationCount = 0;
	EndoscopeType = ZERO;
	m_continuousRegister = false;
	m_videoAugmentationVisibility = true;
	m_visionBasedTrackingVisibility = false;

	this->TTCorrectedTransformation = vctFrm3::Identity();
	this->wristToTipOffset = 9.7;//
	this->m_externalTransformationThreshold = 0.0;

    // Frames
    vctFrm3 wrist, wristToTip, wristToPinTip, wristToPinTop;
    wrist.Translation().Assign(vctDouble3(0.0, 0.0, 0.0));
    Frames[WRIST] = wrist;
    wristToTip.Translation().Assign(vctDouble3(0.0, 0.0, wristToTipOffset));
    Frames[TIP] = wristToTip;
    wristToPinTip.Translation().Assign(vctDouble3(0.0, 0.0, wristToTipOffset));//0.0, 18.0 (length of map pin), wristToTipOffset/2 or wristToTipOffset+10
    Frames[TOOLTIP] = wristToPinTip;
    wristToPinTop.Translation().Assign(vctDouble3(0.0, 0.0, wristToTipOffset));//0.0, -3 (offset), wristToTipOffset/2
    Frames[TOOLTOP] = wristToPinTop;
#if PROVIDED_INTERFACE
	// Configure Provided Interfaces
	SetupProvidedInterfaces();
#endif
}


ManualRegistration::~ManualRegistration()
{
}

double ManualRegistration::SetWristTipOffset(bool flag)
{ 
	if(flag)
		this->wristToTipOffset = 20.0; 
	else 
		this->wristToTipOffset = 9.7; 

    // Frames
    vctFrm3 wrist, wristToTip, wristToPinTip, wristToPinTop;
    wrist.Translation().Assign(vctDouble3(0.0, 0.0, 0.0));
    Frames[WRIST] = wrist;
    wristToTip.Translation().Assign(vctDouble3(0.0, 0.0, wristToTipOffset));
    Frames[TIP] = wristToTip;
    wristToPinTip.Translation().Assign(vctDouble3(0.0, 0.0, wristToTipOffset));//0.0, 18.0 (length of map pin), wristToTipOffset/2 or wristToTipOffset+10
    Frames[TOOLTIP] = wristToPinTip;
    wristToPinTop.Translation().Assign(vctDouble3(0.0, 0.0, wristToTipOffset));//0.0, -3 (offset), wristToTipOffset/2
    Frames[TOOLTOP] = wristToPinTop;

	return this->wristToTipOffset;
}


void ManualRegistration::PositionDepth(void)
{
    ManualRegistrationObjectType::iterator foundModel;
    foundModel = VisibleObjects.find(MODEL);
    if (foundModel == VisibleObjects.end()) {
        return;
    }

    std::cout << "PositionDepth" << std::endl;
    // compute depth of model
    vctFrm3 positionUI3 = (foundModel->second)->GetAbsoluteTransformation();
    prmPositionCartesianGet currentPosition;
    prmPositionCartesianSet newPosition;

    mtsExecutionResult result;
    result = GetPrimaryMasterPosition(currentPosition);
    //if (!result.IsOK()) {
    std::cerr << "PositionDepth, GetPrimaryMasterPosition: " << result << std::endl;
    //}
    newPosition.Goal().Assign(currentPosition.Position());
    newPosition.Goal().Translation().Z() = positionUI3.Translation().Z();
    result = SetPrimaryMasterPosition(newPosition);
    //if (!result.IsOK()) {
    std::cerr << "PositionDepth, SetPrimaryMasterPosition: " << result << std::endl;
    //}

    result = GetSecondaryMasterPosition(currentPosition);
    if (!result.IsOK()) {
        std::cerr << "PositionDepth, GetPrimaryMasterPosition: " << result << std::endl;
    }
    newPosition.Goal().Assign(currentPosition.Position());
    newPosition.Goal().Translation().Z() = positionUI3.Translation().Z();
    result = SetSecondaryMasterPosition(newPosition);
    if (!result.IsOK()) {
        std::cerr << "PositionDepth, SetPrimaryMasterPosition: " << result << std::endl;
    }
}

void ManualRegistration::UpdatePreviousPosition()
{
    bool debugLocal = false;

	// use model/background for global objects
    // get current position in UI3
    ManualRegistrationObjectType::iterator foundModel;
    foundModel = VisibleObjects.find(MODEL);
    if (foundModel == VisibleObjects.end()) {
        return;
    }
    //previous positions saved in ECMRCM
    (foundModel->second)->PreviousPositions[(foundModel->second)->PreviousPositions.size()+1] = (foundModel->second)->GetAbsoluteTransformation();
    (foundModel->second)->PreviousIndex = (foundModel->second)->PreviousPositions.size();
    if (this->BooleanFlags[DEBUG] && debugLocal) {
        std::cout << "Previous Index:" << (foundModel->second)->PreviousIndex << std::endl;
    }

	// use tumor for local objects
	// get current position in UI3
    foundModel = VisibleObjects.find(TUMOR);
    if (foundModel == VisibleObjects.end()) {
        return;
    }
    //previous positions saved in ECMRCM
    (foundModel->second)->PreviousPositions[(foundModel->second)->PreviousPositions.size()+1] = (foundModel->second)->GetAbsoluteTransformation();
    (foundModel->second)->PreviousIndex = (foundModel->second)->PreviousPositions.size();
    if (this->BooleanFlags[DEBUG] && debugLocal) {
        std::cout << "Previous Index:" << (foundModel->second)->PreviousIndex << std::endl;
    }

}

void ManualRegistration::PositionBack(void)
{
	// use model/background for global objects
    ManualRegistrationObjectType::iterator foundModel;
    foundModel = VisibleObjects.find(MODEL);
    if (foundModel == VisibleObjects.end())
        return;
    ManualRegistrationSurfaceVisibleStippleObject::vctFrm3MapType::iterator foundPosition;
    foundPosition = (foundModel->second)->PreviousPositions.find((foundModel->second)->PreviousIndex);
    if (foundPosition != (foundModel->second)->PreviousPositions.end()) {
        this->VisibleListVirtualGlobal->SetTransformation(this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse() * (foundModel->second)->PreviousPositions[(foundModel->second)->PreviousIndex]);
        if (this->BooleanFlags[DEBUG]) {
            std::cout << "Setting back to index:" << (foundModel->second)->PreviousIndex << std::endl;
        }
        (foundModel->second)->PreviousIndex--;
        if ((foundModel->second)->PreviousIndex < 0) {
            (foundModel->second)->PreviousIndex = 0;
        }
    }

	// use tumor for local objects
    foundModel = VisibleObjects.find(TUMOR);
    if (foundModel == VisibleObjects.end())
        return;
    foundPosition = (foundModel->second)->PreviousPositions.find((foundModel->second)->PreviousIndex);
    if (foundPosition != (foundModel->second)->PreviousPositions.end()) {
        this->VisibleListVirtual->SetTransformation(this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse() * (foundModel->second)->PreviousPositions[(foundModel->second)->PreviousIndex]);
        if (this->BooleanFlags[DEBUG]) {
            std::cout << "Setting back to index:" << (foundModel->second)->PreviousIndex << std::endl;
        }
        (foundModel->second)->PreviousIndex--;
        if ((foundModel->second)->PreviousIndex < 0) {
            (foundModel->second)->PreviousIndex = 0;
        }
    }
}


void ManualRegistration::PositionHome(void)
{
    ManualRegistrationObjectType::iterator foundModel;
    foundModel = VisibleObjects.find(MODEL);
    if (foundModel != VisibleObjects.end())
	{
        this->VisibleListVirtual->SetTransformation(this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse()*(foundModel->second)->HomePositionUI3);
		this->VisibleListVirtualGlobal->SetTransformation(this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse()*(foundModel->second)->HomePositionUI3);
	}
}

ManualRegistration::VisibleObjectType ManualRegistration::ToggleFiducials()
{
    switch(this->FiducialToggle)
    {
    case(MODEL):
        this->FiducialToggle = FIDUCIALS_REAL;
        this->BooleanFlags[UPDATE_FIDUCIALS] = true;
	    if (this->BooleanFlags[DEBUG]) {
		    std::cout << "Toggling Fiducial: FIDUCIALS_REAL" << std::endl;
		}
        break;
    case(FIDUCIALS_REAL):
        this->FiducialToggle = TARGETS_REAL;
        this->BooleanFlags[UPDATE_FIDUCIALS] = true;
	    if (this->BooleanFlags[DEBUG]) {
		    std::cout << "Toggling Fiducial: CALIBRATION_REAL" << std::endl;
		}
        break;
    case(TARGETS_REAL):
        this->FiducialToggle = MODEL;
        this->BooleanFlags[UPDATE_FIDUCIALS] = true;
	    if (this->BooleanFlags[DEBUG]) {
		    std::cout << "Toggling Fiducial: CALIBRATION_REAL" << std::endl;
		}
        break;
    case(CALIBRATION_REAL):
        this->FiducialToggle = MODEL;
        this->BooleanFlags[UPDATE_FIDUCIALS] = false;
	    if (this->BooleanFlags[DEBUG]) {
		    std::cout << "Toggling Fiducial: MODEL" << std::endl;
		}
		break;
    default:
        this->FiducialToggle = MODEL;
        this->BooleanFlags[UPDATE_FIDUCIALS] = false;
	    if (this->BooleanFlags[DEBUG]) {
		    std::cout << "Toggling Fiducial: MODEL" << std::endl;
		}
        break;
    }

	return this->FiducialToggle;
}

bool ManualRegistration::SetVideoAugmentationVisibility(bool visible)
{
	m_videoAugmentationVisibility = visible;
	UpdateVisibleList();
	return m_videoAugmentationVisibility;
}

ManualRegistration::VisibleObjectType ManualRegistration::SetVisibleToggle(VisibleObjectType visibility)
{
	this->VisibleToggle = visibility; 
	UpdateVisibleList(); 
	return this->VisibleToggle;
}

ManualRegistration::VisibleObjectType ManualRegistration::ToggleVisibility(void)
{
    bool localDebug = true;
    ManualRegistrationObjectType::iterator foundTumor;
    foundTumor = VisibleObjects.find(TUMOR);

    switch(this->VisibleToggle)
    {
    case(ALL):
        this->VisibleToggle = NO_FIDUCIALS;
        break;
    case(NO_FIDUCIALS):
        this->VisibleToggle = MODEL;
        break;
    case(MODEL):
        if(foundTumor != VisibleObjects.end() || VisibleObjectsVirtualTumors.size() > 0)
            this->VisibleToggle = TUMOR;
        else
            this->VisibleToggle = TARGETS_REAL;
        break;
    case(TUMOR):
        if(VisibleObjectsVirtualFeedback.size() > 0)
            this->VisibleToggle = MODEL_ONLY;
        else
            this->VisibleToggle = TARGETS_REAL;
        break;
    case(MODEL_ONLY):
        this->VisibleToggle = TUMOR_ONLY;
        break;
    case(TUMOR_ONLY):
        this->VisibleToggle = FIDUCIALS_REAL;
        break;
    case(TARGETS_REAL):
        this->VisibleToggle = FIDUCIALS_REAL;
        break;
    case(FIDUCIALS_REAL):
        this->VisibleToggle = NONE;
        break;
    default:
        this->VisibleToggle = ALL;
        break;
    }
    if (this->BooleanFlags[DEBUG] && localDebug) {
        std::cout << "Toggling Visible: " << this->VisibleToggle << std::endl;
    }
    UpdateVisibleList();

	return this->VisibleToggle;
}


void ManualRegistration::ConfigureMenuBar(void)
{
this->MenuBar->AddClickButton("PositionBack",
                                  0,
                                  "undo.png",
                                  &ManualRegistration::PositionBack,
                                  this);
    this->MenuBar->AddClickButton("PositionHome",
                                  1,
                                  "triangle.png",
                                  &ManualRegistration::PositionHome,
                                  this);
    //this->MenuBar->AddClickButton("ToggleFiducials",
    //                              2,
    //                              "map.png",
    //                              &ManualRegistration::ToggleFiducials,
    //                              this);
    //this->MenuBar->AddClickButton("Register",
    //                              3,
    //                              "move.png",
    //                              &ManualRegistration::Register,
    //                              this);
    //this->MenuBar->AddClickButton("ToggleVisibility",
    //                              4,
    //                              "sphere.png",
    //                              &ManualRegistration::ToggleVisibility,
    //                              this);
    this->MenuBar->AddClickButton("PositionDepth",
                                  5,
                                  "iconify-top-left.png",
                                  &ManualRegistration::PositionDepth,
                                  this);
}

void ManualRegistration::UpdateFiducials(void)
{
    prmPositionCartesianGet positionLeft, positionRight, position;;
    this->GetPrimaryMasterPosition(positionRight);
    this->GetSecondaryMasterPosition(positionLeft);
    ManualRegistrationSurfaceVisibleStippleObject* closestFiducial = NULL;
    VisibleObjectType type = ALL;

    switch(this->FiducialToggle)
    {
    case(TARGETS_REAL):
        if(!this->BooleanFlags[BOTH_BUTTON_PRESSED] && this->BooleanFlags[RIGHT_BUTTON] && !this->BooleanFlags[LEFT_BUTTON]) {
            type = TARGETS_REAL;
            position = positionRight;
        }else if(!this->BooleanFlags[BOTH_BUTTON_PRESSED] && !this->BooleanFlags[RIGHT_BUTTON] && this->BooleanFlags[LEFT_BUTTON])
        {
            type = TARGETS_VIRTUAL;
            position = positionLeft;
        }
        break;
    case(FIDUCIALS_REAL):
        if(!this->BooleanFlags[BOTH_BUTTON_PRESSED] && this->BooleanFlags[RIGHT_BUTTON] && !this->BooleanFlags[LEFT_BUTTON]) {
            type = FIDUCIALS_REAL;
            position = positionRight;
        }else if(!this->BooleanFlags[BOTH_BUTTON_PRESSED] && !this->BooleanFlags[RIGHT_BUTTON] && this->BooleanFlags[LEFT_BUTTON])
        {
            type = FIDUCIALS_VIRTUAL;
            position = positionLeft;
        }
        break;
    case(CALIBRATION_REAL):
        //pinch right - just record right tool virtual location
        if(!this->BooleanFlags[BOTH_BUTTON_PRESSED] && this->BooleanFlags[RIGHT_BUTTON] && !this->BooleanFlags[LEFT_BUTTON]) {
            type = CALIBRATION_VIRTUAL;
            position = positionRight;
            //pinch left - add both real and virtual location for right tool
        }else if(!this->BooleanFlags[BOTH_BUTTON_PRESSED] && !this->BooleanFlags[RIGHT_BUTTON] && this->BooleanFlags[LEFT_BUTTON])
        {
            type = CALIBRATION_REAL;
            position = positionRight;
        }
        break;
    default:
        //std::cerr << "Doing nothing to update fiducial of this type: " << type << std::endl;
        return;
    }

    int index;
    closestFiducial = FindClosestFiducial(position.Position(),type,index);
    if(closestFiducial != NULL && closestFiducial->Valid)
    {
        closestFiducial->Valid = false;
        if(this->FiducialToggle == CALIBRATION_REAL)
        {
            std::cout << "Trying to invalidate virual calibration at " << index << std::endl;
            ManualRegistrationObjectType::iterator foundObject = VisibleObjectsVirtualCalibration.find(index);
            if(foundObject != VisibleObjectsVirtualCalibration.end())
                (foundObject->second)->Valid = false;
        }
        ResetButtonEvents();
        UpdateVisibleList();
    }else
    {
        if(this->FiducialToggle == CALIBRATION_REAL)
        {
            if(type == CALIBRATION_REAL)
            {
                AddFiducial(position.Position(),type);
                AddFiducial(GetCurrentCorrectedCartesianPositionSlave(),CALIBRATION_VIRTUAL);
            }
            else if(type == CALIBRATION_VIRTUAL)
            {
                AddFiducial(GetCurrentCorrectedCartesianPositionSlave(),CALIBRATION_VIRTUAL);
            }
        }else
        {
            AddFiducial(position.Position(),type);
        }
        //std::cerr << "MaM position: " << position.Position().Translation() << " slave: " << GetCurrentCorrectedCartesianPositionSlave().Translation() << std::endl;
    }
}

void ManualRegistration::UpdateFiducialRegistration()
{
	m_registrationGUIRight->ComputationalStereo(m_registrationGUILeft);
	vctFrm3 position;

	std::map<int, svlFilterImageRegistrationGUI::PointInternal> points = m_registrationGUIRight->GetRegistrationPoints();
	svlFilterImageRegistrationGUI::PointInternal point;
	for (int i=0;i<std::min<int>(3,points.size());i++) 
	{
		point = points[i];
		position.Translation().Assign(vctDouble3(point.pointLocation.x,point.pointLocation.y,point.pointLocation.z));
		SetFiducial(position,ManualRegistration::VisibleObjectType::FIDUCIALS_REAL,point.Valid,i);
	}

	points = m_registrationGUIRight->GetTargetPoints();
	for (int i=0;i<points.size();i++) 
	{
		point = points[i];
		position.Translation().Assign(vctDouble3(point.pointLocation.x,point.pointLocation.y,point.pointLocation.z));
		SetFiducial(position,ManualRegistration::VisibleObjectType::TARGETS_REAL,point.Valid,i);
		SetFiducial(position,ManualRegistration::VisibleObjectType::TOOLS_REAL,point.Valid,i);
	}
}

void ManualRegistration::Update3DFiducialCameraPressed(void)
{
    vctDynamicVector<vct3> fiducialsVirtual, fiducialsReal;
    GetFiducials(fiducialsVirtual, fiducialsReal,FIDUCIALS_REAL, UI3);

    for(int i=0;i<(int)fiducialsReal.size();i++)
    {
		m_registrationGUIRight->Update3DPosition(cv::Point3f(fiducialsReal[i].X(),fiducialsReal[i].Y(),fiducialsReal[i].Z()),i);
		m_registrationGUILeft->Update3DPosition(cv::Point3f(fiducialsReal[i].X(),fiducialsReal[i].Y(),fiducialsReal[i].Z()),i);
	}
}

void ManualRegistration::Follow()
{
	vctDouble3 axis;
	axis.Assign(1.0,0.0,0.0);
	double angle;
	angle = 0.0;
    vctFrm3 currentUI3toECMRCM, handleCenterECMRCM, displacementECMRCM, displacementECMRCMT, displacementECMRCMR;

	ManualRegistrationObjectType::iterator foundObject = VisibleObjects.find(TUMOR);
	if (!m_continuousRegister && foundObject != VisibleObjects.end() && UpdateExternalTransformation) {
		currentUI3toECMRCM = this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse();
		handleCenterECMRCM.Translation() = (currentUI3toECMRCM*(foundObject->second)->GetAbsoluteTransformation()).Translation();

		// Translation.
		displacementECMRCMT.Translation() = ExternalTransformation.Translation();

		// Rotation (currently not used)
		displacementECMRCMR.Rotation() = ExternalTransformation.Rotation();

		// so we apply rotation on center of handles
		displacementECMRCM = displacementECMRCMT * handleCenterECMRCM * displacementECMRCMR * handleCenterECMRCM.Inverse();

		if(this->BooleanFlags[DEBUG]){
			std::cout << " ExternalTransformation " << ExternalTransformation << std::endl;
			std::cout << " handle " << handleCenterECMRCM.Translation() << std::endl;
			std::cout << " displacementECMRCMT " << displacementECMRCMT.Translation() << std::endl;
			std::cout << " displacementECMRCMR " << displacementECMRCMR.Rotation() << std::endl;
			std::cout << " displacementECMRCM " << displacementECMRCM << std::endl;
		}
		UpdatePreviousPosition();
		// apply transformation in ECMRCM
		this->VisibleListVirtual->SetTransformation(displacementECMRCM * this->VisibleListVirtual->GetTransformation());
		// apply to global
		this->VisibleListVirtualGlobal->SetTransformation(displacementECMRCM * this->VisibleListVirtualGlobal->GetTransformation());
		// apply to real
		this->VisibleListReal->SetTransformation(displacementECMRCM * this->VisibleListReal->GetTransformation());

#if PROVIDED_INTERFACE
		UpdateProvidedInterfaces();
#endif
		ExternalTransformation = vctFrm3::Identity();
		UpdateExternalTransformation = false;
	}

	//if(m_registrationGUIRight->GetValidity() && m_registrationGUILeft->GetValidity())
	//{
		if(m_continuousRegister)
		{
			UpdateFiducialRegistration();
			if(m_registrationGUIRight->GetValidity() && m_registrationGUILeft->GetValidity())
			{
				Register(TUMOR);
				m_visionBasedTrackingVisibility = true;
			}else
			{
				//m_visionBasedTrackingVisibility = false;
			}
			UpdateVisibleList();
		}
	//}
}

void ManualRegistration::FollowMaster(void)
{
    prmPositionCartesianGet positionLeft, positionRight;
    vctFrm3 displacementUI3, displacementUI3T, displacementUI3R;
    vctFrm3 visibleListVirtualPositionUI3, visibleListVirtualPositionUI3New;
    vctFrm3 displacementECMRCM, displacementECMRCMT, displacementECMRCMR;

    this->GetPrimaryMasterPosition(positionRight);
    this->GetSecondaryMasterPosition(positionLeft);

    vctFrm3 currentUI3toECMRCM = this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse();
    vctDouble3 initialMasterRightECMRCM = currentUI3toECMRCM * InitialMasterRight;
    vctDouble3 initialMasterLeftECMRCM = currentUI3toECMRCM * InitialMasterLeft;
    vctDouble3 positionRightECMRCM = currentUI3toECMRCM * positionRight.Position().Translation();
    vctDouble3 positionLeftECMRCM = currentUI3toECMRCM * positionLeft.Position().Translation();

    if (!this->BooleanFlags[BOTH_BUTTON_PRESSED]
            && this->BooleanFlags[RIGHT_BUTTON]
            && !this->BooleanFlags[LEFT_BUTTON]) {
        //translation only using right
        displacementECMRCM.Translation() = positionRightECMRCM - initialMasterRightECMRCM;
    }
    else if (!this->BooleanFlags[BOTH_BUTTON_PRESSED]
             && !this->BooleanFlags[RIGHT_BUTTON]
             && this->BooleanFlags[LEFT_BUTTON]) {
        //translation only using left
        displacementECMRCM.Translation() = positionLeftECMRCM - initialMasterLeftECMRCM;
    } else if (this->BooleanFlags[BOTH_BUTTON_PRESSED]) {

        //rotation using both
        vctDouble3 axis;
        axis.Assign(1.0,0.0,0.0);
        double angle;
        angle = 0.0;
        double object_displacement[3], object_rotation[4];
        vctDouble3 translation;

        vctFrm3 handleCenterECMRCM;
        handleCenterECMRCM.Translation().SumOf(initialMasterRightECMRCM, initialMasterLeftECMRCM);
        handleCenterECMRCM.Translation().Divide(2.0);

        ComputeTransform(initialMasterRightECMRCM.Pointer(),
                         initialMasterLeftECMRCM.Pointer(),
                         positionRightECMRCM.Pointer(),
                         positionLeftECMRCM.Pointer(),
                         object_displacement, object_rotation);

        // Set the Translation.
        translation.Assign(object_displacement);
        // visibleListVirtualPositionUI3.Rotation().ApplyInverseTo(translation, translationInWorld);
        //displacementUI3T.Translation()= translation /*InWorld*/;
        displacementECMRCMT.Translation()= translation;
        // hard coded scale
        translation *= 0.1;

        // Set the Rotation.
        angle = object_rotation[0];
        // hard coded scale to dampen rotation
        angle *= 0.5;
        axis.Assign(object_rotation+1);
        // visibleListVirtualPositionUI3.Rotation().ApplyInverseTo(axis, axisInWorld);
        //displacementUI3R.Rotation().From(vctAxAnRot3(axis /*InWorld*/, angle));
        displacementECMRCMR.Rotation().From(vctAxAnRot3(axis, angle));

        // so we apply rotation on center of handles
        displacementECMRCM = displacementECMRCMT * handleCenterECMRCM * displacementECMRCMR * handleCenterECMRCM.Inverse();
		if(this->BooleanFlags[DEBUG])
		{
			std::cout << " handle " << handleCenterECMRCM.Translation() << std::endl;
			std::cout << " displacementECMRCMT " << displacementECMRCMT.Translation() << std::endl;
			std::cout << " displacementECMRCMR " << displacementECMRCMR.Rotation() << std::endl;
		}
    }

    // save cursor positions
    InitialMasterRight = positionRight.Position().Translation();
    InitialMasterLeft = positionLeft.Position().Translation();

    // apply transformation in ECMRCM
    this->VisibleListVirtual->SetTransformation(displacementECMRCM * this->VisibleListVirtual->GetTransformation());
	this->VisibleListVirtualGlobal->SetTransformation(displacementECMRCM * this->VisibleListVirtualGlobal->GetTransformation());

	if(this->BooleanFlags[DEBUG])
		std::cout << " displacementECMRCM " << displacementECMRCM << std::endl;

#if PROVIDED_INTERFACE
	UpdateProvidedInterfaces();
#endif
    //if(this->BooleanFlags[DEBUG])
    //    std::cerr << "VisibleListVirtual " << this->VisibleListVirtual->GetAbsoluteTransformation().Translation() << " rel " << this->VisibleListVirtual->GetTransformation().Translation() << std::endl;

}



/*!
Compute the object transform from the motion of two grabbed control points.
@param pointa               Right control position.
@param pointb               Left control position.
@param point1               Right cursor pos.
@param point2               Left cursor pos.
@param object_displacement  [dx, dy, dz]
@param object_rotation      [angle, axis_x, axis_y, axis_z]
Author(s):  Simon DiMaio
*/
void ManualRegistration::ComputeTransform(double pointa[3], double pointb[3],
                                          double point1[3], double point2[3],
                                          double object_displacement[3],
                                          double object_rotation[4])
{
    double v1[3], v2[3], v1norm, v2norm, wnorm;
    double w[3], angle, dotarg;

    //cout << "pointa: " << pointa[0] << " " << pointa[1] << " " << pointa[2] << endl;
    //cout << "pointb: " << pointb[0] << " " << pointb[1] << " " << pointb[2] << endl;

    // v1 = ((pb-pa)/norm(pb-pa))
    v1[0] = pointb[0]-pointa[0];
    v1[1] = pointb[1]-pointa[1];
    v1[2] = pointb[2]-pointa[2];
    v1norm = sqrt(v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2]);
    if (v1norm>cmnTypeTraits<double>::Tolerance()) {
        v1[0] /= v1norm;
        v1[1] /= v1norm;
        v1[2] /= v1norm;
    }

    // v2 = ((p2-p1)/norm(p2-p1))
    v2[0] = point2[0]-point1[0];
    v2[1] = point2[1]-point1[1];
    v2[2] = point2[2]-point1[2];
    v2norm = sqrt(v2[0]*v2[0]+v2[1]*v2[1]+v2[2]*v2[2]);
    if (v2norm>cmnTypeTraits<double>::Tolerance())
    {
        v2[0] /= v2norm;
        v2[1] /= v2norm;
        v2[2] /= v2norm;
    }

    // w = (v1 x v2)/norm(v1 x v2)
    w[0] = v1[1]*v2[2] - v1[2]*v2[1];
    w[1] = v1[2]*v2[0] - v1[0]*v2[2];
    w[2] = v1[0]*v2[1] - v1[1]*v2[0];
    wnorm = sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]);
    if (wnorm> cmnTypeTraits<double>::Tolerance()) {
        w[0] /= wnorm;
        w[1] /= wnorm;
        w[2] /= wnorm;
    }
    else {
        w[0] = 1.0;
        w[1] = w[2] = 0.0;
    }

    // theta = arccos(v1.v2/(norm(v1)*norm(v2))
    dotarg = v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
    if (dotarg>-1.0 && dotarg<1.0) {
        angle = acos(dotarg);
    } else {
        angle = 0.0;
    }
    //if (CMN_ISNAN(angle)) angle=0.0;

    // std::cout << "v1: " << v1[0] << " " << v1[1] << " " << v1[2] << std::endl;
    // std::cout << "v2: " << v2[0] << " " << v2[1] << " " << v2[2] << std::endl;
    // std::cout << "w: " << w[0] << " " << w[1] << " " << w[2] << " angle: " << angle*180.0/cmnPI << std::endl;

    // Set object pose updates.
    object_displacement[0] = (point1[0]+point2[0])/2 - (pointa[0]+pointb[0])/2;
    object_displacement[1] = (point1[1]+point2[1])/2 - (pointa[1]+pointb[1])/2;
    object_displacement[2] = (point1[2]+point2[2])/2 - (pointa[2]+pointb[2])/2;

    object_rotation[0] = angle;
    object_rotation[1] = w[0];
    object_rotation[2] = w[1];
    object_rotation[3] = w[2];
}

bool ManualRegistration::RunForeground(void)
{
    if (this->Manager->MastersAsMice() != this->BooleanFlags[PREVIOUS_MAM]) {
        this->BooleanFlags[PREVIOUS_MAM] = this->Manager->MastersAsMice();
        ResetButtonEvents();
    }

    // detect transition, should that be handled as an event?
    // State is used by multiple threads ...
    if (this->State != this->PreviousState) {
        std::cerr << "Entering RunForeground" << std::endl;
        this->PreviousState = this->State;
    }

    // detect active mice
    if (this->BooleanFlags[UPDATE_FIDUCIALS]) {
        UpdateFiducials();
    } else if (this->BooleanFlags[LEFT_BUTTON] || this->BooleanFlags[RIGHT_BUTTON]) {
        FollowMaster();
    }

    // Cursor & Feedback
    int index;
    ManualRegistrationSurfaceVisibleStippleObject* closestFiducial = NULL;
    closestFiducial = FindClosestFiducial(Frames[WRIST]*GetCurrentCorrectedCartesianPositionSlave()*Frames[TOOLTIP],TUMOR,index);

    return true;
}

bool ManualRegistration::RunBackground(void)
{
    // detect transition
    if (this->State != this->PreviousState) {
        this->PreviousState = this->State;
    }

    // Cursor & Feedback
    int index;
    ManualRegistrationSurfaceVisibleStippleObject* closestFiducial = NULL;
    closestFiducial = FindClosestFiducial(Frames[WRIST]*GetCurrentCorrectedCartesianPositionSlave()*Frames[TOOLTIP],TUMOR,index);

    return true;
}

bool ManualRegistration::RunNoInput(void)
{
    if (this->Manager->MastersAsMice() != this->BooleanFlags[PREVIOUS_MAM]) {
        this->BooleanFlags[PREVIOUS_MAM] = this->Manager->MastersAsMice();
        ResetButtonEvents();
    }

    // detect transition
    if (this->State != this->PreviousState) {
        this->PreviousState = this->State;
    }

    // prepare to drop marker if clutch and right MTM are pressed
    if ((this->BooleanFlags[CLUTCH_PRESSED] && this->BooleanFlags[RIGHT_BUTTON] && !this->BooleanFlags[RIGHT_BUTTON_RELEASED])
            &&((this->FiducialToggle == TARGETS_REAL || this->FiducialToggle == FIDUCIALS_REAL) || this->FiducialToggle == CALIBRATION_REAL))
    {
        //Add fiducial
        std::cerr << "Add marker with slave" << std::endl;
        if(this->FiducialToggle == CALIBRATION_REAL)
            AddFiducial(GetCurrentCorrectedCartesianPositionSlave(),CALIBRATION_VIRTUAL);
        else
            AddFiducial(Frames[WRIST]*GetCurrentCorrectedCartesianPositionSlave()*Frames[TOOLTIP],this->FiducialToggle);
    }

    //check if the objects should be updated
    if (this->BooleanFlags[CAMERA_PRESSED]) {
        UpdateCameraPressed();
    }

#if PROVIDED_INTERFACE
	// Update provided Interfaces
	UpdateProvidedInterfaces();
#endif

#if EXTERNAL_TRACKING
	// Update from external
	Follow();
#endif

    // update cursor
	double thresholdA = 2.0;
	vctFrm3 correctedTransformation = Frames[WRIST]*GetCurrentCorrectedCartesianPositionSlave();
	vctFrm3 kinematicsTransformation = Frames[WRIST]*GetCurrentCartesianPositionSlave();
    ManualRegistrationObjectType::iterator foundTooltip, foundTooltop, foundWrist;
    this->Cursor->SetTransformation(correctedTransformation*Frames[TIP]);
    foundTooltip = VisibleObjectsVirtualFeedback.find(TOOLTIP);
    if (foundTooltip != VisibleObjectsVirtualFeedback.end()) {
        (foundTooltip->second)->SetTransformation(correctedTransformation*Frames[TOOLTIP]);
    }
    foundTooltop = VisibleObjectsVirtualFeedback.find(TOOLTOP);
    if (foundTooltop != VisibleObjectsVirtualFeedback.end()) {
        (foundTooltop->second)->SetTransformation(correctedTransformation*Frames[TOOLTOP]);
    }
    foundWrist = VisibleObjectsVirtualFeedback.find(WRIST);
    if (foundWrist != VisibleObjectsVirtualFeedback.end()) {
       (foundWrist->second)->SetPosition(correctedTransformation.Translation());
    }
	if(foundTooltip == VisibleObjectsVirtualFeedback.end() || foundTooltop == VisibleObjectsVirtualFeedback.end() || foundWrist == VisibleObjectsVirtualFeedback.end())
		return true;

    // update depth cues for certain toggles
    if(VisibleToggle == ALL || VisibleToggle == NO_FIDUCIALS || VisibleToggle == TUMOR || VisibleToggle == FIDUCIALS_REAL || VisibleToggle == FIDUCIALS_VIRTUAL)
    {
        int index;
        ManualRegistrationSurfaceVisibleStippleObject* closestFiducial = NULL;
        closestFiducial = FindClosestFiducial(correctedTransformation*Frames[TOOLTIP],TUMOR,index, (double)MARGIN_RADIUS+thresholdA);
        char buffer[33];
        vctDouble3 dist;
        if(closestFiducial != NULL && closestFiducial->Valid)
        {
            dist.DifferenceOf((correctedTransformation*Frames[TOOLTIP]).Translation(), closestFiducial->GetAbsoluteTransformation().Translation());
            double abs = dist.Norm();
#if MARGIN_RADIUS
			if(abs < (double)MARGIN_RADIUS-thresholdA)
			{		
				closestFiducial->SetColor(255.0/255.0, 0.0, 0.0);
				//(foundTooltip->second)->SetColor(255.0/255.0, 0.0, 0.0);
			}else if((abs > (double)MARGIN_RADIUS-thresholdA)&&(abs <= (double)MARGIN_RADIUS))
			{
				closestFiducial->SetColor(255.0/255.0, 255.0/255.0, 0.0);
				//(foundTooltip->second)->SetColor(255.0/255.0, 255.0/255.0, 0.0);
	            
			}else
			// this is only if (double)MARGIN_RADIUS +thresholdA since we bound fiducial search to this
			{
				closestFiducial->SetColor(0.0, 255.0/255.0, 0.0);
				//(foundTooltip->second)->SetColor(0.0, 255.0/255.0, 0.0);
			}
			sprintf(buffer, "%2.2lf", abs-(double)MARGIN_RADIUS);
#else
            if(abs < 4.0)
            {
                if(abs < 1.0)
                    closestFiducial->SetColor(1.0, 0.0, 0.25);
                else if(abs < 2.0)
                    closestFiducial->SetColor(255.0/255.0, 255.0/255.0, 0.0);
                else if(abs < 4.0)
                    closestFiducial->SetColor(153.0/255.0, 255.0/255.0, 153.0/255.0);
            }
            sprintf(buffer, "%2.2lf", abs);
#endif
		}else
        {
			(foundTooltip->second)->SetColor(0.0, 0.0, 0.0);
            sprintf(buffer, ".");
        }
#if TOOL_TRACKING
        (foundWrist->second)->SetText(buffer);
#endif
	}


    return true;
}


double ManualRegistration::UpdateExternalTransformationThreshold(double delta)
{
	double value = 	this->m_externalTransformationThreshold;
	std::min((double)0,value += delta);
	this->m_externalTransformationThreshold = value;
	return this->m_externalTransformationThreshold;
}


void ManualRegistration::UpdateCameraPressed()
{
    vctFrm3 currentECMtoECMRCM = GetCurrentECMtoECMRCM();
    this->VisibleListECMRCM->SetTransformation(currentECMtoECMRCM);
#if PROVIDED_INTERFACE
		UpdateProvidedInterfaces();
#endif
	Update3DFiducialCameraPressed();
    //if(this->BooleanFlags[DEBUG])
    //    std::cerr << "VisibleListECMRCM " << this->VisibleListECMRCM->GetAbsoluteTransformation().Translation() << " rel " << this->VisibleListECMRCM->GetTransformation().Translation() << std::endl;
}


void ManualRegistration::OnQuit(void)
{
}


void ManualRegistration::OnStart(void)
{
    vctFrm3 homePosition, modelHomePosition, currentECMtoECMRCM, staticECMtoUI3;
    ManualRegistrationObjectType::iterator foundObject;

    // VirtualList - Set root transformation at origin
    homePosition.Translation().Assign(0.0,0.0,0.0);
    this->VisibleList->SetTransformation(homePosition);

    // VisibleListECM - Rotate by pi in y to be aligned to console
    staticECMtoUI3.Rotation().From(vctAxAnRot3(vctDouble3(0.0,1.0,0.0), cmnPI));
    this->VisibleListECM->SetTransformation(staticECMtoUI3.Inverse());

    // VisibleListECMRCM
    currentECMtoECMRCM = GetCurrentECMtoECMRCM();
    this->VisibleListECMRCM->SetTransformation(currentECMtoECMRCM);

    // VisibleListVirtual
#if CUBE_DEMO
    // VTK meshes harded coded start location at (0,0,-200)
    modelHomePosition.Translation().Assign(0.0,0.0,-200.0);
#else
    // VTK meshes harded coded start location at (0,0,-200)
    //modelHomePosition.Translation().Assign(100.0,100.0,-200.0);
    modelHomePosition.Translation().Assign(0.0,0.0,0.0);
#endif
    this->VisibleListVirtual->SetTransformation(this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse()*modelHomePosition);
	this->VisibleListVirtualGlobal->SetTransformation(this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse()*modelHomePosition);

    // setup VTK model
    foundObject = VisibleObjects.find(MODEL);
    if (foundObject != VisibleObjects.end()) {
        (foundObject->second)->SetColor(1.0, 0.49, 0.25);
        (foundObject->second)->HomePositionUI3 = modelHomePosition;
        (foundObject->second)->SetOpacity(0.25);
    }
    foundObject = VisibleObjects.find(TUMOR);
    if (foundObject != VisibleObjects.end()) {
        (foundObject->second)->SetColor(1.0, 0.0, 0.25);
        (foundObject->second)->HomePositionUI3 = modelHomePosition;
        (foundObject->second)->SetOpacity(0.1);
    }

    for (ManualRegistrationObjectType::iterator iter = VisibleObjects.begin();
         iter != VisibleObjects.end();
         iter++) {
        (iter->second)->Visible = true;
        (iter->second)->Valid = true;
        (iter->second)->Show();
    }

#if IMPORT_FIDUCIALS
#if CUBE_DEMO
    //ImportFiducialFile("E:/Users/wliu25/MyCommon/data/TORS/CubeCTFids.fcsv", FIDUCIALS_VIRTUAL);
    //ImportFiducialFile("E:/Users/wliu25/MyCommon/data/TORS/CubeCTTargets.fcsv", TARGETS_VIRTUAL);
#else
    //20121017_T2
    //ImportFiducialFile("/home/wen/Images/20121017_T2/demons_fids_flip.fcsv",FIDUCIALS_VIRTUAL);
    //ImportFiducialFile("/home/wen/Images/20121023_T4/demons_fids.fcsv",FIDUCIALS_VIRTUAL);
    //ImportFiducialFile("/home/wen/Images/20121121_Maori/CBCT_F.fcsv",FIDUCIALS_VIRTUAL);
    //ImportFiducialFile("/home/wen/Images/TORS/robotCalibrationReal.fcsv", CALIBRATION_REAL);
    //ImportFiducialFile("/home/wen/Images/TORS/robotCalibrationVirtual.fcsv", CALIBRATION_VIRTUAL);
    //ImportFiducialFile("C:/Users/Wen/Images/daVinciCalibration/robotCalibrationVirtual.fcsv", CALIBRATION_VIRTUAL);
    //ImportFiducialFile("E:/Users/wliu25/MyCommon/data/TORS/TORSPhantomFiducialList.fcsv",FIDUCIALS_VIRTUAL);
    //ImportFiducialFile("E:/Users/wliu25/MyCommon/data/RedSkull/TORSRegistrationFiducials.fcsv", FIDUCIALS_VIRTUAL);
    //ImportFiducialFile("E:/Users/wliu25/MyCommon/data/RedSkull/TORSTargetFiducials.fcsv", TARGETS_VIRTUAL);
    //ImportFiducialFile("E:/Users/wliu25/MyCommon/data/20120223_TORS_Pig_Phantoms/20120223_TORS_PigTongue_sc4_c191100_ROI_Fiducials.fcsv", FIDUCIALS_VIRTUAL);
    //ImportFiducialFile("E:/Users/wliu25/MyCommon/data/20120307_TORS_Pig_Phantoms/T3Final/fiducials.fcsv", FIDUCIALS_VIRTUAL);
#if SUPERFLAB_DEMO
    //ImportFiducialFile("C:/Users/wenl/Projects/Data/20130701_Superflab_Zeego/Slicer/SuperflabPreopFids.fcsv",FIDUCIALS_REAL);
	ImportFiducialFile("C:/Users/wenl/Projects/Data/20130701_Superflab_Zeego/Slicer/SuperflabPreopFids.fcsv",FIDUCIALS_VIRTUAL);
#else
	//ImportFiducialFile("C:/Users/wenl/Projects/Data/20130709_ISI_TORS_Porcine/run1/Fiducials.fcsv",FIDUCIALS_VIRTUAL);
	ImportFiducialFile("C:/Users/wenl/Projects/Data/dev/Experiment/Fiducials.fcsv",FIDUCIALS_VIRTUAL);
	ImportFiducialFile("C:/Users/wenl/Projects/Data/dev/Experiment/Targets.fcsv", TARGETS_VIRTUAL);
#endif
#endif
#endif

#if IMPORT_MULTIPLE_TARGETS
    //ImportFiducialFile("/home/wen/Images/20121121_Maori/demons_targets.fcsv", TUMOR);
#if SUPERFLAB_DEMO
	ImportFiducialFile("C:/Users/wenl/Projects/Data/20130701_Superflab_Zeego/Slicer/SuperflabPreopTargets.fcsv",TUMOR);
#else
	//ImportFiducialFile("C:/Users/wenl/Projects/Data/20130709_ISI_TORS_Porcine/Targets.fcsv", TUMOR);
	//ImportFiducialFile("C:/Users/wenl/Projects/Data/dev/Experiment/Targets.fcsv",TUMOR);
#endif
    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsVirtualFeedback.begin();
         iter != VisibleObjectsVirtualFeedback.end();
         iter++)
    {
        if(iter->first == WRIST)
            (iter->second)->SetOpacity(0.8);
        else
            (iter->second)->SetOpacity(0.2);
    }
    Register();
#endif
    UpdateVisibleList();
}


void ManualRegistration::Startup(void) {

    // To get the joint values, we need to access the component directly
    mtsComponentManager * componentManager = mtsComponentManager::GetInstance();
    CMN_ASSERT(componentManager);
    mtsComponent * daVinci = componentManager->GetComponent("daVinci");
    CMN_ASSERT(daVinci);

    // get PSM1 interface
    mtsInterfaceProvided * interfaceProvided = daVinci->GetInterfaceProvided("PSM1");
    CMN_ASSERT(interfaceProvided);
    mtsCommandRead * command = interfaceProvided->GetCommandRead("GetPositionCartesian");
    CMN_ASSERT(command);
    GetCartesianPositionSlave.Bind(command);

    // get PSM1 interface
    interfaceProvided = daVinci->GetInterfaceProvided("PSM1");
    CMN_ASSERT(interfaceProvided);
    command = interfaceProvided->GetCommandRead("GetPositionCartesianRCM");
    CMN_ASSERT(command);
    GetCartesianPositionRCMSlave.Bind(command);

    // get slave interface
    interfaceProvided = daVinci->GetInterfaceProvided("PSM1");
    CMN_ASSERT(interfaceProvided);
    command = interfaceProvided->GetCommandRead("GetPositionJoint");
    CMN_ASSERT(command);
    GetJointPositionPSM1.Bind(command);

#if TOOL_TRACKING
#ifdef _WIN32
    PSM1 = new robManipulator( "C:/Users/wenl/Projects/dev/cisst/branches/2012-06-06-quadro-sdi/share/models/daVinciSI/T_3000006.rob" );//T_3000006.robT_3000025.rob needle driver
//#else
#endif
#endif

    // get slave interface
    interfaceProvided = daVinci->GetInterfaceProvided("ECM1");
    CMN_ASSERT(interfaceProvided);
    command = interfaceProvided->GetCommandRead("GetPositionJoint");
    CMN_ASSERT(command);
    GetJointPositionECM.Bind(command);

    // get clutch interface
    interfaceProvided = daVinci->GetInterfaceProvided("Clutch");
    CMN_ASSERT(interfaceProvided);
    mtsCommandWrite<ManualRegistration, prmEventButton> * clutchCallbackCommand =
            new mtsCommandWrite<ManualRegistration, prmEventButton>(&ManualRegistration::MasterClutchPedalCallback, this, "Button", prmEventButton());
    CMN_ASSERT(clutchCallbackCommand);
    interfaceProvided->AddObserver("Button", clutchCallbackCommand);

    //get camera control interface
    interfaceProvided = daVinci->GetInterfaceProvided("Camera");
    CMN_ASSERT(interfaceProvided);
    mtsCommandWrite<ManualRegistration, prmEventButton> * cameraCallbackCommand =
            new mtsCommandWrite<ManualRegistration, prmEventButton>(&ManualRegistration::CameraControlPedalCallback, this,
                                                                    "Button", prmEventButton());
    CMN_ASSERT(cameraCallbackCommand);
    interfaceProvided->AddObserver("Button", cameraCallbackCommand);

#if TOOL_TRACKING
    //get tool tracking correction interface
    interfaceProvided = daVinci->GetInterfaceProvided("ToolTrackingCorrection");
    CMN_ASSERT(interfaceProvided);
    mtsCommandWrite<ManualRegistration, vctFrm3> * toolTrackingCorrectionCallbackCommand =
            new mtsCommandWrite<ManualRegistration, vctFrm3>(&ManualRegistration::ToolTrackingCorrectionCallback, this,
                                                                    "ToolTrackingCorrection", vctFrm3());
    CMN_ASSERT(toolTrackingCorrectionCallbackCommand);
    interfaceProvided->AddObserver("Transformation", toolTrackingCorrectionCallbackCommand);
#endif
}

void ManualRegistration::OnStreamSample(svlSample * sample, int streamindex)
{
}

void ManualRegistration::PrimaryMasterButtonCallback(const prmEventButton & event)
{
    if (event.Type() == prmEventButton::PRESSED) {
        this->BooleanFlags[RIGHT_BUTTON] = true;
        this->BooleanFlags[RIGHT_BUTTON_RELEASED] = false;
        UpdatePreviousPosition();
    } else if (event.Type() == prmEventButton::RELEASED) {
        this->BooleanFlags[RIGHT_BUTTON] = false;
        this->BooleanFlags[RIGHT_BUTTON_RELEASED] = true;
    }
    UpdateButtonEvents();
}


void ManualRegistration::SecondaryMasterButtonCallback(const prmEventButton & event)
{
    if (event.Type() == prmEventButton::PRESSED) {
        this->BooleanFlags[LEFT_BUTTON] = true;
        this->BooleanFlags[LEFT_BUTTON_RELEASED] = false;
        UpdatePreviousPosition();
    } else if (event.Type() == prmEventButton::RELEASED) {
        this->BooleanFlags[LEFT_BUTTON] = false;
        this->BooleanFlags[LEFT_BUTTON_RELEASED] = true;
    }
    UpdateButtonEvents();
}


void ManualRegistration::UpdateButtonEvents(void)
{
    prmPositionCartesianGet position;
    if (this->BooleanFlags[RIGHT_BUTTON]) {
        this->GetPrimaryMasterPosition(position);
        InitialMasterRight = position.Position().Translation();
    }

    if (this->BooleanFlags[LEFT_BUTTON]) {
        this->GetSecondaryMasterPosition(position);
        InitialMasterLeft = position.Position().Translation();
    }

    if (this->BooleanFlags[RIGHT_BUTTON] && this->BooleanFlags[LEFT_BUTTON]) {
        this->BooleanFlags[BOTH_BUTTON_PRESSED] = true;
    } else if (this->BooleanFlags[BOTH_BUTTON_PRESSED]){
        this->BooleanFlags[BOTH_BUTTON_PRESSED] = false;
    }

}


/*!

Returns the current position of the center of the tool in the frame of the camera Remote center of motion
@return the frame of the tool wrt to the ECM RCM

*/
vctFrm3 ManualRegistration::GetCurrentECMtoECMRCM(bool tool)
{
    prmPositionJointGet jointsECM;
    vctFrm3 currentECMRCMtoECM;

    vctDouble3 Xaxis;
    Xaxis.Assign(1.0,0.0,0.0);
    vctDouble3 Yaxis;
    Yaxis.Assign(0.0,1.0,0.0);
    vctDouble3 Zaxis;
    Zaxis.Assign(0.0,0.0,1.0);

    // get joint values for ECM
    mtsExecutionResult result = this->GetJointPositionECM(jointsECM);
    if (!result.IsOK()) {
        std::cout << "GetECMtoECMRCM(): ERROR" << result << std::endl;
    }
    // [0] = outer yaw
    // [1] = outer pitch
    // [2] = scope insertion
    // [3] = scope roll

    double yaw0 = jointsECM.Position()[0];
    double pitch1 = jointsECM.Position()[1];
    double insert2 = jointsECM.Position()[2]*1000.0;//convert to mm
    double roll3 = jointsECM.Position()[3];//-180.0*cmnPI/180.0;//30 degree down
    double angle = 30.0*cmnPI/180.0; 

    //create frame for yaw
    vctFrm3 yawFrame0;
    yawFrame0.Rotation() = vctMatRot3(vctAxAnRot3(Yaxis, yaw0));

    //create frame for pitch
    vctFrm3 pitchFrame1;
    pitchFrame1.Rotation() = vctMatRot3(vctAxAnRot3(Xaxis, -pitch1));  // we don't have any logical explanation

    //create frame for insertion
    vctFrm3 insertFrame2;
    insertFrame2.Translation() = vctDouble3(0.0, 0.0, insert2);

    //create frame for the roll
    vctFrm3 rollFrame3;
    rollFrame3.Rotation() = vctMatRot3(vctAxAnRot3(Zaxis, roll3));

    vctFrm3 T_to_horiz;
    T_to_horiz.Rotation() = vctMatRot3(vctAxAnRot3(Xaxis, angle));

	vctFrm3 T_to_30degree;
	if (EndoscopeType == THIRTYUP)
		angle = -1*angle; //'-' for 30 up; '+' for 30 down
    T_to_30degree.Rotation() = vctMatRot3(vctAxAnRot3(Xaxis, angle));

	if((EndoscopeType == THIRTYUP || EndoscopeType == THIRTYDOWN) && !tool)
		currentECMRCMtoECM = yawFrame0 * pitchFrame1 * insertFrame2 * rollFrame3 * T_to_30degree;
	else
		currentECMRCMtoECM = yawFrame0 * pitchFrame1 * insertFrame2 * rollFrame3;
		
	return currentECMRCMtoECM.Inverse();

}

vctFrm3 ManualRegistration::GetCurrentCartesianPositionSlave(void)
{
    // raw cartesian position from slave daVinci, no ui3 correction
    prmPositionCartesianGet slavePosition;
    GetCartesianPositionSlave(slavePosition);

    // Find first virtual object, i.e. Model
    ManualRegistrationObjectType::iterator foundModel;
    foundModel = VisibleObjects.find(MODEL);
    vctFrm3 staticECMtoUI3, currentECMtoECMRCM, currentECMRCMtoUI3;
    staticECMtoUI3.Rotation().From(vctAxAnRot3(vctDouble3(0.0,1.0,0.0), cmnPI));
    currentECMtoECMRCM = GetCurrentECMtoECMRCM(true);
    currentECMRCMtoUI3 = staticECMtoUI3 * currentECMtoECMRCM.Inverse();

    return staticECMtoUI3 * slavePosition.Position();
}

vctFrm3 ManualRegistration::GetCurrentPSM1byJointPositionDH()
{
    vctFrm3 position;
    prmPositionJointGet jointsPSM1;

    vctDouble3 Xaxis;
    Xaxis.Assign(1.0,0.0,0.0);
    vctDouble3 Yaxis;
    Yaxis.Assign(0.0,1.0,0.0);
    vctDouble3 Zaxis;
    Zaxis.Assign(0.0,0.0,1.0);

    // get joint values for ECM
    mtsExecutionResult result = GetJointPositionPSM1(jointsPSM1);

    vctDynamicVector<double> q( 9, 0.0 );
    q[0] = jointsPSM1.Position()[0];
    q[1] = jointsPSM1.Position()[1];
    q[2] = jointsPSM1.Position()[2];
    q[3] = jointsPSM1.Position()[3];
    q[4] = jointsPSM1.Position()[4];
    q[5] = jointsPSM1.Position()[5];
    q[6] = 0.0;//0.0jointsPSM1.Position()[5];//set to zero if needle driver
    q[7] = 0.0;//jointsPSM1.Position()[4];//set to zero if needle driver
    q[8] = 0.0;

    //std::cout << "psm4 joints" << jointsPSM4 << std::endl;
	vctFrame4x4<double> Rts =  PSM1->ForwardKinematics(q);
    position.Rotation().From(Rts.Rotation());
    position.Translation()[0] = Rts.Translation()[0]*1000; //m to mm
    position.Translation()[1] = Rts.Translation()[1]*1000; //m to mm
    position.Translation()[2] = Rts.Translation()[2]*1000; //m to mm
    //Rts.Rotation()
    return position;

}

vctFrm3 ManualRegistration::GetCurrentCorrectedCartesianPositionSlave(void)
{
	// static ECM to UI3
    vctFrm3 staticECMtoUI3, ecm_T_psm, ecm_T_rcm, rcm_T_psm, rcmCorrected_T_rcm;
    staticECMtoUI3.Rotation().From(vctAxAnRot3(vctDouble3(0.0,1.0,0.0), cmnPI));
	
	// raw cartesian position from slave daVinci, no ui3 correction
    prmPositionCartesianGet slavePosition;
    GetCartesianPositionSlave(slavePosition);
	ecm_T_psm = slavePosition.Position();

	// raw cartesian position from slave daVinci, no ui3 correction
    prmPositionCartesianGet slavePositionRCM;
    GetCartesianPositionRCMSlave(slavePositionRCM);
	ecm_T_rcm = slavePositionRCM.Position();

	// tool tip wrt RCM
	rcm_T_psm = ecm_T_rcm.Inverse()*ecm_T_psm;//GetCurrentPSM1byJointPositionDH();
	
	//RCM correction, currently hardcoded
	//rcmCorrected_T_rcm.Translation() = vctDouble3(-9.348, 5.222, -2.939);
	//vctQuaternionRotation3Base<vctFixedSizeVector<double, 4> > quaternionRotation;
	//quaternionRotation.R() = 0.997607;
	//quaternionRotation.X() = -0.024674;
	//quaternionRotation.Y() = -0.061928;
	//quaternionRotation.Z() = 0.018339;
	//rcmCorrected_T_rcm.Rotation().FromRaw(quaternionRotation);
	//rcmCorrected_T_rcm.Rotation().NormalizedSelf();
	//double det = vctDeterminant<3>::Compute(rcmCorrected_T_rcm.Rotation());
	//std::cout << this->TTCorrectedTransformation << std::endl;
	vctFrm3 original = GetCurrentCartesianPositionSlave();
	vctFrm3 corrected = staticECMtoUI3 * ecm_T_rcm * this->TTCorrectedTransformation * rcm_T_psm;
	vctDouble3 distanceVector;
	distanceVector.DifferenceOf(original.Translation(), corrected.Translation());
    double distance = distanceVector.Norm();
	//if(distance > 10.5)
	//	return original;
	//else
	   return corrected;
}

/*!
Function callback triggered by pressing the camera control pedal
Changes the state of the behavior and allows some other features to become active
*/

void ManualRegistration::ToolTrackingCorrectionCallback(const vctFrm3 & payload)
{
	this->BooleanFlags[TOOL_TRACKING_CORRECTION] = true;
	this->TTCorrectedTransformation = payload;
	if (this->BooleanFlags[DEBUG])
		std::cout << "Tool Tracking Correction Event " << payload << std::endl;
}

/*!
Function callback triggered by pressing the camera control pedal
Changes the state of the behavior and allows some other features to become active
*/

void ManualRegistration::CameraControlPedalCallback(const prmEventButton & payload)
{
    if (payload.Type() == prmEventButton::PRESSED) {
        this->BooleanFlags[CAMERA_PRESSED] = true;
        if (this->BooleanFlags[DEBUG])
            std::cout << "Camera pressed" << std::endl;
    } else {
        this->BooleanFlags[CAMERA_PRESSED] = false;
    }
}

/*!
Function callback triggered by pressing the master cluch pedal
Changes the state of the behavior and allows some other features to become active
*/
void ManualRegistration::MasterClutchPedalCallback(const prmEventButton & payload)
{
    if (payload.Type() == prmEventButton::PRESSED) {
        this->BooleanFlags[CLUTCH_PRESSED] = true;
        if (this->BooleanFlags[DEBUG])
            std::cout << "Clutch pressed" << std::endl;
    } else {
        if (this->BooleanFlags[DEBUG])
            std::cout << "Clutch release" << std::endl;
        this->BooleanFlags[CLUTCH_PRESSED] = false;
    }
}

bool ManualRegistration::ImportFiducialFile(const std::string & inputFile, VisibleObjectType type)
{
    int index = 0;
    double count = 0.0;
    vct3 positionFromFile;
    vctFrm3 fiducialPositionUI3, fiducialRotationX, fiducialRotationY, fiducialRotationZ;
    std::string tempLine = "aaaa";
    std::vector <std::string> token;
    ManualRegistrationObjectType::iterator foundModel;
    foundModel = VisibleObjects.find(MODEL);
    if (foundModel == VisibleObjects.end())
        return false;

    if (this->BooleanFlags[DEBUG]) {
        std::cerr << "Importing fiducials from: " << inputFile << std::endl;
    }
    std::ifstream inf(inputFile.c_str());
    while(1) {
        tempLine = "aaaa";
        std::vector <std::string> token;
        std::getline(inf, tempLine);
        Tokenize(tempLine, token, ",");
        if (inf.eof() || token.size() <= 0)
            break;
		std::cerr << cmnData<std::vector<std::string> >::HumanReadable(token) << std::endl;
        if (token.at(0).compare(0,1,"#")) {
            if (token.size() < 4)
                return false;
            if(token.size() >= 5 && strtod(token.at(5).c_str(), NULL) == 0.0)
                continue;
            //assume fiducials are given wrt to model
            positionFromFile =  vct3(strtod(token.at(1).c_str(), NULL),strtod(token.at(2).c_str(), NULL),strtod(token.at(3).c_str(), NULL));
            fiducialPositionUI3.Translation().Assign((foundModel->second)->GetAbsoluteTransformation() * positionFromFile);
            //Add random rotation
            if(type == CALIBRATION_REAL)
            {
                fiducialRotationX.Rotation().From(vctMatRot3(vctAxAnRot3(vctDouble3(1.0,0.0,0.0), cmnPI/(count+1))));
                fiducialRotationY.Rotation().From(vctMatRot3(vctAxAnRot3(vctDouble3(0.0,1.0,0.0), cmnPI/(count+3))));
                fiducialRotationZ.Rotation().From(vctMatRot3(vctAxAnRot3(vctDouble3(0.0,0.0,1.0), cmnPI/(count+4))));
                fiducialPositionUI3.Rotation().From((fiducialRotationX*fiducialRotationY*fiducialRotationZ).Rotation());
            }
            AddFiducial(fiducialPositionUI3, type);
            count++;
            index++;
        }
        token.clear();
    }
    return true;
}


void ManualRegistration::Tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters)
{
    // Skip delimiters at beginning.
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

    while (std::string::npos != pos || std::string::npos != lastPos) {
        // Found a token, add it to the vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }
}

void ManualRegistration::Register(ManualRegistration::VisibleObjectType type)
{
    vctFrm3 displacement, currentUI3toECMRCM;
    vctDynamicVector<vct3> fiducialsVirtual, fiducialsReal, calibrationVirtual, calibrationReal;
    currentUI3toECMRCM = this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse();

    // Register fiducials
    GetFiducials(fiducialsVirtual, fiducialsReal,FIDUCIALS_REAL, ECMRCM);
    double * fre = new double[3*fiducialsVirtual.size()];
    memset(fre, 0, 3*fiducialsVirtual.size() * sizeof(double));

    bool valid = nmrRegistrationRigid(fiducialsVirtual, fiducialsReal, displacement,fre);
    if (valid) {
        // apply transformation in ECMRCM to virtual objects
		switch(type)
		{
			case(TUMOR):
				{
					if(displacement.Translation().Norm() >= this->m_externalTransformationThreshold)
					{
						this->VisibleListVirtual->SetTransformation(displacement * this->VisibleListVirtual->GetTransformation());
						//std::cerr << "Local Registered using # " << fiducialsReal.size() << " fiducials with fre: "<< fre[0] << " type " << type << std::endl;
					}
				break;
				}
			default:
				{
					this->VisibleListVirtual->SetTransformation(displacement * this->VisibleListVirtual->GetTransformation());
					this->VisibleListVirtualGlobal->SetTransformation(displacement * this->VisibleListVirtualGlobal->GetTransformation());
					std::cerr << "Registered using # " << fiducialsReal.size() << " fiducials with fre: "<< fre[0] << " type " << type << std::endl;
				break;
				}
		}
#if PROVIDED_INTERFACE
		UpdateProvidedInterfaces();
#endif
		UpdatePreviousPosition();
    } else {
        std::cerr << "ERROR:ManualRegistration::Register() with # " << fiducialsReal.size() << " real, " << fiducialsVirtual.size()<< " virtual, see log" << std::endl;
    }

    // Calibrate
    GetFiducials(calibrationVirtual, calibrationReal,CALIBRATION_REAL,UI3);
    fre = new double[3*calibrationVirtual.size()];
    memset(fre, 0, 3*calibrationVirtual.size() * sizeof(double));
    if(calibrationVirtual.size() > calibrationCount)
    {
        valid = nmrRegistrationRigid(calibrationVirtual, calibrationReal, displacement,fre);
        if (valid) {
            // save calibration transformation
            Frames[WRIST] = displacement;
            std::cerr << "Calibrated wrist using # " << calibrationReal.size() << " fiducials with fre: "<< fre[0] << " calib:" << Frames[WRIST] << std::endl;
            calibrationCount = calibrationVirtual.size();
        } else {
            std::cerr << "ERROR:ManualRegistration::Register() Calibrated wrist with # " << calibrationReal.size() << " real, " << calibrationVirtual.size()<< " virtual, see log" << std::endl;
        }
    }

    // targets
    //ComputeTRE();
}

void ManualRegistration::ComputeTRE(bool flag)
{
	if(flag)
	{
		//Get fiducials
		vctDynamicVector<vct3> targetsVirtual, targetsReal, targetsRealCamera, toolsReal;
		GetFiducials(targetsVirtual,targetsReal,TARGETS_REAL,ECMRCM,toolsReal);
		targetsRealCamera.resize(targetsReal.size());


		//Error checking
		if(targetsVirtual.size() <= 0 || (targetsVirtual.size() != targetsReal.size())|| (toolsReal.size() != targetsReal.size()))// ||(targetsVirtual.size() != targetsReal.size()/2)))
		{
			std::cerr << "ERROR: ComputeTRE(): virtual#" << targetsVirtual.size() << " real#: " << targetsReal.size() << " tool#: " << toolsReal.size() << std::endl;
			return;
		}

		//Get camera parameters
		vctDynamicVector<vctDouble3> error, toolTrackingError, projectionErrorLeft, projectionErrorRight, errorTriangulation;
		double mTRE=0, maxTRE=-1, mToolTRE=0, maxToolTRE=-1, mTREProjection=0, maxTREProjection=-1, mTRETriangulation=0, maxTRETriangulation=-1;
		error.resize(targetsVirtual.size());
		toolTrackingError.resize(targetsVirtual.size());
		errorTriangulation.resize(targetsVirtual.size());
		projectionErrorLeft.resize(targetsVirtual.size());
		projectionErrorRight.resize(targetsVirtual.size());
		vctDoubleMatRot3 intrinsicsLeft, intrinsicsRight;
		double fcx,fcy,ccx,ccy,a,kc0,kc1,kc2,kc3,kc4;
		int result;
		ui3VTKRenderer * rendererLeft = this->Manager->GetRenderer(SVL_LEFT);
		svlCameraGeometry cameraGeometry = rendererLeft->GetCameraGeometry(); //same geometry stores both left and right, pass in camid
		vctDoubleFrm4x4 extrinsicsLeft(cameraGeometry.GetExtrinsics(SVL_LEFT).frame);
		vctDoubleFrm4x4 extrinsicsRight(cameraGeometry.GetExtrinsics(SVL_RIGHT).frame);
		result = cameraGeometry.GetIntrinsics(fcx,fcy,ccx,ccy,a,kc0,kc1,kc2,kc3,kc4,SVL_LEFT);
		if(result != SVL_OK)
			std::cerr << "ERROR: ComputeTRE, GetIntrinsics(SVL_LEFT)" << std::endl;
		intrinsicsLeft = vctDoubleMatRot3::Identity();
		intrinsicsLeft.Assign(fcx,a,ccx,0.0,fcy,ccy,0.0,0.0,1.0);
		result = cameraGeometry.GetIntrinsics(fcx,fcy,ccx,ccy,a,kc0,kc1,kc2,kc3,kc4,SVL_RIGHT);
		intrinsicsRight = vctDoubleMatRot3::Identity();
		intrinsicsRight.Assign(fcx,a,ccx,0.0,fcy,ccy,0.0,0.0,1.0);
		if(result != SVL_OK)
			std::cerr << "ERROR: ComputeTRE, GetIntrinsics(SVL_RIGHT)" << std::endl;

		//Get PSM1 joint values
		prmPositionJointGet jointsPSM1;
		GetJointPositionPSM1(jointsPSM1);

		//q for forward kinematics
		//vctDynamicVector<double> q( 9, 0.0 );
		//q[0] = jointsPSM1.Position()[0];
		//q[1] = jointsPSM1.Position()[1];
		//q[2] = jointsPSM1.Position()[2];
		//q[3] = jointsPSM1.Position()[3];
		//q[4] = jointsPSM1.Position()[4];
		//q[5] = jointsPSM1.Position()[5];
		//q[6] = 0.0;//0.0jointsPSM1.Position()[5];//set to zero if needle driver
		//q[7] = 0.0;//jointsPSM1.Position()[4];//set to zero if needle driver

		//camera pose (R,t)
		//intrinsic k
		//PtImage = virtual
		//ray r
		//r = R*inv(k)*PtImage-t
		vctDouble3 r, projection;
		for(int i=0;i<(int)targetsVirtual.size();i++)
		{
			//3d point error
			//targetsRealCamera[i] = vct3((targetsReal[j].X()+targetsReal[j+1].X())/2,(targetsReal[j].Y()+targetsReal[j+1].Y())/2,(targetsReal[j].Z()+targetsReal[j+1].Z())/2);
			error[i] = targetsReal[i]-targetsVirtual[i];
			double errorL2Norm = sqrt(error[i].NormSquare());
			mTRE += errorL2Norm;
			fprintf(TRE,"%f, %f, %f\n",error[i].X(),error[i].Y(),error[i].Z());
			if(errorL2Norm > maxTRE)
				maxTRE = errorL2Norm;

			toolTrackingError[i] = targetsReal[i]-toolsReal[i];
			errorL2Norm = sqrt(toolTrackingError[i].NormSquare());
			mToolTRE += errorL2Norm;
			fprintf(TRETool,"%f, %f, %f, %f, %f, %f, %f, %f, %f\n",toolTrackingError[i].X(),toolTrackingError[i].Y(),toolTrackingError[i].Z(),
				jointsPSM1.Position()[0],jointsPSM1.Position()[1],jointsPSM1.Position()[2],jointsPSM1.Position()[3],jointsPSM1.Position()[4],jointsPSM1.Position()[5]);
			if(errorL2Norm > maxToolTRE)
				maxToolTRE = errorL2Norm;

			//project on camera left
			//RayRayIntersect(translationLeft,targetsReal[j], translationRight, targetsReal[j+1], pointA, pointB);
			//targetsRealCamera[i] = vct3((pointA.X()+pointB.X())/2,(pointA.Y()+pointB.Y())/2,(pointA.Z()+pointB.Z())/2);
			//std::cerr << "triangulation: " << targetsRealCamera[i]<< std::endl;
			//errorTriangulation[i] = targetsRealCamera[i]-targetsVirtual[i];
			//double errorTriangulationL2Norm = sqrt(errorTriangulation[i].NormSquare());
			//mTRETriangulation += errorTriangulationL2Norm;
			//fprintf(TRETriangulation,"%f, %f, %f\n",errorTriangulation[i].X(),errorTriangulation[i].Y(),errorTriangulation[i].Z());
			//if(errorTriangulationL2Norm > maxTRETriangulation)
			//   maxTRETriangulation = errorTriangulationL2Norm;
			//c=cbct=virtual,p=translation,q=real; q-p = r
			//l = np.dot((c-p), (q-p)) / np.dot((q-p), (q-p))
			//return p + (l * (q-p))
			r = targetsReal[i]-extrinsicsLeft.Translation();
			projection = extrinsicsLeft.Translation() + (((targetsVirtual[i]-extrinsicsLeft.Translation()).DotProduct(r))/(r.DotProduct(r)))*r;
			projectionErrorLeft[i] = targetsVirtual[i]-projection;
			fprintf(TREProjection,"%f, %f, %f\n",projectionErrorLeft[i].X(),projectionErrorLeft[i].Y(),projectionErrorLeft[i].Z());
			double projectionErrorL2Norm = sqrt(projectionErrorLeft[i].NormSquare());
			mTREProjection += projectionErrorL2Norm;
			if(projectionErrorL2Norm > maxTREProjection)
				maxTREProjection = projectionErrorL2Norm;

			if(this->BooleanFlags[DEBUG])
			{
				std::cerr << "virtual: " << i << " " << targetsVirtual[i] << std::endl;
				//std::cerr << "real: " << i << " " << targetsReal[i]<< std::endl;
				//std::cerr << "projection: " << i << " " << projection<< std::endl;
				std::cerr << "error: " << i << " " << error[i] << std::endl;
				//std::cerr << "projection error: " << i << " " << projectionError[i] << std::endl;
				//PtCBCT = real
				//distance d
				//d=abs|PtCBCT-(t+[(r.(PtCBCT-t))/(r.r))]*r)|
				//std::cerr << "error L1Norm: " << i << " " << error[i].L1Norm()<< std::endl;
				std::cerr << "error L2Norm: " << i << " " << errorL2Norm << std::endl;
				//std::cerr << "error L2NormTriangulation: " << i << " " << errorTriangulationL2Norm << std::endl;
				//std::cerr << "projection error L1Norm: " << i << " " << projectionErrorLeft[i].L1Norm()<< std::endl;
				std::cerr << "projection error left L2Norm: " << i << " " << projectionErrorL2Norm<< std::endl;
			}

			r = targetsReal[i]-extrinsicsRight.Translation();
			projection = extrinsicsRight.Translation() + (((targetsVirtual[i]-extrinsicsRight.Translation()).DotProduct(r))/(r.DotProduct(r)))*r;
			projectionErrorRight[i] = targetsVirtual[i]-projection;
			fprintf(TREProjection,"%f, %f, %f\n",projectionErrorRight[i].X(),projectionErrorRight[i].Y(),projectionErrorRight[i].Z());
			projectionErrorL2Norm = sqrt(projectionErrorRight[i].NormSquare());
			mTREProjection += projectionErrorL2Norm;
			if(projectionErrorL2Norm > maxTREProjection)
				maxTREProjection = projectionErrorL2Norm;

			if(this->BooleanFlags[DEBUG])
			{
				//std::cerr << "virtual: " << i << " " << targetsVirtual[i]<< std::endl;
				//std::cerr << "real: " << i << " " << targetsReal[i]<< std::endl;
				//std::cerr << "projection: " << i << " " << projection<< std::endl;
				//std::cerr << "error: " << i << " " << error[i] << std::endl;
				//std::cerr << "projection error: " << i << " " << projectionError[i] << std::endl;
				//PtCBCT = real
				//distance d
				//d=abs|PtCBCT-(t+[(r.(PtCBCT-t))/(r.r))]*r)|
				//std::cerr << "projection error L1Norm: " << i << " " << projectionErrorRight[i].L1Norm()<< std::endl;
				std::cerr << "projection error right L2Norm: " << i << " " << projectionErrorL2Norm << std::endl;
			}
		}
		std::cerr << "===========================Before All=========================="<<std::endl;
		std::cerr << "MeanTRE: " << MeanTRE/TREFiducialCount << " maxTRE: " << MaxTRE << " count " << TREFiducialCount << std::endl;
		std::cerr << "MeanToolTRE: " << MeanToolTRE/TREFiducialCount << " maxToolTRE: " << MaxToolTRE << " count " << TREFiducialCount << std::endl;
		//std::cerr << "MeanTRETriangulation: " << MeanTRETriangulation/TREFiducialCount << " maxTRE: " << MaxTRETriangulation << std::endl;
		std::cerr << "MeanTREProjection: " << MeanTREProjection/(2*TREFiducialCount) << " maxTREProjection: " << MaxTREProjection << std::endl;
		std::cerr << "====================================================="<<std::endl;

		//update mean TRE
		this->MeanTRE += mTRE;
		this->MeanToolTRE += mToolTRE;
		this->MeanTREProjection += mTREProjection;
		this->MeanTRETriangulation += mTRETriangulation;
		this->TREFiducialCount += targetsVirtual.size();
		if(maxTRE > this->MaxTRE)
			this->MaxTRE = maxTRE;
		if(maxToolTRE > this->MaxToolTRE)
			this->MaxToolTRE = maxToolTRE;
		if(maxTREProjection > this->MaxTREProjection)
			this->MaxTREProjection = maxTREProjection;
		if(maxTRETriangulation > this->MaxTREProjection)
			this->MaxTRETriangulation = maxTRETriangulation;

		std::cerr << "========================After local============================="<<std::endl;
		std::cerr << "mTRE: " << mTRE/targetsVirtual.size() << " maxTRE: " << maxTRE << std::endl;
		std::cerr << "mToolTRE: " << mToolTRE/targetsVirtual.size() << " maxToolTRE: " << maxToolTRE << std::endl;
		//std::cerr << "mTRETriangulation: " << mTRETriangulation/targetsVirtual.size() << " maxTRE: " << maxTRETriangulation << std::endl;
		std::cerr << "mTREProjection: " << mTREProjection/(2*targetsVirtual.size()) << " maxTREProjection: " << maxTREProjection << std::endl;
		std::cerr << "======================After All==============================="<<std::endl;
		std::cerr << "MeanTRE: " << MeanTRE/TREFiducialCount << " maxTRE: " << MaxTRE << " count " << TREFiducialCount << std::endl;
		std::cerr << "MeanToolTRE: " << MeanToolTRE/TREFiducialCount << " maxToolTRE: " << MaxToolTRE << " count " << TREFiducialCount << std::endl;
		//std::cerr << "MeanTRETriangulation: " << MeanTRETriangulation/TREFiducialCount << " maxTRE: " << MaxTRETriangulation << std::endl;
		std::cerr << "MeanTREProjection: " << MeanTREProjection/(2*TREFiducialCount) << " maxTREProjection: " << MaxTREProjection << std::endl;
	}
	else
	{
	    //close TRE log files
		fprintf(TRE,"meanTRE, maxTRE\n");
		fprintf(TRE,"%f, %f\n",MeanTRE/TREFiducialCount,MaxTRE);
		fprintf(TRETool,"meanToolTRE, maxToolTRE\n");
		fprintf(TRETool,"%f, %f\n",MeanToolTRE/TREFiducialCount,MaxToolTRE);
		fprintf(TREProjection,"meanTREProjection, maxTREProjection\n");
		fprintf(TREProjection,"%f, %f\n",MeanTREProjection/(2*TREFiducialCount),MaxTREProjection);
		fclose(TRE);
		fclose(TREProjection);
		fclose(TRETool);	
	}
}

/*
Calculate the line segment PaPb that is the shortest route between
two rays P1P2 and P3P4.
Pa = P1 + mua (P2 - P1)
Pb = P3 + mub (P4 - P3)
Return FALSE if no solution exists.
*/
bool ManualRegistration::RayRayIntersect(vctDouble3 p1,vctDouble3 p2,vctDouble3 p3,vctDouble3 p4,vctDouble3 &pa,vctDouble3 &pb)
{
    vctDouble3 p13,p43,p21;
    double d1343,d4321,d1321,d4343,d2121;
    double numer,denom;
    double mua, mub;
    double eps = 0.0000001;

    p13.X() = p1.X() - p3.X();
    p13.Y() = p1.Y() - p3.Y();
    p13.Z() = p1.Z() - p3.Z();
    p43.X() = p4.X() - p3.X();
    p43.Y() = p4.Y() - p3.Y();
    p43.Z() = p4.Z() - p3.Z();
    if (fabs(p43.X()) < eps && fabs(p43.Y()) < eps && fabs(p43.Z()) < eps)
        return false;
    p21.X() = p2.X() - p1.X();
    p21.Y() = p2.Y() - p1.Y();
    p21.Z() = p2.Z() - p1.Z();
    if (fabs(p21.X()) < eps && fabs(p21.Y()) < eps && fabs(p21.Z()) < eps)
        return false;

    d1343 = p13.X() * p43.X() + p13.Y() * p43.Y() + p13.Z() * p43.Z();
    d4321 = p43.X() * p21.X() + p43.Y() * p21.Y() + p43.Z() * p21.Z();
    d1321 = p13.X() * p21.X() + p13.Y() * p21.Y() + p13.Z() * p21.Z();
    d4343 = p43.X() * p43.X() + p43.Y() * p43.Y() + p43.Z() * p43.Z();
    d2121 = p21.X() * p21.X() + p21.Y() * p21.Y() + p21.Z() * p21.Z();

    denom = d2121 * d4343 - d4321 * d4321;
    if (fabs(denom) < eps)
        return false;
    numer = d1343 * d4321 - d1321 * d4343;

    mua = numer / denom;
    mub = (d1343 + d4321 * mua) / d4343;

    pa.X() = p1.X() + mua * p21.X();
    pa.Y() = p1.Y() + mua * p21.Y();
    pa.Z() = p1.Z() + mua * p21.Z();
    pb.X() = p3.X() + mub * p43.X();
    pb.Y() = p3.Y() + mub * p43.Y();
    pb.Z() = p3.Z() + mub * p43.Z();

    return true;
}

void ManualRegistration::GetFiducials(vctDynamicVector<vct3>& fiducialsVirtual, vctDynamicVector<vct3>& fiducialsReal, VisibleObjectType type, Frame frame,vctDynamicVector<vct3>& toolsReal)
{
    int fiducialIndex = 0;
    bool debugLocal = false;
    ManualRegistrationObjectType localVisibleObjectsVirtual,localVisibleObjectsReal,localVisibleObjectsToolsReal;

    switch(type)
    {
    case(TARGETS_REAL):
        localVisibleObjectsReal = VisibleObjectsRealTargets;
        localVisibleObjectsVirtual = VisibleObjectsVirtualTargets;
		localVisibleObjectsToolsReal = VisibleObjectsRealTool;
        break;
    case(FIDUCIALS_REAL):
        localVisibleObjectsReal = VisibleObjectsRealFiducials;
        localVisibleObjectsVirtual = VisibleObjectsVirtualFiducials;
        break;
    case(CALIBRATION_REAL):
        localVisibleObjectsReal = VisibleObjectsRealCalibration;
        localVisibleObjectsVirtual = VisibleObjectsVirtualCalibration;
        break;
    default:
        //std::cerr << "ERROR: Cannot find fiducial of this type: " << type << std::endl;
        return;
    }

    //real
    for (ManualRegistrationObjectType::iterator iter = localVisibleObjectsReal.begin();
         iter != localVisibleObjectsReal.end() && fiducialsReal.size() < FIDUCIAL_COUNT_MAX;
         iter++) {
        if((iter->second)->Valid)
        {
            fiducialsReal.resize(fiducialIndex + 1);
            switch(frame)
            {
            case(ECMRCM):
                fiducialsReal[fiducialIndex] = this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse() *(iter->second)->GetAbsoluteTransformation().Translation();
                break;
            case(UI3):
                fiducialsReal[fiducialIndex] = (iter->second)->GetAbsoluteTransformation().Translation();
                break;
            default:
                fiducialsReal[fiducialIndex] = this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse() *(iter->second)->GetAbsoluteTransformation().Translation();
                break;
            }
            fiducialIndex++;
            if (this->BooleanFlags[DEBUG] && debugLocal)
                std::cerr << "Getting real fiducial " << fiducialIndex << " at abs positionUI3 " << ((iter->second)->GetAbsoluteTransformation()).Translation()
                          << " relative position: " << ((iter->second)->GetTransformation()).Translation()
                          << " returning " << fiducialsReal[fiducialIndex] << std::endl;
        }
    }

    fiducialIndex = 0;

    //virtual
    for (ManualRegistrationObjectType::iterator iter = localVisibleObjectsVirtual.begin();
         iter != localVisibleObjectsVirtual.end() && fiducialsVirtual.size() < FIDUCIAL_COUNT_MAX;
         iter++) {
        //for fiducials returns no more than # of real fiducials - size must equal for nmrRegistrationRigid
        if((iter->second)->Valid && (fiducialsVirtual.size() < fiducialsReal.size()))
        {
            fiducialsVirtual.resize(fiducialIndex + 1);
            switch(frame)
            {
            case(ECMRCM):
                fiducialsVirtual[fiducialIndex] = this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse() *(iter->second)->GetAbsoluteTransformation().Translation();
                break;
            case(UI3):
                fiducialsVirtual[fiducialIndex] = (iter->second)->GetAbsoluteTransformation().Translation();
                break;
            default:
                fiducialsVirtual[fiducialIndex] = this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse() *(iter->second)->GetAbsoluteTransformation().Translation();
                break;
            }
            fiducialIndex++;
            if (this->BooleanFlags[DEBUG] && debugLocal)
                std::cerr << "Getting virtual fiducial " << fiducialIndex << " at abs positionUI3 " << ((iter->second)->GetAbsoluteTransformation()).Translation()
                          << " relative position: " << ((iter->second)->GetTransformation()).Translation()
                          << " returning " << fiducialsVirtual[fiducialIndex] << std::endl;
        }
    }

    fiducialIndex = 0;

    //tool
    for (ManualRegistrationObjectType::iterator iter = localVisibleObjectsToolsReal.begin();
         iter != localVisibleObjectsToolsReal.end() && toolsReal.size() < FIDUCIAL_COUNT_MAX;
         iter++) {
        //for fiducials returns no more than # of real fiducials - size must equal for nmrRegistrationRigid
        if((iter->second)->Valid && (toolsReal.size() < fiducialsReal.size()))
        {
            toolsReal.resize(fiducialIndex + 1);
            switch(frame)
            {
            case(ECMRCM):
                toolsReal[fiducialIndex] = this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse() *(iter->second)->GetAbsoluteTransformation().Translation();
                break;
            case(UI3):
                toolsReal[fiducialIndex] = (iter->second)->GetAbsoluteTransformation().Translation();
                break;
            default:
                toolsReal[fiducialIndex] = this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse() *(iter->second)->GetAbsoluteTransformation().Translation();
                break;
            }
            fiducialIndex++;
            if (this->BooleanFlags[DEBUG] && debugLocal)
                std::cerr << "Getting real tool fiducial " << fiducialIndex << " at abs positionUI3 " << ((iter->second)->GetAbsoluteTransformation()).Translation()
                          << " relative position: " << ((iter->second)->GetTransformation()).Translation()
                          << " returning " << toolsReal[fiducialIndex] << std::endl;
        }
    }
}

void ManualRegistration::AddFiducial(vctFrm3 positionUI3, VisibleObjectType type)
{
    int fiducialIndex;
    ManualRegistrationObjectType::iterator foundModel;

    // Find first virtual object, i.e. Model
    foundModel = VisibleObjects.find(MODEL);
    if (foundModel == VisibleObjects.end())
        return;

    // Get fiducial index
    switch(type)
    {
    case(TARGETS_REAL):
        fiducialIndex = VisibleObjectsRealTargets.size();
        break;
    case(TARGETS_VIRTUAL):
        fiducialIndex = VisibleObjectsVirtualTargets.size();
        break;
    case(FIDUCIALS_REAL):
        fiducialIndex = VisibleObjectsRealFiducials.size();
        break;
    case(FIDUCIALS_VIRTUAL):
        fiducialIndex = VisibleObjectsVirtualFiducials.size();
        break;
    case(CALIBRATION_REAL):
        fiducialIndex = VisibleObjectsRealCalibration.size();
        break;
    case(CALIBRATION_VIRTUAL):
        fiducialIndex = VisibleObjectsVirtualCalibration.size();
        break;
    case(TOOLS_REAL):
        fiducialIndex = VisibleObjectsRealTool.size();
        break;
    case(TUMOR):
        fiducialIndex = VisibleObjectsVirtualTumors.size();
        break;
    default:
        //std::cerr << "ERROR: Cannot add fiducial of this type: " << type << std::endl;
        return;
    }

    // create new visibleObject for fiducial
    ManualRegistrationSurfaceVisibleStippleObject * newFiducial;
    char buffer[33];
    sprintf(buffer, "%d", fiducialIndex);

    // add visibleObject to visibleList and visibleObjects
    switch(type)
    {
    case(TARGETS_REAL):
        newFiducial = new ManualRegistrationSurfaceVisibleStippleObject(buffer,ManualRegistrationSurfaceVisibleStippleObject::SPHERE,2);
        VisibleObjectsRealTargets[fiducialIndex] = newFiducial;
        this->VisibleListReal->Add(newFiducial);
        break;
    case(TARGETS_VIRTUAL):
        newFiducial = new ManualRegistrationSurfaceVisibleStippleObject(buffer,ManualRegistrationSurfaceVisibleStippleObject::SPHERE,2);
        VisibleObjectsVirtualTargets[fiducialIndex] = newFiducial;
        this->VisibleListVirtual->Add(newFiducial);
        break;
    case(FIDUCIALS_REAL):
        newFiducial = new ManualRegistrationSurfaceVisibleStippleObject(buffer,ManualRegistrationSurfaceVisibleStippleObject::SPHERE,2);
        VisibleObjectsRealFiducials[fiducialIndex] = newFiducial;
        this->VisibleListReal->Add(newFiducial);
        break;
    case(FIDUCIALS_VIRTUAL):
        newFiducial = new ManualRegistrationSurfaceVisibleStippleObject(buffer,ManualRegistrationSurfaceVisibleStippleObject::SPHERE,2);
        VisibleObjectsVirtualFiducials[fiducialIndex] = newFiducial;
        this->VisibleListVirtual->Add(newFiducial);
        break;
    case(CALIBRATION_REAL):
        newFiducial = new ManualRegistrationSurfaceVisibleStippleObject(buffer,ManualRegistrationSurfaceVisibleStippleObject::CYLINDER,1);
        VisibleObjectsRealCalibration[fiducialIndex] = newFiducial;
        this->VisibleListReal->Add(newFiducial);
        break;
    case(CALIBRATION_VIRTUAL):
        newFiducial = new ManualRegistrationSurfaceVisibleStippleObject(buffer,ManualRegistrationSurfaceVisibleStippleObject::CYLINDER,1);
        VisibleObjectsVirtualCalibration[fiducialIndex] = newFiducial;
        this->VisibleListVirtual->Add(newFiducial);
        break;
    case(TOOLS_REAL):
        newFiducial = new ManualRegistrationSurfaceVisibleStippleObject(buffer,ManualRegistrationSurfaceVisibleStippleObject::SPHERE,2);
        VisibleObjectsRealTool[fiducialIndex] = newFiducial;
        this->VisibleListReal->Add(newFiducial);
		break;
    case(TUMOR):
        sprintf(buffer, "%d", type-TUMOR);
        newFiducial = new ManualRegistrationSurfaceVisibleStippleObject(buffer,ManualRegistrationSurfaceVisibleStippleObject::SPHERE,10);
        VisibleObjectsVirtualTumors[fiducialIndex] = newFiducial;
        this->VisibleListVirtual->Add(newFiducial);
        break;
    default:
        return;
    }

    newFiducial->WaitForCreation();
    newFiducial->SetOpacity(0.3);
    newFiducial->Show();
    newFiducial->Visible = true;
    newFiducial->Valid = true;

    switch(type)
    {
    case(TARGETS_REAL):
        newFiducial->SetColor(0.0,1.0,0.0);
        // set position wrt visibleListECMRCM
        newFiducial->SetTransformation(this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse()*positionUI3);
        CMN_LOG_CLASS_RUN_VERBOSE << "PrimaryMasterButtonCallback: added real target: " << fiducialIndex << " "
                                  << newFiducial->GetAbsoluteTransformation().Translation() << std::endl;
        if (this->BooleanFlags[DEBUG])
            std::cerr << "Adding real target " << fiducialIndex << " at positionUI3 " << newFiducial->GetAbsoluteTransformation().Translation() << std::endl;
        break;
    case(TARGETS_VIRTUAL):
        newFiducial->SetColor(0.0,0.0,1.0);
        // set position wrt model
        newFiducial->SetTransformation((foundModel->second)->GetAbsoluteTransformation().Inverse()*positionUI3);
        CMN_LOG_CLASS_RUN_VERBOSE << "PrimaryMasterButtonCallback: added virtual target: " << fiducialIndex << " "
                                  << newFiducial->GetAbsoluteTransformation().Translation() << std::endl;
        if (this->BooleanFlags[DEBUG])
            std::cerr << "Adding virtual target " << fiducialIndex << " at positionUI3 " << newFiducial->GetAbsoluteTransformation().Translation() << std::endl;
        break;
    case(FIDUCIALS_REAL):
        newFiducial->SetColor(0.75,1.0,0.75);
        // set position wrt visibleListECMRCM
        newFiducial->SetTransformation(this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse()*positionUI3);
        CMN_LOG_CLASS_RUN_VERBOSE << "PrimaryMasterButtonCallback: added real fiducial: " << fiducialIndex << " "
                                  << newFiducial->GetAbsoluteTransformation().Translation() << std::endl;
        if (this->BooleanFlags[DEBUG])
		{
            std::cerr << "Adding real fiducial " << fiducialIndex << " at abs " << newFiducial->GetAbsoluteTransformation() << std::endl;
			std::cout << "positionUI3  " << positionUI3 << std::endl;
			std::cout << "UI3_To_ECMRCM  " << this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse() << std::endl;
		}
        break;
    case(FIDUCIALS_VIRTUAL):
        newFiducial->SetColor(0.75,0.75,1.0);
        // set position wrt model
        newFiducial->SetTransformation((foundModel->second)->GetAbsoluteTransformation().Inverse()*positionUI3);
        CMN_LOG_CLASS_RUN_VERBOSE << "PrimaryMasterButtonCallback: added virtual fiducial: " << fiducialIndex << " "
                                  << newFiducial->GetAbsoluteTransformation().Translation() << std::endl;
        if (this->BooleanFlags[DEBUG])
            std::cerr << "Adding virtual fiducial " << fiducialIndex << " at positionUI3 " << newFiducial->GetAbsoluteTransformation().Translation() << std::endl;
        break;
    case(CALIBRATION_REAL):
        newFiducial->SetColor(1.0,0.0,0.0);
        // set position wrt visibleListECMRCM
        newFiducial->SetTransformation(this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse()*positionUI3);
        CMN_LOG_CLASS_RUN_VERBOSE << "PrimaryMasterButtonCallback: added calibration real: " << fiducialIndex << " "
                                  << newFiducial->GetAbsoluteTransformation().Translation() << std::endl;
        if (this->BooleanFlags[DEBUG])
            std::cerr << "Adding calibration real " << fiducialIndex << " at  positionUI3" << (this->VisibleListECMRCM->GetAbsoluteTransformation()*newFiducial->GetTransformation()).Translation() << " " << ((this->VisibleListECMRCM->GetAbsoluteTransformation()*newFiducial->GetTransformation()).Translation().Element(2) + 200) << std::endl;
        break;
    case(CALIBRATION_VIRTUAL):
        newFiducial->SetColor(1.0,0.75,0.75);
        // set position wrt model
        newFiducial->SetTransformation((foundModel->second)->GetAbsoluteTransformation().Inverse()*positionUI3);
        CMN_LOG_CLASS_RUN_VERBOSE << "PrimaryMasterButtonCallback: added calibration virtual : " << fiducialIndex << " "
                                  << newFiducial->GetAbsoluteTransformation().Translation() << std::endl;
        if (this->BooleanFlags[DEBUG])
            std::cerr << "Adding calibration virtual " << fiducialIndex << " at  positionUI3" << ((foundModel->second)->GetAbsoluteTransformation()*newFiducial->GetTransformation()).Translation() << " " << ((foundModel->second)->GetAbsoluteTransformation()*newFiducial->GetTransformation()).Translation().Element(2) + 200<< std::endl;
        break;
    case(TOOLS_REAL):
        newFiducial->SetColor(1.0,0.0,0.0);
        // set position wrt visibleListECMRCM
        newFiducial->SetTransformation(this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse()*positionUI3);
        CMN_LOG_CLASS_RUN_VERBOSE << "PrimaryMasterButtonCallback: added tool real: " << fiducialIndex << " "
                                  << newFiducial->GetAbsoluteTransformation().Translation() << std::endl;
        //if (this->BooleanFlags[DEBUG])
            std::cerr << "Adding tool real " << fiducialIndex << " at  positionUI3" << (this->VisibleListECMRCM->GetAbsoluteTransformation()*newFiducial->GetTransformation()).Translation() << " " << ((this->VisibleListECMRCM->GetAbsoluteTransformation()*newFiducial->GetTransformation()).Translation().Element(2) + 200) << std::endl;
        break;
    case(TUMOR):
        newFiducial->SetColor(0.0, 0.0, 1.0);
        newFiducial->SetOpacity(0.1);
        // set position wrt model
        newFiducial->SetTransformation((foundModel->second)->GetAbsoluteTransformation().Inverse()*positionUI3);
        CMN_LOG_CLASS_RUN_VERBOSE << "Multiple Targets: added virtual target: " << type << " "
                                  << newFiducial->GetAbsoluteTransformation().Translation() << std::endl;
        if (this->BooleanFlags[DEBUG])
            std::cerr << "Adding virtual target " << type << " at positionUI3 " << newFiducial->GetAbsoluteTransformation().Translation() << std::endl;
        break;
    default:
        std::cerr << "ERROR: Cannot add fiducial of this type: " << type << std::endl;
        return;
    }

    ResetButtonEvents();
    UpdateVisibleList();
    return;
}

void ManualRegistration::SetFiducial(vctFrm3 positionUI3, VisibleObjectType type, bool validity, int index, bool overWrite)
{
	ManualRegistrationObjectType::iterator foundObject, foundObjectVirtual;
    ManualRegistrationObjectType localVisibleObjects, localVisibleObjectsVirtual;

    switch(type)
    {
		case(TARGETS_REAL):
		{
			localVisibleObjects = VisibleObjectsRealTargets;
			localVisibleObjectsVirtual = VisibleObjectsVirtualTargets;
		}
		break;
		case(TOOLS_REAL):
		{
			//prmPositionCartesianGet positionMasterRight;
			//this->GetPrimaryMasterPosition(positionMasterRight);
			//vctFrm3 position = positionMasterRight.Position();
			//position.Translation().Assign(positionUI3.Translation());
            //AddFiducial(position,CALIBRATION_REAL);
			localVisibleObjects = VisibleObjectsRealTool;
            positionUI3 = Frames[WRIST]*GetCurrentCorrectedCartesianPositionSlave()*Frames[TIP];
		}
		break;
		case(FIDUCIALS_REAL):
			localVisibleObjects = VisibleObjectsRealFiducials;
			localVisibleObjectsVirtual = VisibleObjectsVirtualFiducials;
			break;
		default:
			return;
	}
	//std::cout << " Setting fiducial " << index << std::endl;
	foundObject = localVisibleObjects.find(index);
	foundObjectVirtual = localVisibleObjectsVirtual.find(index);
    if (foundObject != localVisibleObjects.end()) 
	{
		//std::cout << " from " << (foundObject->second)->GetTransformation() << std::endl;
		//std::cout << " valid " << (foundObject->second)->Valid << std::endl;
		switch(type)
		{
			case(TARGETS_REAL):
			case(FIDUCIALS_REAL):
			case(TOOLS_REAL):
			{
				(foundObject->second)->SetTransformation(this->VisibleListECMRCM->GetAbsoluteTransformation().Inverse()*positionUI3);
			}
			break;
			default:
				break;
		}
		//std::cout << " to " << (foundObject->second)->GetTransformation() << std::endl;
		//std::cout << " valid " << validity << std::endl;
		//std::cout << " using" << positionUI3.Translation() << std::endl;
		(foundObject->second)->Valid = validity;
		(foundObject->second)->Visible = validity;
		if(foundObjectVirtual != localVisibleObjectsVirtual.end())
		{
			(foundObjectVirtual->second)->Valid = validity;
			(foundObjectVirtual->second)->Visible = validity;
		}
	}else
	{
		if(index >= localVisibleObjects.size())
		{
			AddFiducial(positionUI3,type);
		}
	}
	UpdateVisibleList();
}


/*!
find the closest marker to the cursor
*/

ManualRegistrationSurfaceVisibleStippleObject* ManualRegistration::FindClosestFiducial(vctFrm3 positionUI3, VisibleObjectType type, int& index, double distance)
{
    vctFrm3 pos;
    double closestDist = cmnTypeTraits<double>::MaxPositiveValue();
    vctDouble3 dist;
    double abs;
    int currentCount = 0;
    ManualRegistrationSurfaceVisibleStippleObject* closestFiducial = NULL;
    ManualRegistrationObjectType localVisibleObjects;

    switch(type)
    {
    case(TUMOR):
        localVisibleObjects = VisibleObjectsVirtualTumors;
        break;
    case(TARGETS_REAL):
        localVisibleObjects = VisibleObjectsRealTargets;
        break;
    case(TARGETS_VIRTUAL):
        localVisibleObjects = VisibleObjectsVirtualTargets;
        break;
    case(FIDUCIALS_REAL):
        localVisibleObjects = VisibleObjectsRealFiducials;
        break;
    case(FIDUCIALS_VIRTUAL):
        localVisibleObjects = VisibleObjectsVirtualFiducials;
        break;
    case(CALIBRATION_REAL):
        localVisibleObjects = VisibleObjectsRealCalibration;
        break;
    case(CALIBRATION_VIRTUAL):
        localVisibleObjects = VisibleObjectsVirtualCalibration;
        break;
    default:
        //std::cerr << "ERROR: Cannot find fiducial of this type: " << type << std::endl;
        return NULL;
    }

    index = -1;
    char buffer[33];
    for (ManualRegistrationObjectType::iterator iter = localVisibleObjects.begin();
         iter != localVisibleObjects.end();
         iter++) {
        dist.DifferenceOf(positionUI3.Translation(), (iter->second)->GetAbsoluteTransformation().Translation());
        abs = dist.Norm();
        if(abs < closestDist)
        {
            currentCount++;
            if(type == TUMOR)
            {
                closestDist = abs;
                closestFiducial = (iter->second);
                (iter->second)->SetColor(0.0, 0.0, 1.0);
                sprintf(buffer, "%d", iter->first);
                (iter->second)->SetText(buffer);
            }else
            {
                closestDist = abs;
                closestFiducial = (iter->second);
            }
        }
    }


    //if there is one close to the cursor, turn it red
    //return value is that markers count
    //for(iter2 = Markers.begin(); iter2 !=end; iter2++)
    //{
    //    if(closestDist < 2.0 && (*iter2)->count == currentCount)
    //    {
    //        (*iter2)->VisibleObject->SetColor(255.0/255.0, 0.0/255.0, 51.0/255.0);
    //        returnValue = currentCount;
    //    }else{
    //         //otherwise, all the markers should be green, return an invalid number
    //        (*iter2)->VisibleObject->SetColor(153.0/255.0, 255.0/255.0, 153.0/255.0);
    //    }
    //}

    if(closestDist > distance)
    {
        closestFiducial = NULL;
    }
    else{
        //if(this->BooleanFlags[DEBUG])
        //    std::cerr << "Found existing marker at index: " << currentCount << std::endl;
        index = currentCount;
    }

    return closestFiducial;

}

void ManualRegistration::UpdateVisibleList()
{
    for (ManualRegistrationObjectType::iterator iter = VisibleObjects.begin();
         iter != VisibleObjects.end();
         iter++) {
		(iter->second)->Hide();
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if (this->BooleanFlags[VISIBLE] && (iter->second)->Valid && (iter->second)->Visible &&
						(VisibleToggle == ALL || (VisibleToggle == NO_FIDUCIALS)|| (VisibleToggle == iter->first) ||
						 (VisibleToggle == MODEL_ONLY && iter->first == MODEL) ||
						 (VisibleToggle == TUMOR_ONLY && iter->first == TUMOR) ||
						 ((VisibleToggle == FIDUCIALS_REAL || VisibleToggle == FIDUCIALS_VIRTUAL) && iter->first == TUMOR) ))
				{
					(iter->second)->Show();
				} 
			}
		}
	}

    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsVirtualFeedback.begin();
         iter != VisibleObjectsVirtualFeedback.end();
         iter++) {
		(iter->second)->Hide();
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if(VisibleToggle == ALL)
				{
					(iter->second)->Show();
				}
				else if(VisibleToggle == TUMOR)
				{
					if((iter->first) == TOOLTIP)
						(iter->second)->Hide();
					if((iter->first) == WRIST)
						(iter->second)->Show();
				}
				else if(VisibleToggle == NO_FIDUCIALS || VisibleToggle == FIDUCIALS_REAL || VisibleToggle == FIDUCIALS_VIRTUAL)
				{
					if((iter->first) == WRIST || (iter->first) == TOOLTIP)
						(iter->second)->Show();
				}
			}
		}
    }

	this->Cursor->Hide();

    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsVirtualTumors.begin();
         iter != VisibleObjectsVirtualTumors.end();
         iter++) {
		(iter->second)->Hide();
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if (this->BooleanFlags[VISIBLE] && (iter->second)->Valid && (iter->second)->Visible &&
						(VisibleToggle == ALL || VisibleToggle == NO_FIDUCIALS || VisibleToggle == TUMOR || VisibleToggle == MODEL_ONLY || VisibleToggle == TUMOR_ONLY ||
						VisibleToggle == FIDUCIALS_REAL || VisibleToggle == FIDUCIALS_VIRTUAL)) {
					(iter->second)->Show();
				} 
			}
		}
    }

    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsVirtualFiducials.begin();
         iter != VisibleObjectsVirtualFiducials.end();
         iter++) {
		(iter->second)->Hide();
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if (this->BooleanFlags[VISIBLE] && (iter->second)->Valid && (iter->second)->Visible &&
						(VisibleToggle == ALL || VisibleToggle == FIDUCIALS_REAL || VisibleToggle == FIDUCIALS_VIRTUAL)) {
					(iter->second)->Show();
				} 
			}
		}
    }

    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsRealFiducials.begin();
         iter != VisibleObjectsRealFiducials.end();
         iter++) {
		(iter->second)->Hide();
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if (this->BooleanFlags[VISIBLE] && (iter->second)->Valid && (iter->second)->Visible&&
						(VisibleToggle == ALL || VisibleToggle == FIDUCIALS_REAL || VisibleToggle == FIDUCIALS_VIRTUAL))
				{
					(iter->second)->Show();
				} 
			}
		}
    }

    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsVirtualTargets.begin();
         iter != VisibleObjectsVirtualTargets.end();
         iter++) {
		(iter->second)->Hide();
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if (this->BooleanFlags[VISIBLE] && (iter->second)->Valid && (iter->second)->Visible &&
						(VisibleToggle == ALL || VisibleToggle == TARGETS_REAL || VisibleToggle == TARGETS_VIRTUAL)) {
					(iter->second)->Show();
				} 
			}
		}
    }

    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsRealTargets.begin();
         iter != VisibleObjectsRealTargets.end();
         iter++) {
		(iter->second)->Hide();
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if (this->BooleanFlags[VISIBLE] && (iter->second)->Valid && (iter->second)->Visible&&
						(VisibleToggle == ALL || VisibleToggle == TARGETS_REAL || VisibleToggle == TARGETS_VIRTUAL))
				{
					(iter->second)->Show();
				}
			}
		}
    }

    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsVirtualCalibration.begin();
         iter != VisibleObjectsVirtualCalibration.end();
         iter++) {
		(iter->second)->Hide();
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if (this->BooleanFlags[VISIBLE] && (iter->second)->Valid && (iter->second)->Visible&&
						(VisibleToggle == ALL || VisibleToggle == CALIBRATION_REAL || VisibleToggle == CALIBRATION_VIRTUAL))
				{
					(iter->second)->Show();
				} 
			}
		}
    }

    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsRealCalibration.begin();
         iter != VisibleObjectsRealCalibration.end();
         iter++) {
		(iter->second)->Hide();
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if (this->BooleanFlags[VISIBLE] && (iter->second)->Valid && (iter->second)->Visible&&
						(VisibleToggle == ALL || VisibleToggle == CALIBRATION_REAL || VisibleToggle == CALIBRATION_VIRTUAL))
				{
					(iter->second)->Show();
				}
			}
		}
	}

    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsRealTool.begin();
         iter != VisibleObjectsRealTool.end();
         iter++) {
		(iter->second)->Hide();
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if (this->BooleanFlags[VISIBLE] && (iter->second)->Valid && (iter->second)->Visible&&
						(VisibleToggle == ALL || VisibleToggle == TARGETS_REAL || VisibleToggle == TARGETS_VIRTUAL || TOOLS_REAL))
				{
					(iter->second)->Show();
				}
			}
		}
	}
}

void ManualRegistration::SetupProvidedInterfaces(void)
{
	mtsInterfaceRequired * requiresPositionCartesian = AddInterfaceRequired("SlaveArm1");
	if (requiresPositionCartesian) {
		requiresPositionCartesian->AddFunction("SetPositionCartesianSlaveArm1", SetPositionCartesianSlaveArm1);
		requiresPositionCartesian->AddFunction("GetPositionCartesianSlaveArm1", GetPositionCartesianSlaveArm1);
	}

	requiresPositionCartesian = AddInterfaceRequired("ECM_T_ECMRCM");
	if (requiresPositionCartesian) {
		requiresPositionCartesian->AddFunction("SetPositionCartesianECM_T_ECMRCM", SetPositionCartesianECM_T_ECMRCM);
		requiresPositionCartesian->AddFunction("GetPositionCartesianECM_T_ECMRCM", GetPositionCartesianECM_T_ECMRCM);
	}
	requiresPositionCartesian = AddInterfaceRequired("ECMRCM_T_Virtual");
	if (requiresPositionCartesian) {
		requiresPositionCartesian->AddFunction("SetPositionCartesianECMRCM_T_Virtual", SetPositionCartesianECMRCM_T_Virtual);
		requiresPositionCartesian->AddFunction("GetPositionCartesianECMRCM_T_Virtual", GetPositionCartesianECMRCM_T_Virtual);
	}
	requiresPositionCartesian = AddInterfaceRequired("ECMRCM_T_VirtualGlobal");
	if (requiresPositionCartesian) {
		requiresPositionCartesian->AddFunction("SetPositionCartesianECMRCM_T_VirtualGlobal", SetPositionCartesianECMRCM_T_VirtualGlobal);
		requiresPositionCartesian->AddFunction("GetPositionCartesianECMRCM_T_VirtualGlobal", GetPositionCartesianECMRCM_T_VirtualGlobal);
	}
}

void ManualRegistration::UpdateProvidedInterfaces(void)
{
	prmPositionCartesianGet value;

	// ECM_T_ECMRCM
	vctDoubleFrm3 ECM_T_ECMRCM = this->VisibleListECMRCM->GetTransformation();
	value.SetPosition(ECM_T_ECMRCM);
	SetPositionCartesianECM_T_ECMRCM(value);
	//std::cout << "ECM_T_ECMRCM " <<  value << std::endl;

	// ECMRCM_T_Virtual
	vctDoubleFrm3 ECMRCM_T_Virtual = this->VisibleListVirtual->GetTransformation();
	value.SetPosition(ECMRCM_T_Virtual);
	SetPositionCartesianECMRCM_T_Virtual(value);
	//std::cout << "ECM_T_Virtual " << value << std::endl;

	// ECMRCM_T_VirtualGlobal
	vctDoubleFrm3 ECMRCM_T_VirtualGlobal = this->VisibleListVirtualGlobal->GetTransformation();
	value.SetPosition(ECMRCM_T_VirtualGlobal);
	SetPositionCartesianECMRCM_T_VirtualGlobal(value);
	//std::cout << "ECM_T_Virtual " << value << std::endl;


    //ManualRegistrationObjectType::iterator foundObject = VisibleObjectsVirtualFeedback.find(TOOLTIP);
    //if (foundObject != VisibleObjectsVirtualFeedback.end()) {
	//	vctDoubleFrm3 toolTip = (foundObject->second)->GetTransformation();
	//	value.SetPosition(toolTip);
	//	SetPositionCartesianSlaveArm1(value);
	//}

	//raw cartesian position from slave daVinci, no ui3 correction
    //GetCartesianPositionSlave(value);
	vctFrm3 staticECM_T_UI3;
    staticECM_T_UI3.Rotation().From(vctAxAnRot3(vctDouble3(0.0,1.0,0.0), cmnPI));
	// Slicer expects tool position in ECM, will rotate by x(180) to align in own axis
	// Extend wrist to tool tip and needle tip
	vctDoubleFrm3 calibratedWrist = staticECM_T_UI3*Frames[WRIST]*GetCurrentCorrectedCartesianPositionSlave();
	value.SetPosition(calibratedWrist);
	SetPositionCartesianSlaveArm1(value);
	//std::cout << "Slave Arm1 " << value << std::endl;
}

void ManualRegistration::UpdateOpacity(double delta)
{
      for (ManualRegistrationObjectType::iterator iter = VisibleObjects.begin();
         iter != VisibleObjects.end();
         iter++) {
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if (this->BooleanFlags[VISIBLE] && (iter->second)->Valid && (iter->second)->Visible &&
						(VisibleToggle == ALL || (VisibleToggle == NO_FIDUCIALS)|| (VisibleToggle == iter->first) ||
						 (VisibleToggle == MODEL_ONLY && iter->first == MODEL) ||
						 (VisibleToggle == TUMOR_ONLY && iter->first == TUMOR) ||
						 ((VisibleToggle == FIDUCIALS_REAL || VisibleToggle == FIDUCIALS_VIRTUAL) && iter->first == TUMOR) ))
				{
					(iter->second)->SetOpacity((iter->second)->GetOpacity()+delta);
				} 
			}
		}
	  }

    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsVirtualFeedback.begin();
         iter != VisibleObjectsVirtualFeedback.end();
         iter++) {
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if(VisibleToggle == ALL)
				{
					(iter->second)->Show();
				}
				else if(VisibleToggle == NO_FIDUCIALS || VisibleToggle == TUMOR || VisibleToggle == FIDUCIALS_REAL || VisibleToggle == FIDUCIALS_VIRTUAL)
				{
					if((iter->first) == WRIST || (iter->first) == TOOLTIP)
					(iter->second)->SetOpacity((iter->second)->GetOpacity()+delta);
				}
			}
		}
    }

	this->Cursor->Hide();

    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsVirtualTumors.begin();
         iter != VisibleObjectsVirtualTumors.end();
         iter++) {
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if (this->BooleanFlags[VISIBLE] && (iter->second)->Valid && (iter->second)->Visible &&
						(VisibleToggle == ALL || VisibleToggle == NO_FIDUCIALS || VisibleToggle == TUMOR || VisibleToggle == MODEL_ONLY || VisibleToggle == TUMOR_ONLY ||
						VisibleToggle == FIDUCIALS_REAL || VisibleToggle == FIDUCIALS_VIRTUAL)) {
					(iter->second)->SetOpacity((iter->second)->GetOpacity()+delta);
				} 
			}
		}
    }

    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsVirtualFiducials.begin();
         iter != VisibleObjectsVirtualFiducials.end();
         iter++) {
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if (this->BooleanFlags[VISIBLE] && (iter->second)->Valid && (iter->second)->Visible &&
						(VisibleToggle == ALL || VisibleToggle == FIDUCIALS_REAL || VisibleToggle == FIDUCIALS_VIRTUAL)) {
					(iter->second)->SetOpacity((iter->second)->GetOpacity()+delta);
				} 
			}
		}
    }

    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsRealFiducials.begin();
         iter != VisibleObjectsRealFiducials.end();
         iter++) {
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if (this->BooleanFlags[VISIBLE] && (iter->second)->Valid && (iter->second)->Visible&&
						(VisibleToggle == ALL || VisibleToggle == FIDUCIALS_REAL || VisibleToggle == FIDUCIALS_VIRTUAL))
				{
					(iter->second)->SetOpacity((iter->second)->GetOpacity()+delta);
				} 
			}
		}
    }

    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsVirtualTargets.begin();
         iter != VisibleObjectsVirtualTargets.end();
         iter++) {
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if (this->BooleanFlags[VISIBLE] && (iter->second)->Valid && (iter->second)->Visible &&
						(VisibleToggle == ALL || VisibleToggle == TARGETS_REAL || VisibleToggle == TARGETS_VIRTUAL)) {
					(iter->second)->SetOpacity((iter->second)->GetOpacity()+delta);
				} 
			}
		}
    }

    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsRealTargets.begin();
         iter != VisibleObjectsRealTargets.end();
         iter++) {
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if (this->BooleanFlags[VISIBLE] && (iter->second)->Valid && (iter->second)->Visible&&
						(VisibleToggle == ALL || VisibleToggle == TARGETS_REAL || VisibleToggle == TARGETS_VIRTUAL))
				{
					(iter->second)->SetOpacity((iter->second)->GetOpacity()+delta);
				}
			}
		}
    }

    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsVirtualCalibration.begin();
         iter != VisibleObjectsVirtualCalibration.end();
         iter++) {
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if (this->BooleanFlags[VISIBLE] && (iter->second)->Valid && (iter->second)->Visible&&
						(VisibleToggle == ALL || VisibleToggle == CALIBRATION_REAL || VisibleToggle == CALIBRATION_VIRTUAL))
				{
					(iter->second)->SetOpacity((iter->second)->GetOpacity()+delta);
				} 
			}
		}
    }

    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsRealCalibration.begin();
         iter != VisibleObjectsRealCalibration.end();
         iter++) {
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if (this->BooleanFlags[VISIBLE] && (iter->second)->Valid && (iter->second)->Visible&&
						(VisibleToggle == ALL || VisibleToggle == CALIBRATION_REAL || VisibleToggle == CALIBRATION_VIRTUAL))
				{
					(iter->second)->SetOpacity((iter->second)->GetOpacity()+delta);
				}
			}
		}
	}

    for (ManualRegistrationObjectType::iterator iter = VisibleObjectsRealTool.begin();
         iter != VisibleObjectsRealTool.end();
         iter++) {
		if(m_videoAugmentationVisibility)
		{
			if(!m_continuousRegister || (m_continuousRegister && m_visionBasedTrackingVisibility))
			{
				if (this->BooleanFlags[VISIBLE] && (iter->second)->Valid && (iter->second)->Visible&&
						(VisibleToggle == ALL || VisibleToggle == CALIBRATION_REAL || VisibleToggle == CALIBRATION_VIRTUAL))
				{
					(iter->second)->SetOpacity((iter->second)->GetOpacity()+delta);
				}
			}
		}
	}
}
