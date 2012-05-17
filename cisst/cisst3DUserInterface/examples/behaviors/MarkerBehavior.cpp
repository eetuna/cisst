/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Martin Kelly
  Created on: 2012-01-15

  (C) Copyright 2012 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


#include <MarkerBehavior.h>

#include <cisstCommon/cmnPath.h>
#include <cisstOSAbstraction/osaThreadedLogFile.h>
#include <cisstOSAbstraction/osaSleep.h>
#include <cisstMultiTask/mtsTaskManager.h>
#include <cisst3DUserInterface/ui3Manager.h>
#include <cisst3DUserInterface/ui3SlaveArm.h> // bad, ui3 should not have slave arm to start with (adeguet1)
#include <cisstNumerical/nmrRegistrationRigid.h>

#include <vtkActor.h>
#include <vtkAssembly.h>
#include <vtkFollower.h>
#include <vtkGenericDataObjectReader.h>
#include <vtkOutlineFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataReader.h>
#include <vtkProperty.h>
#include <vtkSmartPointer.h>
#include <vtkVectorText.h>

// how close markers need to be to delete (in mm)
#define MARKER_DISTANCE_THRESHOLD (5.0)
// amount of offset from cursor for model display when in offset mode
#define MODEL_OFFSET (20.0)
// z-axis translation between tool eye and tip (in mm)
#define WRIST_TIP_OFFSET (11.0)

#define PROSTATE_MODEL_PATH ("prostate.vtk")
#define URETHRA_MODEL_PATH ("urethra.vtk")
#define REGISTRATION_INPUT_PATH ("registrationInput.txt")

// copied from MapBehaviorTextObject
class MarkerBehaviorTextObject: public ui3VisibleObject
{
	CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_ALLOW_DEFAULT);
public:
    inline MarkerBehaviorTextObject(const std::string & name = "Text"):
        ui3VisibleObject(name),
        Text(0),
        TextMapper(0),
        TextActor(0)
    {}

    inline ~MarkerBehaviorTextObject()
    {}

    inline bool CreateVTKObjects(void) {

        Text = vtkVectorText::New();
        CMN_ASSERT(Text);
        Text->SetText("");

        TextMapper = vtkPolyDataMapper::New();
        CMN_ASSERT(TextMapper);
        TextMapper->SetInputConnection(Text->GetOutputPort());
        TextMapper->ImmediateModeRenderingOn();

        TextActor = vtkFollower::New();
        CMN_ASSERT(TextActor);
        TextActor->SetMapper(TextMapper);
        TextActor->GetProperty()->SetColor(0.0, 0.0, 200.0/255.0 );
        TextActor-> SetScale(2.5);

        this->AddPart(this->TextActor);

        return true;
    }

	inline bool UpdateVTKObjects(void) {
        return true;
    }

    inline void SetText(const std::string & text)
    {
        if (this->Text) {
            this->Text->SetText(text.c_str());
        }
    }

    inline void SetColor(double r, double g, double b)
    {
        if (this->TextActor && (r+g+b)<= 3) {
            this->TextActor->GetProperty()->SetColor(r,g,b);
        }
    }


protected:
    vtkVectorText * Text;
    vtkPolyDataMapper * TextMapper;
    vtkFollower * TextActor;
};
CMN_DECLARE_SERVICES_INSTANTIATION(MarkerBehaviorTextObject);
CMN_IMPLEMENT_SERVICES(MarkerBehaviorTextObject);


// copied from ImageViewerKidneySurfaceVisibleObject
class MarkerBehaviorModelObject: public ui3VisibleObject
{
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);
public:
    inline MarkerBehaviorModelObject(const std::string & name, const std::string & inputFile, bool hasOutline = false):
        ui3VisibleObject(name),
        InputFile(inputFile),
        SurfaceReader(0),
        SurfaceMapper(0),
        SurfaceActor(0),
        HasOutline(hasOutline),
        OutlineData(0),
        OutlineMapper(0),
        OutlineActor(0)
    {}

    inline ~MarkerBehaviorModelObject()
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
        if (this->OutlineData) {
            this->OutlineData->Delete();
            this->OutlineData = 0;
        }
        if (this->OutlineMapper) {
            this->OutlineMapper->Delete();
            this->OutlineMapper = 0;
        }
        if (this->OutlineActor) {
            this->OutlineActor->Delete();
            this->OutlineActor = 0;
        }
    }

    inline bool CreateVTKObjects(void) {
        SurfaceReader = vtkPolyDataReader::New();
        CMN_ASSERT(SurfaceReader);
        CMN_LOG_CLASS_INIT_VERBOSE << "Loading file \"" << InputFile << "\"" << std::endl;
        SurfaceReader->SetFileName(InputFile.c_str());
        CMN_LOG_CLASS_INIT_VERBOSE << "File \"" << InputFile << "\" loaded" << std::endl;
        SurfaceReader->Update();

        SurfaceMapper = vtkPolyDataMapper::New();
        CMN_ASSERT(SurfaceMapper);
        SurfaceMapper->SetInputConnection(SurfaceReader->GetOutputPort());
        SurfaceMapper->ScalarVisibilityOff();
        SurfaceMapper->ImmediateModeRenderingOn();

        SurfaceActor = vtkActor::New();
        CMN_ASSERT(SurfaceActor);
        SurfaceActor->SetMapper(SurfaceMapper);
        SurfaceActor->GetProperty()->SetOpacity(1.0); // change this if you want the model to be slightly transparent

        this->AddPart(this->SurfaceActor);

        // Create a frame for the data volume.
        if (HasOutline) {
            OutlineData = vtkOutlineFilter::New();
            CMN_ASSERT(OutlineData);
            OutlineData->SetInputConnection(SurfaceReader->GetOutputPort());
            OutlineMapper = vtkPolyDataMapper::New();
            CMN_ASSERT(OutlineMapper);
            OutlineMapper->SetInputConnection(OutlineData->GetOutputPort());
            OutlineMapper->ImmediateModeRenderingOn();
            OutlineActor = vtkActor::New();
            CMN_ASSERT(OutlineActor);
            OutlineActor->SetMapper(OutlineMapper);
            OutlineActor->GetProperty()->SetColor(1,1,1);
            
            // Scale the actors.
            OutlineActor->SetScale(0.05);
            this->AddPart(this->OutlineActor);
        }
        return true;
    }

	inline bool UpdateVTKObjects(void) {
        return true;
    }

    inline void SetColor(double r, double g, double b) {
        SurfaceActor->GetProperty()->SetDiffuseColor(r, g, b);
    }

    inline void SetOpacity(double opacity) {
        SurfaceActor->GetProperty()->SetOpacity(opacity);
    }

    vctDouble3 GetCenter(void) {
        vctDouble3 center;
        if (HasOutline) {
            center.Assign(OutlineActor->GetCenter());
        }
        return center;
    }

protected:
    std::string InputFile;
    vtkPolyDataReader * SurfaceReader;
    vtkPolyDataMapper * SurfaceMapper;
    vtkActor * SurfaceActor;
    bool HasOutline;
    vtkOutlineFilter * OutlineData;
    vtkPolyDataMapper * OutlineMapper;
    vtkActor * OutlineActor;
};
CMN_DECLARE_SERVICES_INSTANTIATION(MarkerBehaviorModelObject);
CMN_IMPLEMENT_SERVICES(MarkerBehaviorModelObject);


MarkerBehavior::MarkerBehavior(const std::string & name):
        ui3BehaviorBase(std::string("MarkerBehavior::") + name, 0),
        Ticker(0),
        Following(false),
        RootList(0),
        CursorList(0),
        MarkerList(0),
        ModelList(0),
        FollowCameraList(0),
		TextObject(0),
        MarkerCount(0)
{
    this->RootList = new ui3VisibleList("MarkerBehavior");
    this->MarkerList = new ui3VisibleList("MarkerList");
    this->ModelList = new ui3VisibleList("ModelList");
    this->FollowCameraList = new ui3VisibleList("FollowCameraList");
    this->CursorList = new ui3VisibleList("CursorList");

	this->TextObject = new MarkerBehaviorTextObject;
    this->Cursor = new ui3VisibleAxes;

	cmnPath path;
    path.AddRelativeToCisstShare("/models/dv-3dus");
    std::string prostateName = path.Find(PROSTATE_MODEL_PATH, cmnPath::READ);
	this->ProstateModel = new MarkerBehaviorModelObject("ProstateModel", prostateName);
	std::string urethraName = path.Find(URETHRA_MODEL_PATH, cmnPath::READ);
    this->UrethraModel = new MarkerBehaviorModelObject("UrethraModel", urethraName);
    
    this->ModelList->Add(FollowCameraList);
    this->FollowCameraList->Add(ProstateModel);
    this->FollowCameraList->Add(UrethraModel);

    this->CursorList->Add(Cursor);
    this->CursorList->Add(TextObject);

    this->RootList->Add(MarkerList);
    this->RootList->Add(ModelList);
    this->RootList->Add(CursorList);

    this->RootList->Show();
	this->DisplayMode = DISPLAY_NONE;
	this->ProstateModel->Hide();
	this->UrethraModel->Hide();

    this->Offset.SetAll(0.0);
    this->MarkerCount = 0;
    this->CameraPressed = false;
    this->LeftMTMOpen = true;
    this->RightMTMOpen = true;
    this->ClutchPressed = false;
    this->SettingFiducials = false;

    this->ModelOffset.Translation().Assign(vctDouble3(MODEL_OFFSET, MODEL_OFFSET, 0.0));
    this->WristToTip.Translation().Assign(vctDouble3(0.0, 0.0, WRIST_TIP_OFFSET));
    this->IdentityTransformation.Translation().Assign(vctDouble3(0.0, 0.0, 0.0));

    this->OffsetMode = true;
    this->ModelList->SetTransformation(ModelOffset);
}


MarkerBehavior::~MarkerBehavior()
{
}


void MarkerBehavior::ConfigureMenuBar()
{
    this->MenuBar->AddClickButton("Set Fiducials",
                                  1,
                                  "circle.png",
                                  &MarkerBehavior::SetFiducialsButtonCallback,
                                  this);
    this->MenuBar->AddClickButton("Clear Fiducials",
                                  2,
                                  "redo.png",
                                  &MarkerBehavior::ClearFiducialsButtonCallback,
                                  this);
    this->MenuBar->AddClickButton("Register",
                                  3,
                                  "cylinder.png",
                                  &MarkerBehavior::RegisterButtonCallback,
                                  this);
    this->MenuBar->AddClickButton("Toggle Prostate/Urethra/None",
                                  4,
                                  "slices.png",
                                  &MarkerBehavior::ModelToggleCallback,
                                  this);
    this->MenuBar->AddClickButton("Offset / Augmented Reality",
                                  5,
                                  "iconify-top-left.png",
                                  &MarkerBehavior::SwitchModelModeCallback,
                                  this);
}


void MarkerBehavior::Startup(void)
{
    this->Slave1 = this->Manager->GetSlaveArm("Slave1");
    if (!this->Slave1) {
        CMN_LOG_CLASS_INIT_ERROR << "This behavior requires a slave arm ..." << std::endl;
    }

    this->ECM1 = this->Manager->GetSlaveArm("ECM1");

    // To get the joint values, we need to access the component directly
    mtsComponentManager * componentManager = mtsComponentManager::GetInstance();
    CMN_ASSERT(componentManager);
    mtsComponent * daVinci = componentManager->GetComponent("daVinci");
    CMN_ASSERT(daVinci);
    // get PSM1 interface
    mtsInterfaceProvided * interfaceProvided = daVinci->GetInterfaceProvided("PSM1");
    CMN_ASSERT(interfaceProvided);
    mtsCommandRead * command = interfaceProvided->GetCommandRead("GetPositionJoint");
    CMN_ASSERT(command);
    GetJointPositionSlave.Bind(command);
    command = interfaceProvided->GetCommandRead("GetPositionCartesian");
    CMN_ASSERT(command);
    GetCartesianPositionSlave.Bind(command);
    
    // get slave interface
    interfaceProvided = daVinci->GetInterfaceProvided("ECM1");
    CMN_ASSERT(interfaceProvided);
    command = interfaceProvided->GetCommandRead("GetPositionJoint");
    CMN_ASSERT(command);
    GetJointPositionECM.Bind(command);
    
    // get clutch interface
    interfaceProvided = daVinci->GetInterfaceProvided("Clutch");
    CMN_ASSERT(interfaceProvided);
    mtsCommandWrite<MarkerBehavior, prmEventButton> * clutchCallbackCommand =
            new mtsCommandWrite<MarkerBehavior, prmEventButton>(&MarkerBehavior::MasterClutchPedalCallback, this, "Button", prmEventButton());
    CMN_ASSERT(clutchCallbackCommand);
    interfaceProvided->AddObserver("Button", clutchCallbackCommand);
    
    //get camera control interface
    interfaceProvided = daVinci->GetInterfaceProvided("Camera");
    CMN_ASSERT(interfaceProvided);
    mtsCommandWrite<MarkerBehavior, prmEventButton> * cameraCallbackCommand =
            new mtsCommandWrite<MarkerBehavior, prmEventButton>(&MarkerBehavior::CameraControlPedalCallback, this, "Button", prmEventButton());
    CMN_ASSERT(cameraCallbackCommand);
    interfaceProvided->AddObserver("Button", cameraCallbackCommand);

    this->PreviousSlavePosition.Assign(this->Slave1Position.Position().Translation());

}


void MarkerBehavior::Cleanup(void)
{
    // menu bar will release itself upon destruction
}


bool MarkerBehavior::RunForeground()
{
    this->Ticker++;

    if (this->Manager->MastersAsMice() != this->PreviousMaM) {
        this->PreviousMaM = this->Manager->MastersAsMice();
        this->RootList->Show();
        this->MarkerList->Show();
    }

    // detect transition, should that be handled as an event?
    // state is used by multiple threads ...
    if (this->State != this->PreviousState) {
        this->PreviousState = this->State;
        this->RootList->Show();
        this->MarkerList->Show();
    }
    // running in foreground GUI mode
    prmPositionCartesianGet position;

    this->PreviousCursorPosition.Assign(position.Position().Translation());

    // apply to object
    this->Slave1->GetCartesianPosition(this->Slave1Position);
    // apply wrist to tip transformation
    this->Slave1Position.Position() = this->Slave1Position.Position() * this->WristToTip;

    return true;
}

bool MarkerBehavior::RunBackground()
{
    return true;
}

bool MarkerBehavior::RunNoInput()
{
    this->Ticker++;

    if (this->Manager->MastersAsMice() != this->PreviousMaM) {
        this->PreviousMaM = this->Manager->MastersAsMice();
        this->RootList->Show();
    }
    this->Transition = true;
    this->Slave1->GetCartesianPosition(this->Slave1Position);
    this->Slave1Position.Position().Translation().Add(this->Offset);
    this->ECM1->GetCartesianPosition(this->ECM1Position);
    this->GetJointPositionECM(this->JointsECM);

    UpdateCursorPosition();
    FindClosestMarker();

	// make it possible to add/remove markers again
	if (!ClutchPressed)
	{
		this->MarkerDropped = false;
		this->MarkerRemoved = false;
	}

    // prepare to drop marker if clutch and right MTM are pressed
    if (ClutchPressed && !RightMTMOpen) 
    {
        this->AddMarker();
    }

    // prepare to remove marker if clutch and left MTM are pressed
    if (ClutchPressed && !LeftMTMOpen)
    {
        this->RemoveMarker();
    }

    // check if the map should be updated
    if (CameraPressed ||
       (!ClutchPressed && (PreviousSlavePosition == Slave1Position.Position().Translation())))
    {
        if (this->Cursor->Visible())
        {
            // if the cursor is visible then hide;
            this->Cursor->Hide();
        }
        // update the visible map position when the camera is clutched
        this->UpdateVisibleMap();
    }
	else
	{
		if (SettingFiducials)
		{
			if (!this->Cursor->Visible())
			{
				this->Cursor->Show();
			}
		}
		else
		{
			if (this->Cursor->Visible())
			{
				this->Cursor->Hide();
			}
		}
	}

    PreviousSlavePosition = Slave1Position.Position().Translation();
    return true;
}

void MarkerBehavior::Configure(const std::string & CMN_UNUSED(configFile))
{
    // load settings
}

bool MarkerBehavior::SaveConfiguration(const std::string & CMN_UNUSED(configFile))
{
    // save settings
    return true;
}

void MarkerBehavior::SetFiducialsButtonCallback(void)
{
    CMN_LOG_RUN_VERBOSE << "Behavior \"" << this->GetName() << "\" Set fiducials button pressed" << std::endl;

    SettingFiducials = true;
}

void MarkerBehavior::RegisterButtonCallback(void)
{
    CMN_LOG_RUN_VERBOSE << "Behavior \"" << this->GetName() << "\" Register button pressed" << std::endl;

    // get points from file
    vctDynamicVector<vct3> initialPoints;
    std::ifstream inputFile(REGISTRATION_INPUT_PATH);
    while (inputFile.good())
    {
        std::string line;
        std::getline(inputFile, line);
        std::stringstream ss(line);
        double x;
        double y;
        double z;
        ss >> x;
        ss >> y;
        ss >> z;

        vctDouble3 point(x, y, z);
        initialPoints.resize(initialPoints.size() + 1);
        initialPoints[initialPoints.size()-1] = point;
    }
    inputFile.close();

    // get selected points
    vctDynamicVector<vct3> selectedPoints;
	unsigned int size = 0;
    for (unsigned int i = 0; i < Markers.size(); i++)
	{
		if (Markers[i]->VisibleObject->Visible())
		{
			size++;
			selectedPoints.resize(size);
			selectedPoints[size-1] = Markers[i]->AbsolutePosition.Translation();
		}
    }

    // perform registration
    vctFrm3 registration;
	double error;
    bool success = nmrRegistrationRigid(initialPoints, selectedPoints, registration, &error);
	if (!success)
	{
		CMN_LOG_RUN_WARNING << "MarkerBehavior::RegisterButtonCallback: registration failed;"
			                << " check nmrRegistrationRigid logs" << std::endl;
		return;
	}

    // output registration results
    std::ofstream outputFile(GetRegistrationOutputFilename().c_str());

	outputFile << "initial points:" << std::endl << initialPoints << std::endl
		       << "selected points:" << std::endl << selectedPoints << std::endl;
	registration.ToStream(outputFile);
	outputFile << std::endl << "fiducial registration error: " << error;
    outputFile.close();

	std::cout << "initial points:" << std::endl << initialPoints << std::endl
		       << "selected points:" << std::endl << selectedPoints << std::endl;
	registration.ToStream(std::cout);
	std::cout << std::endl << "fiducial registration error: " << error;
}

void MarkerBehavior::ClearFiducialsButtonCallback(void)
{
    CMN_LOG_RUN_VERBOSE << "Behavior \"" << this->GetName() << "\" Display prostate model pressed" << std::endl;

    // hide all the markers
    for (unsigned int i = 0 ; i < Markers.size(); i++)
	{
        Markers[i]->VisibleObject->Hide();
    }
    // hide map cursor until out of MaM mode so as not to confuse the user into
    // thinking that not all cursors have been cleared
    if (this->Cursor->Visible())
    {
        this->Cursor->Hide();
    }
}


void MarkerBehavior::ModelToggleCallback(void)
{
    CMN_LOG_RUN_VERBOSE << "Behavior \"" << this->GetName() << "\" Toggle prostate/urethra/none" << std::endl;

	switch(this->DisplayMode)
	{
		case DISPLAY_PROSTATE:
			this->DisplayMode = DISPLAY_URETHRA;
			this->ProstateModel->Hide();
			this->UrethraModel->Show();
			break;
		case DISPLAY_URETHRA:
			this->DisplayMode = DISPLAY_NONE;
			this->ProstateModel->Hide();
			this->UrethraModel->Hide();
			break;
		case DISPLAY_NONE:
			this->DisplayMode = DISPLAY_PROSTATE;
			this->ProstateModel->Show();
			this->UrethraModel->Hide();
			break;
		}
}


void MarkerBehavior::SwitchModelModeCallback(void)
{
    CMN_LOG_RUN_VERBOSE << "Behavior \"" << this->GetName() << "\" Offset / augmented reality mode toggled" << std::endl;

    if (OffsetMode)
    {
        OffsetMode = false;
        ModelList->SetTransformation(IdentityTransformation);
    }
    else
    {
        OffsetMode = true;
        ModelList->SetTransformation(ModelOffset);
    }
}


void MarkerBehavior::PrimaryMasterButtonCallback(const prmEventButton & event)
{
    if (event.Type() == prmEventButton::PRESSED) {
        this->RightMTMOpen = false;
        this->Following = true;
    }
    else if (event.Type() == prmEventButton::RELEASED) {
        this->RightMTMOpen = true;
        this->Following = false;
    }
}

/*!
Function callback triggered by the closing of the left master grip.
This action will cause a marker to be removed from the map
*/
void MarkerBehavior::SecondaryMasterButtonCallback(const prmEventButton & event)
{
    if (event.Type() == prmEventButton::PRESSED) {
        this->LeftMTMOpen = false;
    }
    else if (event.Type() == prmEventButton::RELEASED) {
        this->LeftMTMOpen = true;
    }
}

/*!
Function callback triggered by pressing the master cluch pedal
Changes the state of the behavior and allows some other features to become active
*/
void MarkerBehavior::MasterClutchPedalCallback(const prmEventButton & payload)
{
    if (payload.Type() == prmEventButton::PRESSED) {
        this->ClutchPressed = true;
    } else {
        this->ClutchPressed = false;
    }
}

/*!
Function callback triggered by pressing the camera control pedal
Changes the state of the behavior and allows some other features to become active
*/
void MarkerBehavior::CameraControlPedalCallback(const prmEventButton & payload)
{
    if (payload.Type() == prmEventButton::PRESSED) {
        this->CameraPressed = true;
    } else {
        this->CameraPressed = false;
    }
}


/*!
Returns the current position of the center of the tool in the frame of the camera Remote center of motion
@return the frame of the tool wrt to the ECM RCM
*/
vctFrm3 MarkerBehavior::GetCurrentCursorPositionWRTECMRCM(void)
{
    vctDouble3 Xaxis;
    Xaxis.Assign(1.0,0.0,0.0);
    vctDouble3 Yaxis;
    Yaxis.Assign(0.0,1.0,0.0);
    vctDouble3 Zaxis;
    Zaxis.Assign(0.0,0.0,1.0);

    // get joint values for ECM
    this->GetJointPositionECM(this->JointsECM);
    // [0] = outer yaw
    // [1] = outer pitch
    // [2] = scope insertion
    // [3] = scope roll

    double yaw0 = JointsECM.Position()[0];
    double pitch1 = JointsECM.Position()[1];
    double insert2 = JointsECM.Position()[2]*1000.0; // convert to mm
    double roll3 = JointsECM.Position()[3];
    double angle = 30.0*cmnPI/180.0;
 
    // create frame for yaw
    vctFrm3 yawFrame0;
    yawFrame0.Rotation() = vctMatRot3( vctAxAnRot3(Yaxis, yaw0 ) );

    // create frame for pitch
    vctFrm3 pitchFrame1;
    pitchFrame1.Rotation() = vctMatRot3( vctAxAnRot3(Xaxis, -pitch1) );  // we don't have any logical explanation 

    // create frame for insertion
    vctFrm3 insertFrame2;
    insertFrame2.Translation() = vctDouble3(0.0, 0.0, insert2);

    // create frame for the roll
    vctFrm3 rollFrame3;
    rollFrame3.Rotation() = vctMatRot3( vctAxAnRot3(Zaxis, roll3) );

    vctFrm3 T_toHorizontal;
    T_toHorizontal.Rotation() = vctMatRot3( vctAxAnRot3(Xaxis, angle) );
 
    // raw cartesian position from slave daVinci, no ui3 correction
    prmPositionCartesianGet slavePosition;
    GetCartesianPositionSlave(slavePosition);
    // apply wrist to tip transformation
    slavePosition.Position() = slavePosition.Position() * this->WristToTip;

    vctFrm3 finalFrame;

    ECMtoECMRCM = yawFrame0 * pitchFrame1 * insertFrame2 * rollFrame3;
    vctFrm3 imdtframe;
    imdtframe = ECMtoECMRCM * slavePosition.Position(); //* GetCurrentCursorPositionWRTECM(); // working fixed point !!!
    finalFrame = imdtframe;//.Inverse(); // .InverseSelf();
    
    return finalFrame;
}

/*!
updates the map cursor position, by converting the absolute frame to the vtk frame
*/ 
void MarkerBehavior::UpdateCursorPosition(void)
{
    vctFrm3 finalFrame;
    finalFrame = GetCurrentCursorPositionWRTECMRCM();
    vctFrm3 cursorVTK;
    if (MarkerCount == 0)
    {
        vctFrm3 ECMtoVTK;
        ECMtoVTK.Rotation().From( vctAxAnRot3(vctDouble3(0.0,1.0,0.0), cmnPI) );
    
        ECMRCMtoVTK = ECMtoVTK * ECMtoECMRCM.Inverse();
    }

    ECMRCMtoVTK.ApplyTo(finalFrame, cursorVTK);// cursorVTK = ECMRCMtoVTK * finalframe

    vctDouble3 t1;
    t1 = (cursorVTK.Translation());
    cursorVTK.Translation().Assign(t1);
    Cursor->SetTransformation(cursorVTK);
    FollowCameraList->SetTransformation(cursorVTK);
}


/*!
this needs some serious commenting
*/
void MarkerBehavior::UpdateVisibleMap(void)
{
    vctDouble3 corner1, corner2, center;
    MarkersType::iterator iter = Markers.begin();
    const MarkersType::iterator end = Markers.end();
    vctDouble3 currentOrigin;

    vctDouble3 scaleCompTrans(0.0);
    
    // iterate through all elements to build a bounding box
    if (iter != end)
    {
        // initialize the bounding box corners using the first element
        currentOrigin = (*iter)->AbsolutePosition.Translation();
        corner1.Assign(currentOrigin);
        corner2.Assign(currentOrigin);
        iter++;
        // update with all remaining elements
        for (; iter != end; iter++)
        {
            currentOrigin = (*iter)->AbsolutePosition.Translation();
            corner1.ElementwiseMinOf(corner1, currentOrigin);
            corner2.ElementwiseMaxOf(corner2, currentOrigin);
        }
        // computer center of bounding box
        center.SumOf(corner1, corner2);
        center.Divide(2.0);

        vctDynamicVector<vctDouble3> corners, cornersRotated;
        corners.SetSize(8);
        cornersRotated.SetSize(8);

        corners[0].Assign(corner1[0], corner1[1], corner1[2]);
        corners[1].Assign(corner1[0], corner1[1], corner2[2]);
        corners[2].Assign(corner1[0], corner2[1], corner1[2]);
        corners[3].Assign(corner1[0], corner2[1], corner2[2]);

        corners[4].Assign(corner2[0], corner1[1], corner1[2]);
        corners[5].Assign(corner2[0], corner1[1], corner2[2]);
        corners[6].Assign(corner2[0], corner2[1], corner1[2]);
        corners[7].Assign(corner2[0], corner2[1], corner2[2]);

        vctFrm3 ECMtoVTK;
        ECMtoVTK.Rotation().From( vctAxAnRot3(vctDouble3(0.0,1.0,0.0), cmnPI) );
        
        vctFrm3 ECMtoECMRCMInverse = ECMtoECMRCM.Inverse();
        
        vctFrm3 temp;
        temp = ECMtoVTK * ECMtoECMRCMInverse;
        vctDouble3 corner1Rotated, corner2Rotated;
        for (int i = 0; i < 8; i++)
        {
            temp.ApplyTo(corners[i], cornersRotated[i]);
            if (i == 0)
            {
                corner1Rotated = cornersRotated[0];
                corner2Rotated = cornersRotated[0];
            }
            else
            {
                corner1Rotated.ElementwiseMinOf(corner1Rotated, cornersRotated[i]);
                corner2Rotated.ElementwiseMaxOf(corner2Rotated, cornersRotated[i]);
            }
        }
        centerRotated.SumOf(corner1Rotated, corner2Rotated);
        centerRotated.Divide(2.0);

        // computer the transformation to be applied to all absolute coordinates
        // to be display in the SAW coordinate system
        ECMRCMtoVTK = ECMtoVTK * ECMtoECMRCMInverse;
    }

    // apply the transformation to all absolute coordinates
    vctFrm3 positionInSAW;
    for (iter = Markers.begin(); iter != end; iter++)
    {
        vctDouble3 t1;
        ECMRCMtoVTK.ApplyTo((*iter)->AbsolutePosition, positionInSAW);//positionInSAW = ECMRCMtoVTK * Absolutepositon
        t1 = (positionInSAW.Translation() - centerRotated);
        t1 += centerRotated;
        positionInSAW.Translation().Assign(t1);

        (*iter)->VisibleObject->SetTransformation(positionInSAW);
    }
}


/*!
Adds a marker to the list of markers 
the position of the marker is the position of the cursor at the time that it is dropped
*/
void MarkerBehavior::AddMarker(void)
{
	if (MarkerDropped == false && MarkerCount < MARKER_MAX && SettingFiducials)
    {
        // create new marker!
        ui3VisibleAxes * newMarkerVisible = new ui3VisibleAxes;
        // must be added to the list
        MarkerList->Add(newMarkerVisible);
        MyMarkers[MarkerCount] = newMarkerVisible;  // NOTE: this array should be gone, but I am using it to hide the markers when they are being removed
        // make sure the new marker is created and part of the scene before editing it
        newMarkerVisible->WaitForCreation();
        if (MarkerCount < MARKER_MAX)
        {
            MarkerType * newMarker = new MarkerType;
            newMarkerVisible->Show();
            newMarker->VisibleObject = newMarkerVisible;
            newMarker->count = MarkerCount;
            // set the position of the marker based on current cursor position
            newMarker->AbsolutePosition = GetCurrentCursorPositionWRTECMRCM();
            newMarkerVisible->SetTransformation(newMarker->AbsolutePosition);
            // add the marker to the list
            this->Markers.push_back(newMarker); // need to delete them too
            // update the list (updates bounding box and position of all markers
            this->UpdateVisibleMap();

			std::ofstream outputFile("points.txt", std::ios_base::app);
			newMarker->AbsolutePosition.Translation().ToStreamRaw(outputFile, ',');
			outputFile << std::endl;

            MarkerCount++;
        }
        MarkerDropped = true;
    }
}


/*!
Removes the last marker from the list
*/
void MarkerBehavior::RemoveMarker(void)
{
    if (MarkerRemoved == false && SettingFiducials)
    {
        if (MarkerCount > 0)
        {
            int remove = FindClosestMarker();
            if (remove <= MarkerCount)
                MyMarkers[remove]->Hide();// NOTE: this should be done directly on the marker type list

            std::cout << "marker removed" << std::endl;
        }
        else
        {
            std::cout << "There are no more markers to remove" << std::endl;
        }
        std::cout << "Marker Count: " << MarkerCount << std::endl;
        MarkerRemoved = true;
    }
}

/*!
Find the closest marker to the cursor
*/
int MarkerBehavior::FindClosestMarker()
{
    vctFrm3 pos = GetCurrentCursorPositionWRTECMRCM();
    double closestDist = cmnTypeTraits<double>::MaxPositiveValue();
    vctDouble3 dist;
    double abs;
    int currentCount = 0;
    int returnValue = -1;
    MarkersType::iterator iter1, iter2;
    const MarkersType::iterator end = Markers.end();
    for (iter1 = Markers.begin(); iter1 != end; iter1++)
    {
        dist.DifferenceOf(pos.Translation(), (*iter1)->AbsolutePosition.Translation());
        abs = dist.Norm();
        if (abs < closestDist)
        {
            currentCount = (*iter1)->count;
            closestDist = abs;
        }
    }
    
    // if there is one close to the cursor, display a message
    // return value is that marker's count
    for (iter2 = Markers.begin(); iter2 != end; iter2++)
    {
        if (closestDist < MARKER_DISTANCE_THRESHOLD && (*iter2)->count == currentCount)
        {
			this->TextObject->SetText("CLOSE");
            returnValue = currentCount;
        }
    }
    
    if (closestDist > MARKER_DISTANCE_THRESHOLD)
    {
        returnValue = MARKER_MAX + 1;
    }

    return returnValue;
}


// Gets the next registration output file, following the scheme
// "RegistrationOutput1.txt", "RegistrationOutput2.txt", etc.
// Never overwrites an existing file, so it has to find the next numbre to use
std::string MarkerBehavior::GetRegistrationOutputFilename(void)
{
    int i = 1;
    std::stringstream sstream;
    std::ifstream fstream;
    while (true)
    {
        sstream << "RegistrationOutput" << i;
        fstream.open(sstream.str().c_str());
        if (fstream.is_open())
        {
            break;
        }
        sstream.str("");
        sstream.clear();
        i++;
    }
    fstream.close();

    return sstream.str();
}
