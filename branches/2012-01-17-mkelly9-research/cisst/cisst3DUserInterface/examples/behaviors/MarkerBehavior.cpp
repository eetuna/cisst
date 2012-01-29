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

#include <cisstOSAbstraction/osaThreadedLogFile.h>
#include <cisstOSAbstraction/osaSleep.h>
#include <cisstMultiTask/mtsTaskManager.h>
#include <cisst3DUserInterface/ui3Manager.h>
#include <cisst3DUserInterface/ui3SlaveArm.h> // bad, ui3 should not have slave arm to start with (adeguet1)
#include <cisstNumerical/nmrRegistrationRigid.h>

#include <vtkActor.h>
#include <vtkAssembly.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>


// how close markers need to be to delete (in mm)
#define MARKER_DISTANCE_THRESHOLD (5.0)
// z-axis translation between tool eye and tip (in mm)
#define WRIST_TIP_OFFSET (11.0)


struct MarkerType
{
    vctFrm3 AbsolutePosition;
    ui3VisibleAxes * VisibleObject;
    int count;
};


enum OperatingMode
{
    NONE,
    SET_FIDUCIALS
};


MarkerBehavior::MarkerBehavior(const std::string & name):
        ui3BehaviorBase(std::string("MarkerBehavior::") + name, 0),
        Ticker(0),
        Following(false),
        VisibleList(0),
        MarkerList(0),
        MarkerCount(0)
{
    this->VisibleList = new ui3VisibleList("MarkerBehavior");
    this->MarkerList = new ui3VisibleList("MarkerList");
    
    this->MapCursor = new ui3VisibleAxes;
    
    this->MarkerList->Hide();
    
    this->VisibleList->Add(MapCursor);
    this->VisibleList->Add(MarkerList);

    // offset to move points from eye to tooltip (roughly 11mm)
    this->Offset.SetAll(0.0);
    this->MarkerCount = 0;
    this->CameraPressed = false;
    this->LeftMTMOpen = true;
    this->RightMTMOpen = true;
    this->ClutchPressed = false;
    this->ModeSelected = NONE;

    this->WristToTip.Translation().Assign(vctDouble3(0.0, 0.0, WRIST_TIP_OFFSET));
}


MarkerBehavior::~MarkerBehavior()
{
}


void MarkerBehavior::ConfigureMenuBar()
{
    this->MenuBar->AddClickButton("Set Fiducial",
                                  1,
                                  "circle.png",
                                  &MarkerBehavior::SetFiducialButtonCallback,
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
        this->VisibleList->Show();
        this->MarkerList->Show();
    }

    // detect transition, should that be handled as an event?
    // state is used by multiple threads ...
    if (this->State != this->PreviousState) {
        this->PreviousState = this->State;
        this->VisibleList->Show();
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
        this->VisibleList->Show();
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
    if (ClutchPressed && !RightMTMOpen && ModeSelected == SET_FIDUCIALS) 
    {
        this->AddMarker();
    }

    // prepare to remove marker if clutch and left MTM are pressed
    if (ClutchPressed && !LeftMTMOpen && ModeSelected == SET_FIDUCIALS)
    {
        this->RemoveMarker();
    }

    // check if the map should be updated
    if (CameraPressed ||
       (!ClutchPressed && (PreviousSlavePosition == Slave1Position.Position().Translation())))
    {
        if (this->MapCursor->Visible())
        {
            // if the cursor is visible then hide;
            this->MapCursor->Hide();
        }
        // update the visible map position when the camera is clutched
        this->UpdateVisibleMap();
    }
    else
    {
        if (ModeSelected == SET_FIDUCIALS)
        {
            if (!this->MapCursor->Visible())
            {
                this->MapCursor->Show();
            }
        }
        else
        {
            if (this->MapCursor->Visible())
            {
                this->MapCursor->Hide();
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

void MarkerBehavior::SetFiducialButtonCallback(void)
{
    CMN_LOG_RUN_VERBOSE << "Behavior \"" << this->GetName() << "\" Set fiducials button pressed" << std::endl;

    ModeSelected = SET_FIDUCIALS;
}

void MarkerBehavior::RegisterButtonCallback(void)
{
    CMN_LOG_RUN_VERBOSE << "Behavior \"" << this->GetName() << "\" Register button pressed" << std::endl;

    // get points from file
    vctDynamicVector<vct3> initialPoints;
    std::ifstream inputFile("registrationInput.txt");
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
    for (unsigned int i = 0; i < Markers.size(); i++)
	{
		if (Markers[i]->VisibleObject->Visible())
		{
			selectedPoints.resize(selectedPoints.size()+1);
			selectedPoints[i] = this->Markers[i]->AbsolutePosition.Translation();
		}
    }

    // perform registration
    vctFrm3 registration;
    bool success = nmrRegistrationRigid(initialPoints, selectedPoints, registration);
	if (!success)
	{
#if 0
		CMN_LOG_RUN_WARNING << "MarkerBehavior::RegisterButtonCallback: registration failed;"
			                << " check nmrRegistrationRigid logs" << std::endl;
#endif
		std::cerr << "MarkerBehavior::RegisterButtonCallback: registration failed;"
			      << " check nmrRegistrationRigid logs" << std::endl;
		return;
	}

    // output registration results
    std::ofstream outputFile("registrationOutput.txt");
	registration.ToStream(std::cout);
	std::cout << std::endl;
    outputFile.close();
}

void MarkerBehavior::ClearFiducialsButtonCallback(void)
{
    CMN_LOG_RUN_VERBOSE << "Behavior \"" << this->GetName() << "\" Clear fiducials button pressed" << std::endl;

    // hide all the markers
    for (int i = 0 ; i < MarkerCount; i++)
	{
        MyMarkers[i]->Hide();
    }
    // hide map cursor until out of MaM mode so as not to confuse the user into
    // thinking that not all cursors have been cleared
    if (this->MapCursor->Visible())
    {
        this->MapCursor->Hide();
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
    MapCursor->SetTransformation(cursorVTK);
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
	std::cout << "add called marker dropped is " << MarkerDropped << " clutch pressed is " << ClutchPressed << std::endl;
    if (MarkerDropped == false && MarkerCount < MARKER_MAX)
    {
        // create new marker!
        ui3VisibleAxes * newMarkerVisible = new ui3VisibleAxes;
        // must be added to the list
        MarkerList->Add(newMarkerVisible);
        MyMarkers[MarkerCount] = newMarkerVisible;  // NOTE: this array should be gone, but I am using it to hide the markers when they are being removed
        // make sure the new marker is created and part of the scene before editting it
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

			std::ofstream outFile("points.txt", std::ios_base::app);
			newMarker->AbsolutePosition.Translation().ToStreamRaw(outFile, ',');
			outFile << std::endl;

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
    if (MarkerRemoved == false)
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
    
    // if there is one close to the cursor, turn it red
    // return value is that marker's count
    for (iter2 = Markers.begin(); iter2 != end; iter2++)
    {
        if (closestDist < MARKER_DISTANCE_THRESHOLD && (*iter2)->count == currentCount)
        {
            returnValue = currentCount;
        }
    }
    
    if (closestDist > MARKER_DISTANCE_THRESHOLD)
    {
        returnValue = MARKER_MAX + 1;
    }

    return returnValue;
}
