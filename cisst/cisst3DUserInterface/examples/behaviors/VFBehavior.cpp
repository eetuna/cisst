/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id: VFBehavior.cpp 3148 2013-06-26 15:46:31Z oozgune1 $

Author(s):	Orhan Ozguner, Anton Deguet
Created on:	2013-06-26

(C) Copyright 2013 Johns Hopkins University (JHU), All Rights
Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


#include <VFBehavior.h>
#include <cisst3DUserInterface/StayOnPlaneVirtualFixture.h>
#include <cisstParameterTypes/prmFixtureGainCartesianSet.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include <cstdio>
#include <conio.h>
#include <cisstOSAbstraction/osaThreadedLogFile.h>
#include <cisstOSAbstraction/osaSleep.h>
#include <cisstMultiTask/mtsTaskManager.h>
#include <cisst3DUserInterface/ui3Manager.h>
#include <cisstMultiTask/mtsInterfaceRequired.h>
#include <cisst3DUserInterface/ui3SlaveArm.h> // bad, ui3 should not have slave arm to start with (adeguet1)
#include <vtkActor.h>
#include <vtkAssembly.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkCylinderSource.h>

#define POINTVIRTUALFIXTURE 1
#define PLANEVIRTUALFIXTURE 2
#define CURVEVIRTUALFIXTURE 3

/*!

This class creates the VTK object that will become the cursor and markers of the map
@param manager The ui3Manager responsible for this class

*/
class VFMarker: public ui3VisibleObject
{
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);
public:
    inline VFMarker(const std::string & name):
    ui3VisibleObject(name),
        mCylinder(0),
        markerMapper(0),
        marker(0),
        Position()
    { 
    }

    inline ~VFMarker()
    {

    }

    inline bool CreateVTKObjects(void) {

        std::cout << " VF Marker set up: "<< endl;

        mCylinder = vtkCylinderSource::New();
        CMN_ASSERT(mCylinder);
        mCylinder->SetHeight( 4 );
        mCylinder->SetRadius( 1 );
        mCylinder->SetResolution( 25 );

        markerMapper = vtkPolyDataMapper::New();
        CMN_ASSERT(markerMapper);
        markerMapper->SetInputConnection( mCylinder->GetOutputPort() );
        markerMapper->ImmediateModeRenderingOn();


        marker = vtkActor::New();
        CMN_ASSERT(marker);
        marker->SetMapper( markerMapper);

        marker->RotateX(90);
        this->AddPart(this->marker);
        return true;
    }

    inline bool UpdateVTKObjects(void) {
        return false;
    }

    void SetColor(double r, double g, double b) {
        if (this->marker && (r+g+b)<= 3) {
            this->marker->GetProperty()->SetColor(r, g, b);
        }
    }


protected:
    vtkCylinderSource *mCylinder;
    vtkPolyDataMapper *markerMapper;
    vtkActor *marker;
public:
    vctFrm3 Position; // initial position

};

CMN_DECLARE_SERVICES_INSTANTIATION(VFMarker);
CMN_IMPLEMENT_SERVICES(VFMarker);

/*!

The struct to define the marker type that will be used on the map

*/

struct MarkerType
{
    vctFrm3 AbsolutePosition;
    VFMarker * VisibleObject;
    int count;
};

/*!

constructor

*/

VFBehavior::VFBehavior(const std::string & name):
ui3BehaviorBase(std::string("VFBehavior::") + name, 0),
Ticker(0),
Following(false),
VisibleList(0),
MarkerList(0),
MarkerCount(0),
VirtualFixtureEnabled(false),
ForceEnabled(false),
VFEnable(false),
CameraJustReleased(false),
VirtualFixtureType(0)
{
    this->VisibleList= new ui3VisibleList("VFBehavior");
    this->MarkerList = new ui3VisibleList("MarkerList");
    this->MapCursor=new VFMarker("MapCursor");
    this->MarkerList->Hide();    
    this->VisibleList->Add(MapCursor);
    this->VisibleList->Add(MarkerList);
    this->Offset.SetAll(0.0);
    this->MarkerCount = 0;
    this->CameraPressed = false;
    this->LeftMTMOpen = true;
    this->RightMTMOpen = true;
    this->ClutchPressed = false;

    this->requiredForRead = this->AddInterfaceRequired("MTMRRead");

    if (!requiredForRead) {
        std::cout<<"There is a problem in Add Interface"<<std::endl;
    }

    if (!(requiredForRead->AddFunction("GetPositionCartesian",this->FunctionReadGetPositionCartesian,MTS_REQUIRED))) {
        std::cout<<"There is a problem in Function GetPositionCartesian !"<<std::endl;
    }

    this->requiredForWrite = this->AddInterfaceRequired("MTMRWrite");
    if (!requiredForWrite) {
        std::cout<<"There is a problem in Add Interface"<<std::endl;
    }
    if (!(requiredForWrite->AddFunction("EnableVirtualFixture",this->FunctionVoidEnableVF,MTS_REQUIRED))) {
        std::cout<<"There is a problem in Function EnableVirtualFixture !"<<std::endl;
    }
    if (!(requiredForWrite->AddFunction("DisableVirtualFixture",this->FunctionVoidDisableVF,MTS_REQUIRED))) {
        std::cout<<"There is a problem in Function DisableVirtualFixture !"<<std::endl;
    }
    if (!(requiredForWrite->AddFunction("SetVirtualFixture",this->FunctionWriteSetVF,MTS_REQUIRED))) {
        std::cout<<"There is a problem in Function SetVirtualFixture !"<<std::endl;
    }
}


VFBehavior::~VFBehavior()
{
}

void VFBehavior::ConfigureMenuBar()
{
    this->MenuBar->AddClickButton("PointVF",
        1,
        "pointvf.png",
        &VFBehavior::PointVFCallBack,
        this);
    this->MenuBar->AddClickButton("PlaneVF",
        2,
        "planevf.png",
        &VFBehavior::PlaneVFCallBack,
        this);
    this->MenuBar->AddClickButton("CurveVF",
        3,
        "curvevf.png",
        &VFBehavior::CurveVFCallBack,
        this);
    this->MenuBar->AddClickButton("EnableVF",
        4,
        "enablevf.png",
        &VFBehavior::EnableVFCallback,
        this);
    this->MenuBar->AddClickButton("DisableVF",
        5,
        "disablevf.png",
        &VFBehavior::DisableVFCallback,
        this);

}


void VFBehavior::Startup(void)
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
    mtsCommandWrite<VFBehavior, prmEventButton> * clutchCallbackCommand =
        new mtsCommandWrite<VFBehavior, prmEventButton>(&VFBehavior::MasterClutchPedalCallback, this,
        "Button", prmEventButton());
    CMN_ASSERT(clutchCallbackCommand);
    interfaceProvided->AddObserver("Button", clutchCallbackCommand);

    //get camera control interface
    interfaceProvided = daVinci->GetInterfaceProvided("Camera");
    CMN_ASSERT(interfaceProvided);
    mtsCommandWrite<VFBehavior, prmEventButton> * cameraCallbackCommand =
        new mtsCommandWrite<VFBehavior, prmEventButton>(&VFBehavior::CameraControlPedalCallback, this,
        "Button", prmEventButton());
    CMN_ASSERT(cameraCallbackCommand);
    interfaceProvided->AddObserver("Button", cameraCallbackCommand);

    this->PreviousSlavePosition.Assign(this->Slave1Position.Position().Translation());

}


void VFBehavior::Cleanup(void)
{
    // menu bar will release itself upon destruction

}


bool VFBehavior::RunForeground()
{
    this->Ticker++;

    if (this->Manager->MastersAsMice() != this->PreviousMaM) {
        this->PreviousMaM = this->Manager->MastersAsMice();
        this->VisibleList->Show();
        this->MarkerList->Show();
    }

    // detect transition, should that be handled as an event?
    // State is used by multiple threads ...
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
    this->Slave1Position.Position().Translation().Add(this->Offset);

    return true;
}

bool VFBehavior::RunBackground()
{
    this->Ticker++;

    // detect transition
    if (this->State != this->PreviousState) {
        this->PreviousState = this->State;
        this->VisibleList->Show();
        this->MarkerList->Show();
    }

    this->Transition = true;
    this->Slave1->GetCartesianPosition(this->Slave1Position);
    this->Slave1Position.Position().Translation().Add(this->Offset);

    return true;
}

void VFBehavior::UpdateVF(void)
{
    std::cout<<"Update VF call: "<<std::endl;
    if (VFEnable) {
        CMN_LOG_RUN_VERBOSE << "Behavior \"" << this->GetName() << "\" UpdateVF " << std::endl;
        this->result = this->FunctionVoidEnableVF();
        if (!result.IsOK()) {
            CMN_LOG_CLASS_RUN_ERROR << "execution failed, result is \"" << result << "\"" << std::endl;
        }
        ForceEnabled = true;
        vctFrm3 displacement;
        vctFrm3 markerPositionWRTECM;
        vctFrm3 currentMTMRPos;
        vctFrm3 currentPSM1Pos;
        vctFrm3 tempPos;
        vct3 tempDisp;
        vct3 temp;
        int scale = 3.0;
        vctFrm3 planePoints[3]; //points for plane vf
        std::vector<vct3> myPoints; //points for curve vf
        vctFrm3 pointPoint[1]; //point for point vf

        FunctionReadGetPositionCartesian(currentMTMRPosition);
        tempPos = currentMTMRPosition.Position();
        GetCartesianPositionSlave(currentPSM1Position);
        currentPSM1Pos = currentPSM1Position.Position();
        MarkersType::iterator iter;
        unsigned int index = 0;

        //Point Virtual Fixture
        if (Markers.size() > 0 && VirtualFixtureType == POINTVIRTUALFIXTURE) {
            std::cout<<"Point virtual fixture..."<<std::endl;
            vct3 orientPosTorque, orientNegTorque;
            vct3 orientPosDamping, orientNegDamping;
            for (iter = Markers.begin(); index < 1; iter++){
                //Convert marker positions from ECMRCM to ECM
                markerPositionWRTECM = ECMtoECMRCM.Inverse() * (*iter)->AbsolutePosition;
                //Find displacement between markert and current PSM position
                tempDisp = currentPSM1Position.Position().Translation() - markerPositionWRTECM.Translation() ;
                //Scale back to MTM Fine = 3.0
                tempDisp.Multiply(scale);                
                //Add to current MTM position
                tempPos.Translation().Subtract(tempDisp);
                pointPoint[index] = tempPos;
                index++;
            }

            point.setPoint(pointPoint[0].Translation());
            point.update(currentMTMRPosition.Position(), vfParams);
            //orientation torque constants
            orientPosTorque.SetAll(-0.05);
            orientNegTorque.SetAll(-0.05);
            orientPosDamping.SetAll(-0.001);
            orientNegDamping.SetAll(-0.001);
            //set position orientation
            vfParams.SetTorqueOrientation(pointPoint[0].Rotation());
            //orientation torque
            vfParams.SetOrientationStiffnessPos(orientPosTorque);
            vfParams.SetOrientationStiffnessNeg(orientNegTorque);
            //orientation damping
            vfParams.SetOrientationDampingPos(orientPosDamping);
            vfParams.SetOrientationDampingNeg(orientNegDamping);
        }

        //Plane Virtual Fixture
        if(Markers.size() > 2 && VirtualFixtureType == PLANEVIRTUALFIXTURE){
            std::cout<<"Plane virtual fixture..."<<std::endl;
            //reset index
            index = 0;
            for (iter = Markers.begin(); index < 3; iter++){
                //Convert marker positions from ECMRCM to ECM
                markerPositionWRTECM = ECMtoECMRCM.Inverse() * (*iter)->AbsolutePosition;
                //Find displacement between markert and current PSM position
                tempDisp = currentPSM1Position.Position().Translation() - markerPositionWRTECM.Translation() ;
                //Scale back to MTM Fine = 3.0
                tempDisp.Multiply(scale);                
                //Add to current MTM position
                tempPos.Translation().Subtract(tempDisp);
                planePoints[index] = tempPos;
                index++;
            }

            vct3 line12,line23,pNormal;
            plane.setBasePoint((planePoints[0].Translation()+planePoints[1].Translation()+planePoints[2].Translation())/3.0);

            //std::cout<<"Base point: "<<(planePoints[0].Translation()+planePoints[1].Translation()+planePoints[2].Translation())/3.0<<std::endl;
            //std::cout<<"Point 1: "<<planePoints[0].Translation()<<std::endl;
            //std::cout<<"Point 2: "<<planePoints[1].Translation()<<std::endl;
            //std::cout<<"Point 3: "<<planePoints[2].Translation()<<std::endl;

            line12 = planePoints[0].Translation()-planePoints[1].Translation();
            line23 = planePoints[2].Translation()-planePoints[1].Translation();
            pNormal.CrossProductOf(line12,line23);
            pNormal = pNormal / pNormal.Norm();
            plane.setPlaneNormal(pNormal);
            plane.update(currentMTMRPosition.Position(),vfParams );
        }

        //Curve Virtual Fixture
        if(VirtualFixtureType == CURVEVIRTUALFIXTURE){
            std::cout<<"Curve virtual fixture..."<<std::endl;
            //reset index
            index = 0;
            unsigned int count = Markers.size();

            for (iter = Markers.begin(); index < count; iter++){
                //Convert marker positions from ECMRCM to ECM
                markerPositionWRTECM = ECMtoECMRCM.Inverse() * (*iter)->AbsolutePosition;
                //Find displacement between markert and current PSM position
                tempDisp = currentPSM1Position.Position().Translation() - markerPositionWRTECM.Translation() ;
                //Scale back to MTM Fine = 3.0
                tempDisp.Multiply(scale);                
                //Add to current MTM position
                tempPos.Translation().Subtract(tempDisp);                   
                myPoints.push_back(tempPos.Translation());
                //curvePoints[index] = tempPos.Translation();
                index++;
            }
            curve.setPoints(myPoints);
            curve.update(currentMTMRPosition.Position(),vfParams);
        }

    } else {
        this->result = this->FunctionVoidDisableVF();
        if (!result.IsOK()) {
            CMN_LOG_CLASS_RUN_ERROR << "execution failed, result is \"" << result << "\"" << std::endl;
        }
        ForceEnabled = false;
    }
}

bool VFBehavior::RunNoInput()
{  
    this->Ticker++;

    if (this->Manager->MastersAsMice() != this->PreviousMaM) {
        this->PreviousMaM = this->Manager->MastersAsMice();
        this->VisibleList->Show();
        UpdateVF();
    }     
    

    this->Transition = true;
    this->Slave1->GetCartesianPosition(this->Slave1Position);
    this->Slave1Position.Position().Translation().Add(this->Offset);
    this->ECM1->GetCartesianPosition(this->ECM1Position);
    this->GetJointPositionECM(this->JointsECM);

    UpdateCursorPosition();
    FindClosestMarker();

    //prepare to drop marker if clutch and right MTM are pressed
    if(ClutchPressed && !RightMTMOpen) 
    {
        this->AddMarker();
    }

    //prepare to remove marker if clutch and left MTM are pressed
    if(ClutchPressed && !LeftMTMOpen)
    {
        this->RemoveMarker();
    }

    //check if the map should be updated
    if(CameraPressed ||
        (!ClutchPressed && (PreviousSlavePosition == Slave1Position.Position().Translation())))
    {
        if(this->MapCursor->Visible())
        {
            //if the cursor is visible then hide
            this->MapCursor->Hide();
        }
        //Update the visible map position when the camera is clutched
        this->UpdateVisibleMap();
    }else{
        if(!this->MapCursor->Visible())
        {
            this->MapCursor->Show();
        }
    }

    if (CameraJustReleased && (PreviousSlavePosition != Slave1Position.Position().Translation())) {
        CameraJustReleased = false;
        // UpdateVF();
    }

    if (VFEnable && ForceEnabled){

        //set vf
        if (!CameraPressed && !ClutchPressed) {
            FunctionVoidEnableVF(); // adeguet1: we shouldn't have to enable continuously?
            FunctionWriteSetVF(vfParams);
        }

        //update needed for curve vf
        if(VirtualFixtureType == CURVEVIRTUALFIXTURE){
            FunctionReadGetPositionCartesian(MTMRpos);
            curve.update(MTMRpos.Position(),vfParams);
        }
    }

    PreviousSlavePosition = Slave1Position.Position().Translation();
    return true;
}

void VFBehavior::Configure(const std::string & CMN_UNUSED(configFile))
{
    // load settings
}

bool VFBehavior::SaveConfiguration(const std::string & CMN_UNUSED(configFile))
{
    // save settings
    return true;
}

void VFBehavior::EnableVFCallback(void)
{
    VFEnable = true;
}

void VFBehavior::DisableVFCallback(void)
{
    VFEnable = false;
}

void VFBehavior::PointVFCallBack(void)
{
    VirtualFixtureType = POINTVIRTUALFIXTURE;
}

void VFBehavior::PlaneVFCallBack(void)
{
    VirtualFixtureType = PLANEVIRTUALFIXTURE;
}
void VFBehavior::CurveVFCallBack(void)
{
    VirtualFixtureType = CURVEVIRTUALFIXTURE;
}

void VFBehavior::PrimaryMasterButtonCallback(const prmEventButton & event)
{
    if (event.Type() == prmEventButton::PRESSED) {
        this->RightMTMOpen = false;
        this->Following = true;

    } else if (event.Type() == prmEventButton::RELEASED) {
        this->RightMTMOpen = true;
        this->MarkerDropped = false;
        this->Following = false;
    }
}

/*!
Function callback triggered by the closing of the left master grip.
This action will cause a marker to be removed from the map
*/
void VFBehavior::SecondaryMasterButtonCallback(const prmEventButton & event)
{
    if (event.Type() == prmEventButton::PRESSED) {
        this->LeftMTMOpen = false;
    } else if (event.Type() == prmEventButton::RELEASED) {
        this->LeftMTMOpen = true;
        this->MarkerRemoved = false;
    }
}

/*!
Function callback triggered by pressing the master cluch pedal
Changes the state of the behavior and allows some other features to become active
*/

void VFBehavior::MasterClutchPedalCallback(const prmEventButton & payload)
{
    if (payload.Type() == prmEventButton::PRESSED) {
        this->ClutchPressed = true;
        FunctionVoidDisableVF();
    } else {
        this->ClutchPressed = false;
        if (VFEnable) {
            this->UpdateVF();
        }
    }
}

/*!
Function callback triggered by pressing the camera control pedal
Changes the state of the behavior and allows some other features to become active
*/

void VFBehavior::CameraControlPedalCallback(const prmEventButton & payload)
{
    if (payload.Type() == prmEventButton::PRESSED) {
        this->CameraPressed = true;
        FunctionVoidDisableVF();
    } else {
        this->CameraPressed = false;
        CameraJustReleased = true;
    }
}


/*!

Returns the current position of the center of the tool in the frame of the camera Remote center of motion
@return the frame of the tool wrt to the ECM RCM

*/

vctFrm3 VFBehavior::GetCurrentCursorPositionWRTECMRCM(void)
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
    double insert2 = JointsECM.Position()[2]*1000.0;//convert to mm
    double roll3 = JointsECM.Position()[3];
    double angle = 30.0*cmnPI/180.0;

    //create frame for yaw
    vctFrm3 yawFrame0;
    yawFrame0.Rotation() = vctMatRot3( vctAxAnRot3(Yaxis, yaw0 ) );

    //create frame for pitch
    vctFrm3 pitchFrame1;
    pitchFrame1.Rotation() = vctMatRot3( vctAxAnRot3(Xaxis, -pitch1) );  // we don't have any logical explanation 

    //create frame for insertion
    vctFrm3 insertFrame2;
    insertFrame2.Translation() = vctDouble3(0.0, 0.0, insert2);

    //create frame for the roll
    vctFrm3 rollFrame3;
    rollFrame3.Rotation() = vctMatRot3( vctAxAnRot3(Zaxis, roll3) );

    vctFrm3 T_to_horiz;
    T_to_horiz.Rotation() = vctMatRot3( vctAxAnRot3(Xaxis, angle) );

    // raw cartesian position from slave daVinci, no ui3 correction
    prmPositionCartesianGet slavePosition;
    GetCartesianPositionSlave(slavePosition);

    vctFrm3 finalFrame;
    ECMtoECMRCM = yawFrame0 * pitchFrame1 * insertFrame2 * rollFrame3;
    vctFrm3 imdtframe;
    imdtframe = ECMtoECMRCM * slavePosition.Position();
    finalFrame = imdtframe;

    return finalFrame;
}

/*!
updates the map cursor position, by converting the absolute frame into to the vtk frame

*/ 

void VFBehavior::UpdateCursorPosition(void)
{
    vctFrm3 finalFrame;
    finalFrame = GetCurrentCursorPositionWRTECMRCM();
    vctFrm3 cursorVTK;
    if(MarkerCount == 0)
    {
        vctFrm3 ECMtoVTK;
        ECMtoVTK.Rotation().From( vctAxAnRot3(vctDouble3(0.0,1.0,0.0), cmnPI) );

        ECMRCMtoVTK = ECMtoVTK * ECMtoECMRCM.Inverse();
    }

    ECMRCMtoVTK.ApplyTo(finalFrame, cursorVTK);
    vctDouble3 t1;
    t1 = (cursorVTK.Translation());
    cursorVTK.Translation().Assign(t1);
    MapCursor->SetTransformation(cursorVTK);
}


/*!

this needs some serious commenting

*/

void VFBehavior::UpdateVisibleMap(void)
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
        for(int i = 0; i<8; i++)
        {
            temp.ApplyTo(corners[i], cornersRotated[i]);
            if(i == 0)
            {
                corner1Rotated = cornersRotated[0];
                corner2Rotated = cornersRotated[0];
            }else{
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

#if 1
    for (iter = Markers.begin(); iter != end; iter++)
    {
        vctDouble3 t1;
        ECMRCMtoVTK.ApplyTo((*iter)->AbsolutePosition, positionInSAW);//positionInSAW = ECMRCMtoVTK * Absolutepositon
        t1 = (positionInSAW.Translation() - centerRotated);
        t1 += centerRotated;
        positionInSAW.Translation().Assign(t1);

        (*iter)->VisibleObject->SetTransformation(positionInSAW);
    }
#endif
}

/*!

Adds a marker to the list of markers 
the position of the marker is the position of the cursor at the time that it is dropped
*/

void VFBehavior::AddMarker(void)
{
    if(MarkerDropped == false && MarkerCount < MARKER_MAX)
    {
        //create new marker!
        VFMarker * newMarkerVisible = new VFMarker("marker");
        //must be added to the list
        MarkerList->Add(newMarkerVisible);
        MyMarkers[MarkerCount] = newMarkerVisible;  //NOTE: this array should be gone, but I am using it to hide the markers when they are being removed
        //make sure the new marker is created and part of the scene before editting it
        newMarkerVisible->WaitForCreation();
        newMarkerVisible->SetColor(153.0/255.0, 255.0/255.0, 153.0/255.0); 
        if(MarkerCount < MARKER_MAX)
        {
            MarkerType * newMarker = new MarkerType;
            newMarkerVisible->Show();
            std::cout<< "newMarkerVisible: " << newMarkerVisible->Visible() << std::endl;
            newMarker->VisibleObject = newMarkerVisible;
            newMarker->count = MarkerCount;
            // set the position of the marker based on current cursor position
            newMarker->AbsolutePosition = GetCurrentCursorPositionWRTECMRCM();
            newMarkerVisible->SetTransformation(newMarker->AbsolutePosition);
            std::cout << "GetCurrentCursorPositionWRTECMRCM()" << newMarker->AbsolutePosition << std::endl;
            // add the marker to the list
            this->Markers.push_back(newMarker); //need to delete them too
            // update the list (updates bounding box and position of all markers
            this->UpdateVisibleMap();
            std::cout << "AddMarker has been called " << MapCursor->GetTransformation() << std::endl;

            MarkerCount++;
        }
        MarkerDropped = true;
    }
}


/*!

Removes the last marker from the list

*/
void VFBehavior::RemoveMarker(void)
{
    MarkersType::iterator iter = Markers.begin();
    const MarkersType::iterator end = Markers.end();
    if(MarkerRemoved ==false)
    {
        if(MarkerCount > 0)
        {
            int remove = FindClosestMarker();
            if(remove <= MarkerCount)
                MyMarkers[remove]->Hide();//NOTE: this should be done directly on the marker type list

            std::cout << "marker removed" << std::endl;
        }else{
            std::cout<< "There are no more markers to remove" << std::endl;
        }
        std::cout << "Marker Count: " << MarkerCount << std::endl;
        MarkerRemoved = true;
    }
}

/*!
find the closest marker to the cursor
*/

int VFBehavior::FindClosestMarker()
{
    vctFrm3 pos;
    pos = GetCurrentCursorPositionWRTECMRCM();
    double closestDist = cmnTypeTraits<double>::MaxPositiveValue();
    vctDouble3 dist;
    double abs;
    int currentCount = 0, returnValue = -1;
    MarkersType::iterator iter1, iter2;
    const MarkersType::iterator end = Markers.end();
    for (iter1 = Markers.begin(); iter1 != end; iter1++)
    {
        dist.DifferenceOf(pos.Translation(), (*iter1)->AbsolutePosition.Translation());
        abs = dist.Norm();
        if(abs < closestDist)
        {
            currentCount = (*iter1)->count;
            closestDist = abs;
        }
    }

    //if there is one close to the cursor, turn it red
    //return value is that markers count
    for(iter2 = Markers.begin(); iter2 !=end; iter2++)
    {
        if(closestDist < 2.0 && (*iter2)->count == currentCount)
        {
            (*iter2)->VisibleObject->SetColor(255.0/255.0, 0.0/255.0, 51.0/255.0);
            returnValue = currentCount;
        }else{
            //otherwise, all the markers should be green, return an invalid number
            (*iter2)->VisibleObject->SetColor(153.0/255.0, 255.0/255.0, 153.0/255.0);
        }
    }

    if(closestDist >2.0)
    {
        returnValue = MARKER_MAX + 1;
    }

    return returnValue;

}


