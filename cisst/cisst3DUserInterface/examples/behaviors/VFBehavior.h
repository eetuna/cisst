/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id: VFBehavior.h 3148 2013-06-26 15:46:31Z oozgune1 $

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

#include <cisstParameterTypes/prmPositionJointGet.h>
#include <cisst3DUserInterface/ui3BehaviorBase.h>
#include <cisst3DUserInterface/ui3VisibleObject.h>
#include <cisst3DUserInterface/ui3VisibleList.h>
#include <cisst3DUserInterface/VirtualFixture.h>
#include <cisst3DUserInterface/PointVirtualFixture.h>
#include <cisst3DUserInterface/PlaneVirtualFixture.h>
#include <list>
#include <limits>
#define MARKER_MAX 20
class VFMarker;
struct MarkerType;
// Always include last!
#include "ui3BehaviorsExport.h"

class CISST_EXPORT VFBehavior: public ui3BehaviorBase
{
public:
    VFBehavior(const std::string & name);
    ~VFBehavior();

    void Startup(void);
    void Cleanup(void);
    void ConfigureMenuBar(void);
    bool RunForeground(void);
    bool RunBackground(void);
    bool RunNoInput(void);
    void Configure(const std::string & configFile);
    bool SaveConfiguration(const std::string & configFile);
    inline ui3VisibleObject * GetVisibleObject(void) {
        return this->VisibleList;
    }

protected:
    unsigned long Ticker;
    void PointVFCallBack(void);
    void PlaneVFCallBack(void);
    void EnableVFCallback(void);
    void DisableVFCallback(void);
    void PrimaryMasterButtonCallback(const prmEventButton & event);
    void SecondaryMasterButtonCallback(const prmEventButton & event);
    void MasterClutchPedalCallback(const prmEventButton & payload);
    void CameraControlPedalCallback(const prmEventButton & payload);
    StateType PreviousState;
    bool PreviousMaM;
    vctDouble3 PreviousCursorPosition;
    vctDouble3 Offset;
    vctFrm3 Position;
    bool Following;
    bool Transition;

    ui3SlaveArm * Slave1;
    ui3SlaveArm * ECM1;
    prmPositionCartesianGet Slave1Position;
    prmPositionCartesianGet ECM1Position;

    mtsFunctionRead GetJointPositionSlave;
    mtsFunctionRead GetCartesianPositionSlave;
    mtsFunctionRead GetJointPositionECM;
    prmPositionJointGet JointsSlave;
    prmPositionJointGet JointsECM;

    void UpdateCursorPosition(void);
    vctFrm3 GetCurrentCursorPositionWRTECMRCM(void);
    void UpdateVisibleMap(void);
    void AddMarker(void);
    void RemoveMarker(void);
    int FindClosestMarker(void);
    void UpdateVF(void);

    typedef  std::list<MarkerType*> MarkersType;
    MarkersType Markers;

    mtsInterfaceRequired * requiredForWrite;
    mtsInterfaceRequired * requiredForRead;
    mtsFunctionVoid FunctionVoidEnableVF;
    mtsFunctionVoid FunctionVoidDisableVF;
    mtsFunctionWrite FunctionWriteSetVF;
    mtsFunctionRead FunctionReadGetPositionCartesian;
    prmPositionCartesianGet currentMTMRPosition;
    prmPositionCartesianGet currentPSM1Position;

    PointVirtualFixture point;
    PlaneVirtualFixture plane;
    prmFixtureGainCartesianSet vfParams;

    mtsExecutionResult result;

private:

    ui3VisibleList * VisibleList;
    ui3VisibleObject * MapCursor;
    ui3VisibleList * MarkerList;
    VFMarker * MyMarkers[MARKER_MAX];
    bool RightMTMOpen, LeftMTMOpen;
    bool CameraPressed, ClutchPressed;
    bool MarkerDropped, MarkerRemoved;
    int MarkerCount;
    vctDouble3 PreviousSlavePosition;
    vctFrm3 ECMtoECMRCM, ECMRCMtoVTK;
    vctDouble3 centerRotated;
    bool VirtualFixtureEnabled;
    bool ForceEnabled;
    bool VFEnable;
    int VirtualFixtureType;
};

