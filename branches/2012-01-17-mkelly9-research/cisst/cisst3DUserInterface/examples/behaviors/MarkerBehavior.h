
/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):	Martin Kelly
  Created on:	2012-01-16

  (C) Copyright 2012 Johns Hopkins University (JHU), All Rights
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
#include <cisst3DUserInterface/ui3VisibleAxes.h>

#include <list>
#include <limits>
#define MARKER_MAX 50

class MarkerBehaviorTextObject;
struct MarkerType;
enum OperatingMode;

// Always include last!
#include "ui3BehaviorsExport.h"

class CISST_EXPORT MarkerBehavior: public ui3BehaviorBase
{
    public:
        MarkerBehavior(const std::string & name);
        ~MarkerBehavior();

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
        void FirstButtonCallback(void);
        void PrimaryMasterButtonCallback(const prmEventButton & event);
        void SecondaryMasterButtonCallback(const prmEventButton & event);
        void MasterClutchPedalCallback(const prmEventButton & payload);
        void CameraControlPedalCallback(const prmEventButton & payload);
        void SetFiducialButtonCallback(void);
        void RegisterButtonCallback(void);
        void ClearFiducialsButtonCallback(void);
        StateType PreviousState;
        bool PreviousMaM;
        vctDouble3 PreviousCursorPosition;
        vctDouble3 Offset;
        vctFrm3 Position;
		vctFrm3 WristToTip;
        bool Following;
        bool Transition;
		OperatingMode ModeSelected;

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
        void AddMarker();
        void RemoveMarker();
        int FindClosestMarker();

        typedef  std::vector<MarkerType*> MarkersType;
        MarkersType Markers;

    private:
        ui3VisibleList * VisibleList;
		MarkerBehaviorTextObject * TextObject;
        ui3VisibleObject * MapCursor;
        ui3VisibleList * MarkerList;
        ui3VisibleAxes * MyMarkers[MARKER_MAX];
        bool RightMTMOpen, LeftMTMOpen;
        bool CameraPressed, ClutchPressed;
        bool MarkerDropped, MarkerRemoved;
        int MarkerCount;
        vctDouble3 PreviousSlavePosition;

        vctFrm3 ECMtoECMRCM, ECMRCMtoVTK;
        vctDouble3 centerRotated;
};

