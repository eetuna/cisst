/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */
/* $Id: displayTask.cpp 433 2009-06-09 22:10:24Z adeguet1 $ */

/*
  Author(s):  Gorkem Sevinc, Anton Deguet 
  Created on: 2009-07-06

  (C) Copyright 2008 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#include "displayTask.h"

CMN_IMPLEMENT_SERVICES(displayTask);

displayTask::displayTask(const std::string & taskName, double period):
    mtsTaskPeriodic(taskName, period, false, 500),
    ExitFlag(false)
{
    mtsRequiredInterface *requiredInterface = AddRequiredInterface("TeleoperationParameters");
    if(requiredInterface)
    {
        requiredInterface->AddFunction("GetMasterClutch", GetMasterClutch);
        requiredInterface->AddFunction("GetSlaveClutch", GetSlaveClutch);
        requiredInterface->AddFunction("GetMasterSlaveClutch", GetMasterSlaveClutch);
        requiredInterface->AddFunction("GetCollaborativeControlParameter", GetCollaborativeControlParameter);

        requiredInterface->AddFunction("SetMasterClutch", SetMasterClutch);
        requiredInterface->AddFunction("SetSlaveClutch", SetSlaveClutch);
        requiredInterface->AddFunction("SetMasterSlaveClutch", SetMasterSlaveClutch);
        requiredInterface->AddFunction("SetCollaborativeControlParameter", SetCollaborativeControlParameter);
    }

    commandedForceLimit = 0.0;
    commandedLinearGain = 0.0;
    commandedForceCoeff = 0.0;
    commandedMasterClutch = false;
    commandedSlaveClutch = false;
    commandedForceMode = 0;
} 

void displayTask::Startup(void)
{
    UI.DisplayWindow->show(); 
}

void displayTask::Run(void)
{
    bool ParameterChanged = false;
    ProcessQueuedCommands();
    ProcessQueuedEvents();

    Fl::check();
    if (Fl::thread_message() != 0) {
        std::cerr << "GUI Error" << Fl::thread_message() << std::endl;
        return;
    }
    GetCollaborativeControlParameter(MainTaskParameter);
   
    UI.ForceLimitVal->value(MainTaskParameter.ForceLimit());
    UI.ScaleFactorVal->value(MainTaskParameter.LinearGain());
    UI.ForceCoefficientVal->value(MainTaskParameter.ForceFeedbackRatio());

    commandedForceLimit = UI.ForceLimit->value();
    if(MainTaskParameter.LinearGain() != commandedForceLimit) {   
        MainTaskParameter.SetForceLimit(commandedForceLimit);
        ParameterChanged = true;
    }

    commandedLinearGain = UI.ScaleFactor->value();
    if(MainTaskParameter.LinearGain() != commandedLinearGain) {
        MainTaskParameter.SetLinearGain(commandedLinearGain);
        ParameterChanged = true;
    }

    commandedForceCoeff = UI.ForceCoefficient->value();
    if(commandedForceCoeff != MainTaskParameter.ForceFeedbackRatio()) {
        MainTaskParameter.SetForceFeedbackRatio(commandedForceCoeff);
        ParameterChanged = true;
    }

    GetMasterClutch(MasterClutch);
    GetSlaveClutch(SlaveClutch);
    GetMasterSlaveClutch(MasterSlaveClutch);

    const char * ForceModeName;
    if(MainTaskParameter.ForceMode() == robCollaborativeControlForce::ParameterType::RATCHETED) {
        ForceModeName = "Ratchet";
    } else if(MainTaskParameter.ForceMode() == robCollaborativeControlForce::ParameterType::CAPPED) {
        ForceModeName = "Capping";
    } else if(MainTaskParameter.ForceMode() == robCollaborativeControlForce::ParameterType::RAW) {
        ForceModeName = "Raw";
    } else {
        ForceModeName = "";
    }
    UI.CurrentForceMode->value(ForceModeName);

    if(UI.ClutchMaster->value() == 1 && UI.ClutchSlave->value() == 0) {
        commandedMasterClutch = true;
        if(MasterClutch != commandedMasterClutch) {
            SetMasterClutch(commandedMasterClutch);
        }
    } else if(UI.ClutchMaster->value() == 0 && UI.ClutchSlave->value() == 0) {
        commandedMasterClutch = false;
        if(MasterClutch != commandedMasterClutch) {
            SetMasterClutch(commandedMasterClutch);
        }
    }

    if(UI.ClutchSlave->value() == 1 && UI.ClutchMaster->value() == 0) {
        commandedSlaveClutch = true;
        if(SlaveClutch != commandedSlaveClutch) {
            SetSlaveClutch(commandedSlaveClutch);
        }
    } else if(UI.ClutchSlave->value() == 0 && UI.ClutchMaster->value() == 0) {
        commandedSlaveClutch = false;
        if(SlaveClutch != commandedSlaveClutch) {
            SetSlaveClutch(commandedSlaveClutch);
        }
    }

    if(UI.ClutchMaster->value() == 1 && UI.ClutchSlave->value() == 1)
    {
        commandedMasterSlaveClutch = true;
        if(MasterSlaveClutch != commandedMasterSlaveClutch) {
            SetMasterSlaveClutch(commandedMasterSlaveClutch);
            commandedMasterClutch = false;
            SetMasterClutch(commandedMasterClutch);
            SetSlaveClutch(commandedMasterClutch);
        }
    } else {
        commandedMasterSlaveClutch = false;
        if(MasterSlaveClutch != commandedMasterSlaveClutch) {
            SetMasterSlaveClutch(commandedMasterSlaveClutch);
        }
    }

    if(UI.RatchetOn->value() == 1) {
        commandedForceMode = 1;
        if((int)MainTaskParameter.ForceMode() != commandedForceMode.Data) {
            MainTaskParameter.SetForceMode((robCollaborativeControlForce::ParameterType::ForceModeType)commandedForceMode.Data);
            ParameterChanged = true;
        }
    } else if(UI.CappingOn->value() == 1) {
        commandedForceMode = 2;
        if((int)MainTaskParameter.ForceMode() != commandedForceMode.Data) {
            MainTaskParameter.SetForceMode((robCollaborativeControlForce::ParameterType::ForceModeType)commandedForceMode.Data);
            ParameterChanged = true;
        }
    } else if(UI.RawOn->value() == 1) {
        commandedForceMode = 0;
        if((int)MainTaskParameter.ForceMode() != commandedForceMode.Data) {
            MainTaskParameter.SetForceMode((robCollaborativeControlForce::ParameterType::ForceModeType)commandedForceMode.Data);
            ParameterChanged = true;
        }
    }
    
    if(ParameterChanged == true) {
        SetCollaborativeControlParameter(MainTaskParameter);
    }

    if(UI.QuitClicked) {
        this->ExitFlag = true;
    }
    osaSleep(10.0 * cmn_ms);

}
