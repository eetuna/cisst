/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */
/* $Id: C2ServerTask.cpp 1088 2010-01-28 01:47:20Z mjung5 $ */

#include <cisstCommon/cmnConstants.h>
#include "C2ServerTask.h"
#include "fltkMutex.h"

// required to implement the class services, see cisstCommon
CMN_IMPLEMENT_SERVICES(C2ServerTask);

C2ServerTask::C2ServerTask(const std::string & taskName, double period):
    // base constructor, same task name and period.  Set the length of
    // state table to 5000
    mtsTaskPeriodic(taskName, period, false, 5000)
{
    // add ServerData to the StateTable defined in mtsTask
    this->StateTable.AddData(ReadValue, "ReadValue");
    // add one interface, this will create an mtsTaskInterface
    mtsProvidedInterface * provided = AddProvidedInterface("p1");
    if (provided) {
        provided->AddCommandVoid(&C2ServerTask::Void, this, "Void");
        provided->AddCommandWrite(&C2ServerTask::Write, this, "Write");
        provided->AddCommandReadState(this->StateTable, this->ReadValue, "Read");
        provided->AddCommandQualifiedRead(&C2ServerTask::QualifiedRead, this, "QualifiedRead");
        provided->AddEventVoid(this->EventVoid, "EventVoid");
        provided->AddEventWrite(this->EventWrite, "EventWrite", mtsDouble(3.14));
    }

    provided = AddProvidedInterface("p2");
    if (provided) {
        provided->AddCommandVoid(&C2ServerTask::Void, this, "Void");
        provided->AddCommandWrite(&C2ServerTask::Write, this, "Write");
        provided->AddCommandReadState(this->StateTable, this->ReadValue, "Read");
        provided->AddCommandQualifiedRead(&C2ServerTask::QualifiedRead, this, "QualifiedRead");
        provided->AddEventVoid(this->EventVoid, "EventVoid");
        provided->AddEventWrite(this->EventWrite, "EventWrite", mtsDouble(3.14));
    }
}


void C2ServerTask::Void(void)
{
    CMN_LOG_CLASS_RUN_VERBOSE << "Void" << std::endl;
    fltkMutex.Lock();
    {
        if (UI.VoidValue->value() == 0) {
            UI.VoidValue->value(1);
        } else {
            UI.VoidValue->value(0);
        }
    }
    fltkMutex.Unlock();
}


void C2ServerTask::Write(const mtsDouble & data)
{
    CMN_LOG_CLASS_RUN_VERBOSE << "Write" << std::endl;
    fltkMutex.Lock();
    {
        UI.WriteValue->value(data.Data);
    }
    fltkMutex.Unlock();
}


void C2ServerTask::QualifiedRead(const mtsDouble & data, mtsDouble & placeHolder) const
{
    placeHolder.Data = data.Data + UI.ReadValue->value();
}


void C2ServerTask::Startup(void)
{
    // make the UI visible
    fltkMutex.Lock();
    {
        UI.show(0, NULL);
        UI.Opened = true;
    }
    fltkMutex.Unlock();
}

void C2ServerTask::Run(void) {
    if (UIOpened()) {
        // process the commands received, i.e. possible SetServerAmplitude
        ProcessQueuedCommands();
        // compute the new values based on the current time and amplitude
        fltkMutex.Lock();
        {
            if (UI.VoidEventRequested) {
                CMN_LOG_CLASS_RUN_VERBOSE << "Run: VoidEventRequested" << std::endl;
                this->EventVoid();
                UI.VoidEventRequested = false;
            }
            
            if (UI.WriteEventRequested) {
                CMN_LOG_CLASS_RUN_VERBOSE << "Run: WriteEventRequested" << std::endl;
                this->EventWrite(mtsDouble(UI.ReadValue->value()));
                UI.WriteEventRequested = false;
            }
            
            this->ReadValue = UI.ReadValue->value();
            Fl::check();
        }
    fltkMutex.Unlock();
    }
}

/*
  Author(s):  Anton Deguet
  Created on: 2009-08-10

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/
