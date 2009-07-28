/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */
/* $Id: main.cpp 564 2009-07-18 04:09:18Z adeguet1 $ */

#include <cisstCommon.h>
#include <cisstOSAbstraction.h>
#include <cisstMultiTask.h>

#include "sineTask.h"
#include "displayTask.h"
#include "oscilloscopeTask.h"

using namespace std;

int main(void)
{
    // log configuration
    cmnLogger::SetLoD(CMN_LOG_LOD_VERY_VERBOSE);
    cmnLogger::GetMultiplexer()->AddChannel(cout, CMN_LOG_LOD_VERY_VERBOSE);
    // add a log per thread
    osaThreadedLogFile threadedLog("example1-");
    cmnLogger::GetMultiplexer()->AddChannel(threadedLog, CMN_LOG_LOD_VERY_VERBOSE);
    // specify a higher, more verbose log level for these classes
    cmnClassRegister::SetLoD("sineTask", CMN_LOG_LOD_VERY_VERBOSE);
    cmnClassRegister::SetLoD("displayTask", CMN_LOG_LOD_VERY_VERBOSE);
    cmnClassRegister::SetLoD("mtsTaskInterface", CMN_LOG_LOD_VERY_VERBOSE);
    cmnClassRegister::SetLoD("mtsTaskManager", CMN_LOG_LOD_VERY_VERBOSE);

    // create sample sine task
    const double PeriodSine = 1 * cmn_ms; // in milliseconds
    const double PeriodDisplay = 10 * cmn_ms; // in milliseconds
    const double PeriodOscilloscope = 20 * cmn_ms; // in milliseconds
    mtsTaskManager * taskManager = mtsTaskManager::GetInstance();
    sineTask * sineTaskObject = new sineTask("SIN", PeriodSine);
    displayTask * displayTaskObject = new displayTask("DISP", PeriodDisplay);
    displayTaskObject->Configure();
    oscilloscopeTask * oscilloscopeTaskObject = new oscilloscopeTask("OSCILLOSCOPE", PeriodOscilloscope);
    oscilloscopeTaskObject->Configure();

    // add the tasks to the task manager
    taskManager->AddTask(sineTaskObject);
    taskManager->AddTask(displayTaskObject);
    taskManager->AddTask(oscilloscopeTaskObject);

    // connect the tasks, task.RequiresInterface -> task.ProvidesInterface
    taskManager->Connect("DISP", "DataGenerator", "SIN", "MainInterface");

    // create the tasks, i.e. find the commands
    taskManager->CreateAll();
    // start the periodic Run
    taskManager->StartAll();

    // wait until the close button of the UI is pressed
    while (!displayTaskObject->IsTerminated()) {
        osaSleep(10.0 * cmn_ms); // sleep to save CPU
    }
    // cleanup
    taskManager->KillAll();

    osaSleep(PeriodSine * 2);
    while (!sineTaskObject->IsTerminated()) osaSleep(PeriodSine);

    return 0;
}

/*
  Author(s):  Ankur Kapoor, Peter Kazanzides, Anton Deguet
  Created on: 2004-04-30

  (C) Copyright 2004-2008 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/
