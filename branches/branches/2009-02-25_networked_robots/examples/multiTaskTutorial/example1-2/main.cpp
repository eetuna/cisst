/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */
/* $Id: main.cpp 78 2009-02-25 16:13:12Z adeguet1 $ */

#include <cisstCommon.h>
#include <cisstOSAbstraction.h>
#include <cisstMultiTask.h>

#include "sineTask.h"
#include "displayTask.h"
#include "displayUI.h"

using namespace std;

int main(int argc, char * argv[])
{
    // log configuration
    cmnLogger::SetLoD(10);
    cmnLogger::GetMultiplexer()->AddChannel(cout, 10);
    // add a log per thread
    osaThreadedLogFile threadedLog("example1-2_");
    cmnLogger::GetMultiplexer()->AddChannel(threadedLog, 10);
    // specify a higher, more verbose log level for these classes
    cmnClassRegister::SetLoD("sineTask", 10);
    cmnClassRegister::SetLoD("displayTask", 10);
    cmnClassRegister::SetLoD("mtsTaskInterface", 10);
    cmnClassRegister::SetLoD("mtsTaskManager", 10);

    // create our two tasks
    const double PeriodSine = 1 * cmn_ms; // in milliseconds
    const double PeriodDisplay = 50 * cmn_ms; // in milliseconds

    // Get the TaskManager instance
    mtsTaskManager * taskManager = mtsTaskManager::GetInstance();

    // Create a displayTask instance
    displayTask * displayTaskObject = new displayTask("DISP", PeriodDisplay);
    displayTaskObject->Configure();
    taskManager->AddTask(displayTaskObject);

    sineTask * task1 = NULL;
    sineTask * task2 = NULL;
    sineTask * task3 = NULL;
    sineTask * task4 = NULL;
    sineTask * task5 = NULL;

    if (argc == 1) {
        // Server
        task1 = new sineTask("SIN_1", PeriodSine);
        task2 = new sineTask("SIN_2", PeriodSine * 10);
        task3 = new sineTask("SIN_3", PeriodSine * 100);

        taskManager->AddTask(task1);
        taskManager->AddTask(task2);
        taskManager->AddTask(task3);        

        taskManager->Connect("DISP", "DataGenerator", "SIN_1", "MainInterface");

        taskManager->CreateAll();
        taskManager->StartAll();

        taskManager->SetTaskManagerMode(mtsTaskManager::TASK_MANAGER_SERVER);
    } else {
        // Client
        task1 = new sineTask("COS_1", PeriodSine);
        task2 = new sineTask("COS_2", PeriodSine * 2);
        task3 = new sineTask("COS_3", PeriodSine * 20);
        task4 = new sineTask("COS_4", PeriodSine * 200);
        task5 = new sineTask("COS_5", PeriodSine * 2000);

        taskManager->AddTask(task1);
        taskManager->AddTask(task2);
        taskManager->AddTask(task3);
        taskManager->AddTask(task4);
        taskManager->AddTask(task5);

        taskManager->Connect("DISP", "DataGenerator", "COS_1", "MainInterface");

        taskManager->CreateAll();
        taskManager->StartAll();

        taskManager->SetTaskManagerMode(mtsTaskManager::TASK_MANAGER_CLIENT);
    }

    while (1) {
        osaSleep(100 * cmn_ms);
        if (displayTaskObject->GetExitFlag()) {
            break;
        }
    }

    // cleanup
    taskManager->KillAll();

    osaSleep(PeriodDisplay * 2);

#define WAIT_FOR_SAFE_TERMINATION(_instance)\
    if (_instance) {\
        while (!_instance->IsTerminated()) osaSleep(PeriodDisplay);\
    }

    WAIT_FOR_SAFE_TERMINATION(task1);
    WAIT_FOR_SAFE_TERMINATION(task2);
    WAIT_FOR_SAFE_TERMINATION(task3);
    WAIT_FOR_SAFE_TERMINATION(task4);
    WAIT_FOR_SAFE_TERMINATION(task5);
    WAIT_FOR_SAFE_TERMINATION(displayTaskObject);

    return 0;
}

/*
  Author(s):  Ankur Kapoor, Peter Kazanzides, Anton Deguet, Min Yang Jung
  Created on: 2004-04-30

  (C) Copyright 2004-2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/
