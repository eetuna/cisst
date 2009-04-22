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

void help(const char * programName)
{
    cerr << endl 
         << "Usage: multiTaskTutorialExample1-2 [OPTIONS]" << endl << endl
         << "  -s, --server          run Task Manager as a server (global Task Manager)" << endl
         << "  -c, --client          run Task Manager as a client" << endl << endl;
}

int main(int argc, char * argv[])
{
    // Check arguments
    bool RunGlobalTaskManager = false;
    if (argc != 2) {
        help(argv[0]);
        return 1;
    } else {
        if (strcmp(argv[1], "-s") == 0 || strcmp(argv[1], "--server") == 0) {
            RunGlobalTaskManager = true;
        } else if (strcmp(argv[1], "-c") == 0 || strcmp(argv[1], "--client") == 0) {
            RunGlobalTaskManager = false;
        } else {
            help(argv[0]);
            return 1;
        }
    }

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

    // Get the TaskManager instance and set an operation mode
    mtsTaskManager * taskManager = mtsTaskManager::GetInstance();

    // create our two tasks
    const double PeriodSine = 1 * cmn_ms; // in milliseconds
    const double PeriodDisplay = 50 * cmn_ms; // in milliseconds
    sineTask * sineTaskObject = new sineTask("SIN", PeriodSine);
    displayTask * displayTaskObject = new displayTask("DISP", PeriodDisplay);
    displayTaskObject->Configure();

    // add the tasks to the task manager
    taskManager->AddTask(sineTaskObject);
    taskManager->AddTask(displayTaskObject);

    // connect the tasks, task.RequiresInterface -> task.ProvidesInterface
    taskManager->Connect("DISP", "DataGenerator", "SIN", "MainInterface");

    // create the tasks, i.e. find the commands
    taskManager->CreateAll();
    // start the periodic Run
    taskManager->StartAll();

    // Currently don't consider the case that state transition occurs from
    // TASK_MANAGER_CLIENT/SERVER to TASK_MANAGER_LOCAL.
    if (RunGlobalTaskManager) {
        taskManager->SetTaskManagerMode(mtsTaskManager::TASK_MANAGER_SERVER);
    } else {
        taskManager->SetTaskManagerMode(mtsTaskManager::TASK_MANAGER_CLIENT);
    }

    while (1) {
        osaSleep(10 * cmn_ms);
        if (displayTaskObject->GetExitFlag()) {
            break;
        }
    }

    // cleanup
    taskManager->KillAll();

    osaSleep(PeriodDisplay * 2);
    while (!sineTaskObject->IsTerminated()) osaSleep(PeriodDisplay);
    while (!displayTaskObject->IsTerminated()) osaSleep(PeriodDisplay);

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
