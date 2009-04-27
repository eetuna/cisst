/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */
/* $Id: main.cpp 78 2009-02-25 16:13:12Z adeguet1 $ */

#include <cisstCommon.h>
#include <cisstOSAbstraction.h>
#include <cisstMultiTask.h>

#include "sineTask.h"
#include "displayTask.h"
#include "UITask.h"
#include "displayUI.h"

#include <string>

using namespace std;

/*
    Server task : SIN - provided interface

    Client task : DISP - required interface
*/

//
// TODO:
//  1. link task interface proxy instance to mtsTask (differentiate prov. vs. req.)
//  2. Confirm that task interface proxy works well
//  3. transform current methods into network version
//

void help(const char * programName)
{
    cerr << endl 
         << "Usage: multiTaskTutorialExample1-2 [OPTIONS]" << endl 
         << endl
         << "[OPTIONS]" << endl
         << "  -s,    run a server task manager (global task manager)" << endl
         << "  -cs,   run a client task manager with a server task" << endl
         << "  -cc,   run a client task manager with a client task" << endl
         << endl;
}

int main(int argc, char * argv[])
{
    string serverTaskName = "SIN", clientTaskName = "DISP";

    // Check arguments
    bool IsGlobalTaskManager = false;
    bool IsServerTask = false;

    if (argc == 2) {
        if (strcmp(argv[1], "-s") == 0) {
            IsGlobalTaskManager = true;
        } else if (strcmp(argv[1], "-cs") == 0 || strcmp(argv[1], "-cc") == 0) {
            IsGlobalTaskManager = false;

            // Create a server task
            if (strcmp(argv[1], "-cs") == 0) {
                IsServerTask = true;
            } 
            // Create a client task
            else {
                IsServerTask = false;
            }
        } else {
            help(argv[0]);
            return 1;
        }
    } else {
        help(argv[0]);
        return 1;
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

    //-------------------------------------------------------------------------
    // Create default local tasks
    //-------------------------------------------------------------------------
    // Get the TaskManager instance and set operation mode
    mtsTaskManager * taskManager = mtsTaskManager::GetInstance();

    const double PeriodDisplay = 50 * cmn_ms;

    sineTask * sineTaskObject = NULL;
    displayTask * displayTaskObject = NULL;
    UITask * UITaskObject = NULL;

    if (IsGlobalTaskManager) {
        UITaskObject = new UITask("UITask", PeriodDisplay);
        UITaskObject->Configure();

        taskManager->AddTask(UITaskObject);

        taskManager->SetTaskManagerType(mtsTaskManager::TASK_MANAGER_SERVER);
    } else {
        //-------------------------------------------------------------------------
        // Create a task which works over networks
        //-------------------------------------------------------------------------
        const double PeriodSine = 1 * cmn_ms;        

        if (IsServerTask) {
            sineTaskObject = new sineTask(serverTaskName, PeriodSine);
            UITaskObject = new UITask("UITask", PeriodDisplay);
            UITaskObject->Configure();
            
            taskManager->AddTask(UITaskObject);
            taskManager->AddTask(sineTaskObject);
        } else {
            displayTaskObject = new displayTask(clientTaskName, PeriodDisplay);
            displayTaskObject->Configure();

            taskManager->AddTask(displayTaskObject);        
        }

        // Set the type of task manager either as a server or as a client.
        // mtsTaskManager::SetTaskManagerType() should be called before
        // executing mtsTaskManager::Connect()
        taskManager->SetTaskManagerType(mtsTaskManager::TASK_MANAGER_CLIENT);

        //
        // TODO: Hide this waiting routine inside mtsTaskManager using events or other things.
        //
        osaSleep(1 * cmn_s);

        // Connect the tasks across networks
        if (!IsServerTask) {
            taskManager->Connect(clientTaskName, "DataGenerator", serverTaskName, "MainInterface");
        }
    }

    // create the tasks, i.e. find the commands
    taskManager->CreateAll();
    // start the periodic Run
    taskManager->StartAll();

    if (IsGlobalTaskManager) {
        while (1) {
            osaSleep(10 * cmn_ms);
            if (UITaskObject->GetExitFlag()) {
                break;
            }
        }
    } else {
        if (IsServerTask) {
            while (1) {
                osaSleep(10 * cmn_ms);
                if (UITaskObject->GetExitFlag()) {
                    break;
                }
            }
        } else {
            while (1) {
                osaSleep(10 * cmn_ms);
                if (displayTaskObject->GetExitFlag()) {
                    break;
                }
            }
        }
    }

    // cleanup
    taskManager->KillAll();

    osaSleep(PeriodDisplay * 2);

    if (IsGlobalTaskManager) {
        while (!UITaskObject->IsTerminated()) osaSleep(PeriodDisplay);
    } else {
        if (IsServerTask) {
            while (!sineTaskObject->IsTerminated()) osaSleep(PeriodDisplay);
            while (!UITaskObject->IsTerminated()) osaSleep(PeriodDisplay);
        } else {
            while (!displayTaskObject->IsTerminated()) osaSleep(PeriodDisplay);
        }
    }

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
