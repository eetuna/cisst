/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */
/* $Id$ */

#include <cisstVector.h>
#include <cisstOSAbstraction.h>
#include <cisstDevices.h>

#include "displayTaskOmniServer.h"
#include "displayUI.h"

using namespace std;

bool IsGlobalTaskManager = false;
bool IsServerTask = false;
string GlobalTaskManagerIP;
string ServerTaskIP;

int main(void)
{
    // log configuration
    cmnLogger::SetLoD(CMN_LOG_LOD_VERY_VERBOSE);
    cmnLogger::GetMultiplexer()->AddChannel(cout, CMN_LOG_LOD_VERY_VERBOSE);
    // add a log per thread
    osaThreadedLogFile threadedLog("example1-");
    cmnLogger::GetMultiplexer()->AddChannel(threadedLog, CMN_LOG_LOD_VERY_VERBOSE);
    // specify a higher, more verbose log level for these classes
    cmnClassRegister::SetLoD("mtsTaskInterface", CMN_LOG_LOD_VERY_VERBOSE);
    cmnClassRegister::SetLoD("mtsTaskManager", CMN_LOG_LOD_VERY_VERBOSE);
    cmnClassRegister::SetLoD("devSensableHD", CMN_LOG_LOD_VERY_VERBOSE);

    // create our two tasks
    const long PeriodDisplay = 10; // in milliseconds
    mtsTaskManager * taskManager = mtsTaskManager::GetInstance();
    displayTaskOmniServer * displayTaskObject =
        new displayTaskOmniServer("DISP", PeriodDisplay * cmn_ms);
    displayTaskObject->Configure();
    taskManager->AddTask(displayTaskObject);

#if (CISST_DEV_HAS_SENSABLEHD == ON)
    // name as defined in Sensable configuration
    std::string omniName("Omni1");
    devSensableHD * robotObject = new devSensableHD("Omni", "Omni1");    
	taskManager->AddTask(robotObject);
#endif

    taskManager->GlobalTaskManagerIP = "10.162.34.27";
    taskManager->ServerTaskIP = "10.162.34.27";

    // Set the type of task manager either as a server or as a client.
    // mtsTaskManager::SetTaskManagerType() should be called before
    // executing mtsTaskManager::Connect()
    taskManager->SetTaskManagerType(mtsTaskManager::TASK_MANAGER_CLIENT);

    osaSleep(1 * cmn_s);

    // Connect a required interface with a provided interface locally.
#if (CISST_DEV_HAS_SENSABLEHD == ON)
    // connect the tasks
    taskManager->Connect("DISP", "LocalRobot", "Omni", omniName);
    //taskManager->Connect("DISP", "Button1", "Omni", omniName + "Button1");
    taskManager->Connect("DISP", "Button2", "Omni", omniName + "Button2");
#endif

    // generate a nice tasks diagram
    std::ofstream dotFile("example1.dot"); 
    taskManager->ToStreamDot(dotFile);
    dotFile.close();

    // create the tasks, i.e. find the commands
    taskManager->CreateAll();
    // start the periodic Run
    taskManager->StartAll();

    // wait until the close button of the UI is pressed
    while (1) {
        osaSleep(100.0 * cmn_ms); // sleep to save CPU
        if (displayTaskObject->GetExitFlag()) {
            break;
        }
    }
    // cleanup
    taskManager->KillAll();

    osaSleep(2 * PeriodDisplay * cmn_ms);
    while (!displayTaskObject->IsTerminated()) osaSleep(PeriodDisplay);

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
