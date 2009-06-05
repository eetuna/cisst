/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */
/* $Id: main.cpp 332 2009-05-11 00:57:59Z mjung5 $ */

#include <cisstVector.h>
#include <cisstOSAbstraction.h>
#include <cisstDevices.h>

#include "displayTaskOmniClient.h"
#include "displayUIOmniClient.h"

using namespace std;

int main(void)
{
    // log configuration
    cmnLogger::SetLoD(10);
    cmnLogger::GetMultiplexer()->AddChannel(cout, 10);
    // add a log per thread
    osaThreadedLogFile threadedLog("example1-");
    cmnLogger::GetMultiplexer()->AddChannel(threadedLog, 10);
    // specify a higher, more verbose log level for these classes
    cmnClassRegister::SetLoD("mtsTaskInterface", 10);
    cmnClassRegister::SetLoD("mtsTaskManager", 10);
    cmnClassRegister::SetLoD("devSensableHD", 10);

    // create our two tasks
    const long PeriodDisplay = 10; // in milliseconds
    mtsTaskManager * taskManager = mtsTaskManager::GetInstance();
    displayTaskOmniClient * displayTaskObject =
        new displayTaskOmniClient("DISP", PeriodDisplay * cmn_ms);
    taskManager->AddTask(displayTaskObject);

    taskManager->GlobalTaskManagerIP = "127.0.0.1";
    taskManager->ServerTaskIP = "127.0.0.1";

    taskManager->SetTaskManagerType(mtsTaskManager::TASK_MANAGER_CLIENT);
    osaSleep(1 * cmn_s);

    // connect the tasks
#if (CISST_DEV_HAS_SENSABLEHD == ON)
    // connect the tasks
    taskManager->Connect("DISP", "Robot", "Omni", "Omni1");
    // taskManager->Connect("DISP", "Button1", "Omni", omniName + "Button1");
    //taskManager->Connect("DISP", "Button2", "Omni", "Omni1" + "Button2");
#endif

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
