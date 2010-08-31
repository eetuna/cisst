/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */
/* $Id: serverMain.cpp 1660 2010-07-22 20:22:54Z adeguet1 $ */

/*
  Author(s):  Min Yang Jung
  Created on: 2010-08-20

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#include <cisstCommon.h>
#include <cisstOSAbstraction.h>
#include <cisstMultiTask.h>

#include "displayTask.h"

int main(int argc, char * argv[])
{
    // set global component manager IP
    std::string globalComponentManagerIP;
    if (argc == 1) {
        std::cerr << "Use localhost (127.0.0.1) to find global component manager" << std::endl;
        globalComponentManagerIP = "127.0.0.1";
    } else if (argc == 2) {
        globalComponentManagerIP = argv[1];
    } else {
        std::cerr << "Usage: " << argv[0] << " (global component manager IP)" << std::endl;
        return -1;
    }

    // log configuration
    cmnLogger::SetLoD(CMN_LOG_LOD_VERY_VERBOSE);
    cmnLogger::GetMultiplexer()->AddChannel(std::cout, CMN_LOG_LOD_VERY_VERBOSE);
    // add a log per thread
    osaThreadedLogFile threadedLog("DisplayTask");
    cmnLogger::GetMultiplexer()->AddChannel(threadedLog, CMN_LOG_LOD_VERY_VERBOSE);
    // specify a higher, more verbose log level for these classes
    cmnClassRegister::SetLoD("mtsManagerLocal", CMN_LOG_LOD_VERY_VERBOSE);
    cmnClassRegister::SetLoD("sineTask", CMN_LOG_LOD_VERY_VERBOSE);

    // Get the TaskManager instance and set operation mode
    mtsManagerLocal * taskManager = mtsManagerLocal::GetInstance(
        globalComponentManagerIP, "ProcessDisp");

     // create our two tasks
    const double PeriodDisplay = 50 * cmn_ms; // in milliseconds
    displayTask * displayTaskObject = new displayTask("DISP", PeriodDisplay);
    displayTaskObject->Configure();

    // add the tasks to the task manager
    taskManager->AddTask(displayTaskObject);

    // connect the tasks, task.RequiresInterface -> task.ProvidesInterface
    if (!taskManager->Connect("ProcessDisp", "DISP", "DataGenerator", 
                              mtsManagerLocal::ProcessNameOfLCMWithGCM, "SIN", "MainInterface"))
    {
        CMN_LOG_INIT_ERROR << "Failed to connect" << std::endl;
        return 1;
    }

    // create the tasks, i.e. find the commands
    taskManager->CreateAll();
    // start the periodic Run
    taskManager->StartAll();

    // wait until the close button of the UI is pressed
    while (!displayTaskObject->IsTerminated()) {
        osaSleep(100.0 * cmn_ms); // sleep to save CPU
    }

    // cleanup
    taskManager->KillAll();
    taskManager->Cleanup();

    return 0;
}

