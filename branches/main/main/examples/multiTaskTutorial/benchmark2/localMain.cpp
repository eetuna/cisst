/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */
/* $Id: localMain.cpp 671 2009-08-13 02:41:31Z adeguet1 $ */

#include <cisstCommon.h>
#include <cisstOSAbstraction.h>
#include <cisstMultiTask.h>

#include "serverTask.h"
#include "clientTask.h"
#include "configuration.h"

int main(int CMN_UNUSED(argc), char ** CMN_UNUSED(argv))
{
    // log configuration
    cmnLogger::SetLoD(CMN_LOG_LOD_VERY_VERBOSE);
    cmnLogger::GetMultiplexer()->AddChannel(std::cout, CMN_LOG_LOD_RUN_ERROR);
    // add a log per thread
    osaThreadedLogFile threadedLog("benchmark2Local");
    cmnLogger::GetMultiplexer()->AddChannel(threadedLog, CMN_LOG_LOD_RUN_ERROR);

    // create our two tasks
    serverTask * server = new serverTask("Server", confServerPeriod);
    clientTask * client = new clientTask("Client", confClientPeriod);

    // add the tasks to the task manager
    mtsTaskManager * taskManager = mtsTaskManager::GetInstance();
    taskManager->AddTask(client);
    taskManager->AddTask(server);

    // connect the two tasks. The server task provides a provided interface
	// and the client task requires a required interface.
    taskManager->Connect("Client", "Required", "Server", "Provided");

    // create the tasks, i.e. find the commands
    taskManager->CreateAll();
    // start the periodic Run
    taskManager->StartAll();

    while(!client->IsBenchmarkCompleted()) {
        osaSleep(10.0 * cmn_ms);
    }

    // cleanup
    taskManager->Cleanup();
    taskManager->KillAll();

    // To prevent crash due to CMN_LOG_CLASS (cmnLODOutputMultiplexer).
    osaSleep(0.5 * cmn_s);

    return 0;
}

/*
  Author(s):  Min Yang Jung
  Created on: 2009-09-09

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/
