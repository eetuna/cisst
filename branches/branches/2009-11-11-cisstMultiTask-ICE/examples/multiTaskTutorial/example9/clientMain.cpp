/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */
/* $Id$ */

#include <cisstCommon.h>
#include <cisstOSAbstraction.h>
#include <cisstMultiTask.h>

#include "clientTask.h"

int main(int argc, char * argv[])
{
    // TODO: uncomment this
    //if (argc != 2) {
    //    std::cerr << "Usage: " << argv[0] << "[global component manager IP]" << std::endl;
    //    exit(-1);
    //}

    // Set global component manager IP
    //const std::string globalComponentManagerIP(argv[1]);
    const std::string globalComponentManagerIP("127.0.0.1");
    std::cout << "Global component manager IP is set as " << globalComponentManagerIP << std::endl;

    // log configuration
    cmnLogger::SetLoD(CMN_LOG_LOD_VERY_VERBOSE);
    cmnLogger::GetMultiplexer()->AddChannel(std::cout, CMN_LOG_LOD_VERY_VERBOSE);
    // add a log per thread
    osaThreadedLogFile threadedLog("example9Client");
    cmnLogger::GetMultiplexer()->AddChannel(threadedLog, CMN_LOG_LOD_VERY_VERBOSE);
    // specify a higher, more verbose log level for these classes
    cmnClassRegister::SetLoD("mtsTaskInterface", CMN_LOG_LOD_VERY_VERBOSE);
    cmnClassRegister::SetLoD("mtsTaskManager", CMN_LOG_LOD_VERY_VERBOSE);
    cmnClassRegister::SetLoD("clientTask", CMN_LOG_LOD_VERY_VERBOSE);

    // Get the local component manager
    mtsTaskManager * taskManager = mtsTaskManager::GetInstance("example9Client", globalComponentManagerIP);

    // create our server task
    const double PeriodClient = 10 * cmn_ms; // in milliseconds
    clientTask * client = new clientTask("Client", PeriodClient);
    //taskManager->AddTask(client);  // deprecated API
    taskManager->AddComponent(client);

    // Connect the tasks across networks
    //taskManager->Connect("Client", "Required", "Server", "Provided");
    if (!taskManager->Connect("example9Client", "Client", "Required", 
                             "example9Server", "Server", "Provided"))
    {
        CMN_LOG_INIT_ERROR << "Connect failed" << std::endl;
        exit(-1);
    }

    // create the tasks, i.e. find the commands
    taskManager->CreateAll();
    // start the periodic Run
    taskManager->StartAll();

    while (1) {
        osaSleep(10 * cmn_ms);
    }
    
    // cleanup
    taskManager->KillAll();
    taskManager->Cleanup();
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
