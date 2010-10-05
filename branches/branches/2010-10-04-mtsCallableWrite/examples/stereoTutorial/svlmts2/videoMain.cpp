/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
 $Id: $

 Author(s):  Anton Deguet
 Created on: 2010

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
#include <cisstStereoVision.h>

#include "mtsManagerComponent.h"

int main(int argc, char * argv[])
{
    // set global component manager IP
    std::string globalComponentManagerIP;
    if (argc == 1) {
        std::cerr << "Using default, i.e. 127.0.0.1 to find global component manager" << std::endl;
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
    osaThreadedLogFile threadedLog("svlExMultitask2Video-");
    cmnLogger::GetMultiplexer()->AddChannel(threadedLog, CMN_LOG_LOD_VERY_VERBOSE);
    // specify a higher, more verbose log level for these classes
    cmnClassRegister::SetLoDForAllClasses(CMN_LOG_LOD_VERY_VERBOSE);

    svlInitialize();
    mtsComponentManager * componentManager = mtsComponentManager::GetInstance(globalComponentManagerIP, "svlExMultitask2Video");
    mtsManagerComponent * managerComponent = new mtsManagerComponent("Manager");

    componentManager->AddComponent(managerComponent);

    // create the tasks, i.e. find the commands
    componentManager->CreateAll();
    // start all
    componentManager->StartAll();

    osaSleep(2.0 * cmn_s);

    int ch = 0;
    while (ch != 'q') {
        ch = cmnGetChar();
    }

    // cleanup
    componentManager->KillAll();
    componentManager->Cleanup();

    return 0;
}

