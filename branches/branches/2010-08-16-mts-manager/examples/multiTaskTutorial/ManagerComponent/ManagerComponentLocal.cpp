/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: ManagerComponentLocal.h 

  Author(s):  Min Yang Jung
  Created on: 2010-09-01

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstCommon/cmnConstants.h>
#include <cisstOSAbstraction/osaGetTime.h>
#include <cisstOSAbstraction/osaSleep.h>

#include "ManagerComponentLocal.h"

const std::string CounterOddComponentType  = "CounterOddComponent";
const std::string CounterOddComponentName  = "CounterOddComponentObject";
const std::string CounterEvenComponentType = "CounterEvenComponent";
const std::string CounterEvenComponentName = "CounterEvenComponentObject";

const std::string NameCounterOddInterfaceProvided = "CounterOddInterfaceProvided";
const std::string NameCounterOddInterfaceRequired = "CounterOddInterfaceRequired";
const std::string NameCounterEvenInterfaceProvided = "CounterEvenInterfaceProvided";
const std::string NameCounterEvenInterfaceRequired = "CounterEvenInterfaceRequired";

CMN_IMPLEMENT_SERVICES(ManagerComponentLocal);

ManagerComponentLocal::ManagerComponentLocal(const std::string & componentName, double period):
    mtsTaskPeriodic(componentName, period, false, 1000)
{
     UseSeparateLogFileDefault();
}

void ManagerComponentLocal::Run(void) 
{
    ProcessQueuedCommands();

    static double lastTick = 0;
    static int count = 0;

    if (++count == 5) {
        //
        // Create the two components: odd counter and even counter
        //
        std::cout << std::endl << "Creating counter components....." << std::endl;

        std::cout << "> " << CounterOddComponentType << ", " << CounterOddComponentName << ": ";
        if (!RequestComponentCreate(CounterOddComponentType, CounterOddComponentName)) {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }

        std::cout << "> " << CounterEvenComponentType << ", " << CounterEvenComponentName << ": ";
        if (!RequestComponentCreate(CounterEvenComponentType, CounterEvenComponentName)) {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }

        // MJ: needs to be replaced with blocking command with return value
        std::cout << std::endl << "Wait for 5 seconds for \"Component Connect\"...." << std::endl;
        osaSleep(5.0);

        //
        // Connect the two components
        //
        std::cout << std::endl << "Connecting counter components....." << std::endl;
        std::cout << "> Connection 1: ";
        if (!RequestComponentConnect(CounterOddComponentName, NameCounterOddInterfaceRequired, 
            CounterEvenComponentName, NameCounterEvenInterfaceProvided))
        {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }

        std::cout << "> Connection 2: ";
        if (!RequestComponentConnect(CounterEvenComponentName, NameCounterEvenInterfaceRequired,
            CounterOddComponentName, NameCounterOddInterfaceProvided))
        {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }

        // MJ: needs to be replaced with blocking command with return value
        std::cout << std::endl << "Wait for 5 seconds for \"Component Start\"...." << std::endl;
        osaSleep(5.0);

        //
        // Start the two components
        //
        std::cout << std::endl << "Starting counter components....." << std::endl;
        std::cout << "> " << CounterOddComponentName << ": ";
        if (!RequestComponentStart(CounterOddComponentName)) {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }

        std::cout << "> " << CounterEvenComponentName << ": ";
        if (!RequestComponentStart(CounterEvenComponentName)) {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }

        std::cout << std::endl << "Wait for 5 seconds for \"Component Stop\"...." << std::endl;
        osaSleep(5.0);

        //
        // Stop the two components
        //
        std::cout << std::endl << "Stopping counter components....." << std::endl;
        std::cout << "> " << CounterOddComponentName << ": ";
        if (!RequestComponentStop(CounterOddComponentName)) {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }

        std::cout << "> " << CounterEvenComponentName << ": ";
        if (!RequestComponentStop(CounterEvenComponentName)) {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }

        std::cout << std::endl << "Wait for 5 seconds for \"Component Resume\"...." << std::endl;
        osaSleep(5.0);

        //
        // Resume the two components
        //
        std::cout << std::endl << "Resuming counter components....." << std::endl;
        std::cout << "> " << CounterOddComponentName << ": ";
        if (!RequestComponentResume(CounterOddComponentName)) {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }

        std::cout << "> " << CounterEvenComponentName << ": ";
        if (!RequestComponentResume(CounterEvenComponentName)) {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }
    }
}
