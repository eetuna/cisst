/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: ManagerComponent.h 

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

#include "ManagerComponent.h"

const std::string PeerProcessName          = "ProcessCounter";
const std::string CounterOddComponentType  = "CounterOddComponent";
const std::string CounterOddComponentName  = "CounterOddComponentObject";
const std::string CounterEvenComponentType = "CounterEvenComponent";
const std::string CounterEvenComponentName = "CounterEvenComponentObject";

CMN_IMPLEMENT_SERVICES(ManagerComponent);

ManagerComponent::ManagerComponent(const std::string & componentName, double period):
    mtsTaskPeriodic(componentName, period, false, 1000)
{
}

void ManagerComponent::Run(void) 
{
    ProcessQueuedCommands();

    static double lastTick = 0;
    static int count = 0;

#if 1
    std::vector<std::string> processes, components, interfaces, connections;
    if (osaGetTime() - lastTick > 5.0) {
        std::cout << "==================================== Processes" << std::endl;
        if (RequestGetNamesOfProcesses(processes)) {
            for (size_t i = 0; i < processes.size(); ++i) {
                std::cout << processes[i] << std::endl;
            }
        }
        
        std::cout << "==================================== Components" << std::endl;
        for (size_t i = 0; i < processes.size(); ++i) {
            if (RequestGetNamesOfComponents(processes[i], components)) {
                for (size_t j = 0; j < components.size(); ++j) {
                    std::cout << processes[i] << " - " << components[j] << std::endl;
                }
            }
        }

        std::cout << "==================================== Interfaces" << std::endl;
        for (size_t i = 0; i < processes.size(); ++i) {
            if (RequestGetNamesOfInterfaces(processes[i], interfaces)) {
                for (size_t j = 0; j < interfaces.size(); ++j) {
                    std::cout << interfaces[j] << std::endl;
                }
            }
        }

        std::cout << "==================================== Connections" << std::endl;
        if (RequestGetListOfConnections(connections)) {
            for (size_t i = 0; i < connections.size(); ++i) {
                std::cout << connections[i] << std::endl;
            }
        }

        std::cout << std::endl << std::endl;
        std::flush(std::cout);

        lastTick = osaGetTime();
    }
#endif

#if 0
    std::cout << ".... (" << count++ << ") ............" << std::endl;

    if (count == 10) {
        std::cout << std::endl << std::endl << "Creating ODD counter across network: ";
        if (!RequestComponentCreate(PeerProcessName, CounterOddComponentType, CounterOddComponentName)) {
            std::cout << "failure" << std::endl;
        } else {
            std::cout << "success" << std::endl;
        }
    }

    //if (count == 15) {
    //    std::cout << std::endl << std::endl << "Creating EVEN counter across network: ";
    //    if (!RequestComponentCreate(PeerProcessName, CounterEvenComponentType, CounterEvenComponentName)) {
    //        std::cout << "failure" << std::endl;
    //    } else {
    //        std::cout << "success" << std::endl;
    //    }
    //}

    //if (count == 20) {
    //    std::cout << std::cout << std::cout << "Connecting the two counters across network: ";
    //    if (!RequestComponentConnect(PeerProcessName, CounterOddComponentName, "InterfaceRequiredOdd",
    //                                PeerProcessName, CounterEvenComponentName, "InterfaceProvidedEven"))
    //    {
    //        std::cout << "failure" << std::endl;
    //    } else {
    //        std::cout << "success" << std::endl;
    //    }
    //}
#endif
}

