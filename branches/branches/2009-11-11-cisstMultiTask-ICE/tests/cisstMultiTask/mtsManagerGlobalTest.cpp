/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerGlobalTest.cpp 2009-03-05 mjung5 $
  
  Author(s):  Min Yang Jung
  Created on: 2009-11-17
  
  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include "mtsManagerGlobalTest.h"

#include <cisstMultiTask/mtsManagerGlobal.h>

void mtsManagerGlobalTest::TestAddProcess(void)
{
    mtsManagerGlobal managerGlobal;

    const std::string processName1 = "P1", processName2 = "P2";

    CPPUNIT_ASSERT(managerGlobal.AddProcess(processName1));
    CPPUNIT_ASSERT(managerGlobal.ProcessMap.FindItem(processName1));
    CPPUNIT_ASSERT(managerGlobal.ProcessMap.GetItem(processName1) == NULL);
    CPPUNIT_ASSERT_EQUAL((unsigned int) 1, managerGlobal.ProcessMap.size());

    CPPUNIT_ASSERT(!managerGlobal.AddProcess(processName1));
    CPPUNIT_ASSERT(managerGlobal.ProcessMap.FindItem(processName1));
    CPPUNIT_ASSERT_EQUAL((unsigned int) 1, managerGlobal.ProcessMap.size());

    CPPUNIT_ASSERT(managerGlobal.AddProcess(processName2));
    CPPUNIT_ASSERT(managerGlobal.ProcessMap.FindItem(processName1));
    CPPUNIT_ASSERT(managerGlobal.ProcessMap.FindItem(processName2));
    CPPUNIT_ASSERT_EQUAL((unsigned int) 2, managerGlobal.ProcessMap.size());
}

void mtsManagerGlobalTest::TestFindProcess(void)
{
    mtsManagerGlobal managerGlobal;

    const std::string processName = "P1";

    CPPUNIT_ASSERT(!managerGlobal.FindProcess(processName));
    CPPUNIT_ASSERT(managerGlobal.AddProcess(processName));
    CPPUNIT_ASSERT(managerGlobal.FindProcess(processName));

    CPPUNIT_ASSERT(managerGlobal.RemoveProcess(processName));
    CPPUNIT_ASSERT(!managerGlobal.FindProcess(processName));
}

void mtsManagerGlobalTest::TestRemoveProcess(void)
{
    mtsManagerGlobal managerGlobal;

    // Case 1. When only processes are registered
    const std::string processName = "P1";

    CPPUNIT_ASSERT(managerGlobal.AddProcess(processName));
    
    CPPUNIT_ASSERT(managerGlobal.FindProcess(processName));
    CPPUNIT_ASSERT(managerGlobal.RemoveProcess(processName));
    CPPUNIT_ASSERT(!managerGlobal.FindProcess(processName));
    CPPUNIT_ASSERT_EQUAL((unsigned int) 0, managerGlobal.ProcessMap.size());

    //
    // TODO: Add tests for cases that a process is removed when the process has
    // multiple components and multiple interfaces.
    //

    // Case 2. When processes with components are registered

    // Case 3. When processes with components that have interfaces are registered

    // Case 4. When processes with components that have interfaces that have connection
    //         with other interfaces are registered
}

void mtsManagerGlobalTest::TestAddComponent(void)
{
    mtsManagerGlobal managerGlobal;

    const std::string processName = "P1", componentName1 = "C1", componentName2 = "C2";

    // Test adding a component without adding a process first
    CPPUNIT_ASSERT(!managerGlobal.AddComponent(processName, componentName1));
    CPPUNIT_ASSERT_EQUAL((unsigned int) 0, managerGlobal.ProcessMap.GetMap().size());

    // Add a process
    CPPUNIT_ASSERT(managerGlobal.AddProcess(processName));
    CPPUNIT_ASSERT(managerGlobal.AddComponent(processName, componentName1));
    {
        // Check changes in the process map        
        CPPUNIT_ASSERT_EQUAL((unsigned int) 1, managerGlobal.ProcessMap.GetMap().size());

        // Check changes in the component map
        CPPUNIT_ASSERT(NULL != managerGlobal.ProcessMap.GetItem(processName));
        CPPUNIT_ASSERT_EQUAL((unsigned int) 1, managerGlobal.ProcessMap.GetItem(processName)->size());
        CPPUNIT_ASSERT(NULL == managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName1));
    }

    // Test if a same component name in the same process is not allowed
    CPPUNIT_ASSERT(!managerGlobal.AddComponent(processName, componentName1));

    // Test addind another component
    CPPUNIT_ASSERT(managerGlobal.AddComponent(processName, componentName2));
    {
        // Check changes in the process map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 1, managerGlobal.ProcessMap.GetMap().size());

        // Check changes in the component map
        CPPUNIT_ASSERT(NULL != managerGlobal.ProcessMap.GetItem(processName));
        CPPUNIT_ASSERT_EQUAL((unsigned int) 2, managerGlobal.ProcessMap.GetItem(processName)->size());
        CPPUNIT_ASSERT(NULL == managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName2));
    }
}

void mtsManagerGlobalTest::TestFindComponent(void)
{
    mtsManagerGlobal managerGlobal;

    const std::string processName = "P1", componentName = "C1";

    CPPUNIT_ASSERT(!managerGlobal.FindComponent(processName, componentName));

    CPPUNIT_ASSERT(managerGlobal.AddProcess(processName));
    CPPUNIT_ASSERT(!managerGlobal.FindComponent(processName, componentName));

    CPPUNIT_ASSERT(managerGlobal.AddComponent(processName, componentName));
    CPPUNIT_ASSERT(managerGlobal.FindComponent(processName, componentName));

    CPPUNIT_ASSERT(managerGlobal.RemoveComponent(processName, componentName));
    CPPUNIT_ASSERT(!managerGlobal.FindComponent(processName, componentName));
}

void mtsManagerGlobalTest::TestRemoveComponent(void)
{
    //
    // TODO:
    //
    // Case 1. When only components are registered

    // Case 2. When components that have interfaces are registered

    // Case 3. When components that have interfaces that have connection
    //         with other interfaces are registered
}

void mtsManagerGlobalTest::TestAddProvidedInterface(void)
{
    mtsManagerGlobal managerGlobal;

    const std::string processName = "P1", componentName = "C1";
    const std::string providedInterfaceName1 = "p1", providedInterfaceName2 = "p2";

    // Test adding a provided interface before adding a component
    CPPUNIT_ASSERT(!managerGlobal.AddProvidedInterface(processName, componentName, providedInterfaceName1));

    // Test adding a provided interface after adding a component
    CPPUNIT_ASSERT(managerGlobal.AddProcess(processName));
    CPPUNIT_ASSERT(managerGlobal.AddComponent(processName, componentName));
    CPPUNIT_ASSERT(managerGlobal.AddProvidedInterface(processName, componentName, providedInterfaceName1));
    {
        // Check changes in the process map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 1, managerGlobal.ProcessMap.GetMap().size());
        CPPUNIT_ASSERT(NULL != managerGlobal.ProcessMap.GetItem(processName));

        // Check changes in the component map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 1, managerGlobal.ProcessMap.GetItem(processName)->size());
        CPPUNIT_ASSERT(NULL != managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName));

        // Check changes in the interface map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 1, 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->ProvidedInterfaceMap.size());
        CPPUNIT_ASSERT(NULL == 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->ProvidedInterfaceMap.GetItem(providedInterfaceName1));
        CPPUNIT_ASSERT_EQUAL((unsigned int) 0, 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->RequiredInterfaceMap.size());
        CPPUNIT_ASSERT(NULL == 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->RequiredInterfaceMap.GetItem(providedInterfaceName1));
    }

    // Test if a same provided interface name in the same component is not allowed
    CPPUNIT_ASSERT(!managerGlobal.AddProvidedInterface(processName, componentName, providedInterfaceName1));

    // Test addind another component
    CPPUNIT_ASSERT(managerGlobal.AddProvidedInterface(processName, componentName, providedInterfaceName2));
    {
        // Check changes in the process map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 1, managerGlobal.ProcessMap.GetMap().size());
        CPPUNIT_ASSERT(NULL != managerGlobal.ProcessMap.GetItem(processName));

        // Check changes in the component map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 1, managerGlobal.ProcessMap.GetItem(processName)->size());
        CPPUNIT_ASSERT(NULL != managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName));

        // Check changes in the interface map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 2, 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->ProvidedInterfaceMap.size());
        CPPUNIT_ASSERT(NULL == 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->ProvidedInterfaceMap.GetItem(providedInterfaceName2));
        CPPUNIT_ASSERT_EQUAL((unsigned int) 0, 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->RequiredInterfaceMap.size());
        CPPUNIT_ASSERT(NULL == 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->RequiredInterfaceMap.GetItem(providedInterfaceName1));
    }
}

void mtsManagerGlobalTest::TestAddRequiredInterface(void)
{
    mtsManagerGlobal managerGlobal;

    const std::string processName = "P1", componentName = "C1";
    const std::string requiredInterfaceName1 = "r1", requiredInterfaceName2 = "r2";

    // Test adding a required interface before adding a component
    CPPUNIT_ASSERT(!managerGlobal.AddRequiredInterface(processName, componentName, requiredInterfaceName1));

    // Test adding a required interface after adding a component
    CPPUNIT_ASSERT(managerGlobal.AddProcess(processName));
    CPPUNIT_ASSERT(managerGlobal.AddComponent(processName, componentName));
    CPPUNIT_ASSERT(managerGlobal.AddRequiredInterface(processName, componentName, requiredInterfaceName1));
    {
        // Check changes in the process map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 1, managerGlobal.ProcessMap.GetMap().size());
        CPPUNIT_ASSERT(NULL != managerGlobal.ProcessMap.GetItem(processName));

        // Check changes in the component map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 1, managerGlobal.ProcessMap.GetItem(processName)->size());
        CPPUNIT_ASSERT(NULL != managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName));

        // Check changes in the interface map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 1, 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->RequiredInterfaceMap.size());
        CPPUNIT_ASSERT(NULL == 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->RequiredInterfaceMap.GetItem(requiredInterfaceName1));
        CPPUNIT_ASSERT_EQUAL((unsigned int) 0, 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->ProvidedInterfaceMap.size());
        CPPUNIT_ASSERT(NULL == 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->ProvidedInterfaceMap.GetItem(requiredInterfaceName1));
    }

    // Test if a same required interface name in the same component is not allowed
    CPPUNIT_ASSERT(!managerGlobal.AddRequiredInterface(processName, componentName, requiredInterfaceName1));

    // Test addind another component
    CPPUNIT_ASSERT(managerGlobal.AddRequiredInterface(processName, componentName, requiredInterfaceName2));
    {
        // Check changes in the process map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 1, managerGlobal.ProcessMap.GetMap().size());
        CPPUNIT_ASSERT(NULL != managerGlobal.ProcessMap.GetItem(processName));

        // Check changes in the component map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 1, managerGlobal.ProcessMap.GetItem(processName)->size());
        CPPUNIT_ASSERT(NULL != managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName));

        // Check changes in the interface map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 2, 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->RequiredInterfaceMap.size());
        CPPUNIT_ASSERT(NULL == 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->RequiredInterfaceMap.GetItem(requiredInterfaceName2));
        CPPUNIT_ASSERT_EQUAL((unsigned int) 0, 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->ProvidedInterfaceMap.size());
        CPPUNIT_ASSERT(NULL == 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->ProvidedInterfaceMap.GetItem(requiredInterfaceName1));
    }
}

void mtsManagerGlobalTest::TestFindProvidedInterface(void)
{
    mtsManagerGlobal managerGlobal;

    const std::string processName = "P1", componentName = "C1", providedInterfaceName = "p1";

    CPPUNIT_ASSERT(!managerGlobal.FindProvidedInterface(processName, componentName, providedInterfaceName));

    CPPUNIT_ASSERT(managerGlobal.AddProcess(processName));
    CPPUNIT_ASSERT(!managerGlobal.FindProvidedInterface(processName, componentName, providedInterfaceName));

    CPPUNIT_ASSERT(managerGlobal.AddComponent(processName, componentName));
    CPPUNIT_ASSERT(!managerGlobal.FindProvidedInterface(processName, componentName, providedInterfaceName));

    CPPUNIT_ASSERT(managerGlobal.AddProvidedInterface(processName, componentName, providedInterfaceName));
    CPPUNIT_ASSERT(managerGlobal.FindProvidedInterface(processName, componentName, providedInterfaceName));

    CPPUNIT_ASSERT(managerGlobal.RemoveProvidedInterface(processName, componentName, providedInterfaceName));
    CPPUNIT_ASSERT(!managerGlobal.FindProvidedInterface(processName, componentName, providedInterfaceName));
}

void mtsManagerGlobalTest::TestFindRequiredInterface(void)
{
    mtsManagerGlobal managerGlobal;

    const std::string processName = "P1", componentName = "C1", requiredInterfaceName = "r1";

    CPPUNIT_ASSERT(!managerGlobal.FindRequiredInterface(processName, componentName, requiredInterfaceName));

    CPPUNIT_ASSERT(managerGlobal.AddProcess(processName));
    CPPUNIT_ASSERT(!managerGlobal.FindRequiredInterface(processName, componentName, requiredInterfaceName));

    CPPUNIT_ASSERT(managerGlobal.AddComponent(processName, componentName));
    CPPUNIT_ASSERT(!managerGlobal.FindRequiredInterface(processName, componentName, requiredInterfaceName));

    CPPUNIT_ASSERT(managerGlobal.AddProvidedInterface(processName, componentName, requiredInterfaceName));
    CPPUNIT_ASSERT(managerGlobal.FindRequiredInterface(processName, componentName, requiredInterfaceName));

    CPPUNIT_ASSERT(managerGlobal.RemoveProvidedInterface(processName, componentName, requiredInterfaceName));
    CPPUNIT_ASSERT(!managerGlobal.FindRequiredInterface(processName, componentName, requiredInterfaceName));
}

void mtsManagerGlobalTest::TestRemoveProvidedInterface(void)
{
    mtsManagerGlobal managerGlobal;

    const std::string processName = "P1", componentName = "C1", 
        providedInterfaceName1 = "p1", providedInterfaceName2 = "p2";

    // Case 1. When only interfaces that have no connection are registered
    // Test removing a provided interface before adding a component
    CPPUNIT_ASSERT(!managerGlobal.RemoveProvidedInterface(processName, componentName, providedInterfaceName1));

    // Test adding a provided interface after adding a component
    CPPUNIT_ASSERT(!managerGlobal.FindProvidedInterface(processName, componentName, providedInterfaceName1));
    CPPUNIT_ASSERT(managerGlobal.AddProcess(processName));
    CPPUNIT_ASSERT(managerGlobal.AddComponent(processName, componentName));
    CPPUNIT_ASSERT(managerGlobal.AddProvidedInterface(processName, componentName, providedInterfaceName1));

    CPPUNIT_ASSERT(managerGlobal.FindProvidedInterface(processName, componentName, providedInterfaceName1));
    CPPUNIT_ASSERT(managerGlobal.RemoveProvidedInterface(processName, componentName, providedInterfaceName1));
    CPPUNIT_ASSERT(!managerGlobal.FindProvidedInterface(processName, componentName, providedInterfaceName1));
    {
        // Check changes in the process map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 1, managerGlobal.ProcessMap.GetMap().size());
        CPPUNIT_ASSERT(NULL != managerGlobal.ProcessMap.GetItem(processName));

        // Check changes in the component map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 1, managerGlobal.ProcessMap.GetItem(processName)->size());
        CPPUNIT_ASSERT(NULL != managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName));

        // Check changes in the interface map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 0, 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->ProvidedInterfaceMap.size());
        CPPUNIT_ASSERT(NULL == 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->ProvidedInterfaceMap.GetItem(providedInterfaceName1));
        CPPUNIT_ASSERT_EQUAL((unsigned int) 0, 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->RequiredInterfaceMap.size());
        CPPUNIT_ASSERT(NULL == 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->RequiredInterfaceMap.GetItem(providedInterfaceName1));
    }

    //
    // TODO:
    // 
    // Case 2. When interfaces have connection with other interfaces
}
         
void mtsManagerGlobalTest::TestRemoveRequiredInterface(void)
{
    mtsManagerGlobal managerGlobal;

    const std::string processName = "P1", componentName = "C1", 
        requiredInterfaceName1 = "r1", requiredInterfaceName2 = "r2";

    // Case 1. When only interfaces that have no connection are registered
    // Test removing a provided interface before adding a component
    CPPUNIT_ASSERT(!managerGlobal.RemoveRequiredInterface(processName, componentName, requiredInterfaceName1));

    // Test adding a provided interface after adding a component
    CPPUNIT_ASSERT(!managerGlobal.FindRequiredInterface(processName, componentName, requiredInterfaceName1));
    CPPUNIT_ASSERT(managerGlobal.AddProcess(processName));
    CPPUNIT_ASSERT(managerGlobal.AddComponent(processName, componentName));
    CPPUNIT_ASSERT(managerGlobal.AddRequiredInterface(processName, componentName, requiredInterfaceName1));

    CPPUNIT_ASSERT(managerGlobal.FindRequiredInterface(processName, componentName, requiredInterfaceName1));
    CPPUNIT_ASSERT(managerGlobal.RemoveRequiredInterface(processName, componentName, requiredInterfaceName1));
    CPPUNIT_ASSERT(!managerGlobal.FindRequiredInterface(processName, componentName, requiredInterfaceName1));
    {
        // Check changes in the process map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 1, managerGlobal.ProcessMap.GetMap().size());
        CPPUNIT_ASSERT(NULL != managerGlobal.ProcessMap.GetItem(processName));

        // Check changes in the component map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 1, managerGlobal.ProcessMap.GetItem(processName)->size());
        CPPUNIT_ASSERT(NULL != managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName));

        // Check changes in the interface map
        CPPUNIT_ASSERT_EQUAL((unsigned int) 0, 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->ProvidedInterfaceMap.size());
        CPPUNIT_ASSERT(NULL == 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->ProvidedInterfaceMap.GetItem(requiredInterfaceName1));
        CPPUNIT_ASSERT_EQUAL((unsigned int) 0, 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->RequiredInterfaceMap.size());
        CPPUNIT_ASSERT(NULL == 
            managerGlobal.ProcessMap.GetItem(processName)->GetItem(componentName)->RequiredInterfaceMap.GetItem(requiredInterfaceName1));
    }

    //
    // TODO:
    // 
    // Case 2. When interfaces have connection with other interfaces
}

/*
//-----------------------------------------------------------------------------
//	Tests for public variables and methods
//-----------------------------------------------------------------------------
void mtsManagerGlobalTest::TestAddComponent(void)
{
    mtsManagerGlobal globalManager;

    CPPUNIT_ASSERT(globalManager.AddComponent("P1", "C1"));
    CPPUNIT_ASSERT_EQUAL((unsigned int) 1, globalManager.ProcessMap.GetItem("P1")->size());
    CPPUNIT_ASSERT(globalManager.AddComponent("P1", "C2"));
    CPPUNIT_ASSERT_EQUAL((unsigned int) 2, globalManager.ProcessMap.GetItem("P1")->size());

    CPPUNIT_ASSERT(globalManager.AddComponent("P2", "C1"));
    CPPUNIT_ASSERT_EQUAL((unsigned int) 2, globalManager.ProcessMap.GetItem("P1")->size());
    CPPUNIT_ASSERT_EQUAL((unsigned int) 1, globalManager.ProcessMap.GetItem("P2")->size());
    CPPUNIT_ASSERT(globalManager.AddComponent("P2", "C2"));
    CPPUNIT_ASSERT_EQUAL((unsigned int) 2, globalManager.ProcessMap.GetItem("P2")->size());
    CPPUNIT_ASSERT(globalManager.AddComponent("", "C1"));
    CPPUNIT_ASSERT_EQUAL((unsigned int) 1, globalManager.ProcessMap.GetItem("")->size());
    CPPUNIT_ASSERT(globalManager.AddComponent("", "C2"));
    CPPUNIT_ASSERT_EQUAL((unsigned int) 2, globalManager.ProcessMap.GetItem("")->size());

    CPPUNIT_ASSERT(!globalManager.AddComponent("P1", "C1"));
    CPPUNIT_ASSERT(!globalManager.AddComponent("P1", "C2"));
    CPPUNIT_ASSERT(!globalManager.AddComponent("P2", "C1"));
    CPPUNIT_ASSERT(!globalManager.AddComponent("P2", "C2"));
    CPPUNIT_ASSERT(!globalManager.AddComponent("", "C1"));
    CPPUNIT_ASSERT(!globalManager.AddComponent("", "C2"));
}

void mtsManagerGlobalTest::TestFindComponent(void)
{
    mtsManagerGlobal globalManager;

    CPPUNIT_ASSERT(!globalManager.FindComponent("P1", "C1"));
    CPPUNIT_ASSERT(globalManager.AddComponent("P1", "C1"));
    CPPUNIT_ASSERT(globalManager.FindComponent("P1", "C1"));
}

void mtsManagerGlobalTest::TestRemoveComponent(void)
{
    mtsManagerGlobal globalManager;

    CPPUNIT_ASSERT(globalManager.AddComponent("P1", "C1"));
    CPPUNIT_ASSERT(globalManager.AddComponent("P1", "C2"));
    CPPUNIT_ASSERT(globalManager.AddComponent("P1", "C3"));
    CPPUNIT_ASSERT(globalManager.AddComponent("P2", "C1"));

    CPPUNIT_ASSERT_EQUAL((unsigned int) 2, globalManager.ProcessMap.size());
    CPPUNIT_ASSERT_EQUAL((unsigned int) 3, globalManager.ProcessMap.GetItem("P1")->size());
    CPPUNIT_ASSERT_EQUAL((unsigned int) 1, globalManager.ProcessMap.GetItem("P2")->size());

    CPPUNIT_ASSERT(globalManager.RemoveComponent("P1", "C1"));
    CPPUNIT_ASSERT(!globalManager.FindComponent("P1", "C1"));
    CPPUNIT_ASSERT(globalManager.FindComponent("P1", "C2"));
    CPPUNIT_ASSERT(globalManager.FindComponent("P1", "C3"));
    CPPUNIT_ASSERT_EQUAL((unsigned int) 2, globalManager.ProcessMap.size());
    CPPUNIT_ASSERT_EQUAL((unsigned int) 2, globalManager.ProcessMap.GetItem("P1")->size());
    CPPUNIT_ASSERT_EQUAL((unsigned int) 1, globalManager.ProcessMap.GetItem("P2")->size());
    
    CPPUNIT_ASSERT(globalManager.RemoveComponent("P2", "C1"));
    CPPUNIT_ASSERT(!globalManager.FindComponent("P2", "C1"));
    CPPUNIT_ASSERT_EQUAL((unsigned int) 1, globalManager.ProcessMap.size());
    CPPUNIT_ASSERT_EQUAL((unsigned int) 2, globalManager.ProcessMap.GetItem("P1")->size());
    CPPUNIT_ASSERT(0 == globalManager.ProcessMap.GetItem("P2"));

    //
    // TODO: add tests for ConnectionMap
    //
}

void mtsManagerGlobalTest::TestConnect(void)
{    
    mtsManagerGlobal globalManager;

    // Check if the interfaces specified actually exist.
    // These test cases are described in the project wiki.
    // (see https://trac.lcsr.jhu.edu/cisst/wiki/Private/cisstMultiTaskNetwork)
    CPPUNIT_ASSERT(!globalManager.Connect("P1", "C1", "r1", "P2", "C2", "p1"));
    CPPUNIT_ASSERT(!globalManager.Connect("P1", "C1", "r2", "P2", "C2", "p2"));
    CPPUNIT_ASSERT(!globalManager.Connect("P1", "C2", "r1", "P2", "C2", "p2"));
    CPPUNIT_ASSERT(!globalManager.Connect("P2", "C3", "r1", "P2", "C2", "p2"));

    globalManager.AddComponent("P1", "C1");
    globalManager.AddComponent("P2", "C2");

    // Connect two interfaces
    CPPUNIT_ASSERT(globalManager.Connect("P1", "C1", "r1", "P2", "C2", "p1"));

    mtsManagerGlobal::ConnectionMapType * connectionMap;
    mtsManagerGlobal::ConnectedInterfaceInfo * connectedInterfaceInfo;

    // Check if the information at the client side is correct.
    connectionMap = globalManager.GetConnectionMap("P1", "C1");
    CPPUNIT_ASSERT(connectionMap);
    CPPUNIT_ASSERT_EQUAL((unsigned int) 1, connectionMap->size());
    
    connectedInterfaceInfo = connectionMap->GetItem("r1");
    CPPUNIT_ASSERT(connectedInterfaceInfo);
    CPPUNIT_ASSERT(connectedInterfaceInfo->ProcessName == "P2");
    CPPUNIT_ASSERT(connectedInterfaceInfo->ComponentName == "C2");
    CPPUNIT_ASSERT(connectedInterfaceInfo->InterfaceName == "p2");

    // Check if the information at the client side is correct.
    connectionMap = globalManager.GetConnectionMap("P2", "C2");
    CPPUNIT_ASSERT(connectionMap);
    CPPUNIT_ASSERT_EQUAL((unsigned int) 1, connectionMap->size());

    connectedInterfaceInfo = connectionMap->GetItem("p1");
    CPPUNIT_ASSERT(connectedInterfaceInfo);
    CPPUNIT_ASSERT(connectedInterfaceInfo->ProcessName == "P1");
    CPPUNIT_ASSERT(connectedInterfaceInfo->ComponentName == "C1");
    CPPUNIT_ASSERT(connectedInterfaceInfo->InterfaceName == "r1");
}

void mtsManagerGlobalTest::TestDisconnect(void)
{
    // TODO: implement this
}

void mtsManagerGlobalTest::TestGetConnectionMap(void)
{
    mtsManagerGlobal globalManager;

    CPPUNIT_ASSERT(!globalManager.GetConnectionMap("P1", "C1"));
    CPPUNIT_ASSERT(globalManager.AddComponent("P1", "C1"));
    CPPUNIT_ASSERT(globalManager.GetConnectionMap("P1", "C1"));
}

//-----------------------------------------------------------------------------
//	Tests for private variables and methods
//-----------------------------------------------------------------------------
*/
CPPUNIT_TEST_SUITE_REGISTRATION(mtsManagerGlobalTest);
