/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Min Yang Jung
  Created on: 2009-11-17

  (C) Copyright 2009-2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include "mtsManagerLocalTest.h"

#include <cisstMultiTask/mtsManagerGlobal.h>
#include <cisstMultiTask/mtsManagerLocal.h>
#include <cisstMultiTask/mtsStateTable.h>

#include "mtsManagerTestClasses.h"

#define P1 "P1"
#define P2 "P2"
#define P1_OBJ localManager1
#define P2_OBJ localManager2

#define DEFAULT_PROCESS_NAME "LCM"

using namespace std;

mtsManagerLocalTest::mtsManagerLocalTest()
{
    mtsManagerLocal::UnitTestEnabled = true;
#if !CISST_MTS_HAS_ICE
    mtsManagerLocal::UnitTestNetworkProxyEnabled = false;
#else
    mtsManagerLocal::UnitTestNetworkProxyEnabled = true;
#endif
}

void mtsManagerLocalTest::setUp(void)
{
    //localManager1 = new mtsManagerLocal();
    //localManager2 = new mtsManagerLocal();

    //localManager1->ProcessName = P1;
    //localManager2->ProcessName = P2;
}

void mtsManagerLocalTest::tearDown(void)
{
    //delete localManager1;
    //delete localManager2;
}

void mtsManagerLocalTest::TestInitialize(void)
{
    // Add __os_init() test if needed.
}

void mtsManagerLocalTest::TestConstructor(void)
{
    mtsManagerLocal localManager;
    CPPUNIT_ASSERT_EQUAL(localManager.ProcessName, string(DEFAULT_PROCESS_NAME));
    CPPUNIT_ASSERT(localManager.ManagerGlobal);

    mtsManagerGlobal * GCM = dynamic_cast<mtsManagerGlobal*>(localManager.ManagerGlobal);
    CPPUNIT_ASSERT(GCM);

    CPPUNIT_ASSERT(GCM->FindProcess(localManager.ProcessName));
    CPPUNIT_ASSERT(GCM->GetProcessObject(localManager.ProcessName) == &localManager);
}

void mtsManagerLocalTest::TestCleanup(void)
{
    mtsManagerLocal managerLocal;

    CPPUNIT_ASSERT(managerLocal.ManagerGlobal);
    mtsManagerTestDevice1 * dummy = new mtsManagerTestDevice1;
    CPPUNIT_ASSERT(managerLocal.ComponentMap.AddItem("dummy", dummy));
    CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(1), managerLocal.ComponentMap.size());

    managerLocal.Cleanup();

    CPPUNIT_ASSERT(managerLocal.ManagerGlobal == 0);
    CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(0), managerLocal.ComponentMap.size());

    // Add __os_exit() test if needed.
}

void mtsManagerLocalTest::TestGetInstance(void)
{
    mtsManagerLocal * managerLocal = mtsManagerLocal::GetInstance();

    CPPUNIT_ASSERT(managerLocal);
    CPPUNIT_ASSERT(managerLocal->ManagerGlobal);
    CPPUNIT_ASSERT_EQUAL(managerLocal, mtsManagerLocal::Instance);
    CPPUNIT_ASSERT(managerLocal->ManagerGlobal->FindProcess(DEFAULT_PROCESS_NAME));
}

void mtsManagerLocalTest::TestAddComponent(void)
{
    mtsManagerLocal localManager1;

    // Test with mtsComponent type components
    mtsManagerTestDevice2 * device2 = new mtsManagerTestDevice2;

    // Invalid argument test
    CPPUNIT_ASSERT(!localManager1.AddComponent(NULL));

    // Check with the global component manager.
    // Should fail if a component has already been registered before
    CPPUNIT_ASSERT(localManager1.ManagerGlobal->AddComponent(DEFAULT_PROCESS_NAME, device2->GetName()));
    CPPUNIT_ASSERT(!localManager1.AddComponent(device2));

    // Should succeed if a component is new
    CPPUNIT_ASSERT(localManager1.ManagerGlobal->RemoveComponent(DEFAULT_PROCESS_NAME, device2->GetName()));
    CPPUNIT_ASSERT(localManager1.AddComponent(device2));
    CPPUNIT_ASSERT(localManager1.ComponentMap.FindItem(device2->GetName()));

    // Check if all the existing required interfaces and provided interfaces are
    // added to the global component manager.
    CPPUNIT_ASSERT(localManager1.ManagerGlobal->FindInterfaceRequiredOrInput(DEFAULT_PROCESS_NAME, device2->GetName(), "r1"));
    CPPUNIT_ASSERT(localManager1.ManagerGlobal->FindInterfaceProvidedOrOutput(DEFAULT_PROCESS_NAME, device2->GetName(), "p1"));
    CPPUNIT_ASSERT(localManager1.ManagerGlobal->FindInterfaceProvidedOrOutput(DEFAULT_PROCESS_NAME, device2->GetName(), "p2"));

    mtsManagerLocal localManager2;

    // Test with mtsTask type components
    mtsManagerTestContinuous1 * continuous1 = new mtsManagerTestContinuous1;

    // Invalid argument test
    CPPUNIT_ASSERT(!localManager2.AddComponent(NULL));

    // Check with the global component manager.
    // Should fail if a component to be added has already been registered before
    CPPUNIT_ASSERT(localManager2.ManagerGlobal->AddComponent(DEFAULT_PROCESS_NAME, continuous1->GetName()));
    CPPUNIT_ASSERT(!localManager2.AddComponent(continuous1));

    // Should succeed if a component is new
    CPPUNIT_ASSERT(localManager2.ManagerGlobal->RemoveComponent(DEFAULT_PROCESS_NAME, continuous1->GetName()));
    CPPUNIT_ASSERT(localManager2.AddComponent(continuous1));
    CPPUNIT_ASSERT(localManager2.ComponentMap.FindItem(continuous1->GetName()));

    // Check if all the existing required interfaces and provided interfaces are
    // added to the global component manager.
    CPPUNIT_ASSERT(localManager2.ManagerGlobal->FindInterfaceRequiredOrInput(DEFAULT_PROCESS_NAME, continuous1->GetName(), "r1"));
    CPPUNIT_ASSERT(localManager2.ManagerGlobal->FindInterfaceProvidedOrOutput(DEFAULT_PROCESS_NAME, continuous1->GetName(), "p1"));
    CPPUNIT_ASSERT(localManager2.ManagerGlobal->FindInterfaceProvidedOrOutput(DEFAULT_PROCESS_NAME, continuous1->GetName(), "p2"));
}

void mtsManagerLocalTest::TestFindComponent(void)
{
    mtsManagerLocal localManager1;
    mtsManagerTestDevice1 * device1 = new mtsManagerTestDevice1;
    const std::string componentName = device1->GetName();

    CPPUNIT_ASSERT(!localManager1.FindComponent(componentName));
    CPPUNIT_ASSERT(localManager1.AddComponent(device1));
    CPPUNIT_ASSERT(localManager1.FindComponent(componentName));

    CPPUNIT_ASSERT(localManager1.RemoveComponent(componentName));
    CPPUNIT_ASSERT(!localManager1.FindComponent(componentName));
}

void mtsManagerLocalTest::TestRemoveComponent(void)
{
    // Test with mtsComponent type components
    mtsManagerLocal localManager1;
    mtsManagerTestDevice1 * device1 = new mtsManagerTestDevice1;
    const std::string componentName1 = device1->GetName();

    // Invalid argument test
    CPPUNIT_ASSERT(!localManager1.RemoveComponent(NULL));
    CPPUNIT_ASSERT(!localManager1.RemoveComponent("dummy"));

    CPPUNIT_ASSERT(localManager1.AddComponent(device1));
    CPPUNIT_ASSERT(localManager1.FindComponent(componentName1));
    CPPUNIT_ASSERT(localManager1.RemoveComponent(componentName1));
    CPPUNIT_ASSERT(!localManager1.FindComponent(componentName1));

    device1 = new mtsManagerTestDevice1;
    CPPUNIT_ASSERT(localManager1.AddComponent(device1));
    CPPUNIT_ASSERT(localManager1.FindComponent(componentName1));
    CPPUNIT_ASSERT(localManager1.RemoveComponent(device1));
    CPPUNIT_ASSERT(!localManager1.FindComponent(componentName1));

    // Test with mtsComponent type components
    mtsManagerLocal localManager2;
    mtsManagerTestPeriodic1 * periodic1 = new mtsManagerTestPeriodic1;
    const std::string componentName2 = periodic1->GetName();

    CPPUNIT_ASSERT(localManager2.AddComponent(periodic1));
    CPPUNIT_ASSERT(localManager2.FindComponent(componentName2));
    CPPUNIT_ASSERT(localManager2.RemoveComponent(componentName2));
    CPPUNIT_ASSERT(!localManager2.FindComponent(componentName2));

    periodic1 = new mtsManagerTestPeriodic1;
    CPPUNIT_ASSERT(localManager2.AddComponent(periodic1));
    CPPUNIT_ASSERT(localManager2.FindComponent(componentName2));
    CPPUNIT_ASSERT(localManager2.RemoveComponent(periodic1));
    CPPUNIT_ASSERT(!localManager2.FindComponent(componentName2));
}

void mtsManagerLocalTest::TestRegisterInterfaces(void)
{
    mtsManagerLocal localManager;
    mtsManagerGlobal * globalManager = dynamic_cast<mtsManagerGlobal *>(localManager.ManagerGlobal);
    CPPUNIT_ASSERT(globalManager);

    mtsManagerTestDevice2 * component = new mtsManagerTestDevice2;
    const std::string componentName = component->GetName();

    // Check initial values of GCM
    CPPUNIT_ASSERT(!globalManager->FindInterfaceRequiredOrInput("LCM", componentName, "r1"));
    CPPUNIT_ASSERT(!globalManager->FindInterfaceProvidedOrOutput("LCM", componentName, "p1"));
    CPPUNIT_ASSERT(!globalManager->FindInterfaceProvidedOrOutput("LCM", componentName, "p2"));
    // This should fail because no component is registered yet
    CPPUNIT_ASSERT(!localManager.RegisterInterfaces(component));

    // Add the component. This includes registration of interfaces that have been added so far.
    CPPUNIT_ASSERT(localManager.AddComponent(component));

    // Check updated values of GCM
    CPPUNIT_ASSERT(globalManager->FindInterfaceRequiredOrInput("LCM", componentName, "r1"));
    CPPUNIT_ASSERT(globalManager->FindInterfaceProvidedOrOutput("LCM", componentName, "p1"));
    CPPUNIT_ASSERT(globalManager->FindInterfaceProvidedOrOutput("LCM", componentName, "p2"));

    // Now, create a new required and provided interface which have not been added.
    mtsInterfaceRequired * requiredInterface = component->AddInterfaceRequired("newRequiredInterface");
    CPPUNIT_ASSERT(requiredInterface);
    mtsInterfaceProvided * providedInterface = component->AddInterfaceProvided("newProvidedInterface");
    CPPUNIT_ASSERT(providedInterface);

    // Check initial values of GCM
    CPPUNIT_ASSERT(!globalManager->FindInterfaceRequiredOrInput("LCM", componentName, requiredInterface->GetName()));
    CPPUNIT_ASSERT(!globalManager->FindInterfaceProvidedOrOutput("LCM", componentName, providedInterface->GetName()));

    // Register the new interfaces
    CPPUNIT_ASSERT(localManager.RegisterInterfaces(component));

    // Check updated values of GCM
    CPPUNIT_ASSERT(globalManager->FindInterfaceRequiredOrInput("LCM", componentName, requiredInterface->GetName()));
    CPPUNIT_ASSERT(globalManager->FindInterfaceProvidedOrOutput("LCM", componentName, providedInterface->GetName()));
}


void mtsManagerLocalTest::TestGetComponent(void)
{
    mtsManagerLocal localManager;
    mtsManagerTestDevice1 * device1 = new mtsManagerTestDevice1;
    mtsManagerTestDevice2 * device2 = new mtsManagerTestDevice2;
    mtsManagerTestDevice3 * device3 = new mtsManagerTestDevice3;
    mtsComponent * nullComponent = 0;

    CPPUNIT_ASSERT_EQUAL(nullComponent, localManager.GetComponent(device1->GetName()));
    CPPUNIT_ASSERT_EQUAL(nullComponent, localManager.GetComponent(device2->GetName()));
    CPPUNIT_ASSERT_EQUAL(nullComponent, localManager.GetComponent(device3->GetName()));

    CPPUNIT_ASSERT(localManager.AddComponent(device1));
    CPPUNIT_ASSERT(localManager.AddComponent(device2));
    CPPUNIT_ASSERT(localManager.AddComponent(device3));

    CPPUNIT_ASSERT(device1 == localManager.GetComponent(device1->GetName()));
    CPPUNIT_ASSERT(device2 == localManager.GetComponent(device2->GetName()));
    CPPUNIT_ASSERT(device3 == localManager.GetComponent(device3->GetName()));
}

void mtsManagerLocalTest::TestGetNamesOfComponents(void)
{
    mtsManagerLocal localManager;
    mtsManagerTestDevice1 * device1 = new mtsManagerTestDevice1;
    mtsManagerTestDevice2 * device2 = new mtsManagerTestDevice2;
    mtsManagerTestDevice3 * device3 = new mtsManagerTestDevice3;

    CPPUNIT_ASSERT(localManager.AddComponent(device1));
    CPPUNIT_ASSERT(localManager.AddComponent(device2));
    CPPUNIT_ASSERT(localManager.AddComponent(device3));

    // return value
    std::vector<std::string> namesOfComponents1 = localManager.GetNamesOfComponents();
    CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(3), namesOfComponents1.size());
    for (size_t i = 0; i < 3; ++i) {
        CPPUNIT_ASSERT(namesOfComponents1[i] == device1->GetName() ||
                       namesOfComponents1[i] == device2->GetName() ||
                       namesOfComponents1[i] == device3->GetName());
    }

    // using placeholder
    std::vector<std::string> namesOfComponents2;
    localManager.GetNamesOfComponents(namesOfComponents2);
    CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(3), namesOfComponents2.size());
    for (size_t i = 0; i < 3; ++i) {
        CPPUNIT_ASSERT(namesOfComponents2[i] == device1->GetName() ||
                       namesOfComponents2[i] == device2->GetName() ||
                       namesOfComponents2[i] == device3->GetName());
    }
}

void mtsManagerLocalTest::TestGetNamesOfTasks(void)
{
    mtsManagerLocal localManager;
    mtsManagerTestDevice1 * device1 = new mtsManagerTestDevice1;
    mtsManagerTestDevice2 * device2 = new mtsManagerTestDevice2;
    mtsManagerTestPeriodic1 * periodic1 = new mtsManagerTestPeriodic1;
    mtsManagerTestContinuous1 * continuous1 = new mtsManagerTestContinuous1;

    CPPUNIT_ASSERT(localManager.AddComponent(device1));
    CPPUNIT_ASSERT(localManager.AddComponent(device2));
    CPPUNIT_ASSERT(localManager.AddComponent(periodic1));
    CPPUNIT_ASSERT(localManager.AddComponent(continuous1));

    std::vector<std::string> namesOfTasks1 = localManager.GetNamesOfTasks();
    CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(2), namesOfTasks1.size());
    for (size_t i = 0; i < 2; ++i) {
        CPPUNIT_ASSERT(namesOfTasks1[i] == continuous1->GetName() ||
                       namesOfTasks1[i] == periodic1->GetName());
    }

    std::vector<std::string> namesOfTasks2;
    localManager.GetNamesOfTasks(namesOfTasks2);
    CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(2), namesOfTasks2.size());
    for (size_t i = 0; i < 2; ++i) {
        CPPUNIT_ASSERT(namesOfTasks2[i] == continuous1->GetName() ||
                       namesOfTasks2[i] == periodic1->GetName());
    }
}

void mtsManagerLocalTest::TestGetNamesOfDevices(void)
{
    mtsManagerLocal localManager;
    mtsManagerTestDevice1 * device1 = new mtsManagerTestDevice1;
    mtsManagerTestDevice2 * device2 = new mtsManagerTestDevice2;
    mtsManagerTestPeriodic1 * periodic1 = new mtsManagerTestPeriodic1;
    mtsManagerTestContinuous1 * continuous1 = new mtsManagerTestContinuous1;

    CPPUNIT_ASSERT(localManager.AddComponent(device1));
    CPPUNIT_ASSERT(localManager.AddComponent(device2));
    CPPUNIT_ASSERT(localManager.AddComponent(periodic1));
    CPPUNIT_ASSERT(localManager.AddComponent(continuous1));

    std::vector<std::string> namesOfDevices1 = localManager.GetNamesOfDevices();
    CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(2), namesOfDevices1.size());
    for (size_t i = 0; i < 2; ++i) {
        CPPUNIT_ASSERT(namesOfDevices1[i] == device1->GetName() ||
                       namesOfDevices1[i] == device2->GetName());
    }

    std::vector<std::string> namesOfDevices2;
    localManager.GetNamesOfDevices(namesOfDevices2);
    CPPUNIT_ASSERT_EQUAL(static_cast<size_t>(2), namesOfDevices2.size());
    for (size_t i = 0; i < 2; ++i) {
        CPPUNIT_ASSERT(namesOfDevices2[i] == device1->GetName() ||
                       namesOfDevices2[i] == device2->GetName());
    }
}

void mtsManagerLocalTest::TestGetTimeServer(void)
{
    mtsManagerLocal localManager;
    CPPUNIT_ASSERT(&localManager.GetTimeServer() == &localManager.TimeServer);
}

void mtsManagerLocalTest::TestGetProcessName(void)
{
    mtsManagerLocal localManager;
    CPPUNIT_ASSERT_EQUAL(localManager.GetProcessName(), std::string(DEFAULT_PROCESS_NAME));
}


void mtsManagerLocalTest::TestStates(void)
{
    mtsManagerLocal localManager;
    mtsManagerTestPeriodic1 * periodic1 = new mtsManagerTestPeriodic1;
    mtsManagerTestContinuous1 * continuous1 = new mtsManagerTestContinuous1;
    mtsManagerTestFromCallback1 * fromCallback1 = new mtsManagerTestFromCallback1;
    mtsManagerTestCallbackTrigger * callbackTrigger = new mtsManagerTestCallbackTrigger(fromCallback1);
    mtsManagerTestFromSignal1 * fromSignal1 = new mtsManagerTestFromSignal1;
    mtsManagerTestDevice2 * device2 = new mtsManagerTestDevice2;

    CPPUNIT_ASSERT_EQUAL(mtsTask::CONSTRUCTED, periodic1->GetTaskState());
    CPPUNIT_ASSERT_EQUAL(mtsTask::CONSTRUCTED, continuous1->GetTaskState());
    CPPUNIT_ASSERT_EQUAL(mtsTask::CONSTRUCTED, fromCallback1->GetTaskState());
    CPPUNIT_ASSERT_EQUAL(mtsTask::CONSTRUCTED, fromSignal1->GetTaskState());

    CPPUNIT_ASSERT(localManager.AddComponent(periodic1));
    CPPUNIT_ASSERT(localManager.AddComponent(continuous1));
    CPPUNIT_ASSERT(localManager.AddComponent(fromCallback1));
    CPPUNIT_ASSERT(localManager.AddComponent(fromSignal1));
    CPPUNIT_ASSERT(localManager.AddComponent(device2));

    // Establish connections between the three components of mtsTask type
    CPPUNIT_ASSERT(localManager.Connect(periodic1->GetName(), "r1", continuous1->GetName(), "p1"));
    CPPUNIT_ASSERT(localManager.Connect(periodic1->GetName(), "r2", continuous1->GetName(), "p2"));
    CPPUNIT_ASSERT(localManager.Connect(continuous1->GetName(), "r1", device2->GetName(), "p1"));
    CPPUNIT_ASSERT(localManager.Connect(fromCallback1->GetName(), "r1", continuous1->GetName(), "p2"));
    CPPUNIT_ASSERT(localManager.Connect(fromSignal1->GetName(), "r1", continuous1->GetName(), "p2"));
    CPPUNIT_ASSERT(localManager.Connect(device2->GetName(), "r1", continuous1->GetName(), "p2"));

    localManager.CreateAll();
    CPPUNIT_ASSERT((periodic1->GetTaskState() == mtsTask::INITIALIZING) ||
                   (periodic1->GetTaskState() == mtsTask::READY));
    CPPUNIT_ASSERT((continuous1->GetTaskState() == mtsTask::INITIALIZING) ||
                   (continuous1->GetTaskState() == mtsTask::READY));
    CPPUNIT_ASSERT((fromCallback1->GetTaskState() == mtsTask::INITIALIZING) || 
                   (fromCallback1->GetTaskState() == mtsTask::READY));
    CPPUNIT_ASSERT((fromSignal1->GetTaskState() == mtsTask::INITIALIZING) ||
                   (fromSignal1->GetTaskState() == mtsTask::READY));

    // let all tasks get initialized
    osaSleep(1.0 * cmn_s);
    CPPUNIT_ASSERT_EQUAL(mtsTask::READY, periodic1->GetTaskState());
    CPPUNIT_ASSERT_EQUAL(mtsTask::READY, continuous1->GetTaskState());
    CPPUNIT_ASSERT_EQUAL(mtsTask::READY, fromCallback1->GetTaskState());
    CPPUNIT_ASSERT_EQUAL(mtsTask::READY, fromSignal1->GetTaskState());

    localManager.StartAll();
    // let all tasks start
    osaSleep(1.0 * cmn_s);
    CPPUNIT_ASSERT_EQUAL(mtsTask::ACTIVE, periodic1->GetTaskState());
    CPPUNIT_ASSERT_EQUAL(mtsTask::ACTIVE, continuous1->GetTaskState());
    CPPUNIT_ASSERT_EQUAL(mtsTask::ACTIVE, fromCallback1->GetTaskState());

    localManager.KillAll();
    CPPUNIT_ASSERT(periodic1->GetTaskState() == mtsTask::FINISHING ||
                   periodic1->GetTaskState() == mtsTask::FINISHED);
    CPPUNIT_ASSERT(continuous1->GetTaskState() == mtsTask::FINISHING ||
                   continuous1->GetTaskState() == mtsTask::FINISHED);
    CPPUNIT_ASSERT(fromCallback1->GetTaskState() == mtsTask::FINISHING ||
                   fromCallback1->GetTaskState() == mtsTask::FINISHED);
    CPPUNIT_ASSERT(fromSignal1->GetTaskState() == mtsTask::FINISHING ||
                   fromSignal1->GetTaskState() == mtsTask::FINISHED);

    // let all tasks stop
    osaSleep(1.0 * cmn_s);
    CPPUNIT_ASSERT_EQUAL(mtsTask::FINISHED, periodic1->GetTaskState());
    CPPUNIT_ASSERT_EQUAL(mtsTask::FINISHED, continuous1->GetTaskState());
    CPPUNIT_ASSERT_EQUAL(mtsTask::FINISHED, fromCallback1->GetTaskState());
    CPPUNIT_ASSERT_EQUAL(mtsTask::FINISHED, fromSignal1->GetTaskState());

    delete periodic1;
    delete continuous1;
    callbackTrigger->Stop();
    delete callbackTrigger;
    delete fromCallback1;
    delete fromSignal1;
    delete device2;
}


void mtsManagerLocalTest::TestConnectDisconnect(void)
{
    // Local connection test
    mtsManagerLocal localManager;
    mtsManagerTestPeriodic1 * periodic1 = new mtsManagerTestPeriodic1;
    mtsManagerTestContinuous1 * continuous1 = new mtsManagerTestContinuous1;
    mtsManagerTestFromCallback1 * fromCallback1 = new mtsManagerTestFromCallback1;
    mtsManagerTestDevice2 * device2 = new mtsManagerTestDevice2;

    CPPUNIT_ASSERT(localManager.AddComponent(periodic1));
    CPPUNIT_ASSERT(localManager.AddComponent(continuous1));
    CPPUNIT_ASSERT(localManager.AddComponent(fromCallback1));
    CPPUNIT_ASSERT(localManager.AddComponent(device2));

    // Establish connections between the three components of mtsTask type
    CPPUNIT_ASSERT(localManager.Connect(periodic1->GetName(), "r1", continuous1->GetName(), "p1"));
    CPPUNIT_ASSERT(localManager.Connect(periodic1->GetName(), "r2", continuous1->GetName(), "p2"));
    CPPUNIT_ASSERT(localManager.Connect(device2->GetName(), "r1", continuous1->GetName(), "p2"));
    CPPUNIT_ASSERT(localManager.Connect(fromCallback1->GetName(), "r1", continuous1->GetName(), "p2"));

    // Should fail: already established connections
    CPPUNIT_ASSERT(!localManager.Connect(periodic1->GetName(), "r1", continuous1->GetName(), "p1"));
    CPPUNIT_ASSERT(!localManager.Connect(periodic1->GetName(), "r2", continuous1->GetName(), "p2"));
    CPPUNIT_ASSERT(!localManager.Connect(device2->GetName(), "r1", continuous1->GetName(), "p2"));
    CPPUNIT_ASSERT(!localManager.Connect(fromCallback1->GetName(), "r1", continuous1->GetName(), "p2"));

    // Disconnect all current connections
    CPPUNIT_ASSERT(localManager.Disconnect(periodic1->GetName(), "r1", continuous1->GetName(), "p1"));
    CPPUNIT_ASSERT(localManager.Disconnect(periodic1->GetName(), "r2", continuous1->GetName(), "p2"));
    CPPUNIT_ASSERT(localManager.Disconnect(device2->GetName(), "r1", continuous1->GetName(), "p2"));
    CPPUNIT_ASSERT(localManager.Disconnect(fromCallback1->GetName(), "r1", continuous1->GetName(), "p2"));

    // Should success: new connections
    CPPUNIT_ASSERT(localManager.Connect(periodic1->GetName(), "r1", continuous1->GetName(), "p1"));
    CPPUNIT_ASSERT(localManager.Connect(periodic1->GetName(), "r2", continuous1->GetName(), "p2"));
    CPPUNIT_ASSERT(localManager.Connect(device2->GetName(), "r1", continuous1->GetName(), "p2"));
    CPPUNIT_ASSERT(localManager.Connect(fromCallback1->GetName(), "r1", continuous1->GetName(), "p2"));
}

void mtsManagerLocalTest::TestConnectLocally(void)
{
    mtsManagerLocal localManager;
    mtsManagerTestDevice1 * client = new mtsManagerTestDevice1;
    mtsManagerTestDevice2 * server = new mtsManagerTestDevice2;

#define FAIL    false
#define SUCCESS true
    // test with invalid arguments
    CPPUNIT_ASSERT_EQUAL(FAIL, localManager.ConnectLocally("", "", "", ""));

    CPPUNIT_ASSERT(localManager.AddComponent(client));
    CPPUNIT_ASSERT_EQUAL(FAIL, localManager.ConnectLocally(client->GetName(), "", "", ""));

    CPPUNIT_ASSERT(localManager.AddComponent(server));
    CPPUNIT_ASSERT_EQUAL(FAIL, localManager.ConnectLocally(client->GetName(), "", server->GetName(), ""));

    CPPUNIT_ASSERT_EQUAL(FAIL, localManager.ConnectLocally(client->GetName(), "", server->GetName(), "p1"));

    CPPUNIT_ASSERT(client->GetInterfaceRequired("r1")->InterfaceProvidedOrOutput == 0);
    CPPUNIT_ASSERT_EQUAL(SUCCESS, localManager.ConnectLocally(client->GetName(), "r1", server->GetName(), "p1"));
    CPPUNIT_ASSERT(client->GetInterfaceRequired("r1")->InterfaceProvidedOrOutput == server->GetInterfaceProvided("p1"));
}

#if CISST_MTS_HAS_ICE
void mtsManagerLocalTest::TestGetIPAddressList(void)
{
    vector<string> ipList1, ipList2;
    ipList1 = mtsManagerLocal::GetIPAddressList();
    mtsManagerLocal::GetIPAddressList(ipList2);

    CPPUNIT_ASSERT_EQUAL(ipList1.size(), ipList2.size());
    for (size_t i = 0; i < ipList1.size(); ++i) {
        CPPUNIT_ASSERT_EQUAL(ipList1[i], ipList2[i]);
    }
}

void mtsManagerLocalTest::TestGetName(void)
{
}

void mtsManagerLocalTest::TestConnectServerSideInterface(void)
{
}

void mtsManagerLocalTest::TestCreateInterfaceRequiredProxy(void)
{
}

void mtsManagerLocalTest::TestCreateInterfaceProvidedProxy(void)
{
}

void mtsManagerLocalTest::TestRemoveInterfaceRequiredProxy(void)
{
}

void mtsManagerLocalTest::TestRemoveInterfaceProvidedProxy(void)
{
}

void mtsManagerLocalTest::TestRemoteCommandsAndEvents(void)
{
}
#endif

/*
    //
    // Remote connection test
    //

    // Test with invalid arguments.
    managerLocal.UnitTestEnabled = true; // run in unit test mode
    managerLocal.UnitTestNetworkProxyEnabled = false; // but disable network proxy processings
    CPPUNIT_ASSERT(!managerLocal.Connect(P1, C1, r1, P2, C2, p1));

    mtsManagerGlobal managerGlobal;

    // Prepare local managers for this test
    mtsManagerTestDevice1 * P1C1 = new mtsManagerTestDevice1;
    mtsManagerTestDevice2 * P1C2 = new mtsManagerTestDevice2;
    mtsManagerTestDevice2 * P2C2 = new mtsManagerTestDevice2;
    mtsManagerTestDevice3 * P2C3 = new mtsManagerTestDevice3;

    mtsManagerLocalInterface * managerLocal1 = new mtsManagerLocal(P1);
    mtsManagerLocal * managerLocal1Object = dynamic_cast<mtsManagerLocal*>(managerLocal1);
    managerLocal1Object->ManagerGlobal = &managerGlobal;
    managerGlobal.AddProcess(managerLocal1->GetProcessName());
    managerLocal1Object->AddComponent(P1C1);
    managerLocal1Object->AddComponent(P1C2);
    managerLocal1Object->UnitTestEnabled = true; // run in unit test mode
    managerLocal1Object->UnitTestNetworkProxyEnabled = true; // but disable network proxy processings

    mtsManagerLocalInterface * managerLocal2 = new mtsManagerLocal(P2);
    mtsManagerLocal * managerLocal2Object = dynamic_cast<mtsManagerLocal*>(managerLocal2);
    managerLocal2Object->ManagerGlobal = &managerGlobal;
    managerGlobal.AddProcess(managerLocal2->GetProcessName());
    managerLocal2Object->AddComponent(P2C2);
    managerLocal2Object->AddComponent(P2C3);
    managerLocal2Object->UnitTestEnabled = true; // run in unit test mode
    managerLocal2Object->UnitTestNetworkProxyEnabled = true; // but disable network proxy processings

    // Connecting two interfaces for the first time should success.
    CPPUNIT_ASSERT(managerLocal1Object->Connect(P1, C1, r1, P2, C2, p1));
    CPPUNIT_ASSERT(managerLocal1Object->Connect(P1, C1, r2, P2, C2, p2));
    CPPUNIT_ASSERT(managerLocal1Object->Connect(P1, C2, r1, P2, C2, p2));
    CPPUNIT_ASSERT(managerLocal2Object->Connect(P2, C3, r1, P2, C2, p2));

    // Connecting two interfaces that are already connected should fail.
    CPPUNIT_ASSERT(!managerLocal1Object->Connect(P1, C1, r1, P2, C2, p1));
    CPPUNIT_ASSERT(!managerLocal1Object->Connect(P1, C1, r2, P2, C2, p2));
    CPPUNIT_ASSERT(!managerLocal1Object->Connect(P1, C2, r1, P2, C2, p2));
    CPPUNIT_ASSERT(!managerLocal2Object->Connect(P2, C3, r1, P2, C2, p2));

    // Disconnect all the connections for the next tests
    CPPUNIT_ASSERT(managerLocal1Object->Disconnect(P1, C1, r1, P2, C2, p1));
    CPPUNIT_ASSERT(managerLocal1Object->Disconnect(P1, C1, r2, P2, C2, p2));
    CPPUNIT_ASSERT(managerLocal1Object->Disconnect(P1, C2, r1, P2, C2, p2));
    CPPUNIT_ASSERT(managerLocal2Object->Disconnect(P2, C3, r1, P2, C2, p2));

    // Disconnect should fail if disconnecting non-connected interfaces.
    CPPUNIT_ASSERT(!managerLocal1Object->Disconnect(P1, C1, r1, P2, C2, p1));
    CPPUNIT_ASSERT(!managerLocal1Object->Disconnect(P1, C1, r2, P2, C2, p2));
    CPPUNIT_ASSERT(!managerLocal1Object->Disconnect(P1, C2, r1, P2, C2, p2));
    CPPUNIT_ASSERT(!managerLocal2Object->Disconnect(P2, C3, r1, P2, C2, p2));

    //
    // TODO: After implementing proxy clean-up codes (WHEN DISCONNECT() IS CALLED),
    // enable the following tests!!!
    //
    return;

    // Connection should be established correctly regardless whoever calls Connect() method.
    CPPUNIT_ASSERT(managerLocal2Object->Connect(P1, C1, r1, P2, C2, p1));
    CPPUNIT_ASSERT(managerLocal2Object->Connect(P1, C1, r2, P2, C2, p2));
    CPPUNIT_ASSERT(managerLocal2Object->Connect(P1, C2, r1, P2, C2, p2));
*/

void mtsManagerLocalTest::TestLocalCommandsAndEvents(void)
{
    mtsManagerLocal localManager;

    mtsManagerTestDevice2 * P2C2 = new mtsManagerTestDevice2;
    mtsManagerTestDevice3 * P2C3 = new mtsManagerTestDevice3;
    CPPUNIT_ASSERT(localManager.AddComponent(P2C2));
    CPPUNIT_ASSERT(localManager.AddComponent(P2C3));

    // Connect two interfaces (establish local connection) and test if commands
    // and events work correctly.
    CPPUNIT_ASSERT(localManager.Connect(P2C3->GetName(), "r1", P2C2->GetName(), "p2"));

    // Check initial values
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceRequired1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceProvided1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceProvided2.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C3->InterfaceRequired1.GetValue());

    // Test void command
    P2C3->InterfaceRequired1.CommandVoid();
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceRequired1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceProvided1.GetValue());
    CPPUNIT_ASSERT_EQUAL(0,  P2C2->InterfaceProvided2.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C3->InterfaceRequired1.GetValue());

    // Test write command
    mtsInt valueWrite;
    valueWrite.Data = 2;
    P2C3->InterfaceRequired1.CommandWrite(valueWrite);
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceRequired1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceProvided1.GetValue());
    CPPUNIT_ASSERT_EQUAL(valueWrite.Data,  P2C2->InterfaceProvided2.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C3->InterfaceRequired1.GetValue());

    // Test read command
    mtsInt valueRead;
    valueRead.Data = 0;
    P2C3->InterfaceRequired1.CommandRead(valueRead);
    CPPUNIT_ASSERT_EQUAL(valueWrite.Data, valueRead.Data);
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceRequired1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceProvided1.GetValue());
    CPPUNIT_ASSERT_EQUAL(valueWrite.Data,  P2C2->InterfaceProvided2.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C3->InterfaceRequired1.GetValue());

    // Test qualified read command
    valueWrite.Data = 3;
    valueRead.Data = 0;
    P2C3->InterfaceRequired1.CommandQualifiedRead(valueWrite, valueRead);
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceRequired1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceProvided1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C3->InterfaceRequired1.GetValue());

    // Test void event
    P2C2->InterfaceProvided2.EventVoid();
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceRequired1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceProvided1.GetValue());
    CPPUNIT_ASSERT_EQUAL(0, P2C3->InterfaceRequired1.GetValue());

    // Test write event
    valueWrite.Data = 4;
    P2C2->InterfaceProvided2.EventWrite(valueWrite);
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceRequired1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceProvided1.GetValue());
    CPPUNIT_ASSERT_EQUAL(valueWrite.Data, P2C3->InterfaceRequired1.GetValue());
}

/*
void mtsManagerLocalTest::TestRemoteCommandsAndEvents(void)
{
    mtsManagerGlobal managerGlobal;

    // Prepare local managers for this test
    mtsManagerTestDevice1 * P1C1 = new mtsManagerTestDevice1;
    mtsManagerTestDevice2 * P1C2 = new mtsManagerTestDevice2;
    mtsManagerTestDevice2 * P2C2 = new mtsManagerTestDevice2;
    mtsManagerTestDevice3 * P2C3 = new mtsManagerTestDevice3;

    mtsManagerLocalInterface * managerLocal1 = new mtsManagerLocal(P1);
    mtsManagerLocal * managerLocal1Object = dynamic_cast<mtsManagerLocal*>(managerLocal1);
    managerLocal1Object->ManagerGlobal = &managerGlobal;
    managerGlobal.AddProcess(managerLocal1->GetProcessName());
    managerLocal1Object->AddComponent(P1C1);
    managerLocal1Object->AddComponent(P1C2);
    managerLocal1Object->UnitTestEnabled = true; // run in unit test mode
    managerLocal1Object->UnitTestNetworkProxyEnabled = false; // but disable network proxy processings

    mtsManagerLocalInterface * managerLocal2 = new mtsManagerLocal(P2);
    mtsManagerLocal * managerLocal2Object = dynamic_cast<mtsManagerLocal*>(managerLocal2);
    managerLocal2Object->ManagerGlobal = &managerGlobal;
    managerGlobal.AddProcess(managerLocal2->GetProcessName());
    managerLocal2Object->AddComponent(P2C2);
    managerLocal2Object->AddComponent(P2C3);
    managerLocal2Object->UnitTestEnabled = true; // run in unit test mode
    managerLocal2Object->UnitTestNetworkProxyEnabled = false; // but disable network proxy processings

    // Connect two interfaces (establish remote connection) and test if commands
    // and events work correctly.
    CPPUNIT_ASSERT(managerLocal1Object->Connect(P1, C1, r1, P2, C2, p1));

    // Check initial values
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceRequired1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceProvided1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceProvided2.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C3->InterfaceRequired1.GetValue());

    // Test void command
    P2C3->InterfaceRequired1.CommandVoid();
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceRequired1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceProvided1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(0,  P2C2->InterfaceProvided2.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C3->InterfaceRequired1.GetValue());

    // Test write command
    //mtsInt valueWrite;
    //valueWrite.Data = 2;
    //P2C3->InterfaceRequired1.CommandWrite(valueWrite);
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceRequired1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceProvided1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(valueWrite.Data,  P2C2->InterfaceProvided2.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C3->InterfaceRequired1.GetValue());

    //// Test read command
    //mtsInt valueRead;
    //valueRead.Data = 0;
    //P2C3->InterfaceRequired1.CommandRead(valueRead);
    //CPPUNIT_ASSERT_EQUAL(valueWrite.Data, valueRead.Data);
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceRequired1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceProvided1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(valueWrite.Data,  P2C2->InterfaceProvided2.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C3->InterfaceRequired1.GetValue());

    //// Test qualified read command
    //valueWrite.Data = 3;
    //valueRead.Data = 0;
    //P2C3->InterfaceRequired1.CommandQualifiedRead(valueWrite, valueRead);
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceRequired1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceProvided1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C3->InterfaceRequired1.GetValue());

    //// Test void event
    //P2C2->InterfaceProvided2.EventVoid();
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceRequired1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceProvided1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(0, P2C3->InterfaceRequired1.GetValue());

    //// Test write event
    //valueWrite.Data = 4;
    //P2C2->InterfaceProvided2.EventWrite(valueWrite);
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceRequired1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->InterfaceProvided1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(valueWrite.Data, P2C3->InterfaceRequired1.GetValue());
}
*/

CPPUNIT_TEST_SUITE_REGISTRATION(mtsManagerLocalTest);
