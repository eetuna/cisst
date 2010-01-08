/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerLocalTest.cpp 2009-03-05 mjung5 $
  
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
#include <cisstOSAbstraction/osaSleep.h>

#include "mtsManagerTestClasses.h"

#define P1 "P1"
#define P2 "P2"
#define C1 "C1"
#define C2 "C2"
#define C3 "C3"
#define C4 "C4"
#define p1 "p1"
#define p2 "p2"
#define r1 "r1"
#define r2 "r2"
#define P1_OBJ localManager1
#define P2_OBJ localManager2

using namespace std;

void mtsManagerLocalTest::setUp(void)
{
    localManager1 = new mtsManagerLocal(P1, "localhost");
    localManager2 = new mtsManagerLocal(P2, "localhost");
}

void mtsManagerLocalTest::tearDown(void)
{
    delete localManager1;
    delete localManager2;
}

// TODO: Please see some comments for mtsManagerLocal constructor in mtsManagerLocal.h.
// Need to check a couple of issues (TODOs) there.
void mtsManagerLocalTest::TestConstructor(void)
{
    // In this test, we can assume that the constructor is always called with 
    // two arguments because users cannot directly call the constructor.
    // (Singleton implementation: protected constructor)

    // Standalone mode 
    mtsManagerLocal managerLocal1("", "");
    CPPUNIT_ASSERT(managerLocal1.ManagerGlobal);
    CPPUNIT_ASSERT(managerLocal1.ProcessName == "");
    CPPUNIT_ASSERT(managerLocal1.ProcessIP == "");
    CPPUNIT_ASSERT(managerLocal1.ManagerGlobal->FindProcess(""));

    mtsManagerLocal managerLocal2("localhost", "");
    CPPUNIT_ASSERT(managerLocal2.ManagerGlobal);
    CPPUNIT_ASSERT(managerLocal2.ProcessName == "localhost");
    CPPUNIT_ASSERT(managerLocal2.ProcessIP == "");
    CPPUNIT_ASSERT(managerLocal2.ManagerGlobal->FindProcess("localhost"));

    mtsManagerLocal managerLocal3("", "localhost");
    CPPUNIT_ASSERT(managerLocal3.ManagerGlobal);
    CPPUNIT_ASSERT(managerLocal3.ProcessName == "");
    CPPUNIT_ASSERT(managerLocal3.ProcessIP == "localhost");
    CPPUNIT_ASSERT(managerLocal3.ManagerGlobal->FindProcess(""));

    // Network mode
    mtsManagerLocal managerLocal4("ProcessName", "ProcessIP");
    CPPUNIT_ASSERT(managerLocal4.ManagerGlobal == NULL); // TODO: not yet implemented
    CPPUNIT_ASSERT(managerLocal4.ProcessName == "ProcessName");
    CPPUNIT_ASSERT(managerLocal4.ProcessIP == "ProcessIP");
}

void mtsManagerLocalTest::TestCleanup(void)
{
    mtsManagerLocal managerLocal("", "");
    CPPUNIT_ASSERT(managerLocal.ManagerGlobal);

    managerLocal.Cleanup();
    CPPUNIT_ASSERT(managerLocal.ManagerGlobal == NULL);
}

void mtsManagerLocalTest::TestGetInstance(void)
{
    mtsManagerLocal * managerLocal = mtsManagerLocal::GetInstance();

    CPPUNIT_ASSERT(managerLocal);
    CPPUNIT_ASSERT(managerLocal->ManagerGlobal);
    CPPUNIT_ASSERT(mtsManagerLocal::Instance);
    CPPUNIT_ASSERT(managerLocal->ProcessName == "");
    CPPUNIT_ASSERT(managerLocal->ProcessIP == "");
    CPPUNIT_ASSERT(managerLocal->ManagerGlobal->FindProcess(""));

    // Clear current singleton instance. Note that this is done only for unit-test
    // purpose. This instance should not be touched from the outside.
    mtsManagerLocal * temp = managerLocal;
    mtsManagerLocal::Instance = NULL;   // avoid a crash due to duplicate memory release
    delete temp;

    managerLocal = mtsManagerLocal::GetInstance(P1, "");
    CPPUNIT_ASSERT(managerLocal);
    CPPUNIT_ASSERT(managerLocal->ManagerGlobal);
    CPPUNIT_ASSERT(mtsManagerLocal::Instance);
    CPPUNIT_ASSERT(managerLocal->ProcessName == P1);
    CPPUNIT_ASSERT(managerLocal->ProcessIP == "");
    CPPUNIT_ASSERT(managerLocal->ManagerGlobal->FindProcess(P1));
}

void mtsManagerLocalTest::TestAddComponent(void)
{
    //-----------------------------------------------------
    // Test with mtsDevice type components
    mtsManagerLocal managerLocal1(P1, "");
    mtsManagerTestC2Device * c2Device = new mtsManagerTestC2Device;

    // Invalid argument test
    CPPUNIT_ASSERT(!managerLocal1.AddComponent(NULL));

    // Check with the global component manager.    
    // Should fail if a component to be added has already been registered before
    CPPUNIT_ASSERT(managerLocal1.ManagerGlobal->AddComponent(P1, C2));
    CPPUNIT_ASSERT(!managerLocal1.AddComponent(c2Device));

    // Should succeed if a component is added for the first time
    CPPUNIT_ASSERT(managerLocal1.ManagerGlobal->RemoveComponent(P1, C2));
    CPPUNIT_ASSERT(managerLocal1.AddComponent(c2Device));
    CPPUNIT_ASSERT(managerLocal1.ComponentMap.FindItem(C2));
    
    // The C2 has both provided interfaces (p1, p2) and a required interface (r1).
    CPPUNIT_ASSERT(managerLocal1.ManagerGlobal->FindRequiredInterface(P1, C2, r1));
    CPPUNIT_ASSERT(managerLocal1.ManagerGlobal->FindProvidedInterface(P1, C2, p1));
    CPPUNIT_ASSERT(managerLocal1.ManagerGlobal->FindProvidedInterface(P1, C2, p2));

    //-----------------------------------------------------
    // Test with mtsDevice type components
    mtsManagerLocal managerLocal2(P1, "");
    mtsManagerTestC2 * c2Task = new mtsManagerTestC2;

    // Invalid argument test
    CPPUNIT_ASSERT(!managerLocal2.AddComponent(NULL));

    // Check with the global component manager.    
    // Should fail if a component to be added has already been registered before
    CPPUNIT_ASSERT(managerLocal2.ManagerGlobal->AddComponent(P1, C2));
    CPPUNIT_ASSERT(!managerLocal2.AddComponent(c2Task));

    // Should succeed if a component is added for the first time
    CPPUNIT_ASSERT(managerLocal2.ManagerGlobal->RemoveComponent(P1, C2));
    CPPUNIT_ASSERT(managerLocal2.AddComponent(c2Task));
    CPPUNIT_ASSERT(managerLocal2.ComponentMap.FindItem(C2));
    
    // The C2 has both provided interfaces (p1, p2) and a required interface (r1).
    CPPUNIT_ASSERT(managerLocal2.ManagerGlobal->FindRequiredInterface(P1, C2, r1));
    CPPUNIT_ASSERT(managerLocal2.ManagerGlobal->FindProvidedInterface(P1, C2, p1));
    CPPUNIT_ASSERT(managerLocal2.ManagerGlobal->FindProvidedInterface(P1, C2, p2));
}

void mtsManagerLocalTest::TestRemoveComponent(void)
{
    //-----------------------------------------------------
    // Test with mtsDevice type components
    mtsManagerLocal managerLocal1(P1, "");
    mtsManagerTestC1Device * c1Device = new mtsManagerTestC1Device;

    // Invalid argument test
    CPPUNIT_ASSERT(!managerLocal1.RemoveComponent(NULL));

    CPPUNIT_ASSERT(managerLocal1.AddComponent(c1Device));

    CPPUNIT_ASSERT(managerLocal1.ManagerGlobal->FindComponent(P1, C1));
    CPPUNIT_ASSERT(managerLocal1.RemoveComponent(c1Device));
    CPPUNIT_ASSERT(!managerLocal1.ManagerGlobal->FindComponent(P1, C1));
    
    //-----------------------------------------------------
    // Test with mtsTask type components
    mtsManagerLocal managerLocal2(P1, "");
    mtsManagerTestC1 * c1Task = new mtsManagerTestC1;

    // Invalid argument test
    CPPUNIT_ASSERT(!managerLocal2.RemoveComponent(NULL));

    CPPUNIT_ASSERT(managerLocal2.AddComponent(c1Task));

    CPPUNIT_ASSERT(managerLocal2.ManagerGlobal->FindComponent(P1, C1));
    CPPUNIT_ASSERT(managerLocal2.RemoveComponent(c1Task));
    CPPUNIT_ASSERT(!managerLocal2.ManagerGlobal->FindComponent(P1, C1));

    //-----------------------------------------------------
    // Test using component name instead of component object
    mtsManagerLocal managerLocal3(P1, "");
    
    // should fail: not yet registered component name
    CPPUNIT_ASSERT(!managerLocal3.RemoveComponent(C1));

    CPPUNIT_ASSERT(managerLocal3.AddComponent(c1Device));

    CPPUNIT_ASSERT(managerLocal3.ManagerGlobal->FindComponent(P1, C1));
    CPPUNIT_ASSERT(managerLocal3.RemoveComponent(C1));
    CPPUNIT_ASSERT(!managerLocal3.ManagerGlobal->FindComponent(P1, C1));
}

void mtsManagerLocalTest::TestGetNamesOfComponents(void)
{
    mtsManagerLocal managerLocal(P1, "");
    mtsManagerTestC1Device * c1Device = new mtsManagerTestC1Device;
    mtsManagerTestC2Device * c2Device = new mtsManagerTestC2Device;
    mtsManagerTestC3Device * c3Device = new mtsManagerTestC3Device;

    CPPUNIT_ASSERT(managerLocal.AddComponent(c1Device));
    CPPUNIT_ASSERT(managerLocal.AddComponent(c2Device));
    CPPUNIT_ASSERT(managerLocal.AddComponent(c3Device));

    std::vector<std::string> namesOfComponents1 = managerLocal.GetNamesOfComponents();
    CPPUNIT_ASSERT_EQUAL((unsigned int) 3, namesOfComponents1.size());
    for (int i = 0; i < 3; ++i) {
        CPPUNIT_ASSERT(namesOfComponents1[i] == c1Device->GetName() ||
                       namesOfComponents1[i] == c2Device->GetName() ||
                       namesOfComponents1[i] == c3Device->GetName());
    }

    std::vector<std::string> namesOfComponents2;
    managerLocal.GetNamesOfComponents(namesOfComponents2);
    CPPUNIT_ASSERT_EQUAL((unsigned int) 3, namesOfComponents2.size());
    for (int i = 0; i < 3; ++i) {
        CPPUNIT_ASSERT(namesOfComponents2[i] == c1Device->GetName() ||
                       namesOfComponents2[i] == c2Device->GetName() ||
                       namesOfComponents2[i] == c3Device->GetName());
    }
}

void mtsManagerLocalTest::TestGetComponent(void)
{
    mtsManagerLocal managerLocal(P1, "");
    mtsManagerTestC1Device * c1Device = new mtsManagerTestC1Device;
    mtsManagerTestC2Device * c2Device = new mtsManagerTestC2Device;
    mtsManagerTestC3Device * c3Device = new mtsManagerTestC3Device;

    CPPUNIT_ASSERT(NULL == managerLocal.GetComponent(C1));
    CPPUNIT_ASSERT(NULL == managerLocal.GetComponent(C2));
    CPPUNIT_ASSERT(NULL == managerLocal.GetComponent(C3));

    CPPUNIT_ASSERT(managerLocal.AddComponent(c1Device));
    CPPUNIT_ASSERT(managerLocal.AddComponent(c2Device));
    CPPUNIT_ASSERT(managerLocal.AddComponent(c3Device));

    CPPUNIT_ASSERT(c1Device == managerLocal.GetComponent(C1));
    CPPUNIT_ASSERT(c2Device == managerLocal.GetComponent(C2));
    CPPUNIT_ASSERT(c3Device == managerLocal.GetComponent(C3));
}

void mtsManagerLocalTest::TestGetProcessName(void)
{
    mtsManagerLocal managerLocal1(P1, "");
    CPPUNIT_ASSERT(managerLocal1.GetProcessName() == P1);

    mtsManagerLocal managerLocal2(P2, "");
    CPPUNIT_ASSERT(managerLocal2.GetProcessName() == P2);
}

void mtsManagerLocalTest::TestConnectDisconnect(void)
{
    //
    // Local connection test
    //
    mtsManagerLocal managerLocal(P1, "");

    // Test with invalid arguments
    CPPUNIT_ASSERT(!managerLocal.Connect(C1, r1, C2, p1));

    mtsManagerTestC1Device * c1Device = new mtsManagerTestC1Device;
    mtsManagerTestC2Device * c2Device = new mtsManagerTestC2Device;
    CPPUNIT_ASSERT(managerLocal.AddComponent(c1Device));
    CPPUNIT_ASSERT(managerLocal.AddComponent(c2Device));

    CPPUNIT_ASSERT(managerLocal.Connect(C1, r1, C2, p1));

    //
    // Remote connection test
    //

    // Test with invalid arguments
    CPPUNIT_ASSERT(!managerLocal.Connect(P1, C1, r1, P2, C2, p1));

    mtsManagerGlobal managerGlobal;

    // Prepare local managers for this test
    mtsManagerTestC1Device * P1C1 = new mtsManagerTestC1Device;
    mtsManagerTestC2Device * P1C2 = new mtsManagerTestC2Device;
    mtsManagerTestC2Device * P2C2 = new mtsManagerTestC2Device;
    mtsManagerTestC3Device * P2C3 = new mtsManagerTestC3Device;

    mtsManagerLocalInterface * managerLocal1 = new mtsManagerLocal(P1);
    mtsManagerLocal * managerLocal1Object = dynamic_cast<mtsManagerLocal*>(managerLocal1);
    managerLocal1Object->ManagerGlobal = &managerGlobal;
    managerGlobal.AddProcess(managerLocal1);
    managerLocal1Object->AddComponent(P1C1);
    managerLocal1Object->AddComponent(P1C2);

    mtsManagerLocalInterface * managerLocal2 = new mtsManagerLocal(P2);
    mtsManagerLocal * managerLocal2Object = dynamic_cast<mtsManagerLocal*>(managerLocal2);
    managerLocal2Object->ManagerGlobal = &managerGlobal;
    managerGlobal.AddProcess(managerLocal2);
    managerLocal2Object->AddComponent(P2C2);
    managerLocal2Object->AddComponent(P2C3);

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

    // Connection should be established correctly regardless whoever calls Connect() method.
    CPPUNIT_ASSERT(managerLocal2Object->Connect(P1, C1, r1, P2, C2, p1));
    CPPUNIT_ASSERT(managerLocal2Object->Connect(P1, C1, r2, P2, C2, p2));
    CPPUNIT_ASSERT(managerLocal2Object->Connect(P1, C2, r1, P2, C2, p2));    
}

void mtsManagerLocalTest::TestCreateAll(void)
{
    mtsManagerLocal managerLocal(P1, "");
    mtsManagerTestC1 * c1Task = new mtsManagerTestC1;    // C1 is of mtsTaskPeriodic type
    mtsManagerTestC2 * c2Task = new mtsManagerTestC2;    // C2 is of mtsTaskContinuous type
    mtsManagerTestC3 * c3Task = new mtsManagerTestC3;    // C3 is of mtsTaskCallback type

    CPPUNIT_ASSERT(managerLocal.AddComponent(c1Task));
    CPPUNIT_ASSERT(managerLocal.AddComponent(c2Task));
    CPPUNIT_ASSERT(managerLocal.AddComponent(c3Task));
    
    // Check internal states before calling CreateAll()
    CPPUNIT_ASSERT_EQUAL(mtsTask::CONSTRUCTED, c1Task->TaskState);
    CPPUNIT_ASSERT_EQUAL(mtsTask::CONSTRUCTED, c2Task->TaskState);
    CPPUNIT_ASSERT_EQUAL(mtsTask::CONSTRUCTED, c3Task->TaskState);

    managerLocal.CreateAll();

    // TODO: Resolve this unit-test
    //CPPUNIT_ASSERT_EQUAL(mtsTask::INITIALIZING, c1Task->TaskState);
    CPPUNIT_ASSERT_EQUAL(mtsTask::INITIALIZING, c2Task->TaskState);
    CPPUNIT_ASSERT_EQUAL(mtsTask::INITIALIZING, c3Task->TaskState);

    managerLocal.KillAll();
}

void mtsManagerLocalTest::TestStartAll(void)
{
    mtsManagerLocal managerLocal(P1, "");
    mtsManagerTestC1 * c1Task = new mtsManagerTestC1;    // C1 is of mtsTaskPeriodic type
    mtsManagerTestC2 * c2Task = new mtsManagerTestC2;    // C2 is of mtsTaskContinuous type
    mtsManagerTestC3 * c3Task = new mtsManagerTestC3;    // C3 is of mtsTaskCallback type

    CPPUNIT_ASSERT(managerLocal.AddComponent(c1Task));
    CPPUNIT_ASSERT(managerLocal.AddComponent(c2Task));
    CPPUNIT_ASSERT(managerLocal.AddComponent(c3Task));

    // Establish connections between the three components of mtsTask type
    // Connection: (P1, C1, r1) ~ (P2, C2, p1)
    CPPUNIT_ASSERT(managerLocal.Connect(C1, r1, C2, p1));
    // Connection: (P1, C1, r2) ~ (P2, C2, p2)
    CPPUNIT_ASSERT(managerLocal.Connect(C1, r2, C2, p2));
    // Connection: (P1, C2, r1) ~ (P2, C2, p2)
    CPPUNIT_ASSERT(managerLocal.Connect(C2, r1, C2, p2));
    // Connection: (P2, C3, r1) ~ (P2, C2, p2)
    CPPUNIT_ASSERT(managerLocal.Connect(C3, r1, C2, p2));
    
    managerLocal.CreateAll();

    // TODO: Resolve this unit-test
    //CPPUNIT_ASSERT_EQUAL(mtsTask::INITIALIZING, c1Task->TaskState);
    CPPUNIT_ASSERT_EQUAL(mtsTask::INITIALIZING, c2Task->TaskState);
    CPPUNIT_ASSERT_EQUAL(mtsTask::INITIALIZING, c3Task->TaskState);

    managerLocal.StartAll();

    CPPUNIT_ASSERT_EQUAL(mtsTask::ACTIVE, c1Task->TaskState);
    CPPUNIT_ASSERT_EQUAL(mtsTask::ACTIVE, c2Task->TaskState);
    CPPUNIT_ASSERT_EQUAL(mtsTask::INITIALIZING, c3Task->TaskState);

    managerLocal.KillAll();

    osaSleep(1 * cmn_ms);
}

void mtsManagerLocalTest::TestKillAll(void)
{
    mtsManagerLocal managerLocal(P1, "");
    mtsManagerTestC1 * c1Task = new mtsManagerTestC1;    // C1 is of mtsTaskPeriodic type
    mtsManagerTestC2 * c2Task = new mtsManagerTestC2;    // C2 is of mtsTaskContinuous type
    mtsManagerTestC3 * c3Task = new mtsManagerTestC3;    // C3 is of mtsTaskCallback type

    CPPUNIT_ASSERT(managerLocal.AddComponent(c1Task));
    CPPUNIT_ASSERT(managerLocal.AddComponent(c2Task));
    CPPUNIT_ASSERT(managerLocal.AddComponent(c3Task));

    // Establish connections between the three components of mtsTask type
    // Connection: (P1, C1, r1) ~ (P2, C2, p1)
    CPPUNIT_ASSERT(managerLocal.Connect(C1, r1, C2, p1));
    // Connection: (P1, C1, r2) ~ (P2, C2, p2)
    CPPUNIT_ASSERT(managerLocal.Connect(C1, r2, C2, p2));
    // Connection: (P1, C2, r1) ~ (P2, C2, p2)
    CPPUNIT_ASSERT(managerLocal.Connect(C2, r1, C2, p2));
    // Connection: (P2, C3, r1) ~ (P2, C2, p2)
    CPPUNIT_ASSERT(managerLocal.Connect(C3, r1, C2, p2));
    
    managerLocal.CreateAll();
    managerLocal.StartAll();
    managerLocal.KillAll();

    osaSleep(1 * cmn_ms);

    CPPUNIT_ASSERT(c1Task->TaskState == mtsTask::FINISHING || 
                   c1Task->TaskState == mtsTask::FINISHED);
    CPPUNIT_ASSERT(c2Task->TaskState == mtsTask::FINISHING || 
                   c2Task->TaskState == mtsTask::FINISHED);
    CPPUNIT_ASSERT(c3Task->TaskState == mtsTask::FINISHING || 
                   c3Task->TaskState == mtsTask::FINISHED);
}

void mtsManagerLocalTest::TestLocalCommandsAndEvents(void)
{
    mtsManagerGlobal managerGlobal;

    // Prepare local managers for this test
    mtsManagerTestC2Device * P2C2 = new mtsManagerTestC2Device;
    mtsManagerTestC3Device * P2C3 = new mtsManagerTestC3Device;

    mtsManagerLocalInterface * managerLocal2 = new mtsManagerLocal(P2);
    mtsManagerLocal * managerLocal2Object = dynamic_cast<mtsManagerLocal*>(managerLocal2);
    managerLocal2Object->ManagerGlobal = &managerGlobal;
    managerGlobal.AddProcess(managerLocal2);
    managerLocal2Object->AddComponent(P2C2);
    managerLocal2Object->AddComponent(P2C3);

    // Connect two interfaces (establish local connection) and test if commands 
    // and events work correctly.
    CPPUNIT_ASSERT(managerLocal2Object->Connect(P2, C3, r1, P2, C2, p2));

    // Check initial values
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->RequiredInterface1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->ProvidedInterface1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->ProvidedInterface2.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C3->RequiredInterface1.GetValue());

    // Test void command
    P2C3->RequiredInterface1.CommandVoid();
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->RequiredInterface1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->ProvidedInterface1.GetValue());
    CPPUNIT_ASSERT_EQUAL(0,  P2C2->ProvidedInterface2.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C3->RequiredInterface1.GetValue());

    // Test write command
    mtsInt valueWrite;
    valueWrite.Data = 2;
    P2C3->RequiredInterface1.CommandWrite(valueWrite);
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->RequiredInterface1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->ProvidedInterface1.GetValue());
    CPPUNIT_ASSERT_EQUAL(valueWrite.Data,  P2C2->ProvidedInterface2.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C3->RequiredInterface1.GetValue());

    // Test read command
    mtsInt valueRead;
    valueRead.Data = 0;
    P2C3->RequiredInterface1.CommandRead(valueRead);
    CPPUNIT_ASSERT_EQUAL(valueWrite.Data, valueRead.Data);
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->RequiredInterface1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->ProvidedInterface1.GetValue());
    CPPUNIT_ASSERT_EQUAL(valueWrite.Data,  P2C2->ProvidedInterface2.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C3->RequiredInterface1.GetValue());    

    // Test qualified read command
    valueWrite.Data = 3;
    valueRead.Data = 0;
    P2C3->RequiredInterface1.CommandQualifiedRead(valueWrite, valueRead);
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->RequiredInterface1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->ProvidedInterface1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C3->RequiredInterface1.GetValue());    

    // Test void event
    P2C2->ProvidedInterface2.EventVoid();
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->RequiredInterface1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->ProvidedInterface1.GetValue());
    CPPUNIT_ASSERT_EQUAL(0, P2C3->RequiredInterface1.GetValue());

    // Test write event
    valueWrite.Data = 4;
    P2C2->ProvidedInterface2.EventWrite(valueWrite);
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->RequiredInterface1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->ProvidedInterface1.GetValue());
    CPPUNIT_ASSERT_EQUAL(valueWrite.Data, P2C3->RequiredInterface1.GetValue());
}

void mtsManagerLocalTest::TestRemoteCommandsAndEvents(void)
{
    mtsManagerGlobal managerGlobal;

    // Prepare local managers for this test
    mtsManagerTestC1Device * P1C1 = new mtsManagerTestC1Device;
    mtsManagerTestC2Device * P1C2 = new mtsManagerTestC2Device;
    mtsManagerTestC2Device * P2C2 = new mtsManagerTestC2Device;
    mtsManagerTestC3Device * P2C3 = new mtsManagerTestC3Device;

    mtsManagerLocalInterface * managerLocal1 = new mtsManagerLocal(P1);
    mtsManagerLocal * managerLocal1Object = dynamic_cast<mtsManagerLocal*>(managerLocal1);
    managerLocal1Object->ManagerGlobal = &managerGlobal;
    managerGlobal.AddProcess(managerLocal1);
    managerLocal1Object->AddComponent(P1C1);
    managerLocal1Object->AddComponent(P1C2);

    mtsManagerLocalInterface * managerLocal2 = new mtsManagerLocal(P2);
    mtsManagerLocal * managerLocal2Object = dynamic_cast<mtsManagerLocal*>(managerLocal2);
    managerLocal2Object->ManagerGlobal = &managerGlobal;
    managerGlobal.AddProcess(managerLocal2);
    managerLocal2Object->AddComponent(P2C2);
    managerLocal2Object->AddComponent(P2C3);

    // Connect two interfaces (establish remote connection) and test if commands
    // and events work correctly.
    CPPUNIT_ASSERT(managerLocal2Object->Connect(P1, C1, r1, P2, C2, p1));

    // Check initial values
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->RequiredInterface1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->ProvidedInterface1.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C2->ProvidedInterface2.GetValue());
    CPPUNIT_ASSERT_EQUAL(-1, P2C3->RequiredInterface1.GetValue());

    // Test void command
    P2C3->RequiredInterface1.CommandVoid();
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->RequiredInterface1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->ProvidedInterface1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(0,  P2C2->ProvidedInterface2.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C3->RequiredInterface1.GetValue());

    // Test write command
    //mtsInt valueWrite;
    //valueWrite.Data = 2;
    //P2C3->RequiredInterface1.CommandWrite(valueWrite);
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->RequiredInterface1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->ProvidedInterface1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(valueWrite.Data,  P2C2->ProvidedInterface2.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C3->RequiredInterface1.GetValue());

    //// Test read command
    //mtsInt valueRead;
    //valueRead.Data = 0;
    //P2C3->RequiredInterface1.CommandRead(valueRead);
    //CPPUNIT_ASSERT_EQUAL(valueWrite.Data, valueRead.Data);
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->RequiredInterface1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->ProvidedInterface1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(valueWrite.Data,  P2C2->ProvidedInterface2.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C3->RequiredInterface1.GetValue());    

    //// Test qualified read command
    //valueWrite.Data = 3;
    //valueRead.Data = 0;
    //P2C3->RequiredInterface1.CommandQualifiedRead(valueWrite, valueRead);
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->RequiredInterface1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->ProvidedInterface1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C3->RequiredInterface1.GetValue());    

    //// Test void event
    //P2C2->ProvidedInterface2.EventVoid();
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->RequiredInterface1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->ProvidedInterface1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(0, P2C3->RequiredInterface1.GetValue());

    //// Test write event
    //valueWrite.Data = 4;
    //P2C2->ProvidedInterface2.EventWrite(valueWrite);
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->RequiredInterface1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(-1, P2C2->ProvidedInterface1.GetValue());
    //CPPUNIT_ASSERT_EQUAL(valueWrite.Data, P2C3->RequiredInterface1.GetValue());
}

CPPUNIT_TEST_SUITE_REGISTRATION(mtsManagerLocalTest);
