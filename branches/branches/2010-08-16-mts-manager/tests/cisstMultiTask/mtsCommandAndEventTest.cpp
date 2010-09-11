/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Min Yang Jung, Anton Deguet
  Created on: 2009-11-17

  (C) Copyright 2009-2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include "mtsCommandAndEventTest.h"

#include <cisstMultiTask/mtsManagerGlobal.h>
#include <cisstMultiTask/mtsManagerLocal.h>

#include "mtsTestComponents.h"

#define P1 "P1"
#define P2 "P2"
#define P1_OBJ localManager1
#define P2_OBJ localManager2

#define DEFAULT_PROCESS_NAME "LCM"


mtsCommandAndEventTest::mtsCommandAndEventTest()
{
    mtsManagerLocal::UnitTestEnabled = true;
#if !CISST_MTS_HAS_ICE
    mtsManagerLocal::UnitTestNetworkProxyEnabled = false;
#else
    mtsManagerLocal::UnitTestNetworkProxyEnabled = true;
#endif
}


void mtsCommandAndEventTest::setUp(void)
{
}


void mtsCommandAndEventTest::tearDown(void)
{
}

template <class _clientType, class _serverType>
void mtsCommandAndEventTest::TestExecution(_clientType * client, _serverType * server,
                                           double clientExecutionDelay, double serverExecutionDelay,
                                           double blockingDelay)
{
    const double queuingDelay = 10.0 * cmn_ms;
    const osaTimeServer & timeServer = mtsComponentManager::GetInstance()->GetTimeServer();
    double startTime, stopTime;

    // check initial values
    CPPUNIT_ASSERT_EQUAL(-1, server->InterfaceProvided1.GetValue()); // initial value
    CPPUNIT_ASSERT_EQUAL(-1, client->InterfaceRequired1.GetValue()); // initial value

    // value we used to make sure commands are processed, default is
    // -1, void command set to 0
    mtsInt valueWrite;
    valueWrite.Data = 4;

    // loop over void and write commands to alternate blocking and non
    // blocking commands
    unsigned int index;
    for (index = 0; index < 3; index++) {
        // test void command non blocking
        startTime = timeServer.GetRelativeTime();
        client->InterfaceRequired1.CommandVoid();
        stopTime = timeServer.GetRelativeTime();
        CPPUNIT_ASSERT((stopTime - startTime) <= queuingDelay); // make sure execution is fast
        osaSleep(serverExecutionDelay + blockingDelay); // time to dequeue and let command execute
        CPPUNIT_ASSERT_EQUAL(0,  server->InterfaceProvided1.GetValue()); // reset
        CPPUNIT_ASSERT_EQUAL(-1, client->InterfaceRequired1.GetValue()); // unchanged

        // test write command
        startTime = timeServer.GetRelativeTime();
        client->InterfaceRequired1.CommandWrite(valueWrite);
        stopTime = timeServer.GetRelativeTime();
        CPPUNIT_ASSERT((stopTime - startTime) <= queuingDelay); // make sure execution is fast
        osaSleep(serverExecutionDelay + blockingDelay);  // time to dequeue and let command execute
        CPPUNIT_ASSERT_EQUAL(valueWrite.Data, server->InterfaceProvided1.GetValue()); // set to new value
        CPPUNIT_ASSERT_EQUAL(-1, client->InterfaceRequired1.GetValue()); // unchanged
        
        // test void command blocking
        if (blockingDelay > 0.0) {
            startTime = timeServer.GetRelativeTime();
            client->InterfaceRequired1.CommandVoid.ExecuteBlocking();
            stopTime = timeServer.GetRelativeTime();
            std::stringstream message;
            message << "Actual: " << (stopTime - startTime) << " >= " << (blockingDelay * 0.9);
            CPPUNIT_ASSERT_MESSAGE(message.str(), (stopTime - startTime) >= (blockingDelay * 0.9));
        } else {
            // no significant delay but result should be garanteed without sleep
            client->InterfaceRequired1.CommandVoid.ExecuteBlocking();
        }
        CPPUNIT_ASSERT_EQUAL(0,  server->InterfaceProvided1.GetValue()); // reset
        CPPUNIT_ASSERT_EQUAL(-1, client->InterfaceRequired1.GetValue()); // unchanged

        // test write command blocking
        if (blockingDelay > 0.0) {
            startTime = timeServer.GetRelativeTime();
            client->InterfaceRequired1.CommandWrite.ExecuteBlocking(valueWrite);
            stopTime = timeServer.GetRelativeTime();
            std::stringstream message;
            message << "Actual: " << (stopTime - startTime) << " >= " << (blockingDelay * 0.9);
            CPPUNIT_ASSERT_MESSAGE(message.str(), (stopTime - startTime) >= (blockingDelay * 0.9));
        } else {
            // no significant delay but result should be garanteed without sleep
            client->InterfaceRequired1.CommandWrite.ExecuteBlocking(valueWrite);
        }
        CPPUNIT_ASSERT_EQUAL(valueWrite.Data, server->InterfaceProvided1.GetValue()); // set to new value
        CPPUNIT_ASSERT_EQUAL(-1, client->InterfaceRequired1.GetValue()); // unchanged
    }

    // test read command
    mtsInt valueRead;
    valueRead.Data = 0;
    client->InterfaceRequired1.CommandRead(valueRead);
    CPPUNIT_ASSERT_EQUAL(valueWrite.Data, valueRead.Data);
    CPPUNIT_ASSERT_EQUAL(valueWrite.Data, server->InterfaceProvided1.GetValue()); // unchanged
    CPPUNIT_ASSERT_EQUAL(-1, client->InterfaceRequired1.GetValue()); // unchanged

    // test qualified read command
    valueRead.Data = 0;
    client->InterfaceRequired1.CommandQualifiedRead(valueWrite, valueRead);
    CPPUNIT_ASSERT_EQUAL(valueWrite.Data + 1, valueRead.Data);
    CPPUNIT_ASSERT_EQUAL(valueWrite.Data, server->InterfaceProvided1.GetValue()); // unchanged
    CPPUNIT_ASSERT_EQUAL(-1, client->InterfaceRequired1.GetValue()); // unchanged

    // test void event
    server->InterfaceProvided1.EventVoid();
    osaSleep(clientExecutionDelay);
    CPPUNIT_ASSERT_EQUAL(valueWrite.Data, server->InterfaceProvided1.GetValue()); // unchanged
    CPPUNIT_ASSERT_EQUAL(0, client->InterfaceRequired1.GetValue()); // reset by void event

    // test write event
    server->InterfaceProvided1.EventWrite(valueWrite);
    osaSleep(clientExecutionDelay);
    CPPUNIT_ASSERT_EQUAL(valueWrite.Data, server->InterfaceProvided1.GetValue()); // unchanged
    CPPUNIT_ASSERT_EQUAL(valueWrite.Data, client->InterfaceRequired1.GetValue()); // set by write event
}


void mtsCommandAndEventTest::TestLocalDeviceDevice(void)
{
    mtsComponentManager * manager = mtsComponentManager::GetInstance();
    mtsTestDevice2 * client = new mtsTestDevice2;
    mtsTestDevice3 * server = new mtsTestDevice3;
    
    manager->AddComponent(client);
    manager->AddComponent(server);
    manager->Connect(client->GetName(), "r1", server->GetName(), "p1");
    manager->CreateAll();
    manager->StartAll();
    TestExecution(client, server, 0.0, 0.0);
    manager->KillAll();
    osaSleep(0.1 * cmn_s);
    manager->Disconnect(client->GetName(), "r1", server->GetName(), "p1");
    manager->RemoveComponent(client);
    manager->RemoveComponent(server);

    delete client;
    delete server;
}


void mtsCommandAndEventTest::TestLocalPeriodicPeriodic(void)
{
    mtsComponentManager * manager = mtsComponentManager::GetInstance();
    mtsTestPeriodic1 * client = new mtsTestPeriodic1("mtsTestPeriodic1Client");
    mtsTestPeriodic1 * server = new mtsTestPeriodic1("mtsTestPeriodic1Server");

    // these delays are OS dependent, we might need to increase them later
    const double clientExecutionDelay = 0.1 * cmn_s;
    const double serverExecutionDelay = 0.1 * cmn_s;

    manager->AddComponent(client);
    manager->AddComponent(server);
    manager->Connect(client->GetName(), "r1", server->GetName(), "p1");
    manager->CreateAll();
    manager->StartAll();
    TestExecution(client, server, clientExecutionDelay, serverExecutionDelay);
    manager->KillAll();
    osaSleep(0.1 * cmn_s);
    manager->Disconnect(client->GetName(), "r1", server->GetName(), "p1");
    manager->RemoveComponent(client);
    manager->RemoveComponent(server);

    delete client;
    delete server;
}


void mtsCommandAndEventTest::TestLocalContinuousContinuous(void)
{
    mtsComponentManager * manager = mtsComponentManager::GetInstance();
    mtsTestContinuous1 * client = new mtsTestContinuous1("mtsTestContinuous1Client");
    mtsTestContinuous1 * server = new mtsTestContinuous1("mtsTestContinuous1Server");

    // these delays are OS dependent, we might need to increase them later
    const double clientExecutionDelay = 0.1 * cmn_s;
    const double serverExecutionDelay = 0.1 * cmn_s;

    manager->AddComponent(client);
    manager->AddComponent(server);
    manager->Connect(client->GetName(), "r1", server->GetName(), "p1");
    manager->CreateAll();
    manager->StartAll();
    TestExecution(client, server, clientExecutionDelay, serverExecutionDelay);
    manager->KillAll();
    osaSleep(0.1 * cmn_s);
    manager->Disconnect(client->GetName(), "r1", server->GetName(), "p1");
    manager->RemoveComponent(client);
    manager->RemoveComponent(server);

    delete client;
    delete server;
}


void mtsCommandAndEventTest::TestLocalFromCallbackFromCallback(void)
{
    mtsComponentManager * manager = mtsComponentManager::GetInstance();
    mtsTestFromCallback1 * client = new mtsTestFromCallback1("mtsTestFromCallback1Client");
    mtsTestCallbackTrigger * clientTrigger = new mtsTestCallbackTrigger(client);
    mtsTestFromCallback1 * server = new mtsTestFromCallback1("mtsTestFromCallback1Server");
    mtsTestCallbackTrigger * serverTrigger = new mtsTestCallbackTrigger(server);

    // these delays are OS dependent, we might need to increase them later
    const double clientExecutionDelay = 0.1 * cmn_s;
    const double serverExecutionDelay = 0.1 * cmn_s;

    manager->AddComponent(client);
    manager->AddComponent(server);
    manager->Connect(client->GetName(), "r1", server->GetName(), "p1");
    manager->CreateAll();
    manager->StartAll();
    TestExecution(client, server, clientExecutionDelay, serverExecutionDelay);
    manager->KillAll();
    osaSleep(0.1 * cmn_s);
    manager->Disconnect(client->GetName(), "r1", server->GetName(), "p1");
    manager->RemoveComponent(client);
    manager->RemoveComponent(server);

    clientTrigger->Stop();
    delete clientTrigger;
    delete client;
    serverTrigger->Stop();
    delete serverTrigger;
    delete server;
}


void mtsCommandAndEventTest::TestLocalFromSignalFromSignal(void)
{
    mtsComponentManager * manager = mtsComponentManager::GetInstance();
    mtsTestFromSignal1 * client = new mtsTestFromSignal1("mtsTestFromSignal1Client");
    mtsTestFromSignal1 * server = new mtsTestFromSignal1("mtsTestFromSignal1Server");

    // these delays are OS dependent, we might need to increase them later
    const double clientExecutionDelay = 0.1 * cmn_s;
    const double serverExecutionDelay = 0.1 * cmn_s;

    manager->AddComponent(client);
    manager->AddComponent(server);
    manager->Connect(client->GetName(), "r1", server->GetName(), "p1");
    manager->CreateAll();
    manager->StartAll();
    TestExecution(client, server, clientExecutionDelay, serverExecutionDelay);
    manager->KillAll();
    osaSleep(0.1 * cmn_s);
    manager->Disconnect(client->GetName(), "r1", server->GetName(), "p1");
    manager->RemoveComponent(client);
    manager->RemoveComponent(server);

    delete client;
    delete server;
}


void mtsCommandAndEventTest::TestLocalPeriodicPeriodicBlocking(void)
{
    mtsComponentManager * manager = mtsComponentManager::GetInstance();
    const double blockingDelay = 0.5 * cmn_s;
    mtsTestPeriodic1 * client = new mtsTestPeriodic1("mtsTestPeriodic1Client");
    mtsTestPeriodic1 * server = new mtsTestPeriodic1("mtsTestPeriodic1Server", blockingDelay);

    // these delays are OS dependent, we might need to increase them later
    const double clientExecutionDelay = 0.1 * cmn_s;
    const double serverExecutionDelay = 0.1 * cmn_s;

    manager->AddComponent(client);
    manager->AddComponent(server);
    manager->Connect(client->GetName(), "r1", server->GetName(), "p1");
    manager->CreateAll();
    manager->StartAll();
    TestExecution(client, server, clientExecutionDelay, serverExecutionDelay, blockingDelay);
    manager->KillAll();
    osaSleep(0.1 * cmn_s);
    manager->Disconnect(client->GetName(), "r1", server->GetName(), "p1");
    manager->RemoveComponent(client);
    manager->RemoveComponent(server);

    delete client;
    delete server;
}


void mtsCommandAndEventTest::TestLocalContinuousContinuousBlocking(void)
{
    mtsComponentManager * manager = mtsComponentManager::GetInstance();
    const double blockingDelay = 0.5 * cmn_s;
    mtsTestContinuous1 * client = new mtsTestContinuous1("mtsTestContinuous1Client");
    mtsTestContinuous1 * server = new mtsTestContinuous1("mtsTestContinuous1Server", blockingDelay);

    // these delays are OS dependent, we might need to increase them later
    const double clientExecutionDelay = 0.1 * cmn_s;
    const double serverExecutionDelay = 0.1 * cmn_s;

    manager->AddComponent(client);
    manager->AddComponent(server);
    manager->Connect(client->GetName(), "r1", server->GetName(), "p1");
    manager->CreateAll();
    manager->StartAll();
    TestExecution(client, server, clientExecutionDelay, serverExecutionDelay, blockingDelay);
    manager->KillAll();
    osaSleep(0.1 * cmn_s);
    manager->Disconnect(client->GetName(), "r1", server->GetName(), "p1");
    manager->RemoveComponent(client);
    manager->RemoveComponent(server);

    delete client;
    delete server;
}


void mtsCommandAndEventTest::TestLocalFromCallbackFromCallbackBlocking(void)
{
    mtsComponentManager * manager = mtsComponentManager::GetInstance();
    const double blockingDelay = 0.5 * cmn_s;
    mtsTestFromCallback1 * client = new mtsTestFromCallback1("mtsTestFromCallback1Client");
    mtsTestCallbackTrigger * clientTrigger = new mtsTestCallbackTrigger(client);
    mtsTestFromCallback1 * server = new mtsTestFromCallback1("mtsTestFromCallback1Server", blockingDelay);
    mtsTestCallbackTrigger * serverTrigger = new mtsTestCallbackTrigger(server);

    // these delays are OS dependent, we might need to increase them later
    const double clientExecutionDelay = 0.1 * cmn_s;
    const double serverExecutionDelay = 0.1 * cmn_s;

    manager->AddComponent(client);
    manager->AddComponent(server);
    manager->Connect(client->GetName(), "r1", server->GetName(), "p1");
    manager->CreateAll();
    manager->StartAll();
    TestExecution(client, server, clientExecutionDelay, serverExecutionDelay, blockingDelay);
    manager->KillAll();
    osaSleep(0.1 * cmn_s);
    manager->Disconnect(client->GetName(), "r1", server->GetName(), "p1");
    manager->RemoveComponent(client);
    manager->RemoveComponent(server);

    clientTrigger->Stop();
    delete clientTrigger;
    delete client;
    serverTrigger->Stop();
    delete serverTrigger;
    delete server;
}


void mtsCommandAndEventTest::TestLocalFromSignalFromSignalBlocking(void)
{
    mtsComponentManager * manager = mtsComponentManager::GetInstance();
    const double blockingDelay = 0.5 * cmn_s;
    mtsTestFromSignal1 * client = new mtsTestFromSignal1("mtsTestFromSignal1Client");
    mtsTestFromSignal1 * server = new mtsTestFromSignal1("mtsTestFromSignal1Server", blockingDelay);

    // these delays are OS dependent, we might need to increase them later
    const double clientExecutionDelay = 0.1 * cmn_s;
    const double serverExecutionDelay = 0.1 * cmn_s;

    manager->AddComponent(client);
    manager->AddComponent(server);
    manager->Connect(client->GetName(), "r1", server->GetName(), "p1");
    manager->CreateAll();
    manager->StartAll();
    TestExecution(client, server, clientExecutionDelay, serverExecutionDelay, blockingDelay);
    manager->KillAll();
    osaSleep(0.1 * cmn_s);
    manager->Disconnect(client->GetName(), "r1", server->GetName(), "p1");
    manager->RemoveComponent(client);
    manager->RemoveComponent(server);

    delete client;
    delete server;
}


/*
void mtsCommandAndEventTest::TestRemoteDeviceDevice(void)
{
    mtsManagerGlobal managerGlobal;

    // Prepare local managers for this test
    mtsTestDevice1 * P1C1 = new mtsTestDevice1;
    mtsTestDevice2 * P1C2 = new mtsTestDevice2;
    mtsTestDevice2 * P2C2 = new mtsTestDevice2;
    mtsTestDevice3 * P2C3 = new mtsTestDevice3;

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

CPPUNIT_TEST_SUITE_REGISTRATION(mtsCommandAndEventTest);
