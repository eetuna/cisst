/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerLocalTest.cpp 2009-03-05 mjung5 $
  
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

#include "mtsManagerLocalTest.h"

#include <cisstMultiTask/mtsManagerGlobal.h>
#include <cisstMultiTask/mtsManagerLocal.h>

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
    CPPUNIT_ASSERT(managerLocal->ProcessName == "");
    CPPUNIT_ASSERT(managerLocal->ProcessIP == "");
    CPPUNIT_ASSERT(managerLocal->ManagerGlobal->FindProcess(""));
}

void mtsManagerLocalTest::TestAddComponent(void)
{
    mtsManagerLocal * managerLocal = mtsManagerLocal::GetInstance();

    // Invalid argument test
    //CPPUNIT_ASSERT(!managerLocal.AddComponent(NULL));

    // Check if this works correctly with the global component manager
}

CPPUNIT_TEST_SUITE_REGISTRATION(mtsManagerLocalTest);
