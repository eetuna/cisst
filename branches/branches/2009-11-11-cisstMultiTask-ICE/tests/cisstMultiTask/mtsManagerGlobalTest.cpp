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
