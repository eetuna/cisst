/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsGlobalManagerTest.cpp 2009-03-05 mjung5 $
  
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

#include "mtsGlobalManagerTest.h"

#include <cisstMultiTask/mtsGlobalManager.h>

//-----------------------------------------------------------------------------
//	Tests for public variables and methods
//-----------------------------------------------------------------------------
void mtsGlobalManagerTest::TestAddComponent(void)
{
    mtsGlobalManager globalManager;

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

void mtsGlobalManagerTest::TestFindComponent(void)
{
    mtsGlobalManager globalManager;

    CPPUNIT_ASSERT(!globalManager.FindComponent("P1", "C1"));
    CPPUNIT_ASSERT(globalManager.AddComponent("P1", "C1"));
    CPPUNIT_ASSERT(globalManager.FindComponent("P1", "C1"));
}

void mtsGlobalManagerTest::TestRemoveComponent(void)
{
    mtsGlobalManager globalManager;

    CPPUNIT_ASSERT(globalManager.AddComponent("P1", "C1"));
    CPPUNIT_ASSERT(globalManager.FindComponent("P1", "C1"));
    CPPUNIT_ASSERT(globalManager.RemoveComponent("P1", "C1"));
    CPPUNIT_ASSERT(!globalManager.FindComponent("P1", "C1"));
}


void mtsGlobalManagerTest::TestAddInterface(void)
{
    // TODO: implement this
}

void mtsGlobalManagerTest::TestFindInterface(void)
{
    mtsGlobalManager globalManager;

    const std::string providedInterfaceName("ProvidedInterface1");
    CPPUNIT_ASSERT(!globalManager.FindInterface("P1", "C1", providedInterfaceName));
    CPPUNIT_ASSERT(globalManager.AddInterface("P1", "C1", providedInterfaceName));
    CPPUNIT_ASSERT(globalManager.FindInterface("P1", "C1", providedInterfaceName));

    const std::string requiredInterfaceName("RequiredInterface1");
    CPPUNIT_ASSERT(!globalManager.FindInterface("P1", "C2", requiredInterfaceName, false));
    CPPUNIT_ASSERT(globalManager.AddInterface("P1", "C2", requiredInterfaceName, false));
    CPPUNIT_ASSERT(globalManager.FindInterface("P1", "C2", requiredInterfaceName, false));

    CPPUNIT_ASSERT(!globalManager.FindInterface("P1", "C1", requiredInterfaceName));
    CPPUNIT_ASSERT(!globalManager.FindInterface("P1", "C2", providedInterfaceName));
}

void mtsGlobalManagerTest::TestRemoveInterface(void)
{
    // TODO: implement this
}

void mtsGlobalManagerTest::TestConnect(void)
{
    // TODO: implement this
}

void mtsGlobalManagerTest::TestDisconnect(void)
{
    // TODO: implement this
}

//-----------------------------------------------------------------------------
//	Tests for private variables and methods
//-----------------------------------------------------------------------------

CPPUNIT_TEST_SUITE_REGISTRATION(mtsGlobalManagerTest);
