/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerLocalTest.h 2009-03-05 mjung5 $
  
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

#include <cppunit/TestCase.h>
#include <cppunit/extensions/HelperMacros.h>

class mtsManagerLocalInterface;

class mtsManagerLocalTest: public CppUnit::TestFixture
{
private:
    mtsManagerLocalInterface *localManager1, *localManager2;

    CPPUNIT_TEST_SUITE(mtsManagerLocalTest);
    {
        CPPUNIT_TEST(TestConstructor);
        CPPUNIT_TEST(TestGetInstance);
        CPPUNIT_TEST(TestAddComponent);
        CPPUNIT_TEST(TestRemoveComponent);
        CPPUNIT_TEST(TestGetComponent);
        CPPUNIT_TEST(TestConnect);
        CPPUNIT_TEST(TestDisconnect);
        //CPPUNIT_TEST(TestCreateAll);
        //CPPUNIT_TEST(TestStartAll);
        //CPPUNIT_TEST(TestKillAll);
        CPPUNIT_TEST(TestCleanup);
        CPPUNIT_TEST(TestGetNamesOfComponents);
        //CPPUNIT_TEST(TestGetNamesOfTasks);
        //CPPUNIT_TEST(TestGetNamesOfDevices);
        //CPPUNIT_TEST(TestCreateRequiredInterfaceProxy);
        //CPPUNIT_TEST(TestCreateProvidedInterfaceProxy);
        //CPPUNIT_TEST(TestRemoveRequiredInterfaceProxy);
        //CPPUNIT_TEST(TestRemoveProvidedInterfaceProxy);
        CPPUNIT_TEST(TestGetProcessName);
        
	}
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp(void);
    void tearDown(void);

    void TestConstructor(void);
    void TestGetInstance(void);
    void TestAddComponent(void);
    void TestRemoveComponent(void);
    void TestGetComponent(void);
    void TestConnect(void);
    void TestDisconnect(void);
    void TestCreateAll(void);
    void TestStartAll(void);
    void TestKillAll(void);
    void TestCleanup(void);
    void TestGetNamesOfComponents(void);
    void TestGetProcessName(void);
};
