/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsGlobalManagerTest.h 2009-03-05 mjung5 $
  
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

#include <cisstCommon/cmnGenericObjectProxy.h>

class mtsGlobalManagerTest: public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(mtsGlobalManagerTest);
	{		
		CPPUNIT_TEST(TestAddComponent);
        CPPUNIT_TEST(TestFindComponent);
        CPPUNIT_TEST(TestRemoveComponent);

        CPPUNIT_TEST(TestAddInterface);
        CPPUNIT_TEST(TestFindInterface);
        CPPUNIT_TEST(TestRemoveInterface);

        CPPUNIT_TEST(TestConnect);
        CPPUNIT_TEST(TestDisconnect);
	}
    CPPUNIT_TEST_SUITE_END();
	
public:
    //void setUp(void) {
    //}
    //
    //void tearDown(void) {
    //}
    
	void TestAddComponent(void);
    void TestFindComponent(void);
    void TestRemoveComponent(void);

    void TestAddInterface(void);
    void TestFindInterface(void);
    void TestRemoveInterface(void);

    void TestConnect(void);
    void TestDisconnect(void);
};
