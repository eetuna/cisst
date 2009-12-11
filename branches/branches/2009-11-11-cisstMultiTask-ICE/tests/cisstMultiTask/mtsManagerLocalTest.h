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

#include "mtsManagerTestClasses.h"

class mtsManagerLocalInterface;

class mtsManagerLocalTest: public CppUnit::TestFixture
{
private:
    mtsManagerLocalInterface *localManager1, *localManager2;

    CPPUNIT_TEST_SUITE(mtsManagerLocalTest);
    {
        CPPUNIT_TEST(TestConstructor);
        CPPUNIT_TEST(TestCleanup);
        CPPUNIT_TEST(TestGetInstance);
        CPPUNIT_TEST(TestAddComponent);
	}
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp(void);
    void tearDown(void);

    void TestConstructor(void);
    void TestCleanup(void);
    void TestGetInstance(void);
    void TestAddComponent(void);
};
