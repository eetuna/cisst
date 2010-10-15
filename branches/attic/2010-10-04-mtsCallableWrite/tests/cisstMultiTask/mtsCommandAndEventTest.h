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

#include <cppunit/TestCase.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cisstMultiTask/mtsConfig.h>

/*
  Test that commands and events are executed for different types of
  components using a reasonable execution delay and blocking or not
  blocking commands.  These tests rely on the components defined in
  mtsTestComponents.h

  Name of tests are Test{Local,Remote}<_clientType><_serverType>{,Blocking}.
*/
class mtsCommandAndEventTest: public CppUnit::TestFixture
{
private:
    CPPUNIT_TEST_SUITE(mtsCommandAndEventTest);
    {
        CPPUNIT_TEST(TestLocalDeviceDevice);
        CPPUNIT_TEST(TestLocalPeriodicPeriodic);
        CPPUNIT_TEST(TestLocalContinuousContinuous);
        CPPUNIT_TEST(TestLocalFromCallbackFromCallback);
        CPPUNIT_TEST(TestLocalFromSignalFromSignal);

        CPPUNIT_TEST(TestLocalPeriodicPeriodicBlocking);
        CPPUNIT_TEST(TestLocalContinuousContinuousBlocking);
        CPPUNIT_TEST(TestLocalFromCallbackFromCallbackBlocking);
        CPPUNIT_TEST(TestLocalFromSignalFromSignalBlocking);

#if CISST_MTS_HAS_ICE
        //        CPPUNIT_TEST(TestRemoteDeviceDevice);
#endif
    }
    CPPUNIT_TEST_SUITE_END();

public:
    mtsCommandAndEventTest();

    void setUp(void);
    void tearDown(void);

    template <class _clientType, class _serverType>
    void TestExecution(_clientType * client, _serverType * server,
                       double clientExecutionDelay, double serverExecutionDelay,
                       double blockingDelay = 0.0);
    void TestLocalDeviceDevice(void);
    void TestLocalPeriodicPeriodic(void);
    void TestLocalContinuousContinuous(void);
    void TestLocalFromCallbackFromCallback(void);
    void TestLocalFromSignalFromSignal(void);

    void TestLocalPeriodicPeriodicBlocking(void);
    void TestLocalContinuousContinuousBlocking(void);
    void TestLocalFromCallbackFromCallbackBlocking(void);
    void TestLocalFromSignalFromSignalBlocking(void);

#if CISST_MTS_HAS_ICE
    void TestRemoteDeviceDevice(void);
#endif
};
