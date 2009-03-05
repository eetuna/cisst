/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCollectorTest.h 2009-03-02 mjung5 $
  
  Author(s):  Min Yang Jung
  Created on: 2009-03-02
  
  (C) Copyright 2008-2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cppunit/TestCase.h>
#include <cppunit/extensions/HelperMacros.h>

//#include <cisstCommon/cmnClassServices.h>
//#include <cisstCommon/cmnClassRegister.h>
#include <cisstMultiTask/mtsCollector.h>
#include <cisstMultiTask/mtsStateData.h>

#include <string>

// To be used for TestAddSignal() ---------------------------------------------
class mtsCollectorTestTask : public mtsTaskPeriodic {
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

protected:    
	mtsStateData<cmnDouble> TestData;

public:
	mtsCollectorTestTask(const std::string & collectorName, double period);
	virtual ~mtsCollectorTestTask() {}

	// implementation of four methods that are pure virtual in mtsTask
    void Configure(const std::string) {}
	void Startup(void)	{}
	void Run(void)		{}
    void Cleanup(void)	{}

	void AddDataToStateTable(const std::string & dataName);
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsCollectorTestTask);

// Tester class ---------------------------------------------------------------
class mtsCollectorTest: public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(mtsCollectorTest);
	{		
		// public variables and methods
	    CPPUNIT_TEST(TestGetCollectorCount);
		CPPUNIT_TEST(TestGetSignalCount);

		CPPUNIT_TEST(TestAddSignal);		
		CPPUNIT_TEST(TestRemoveSignal);
		CPPUNIT_TEST(TestFindSignal);
		
		// private variables and methods
		CPPUNIT_TEST(TestInit);

	}
    CPPUNIT_TEST_SUITE_END();
	
private:
	//mtsCollector * Collector;
    
public:
    void setUp(void) {
		//Collector = new mtsCollector("collector", 10 * cmn_ms);
    }
    
    void tearDown(void) {
		//delete Collector;
    }
    
	// public variables and methods
    void TestGetCollectorCount(void);
	void TestGetSignalCount(void);

	void TestAddSignal(void);
	void TestRemoveSignal(void);	
	void TestFindSignal(void);
	
	// private variables and methods
	void TestInit(void);
};
