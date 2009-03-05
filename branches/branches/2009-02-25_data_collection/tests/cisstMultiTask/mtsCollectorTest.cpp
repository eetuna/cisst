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

#include <cisstCommon/cmnUnits.h>
#include <cisstMultiTask/mtsCollector.h>
#include <cisstMultiTask/mtsTaskManager.h>

#include "mtsCollectorTest.h"

#include <string.h>

CMN_IMPLEMENT_SERVICES(mtsCollectorTestTask);

//-----------------------------------------------------------------------------
mtsCollectorTestTask::mtsCollectorTestTask(const std::string & collectorName, 
										   double period) :
	mtsTaskPeriodic(collectorName, period, false, 5000)
{
}

void mtsCollectorTestTask::AddDataToStateTable(const std::string & dataName)
{ 
	TestData.AddToStateTable(StateTable, dataName); 
}

//-----------------------------------------------------------------------------
//	Tests for public variables and methods
//-----------------------------------------------------------------------------
void mtsCollectorTest::TestGetCollectorCount(void)
{
	mtsCollector a("collector-1", 10 * cmn_ms);
	CPPUNIT_ASSERT_EQUAL((unsigned int) 1, mtsCollector::GetCollectorCount());

	mtsCollector b("collector-2", 10 * cmn_ms);
    CPPUNIT_ASSERT_EQUAL((unsigned int) 2, mtsCollector::GetCollectorCount());    
}

void mtsCollectorTest::TestAddSignal(void)
{	
	const std::string taskName = "Task_TestAddSignal";
	const std::string signalName = "Data_TestAddSignal";	

	mtsCollector collector("collector", 10 * cmn_ms);
	mtsCollectorTestTask TaskA(taskName, 10 * cmn_ms );

	mtsTaskManager * taskManager = mtsTaskManager::GetInstance();
	CPPUNIT_ASSERT(taskManager);

	// In case of a task that are not under the control of the manager	
	CPPUNIT_ASSERT(!collector.AddSignal(taskName, "", ""));

	// Put it under the control of the manager.	
	CPPUNIT_ASSERT(taskManager->AddTask(&TaskA));

	//TaskA.AddDataToStateTable(signalName);
	CPPUNIT_ASSERT(collector.AddSignal(taskName, signalName, ""));
	
	// Prevent duplicate signal registration
	CPPUNIT_ASSERT(!collector.AddSignal(taskName, signalName, ""));

	// Remove a task not to cause potential side-effect when using mtsTaskManager
	//taskManager->
		
	// Throw an exception if already collecting
	//
	// TODO: IMPLEMENT ME~~~~!!!!!
	//

	/*
	// 1. Test if a task of which name is taskName actually exists.
	// 1) In case of non-registered task: should return false
	const std::string nonRegisteredTaskName = "@!Non_Registered_Task!@";
	const std::string registeredTaskName = "mtsCollectorTestTask";
	
	const cmnClassServicesBase * classService = NULL;
	classService = cmnClassRegister::FindClassServices(nonRegisteredTaskName);
    CPPUNIT_ASSERT(!classService);		
	CPPUNIT_ASSERT_EQUAL(false, collector.AddSignal(nonRegisteredTaskName, "", ""));

	// 2) In case of registered task: return true or false depending on a signal name
	classService = cmnClassRegister::FindClassServices(registeredTaskName);
    CPPUNIT_ASSERT(classService);
	{
		// 2. Test if a specified signal exists s.t. a name is signalName and is bound with
		// the task of which name is taskName.
		mtsCollectorTestTask collectorTestTaskA("TaskA", 10 * cmn_ms);		
		mtsCollectorTestTask collectorTestTaskB("TaskB", 10 * cmn_ms);

		// Only add data of TaskA to State Table
		collectorTestTaskA.AddDataToStateTable();

		CPPUNIT_ASSERT_EQUAL(true, collector.AddSignal(registeredTaskName, "TaskA", ""));
		CPPUNIT_ASSERT_EQUAL(false, collector.AddSignal(registeredTaskName, "TaskB", ""));
	}
	*/

	// 3. format option test (to be implemented)
	// NOP
}

void mtsCollectorTest::TestRemoveSignal(void)
{
	const std::string taskName = "Task_TestRemoveSignal";
	const std::string signalName = "Data_TestRemoveSignal";	

	mtsCollector collector("collector", 10 * cmn_ms);
	mtsCollectorTestTask TaskA(taskName, 10 * cmn_ms );

	// 1. Try removing a signal from the empty signal list
	//
	//	TODO: need a method such as mtsTaskManager::RemoveTask() to test the following code.
	//
	//CPPUNIT_ASSERT(!collector.RemoveSignal(taskName, signalName));

	// Add a signal
	mtsTaskManager * taskManager = mtsTaskManager::GetInstance();
	CPPUNIT_ASSERT(taskManager);
	CPPUNIT_ASSERT(taskManager->AddTask(&TaskA));
	//TaskA.AddDataToStateTable(signalName);
	CPPUNIT_ASSERT(collector.AddSignal(taskName, signalName, ""));

	// 2. Try removing a signal with incorrect task name
	CPPUNIT_ASSERT(!collector.RemoveSignal(taskName + "1234", signalName));

	// 3. Try removing a signal with incorrect signal name
	CPPUNIT_ASSERT(!collector.RemoveSignal(taskName, signalName + "1234"));

	// 4. Try removing a signal in a correct way
	CPPUNIT_ASSERT(collector.RemoveSignal(taskName, signalName));
}

void mtsCollectorTest::TestFindSignal(void)
{
	const std::string taskName = "Task_TestFindSignal";
	const std::string signalName = "Data_TestFindSignal";
	
	mtsCollector collector("collector", 10 * cmn_ms);
	mtsCollectorTestTask TaskA(taskName, 10 * cmn_ms );
	
	// return false if the signal list is empty
	CPPUNIT_ASSERT(!collector.FindSignal(taskName, signalName));
	
	// Add a signal
	mtsTaskManager * taskManager = mtsTaskManager::GetInstance();
	CPPUNIT_ASSERT(taskManager);
	CPPUNIT_ASSERT(taskManager->AddTask(&TaskA));
	//TaskA.AddDataToStateTable(signalName);
	CPPUNIT_ASSERT(collector.AddSignal(taskName, signalName, ""));
	
	// return false if finding a nonregistered task or signal
	CPPUNIT_ASSERT(!collector.FindSignal(taskName, "nonregistered_signal"));
	CPPUNIT_ASSERT(!collector.FindSignal("nonregistered_task", signalName));
	
	// return true if finding a correct one
	CPPUNIT_ASSERT(collector.FindSignal(taskName, signalName));
	
}

void mtsCollectorTest::TestGetSignalCount(void)
{
	CPPUNIT_ASSERT_EQUAL((unsigned int) 0, mtsCollector::GetCollectorCount());
	
	mtsCollector collector1("collector1", 10 * cmn_ms);
	CPPUNIT_ASSERT_EQUAL((unsigned int) 1, mtsCollector::GetCollectorCount());
	
	mtsCollector collector2("collector2", 10 * cmn_ms);
	CPPUNIT_ASSERT_EQUAL((unsigned int) 2, mtsCollector::GetCollectorCount());
}

//-----------------------------------------------------------------------------
//	Tests for private variables and methods
//
//	Be sure that _OPEN_PRIVATE_FOR_UNIT_TEST_ macro is enabled at mtsCollector.h
//-----------------------------------------------------------------------------
void mtsCollectorTest::TestInit(void)
{
	mtsCollector collector("collector", 10 * cmn_ms);
	
	CPPUNIT_ASSERT(collector.SignalCollection.empty());
	collector.SignalCollection.insert(make_pair(std::string("a"), std::string("1")));
	CPPUNIT_ASSERT(1 == collector.SignalCollection.size());
	
	collector.SignalCollection.clear();
	CPPUNIT_ASSERT(collector.SignalCollection.empty());
}

CPPUNIT_TEST_SUITE_REGISTRATION(mtsCollectorTest);
