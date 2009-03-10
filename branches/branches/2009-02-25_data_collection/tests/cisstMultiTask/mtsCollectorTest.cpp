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

	mtsCollector * b = new mtsCollector("collector-2", 10 * cmn_ms);
    CPPUNIT_ASSERT_EQUAL((unsigned int) 2, mtsCollector::GetCollectorCount());
	
	delete b;
	CPPUNIT_ASSERT_EQUAL((unsigned int) 1, mtsCollector::GetCollectorCount());
}

void mtsCollectorTest::TestAddSignal(void)
{	
	const std::string taskName = "Task_TestAddSignal";
	const std::string signalName = "Data_TestAddSignal";	

	mtsCollector collector("collector", 10 * cmn_ms);
    // The following object has to be created in a dynamic way so that the object can
    // be deleted outside this unit test. (It'll be deleted at
    // mtsCollectorTest::TestAddSignalCleanUp().)
	mtsCollectorTestTask * TaskA = new mtsCollectorTestTask(taskName, 10 * cmn_ms );

	mtsTaskManager * taskManager = mtsTaskManager::GetInstance();
	CPPUNIT_ASSERT(taskManager);

	// In case of a task that are not under the control of the manager
    CPPUNIT_ASSERT(!collector.AddSignal(taskName, "", ""));

	// Put it under the control of the manager.	
	CPPUNIT_ASSERT(taskManager->AddTask(TaskA));

	CPPUNIT_ASSERT(collector.AddSignal(taskName, signalName, ""));

	// Prevent duplicate signal registration
    // The following line should throw a mtsCollectorException.
	CPPUNIT_ASSERT(!collector.AddSignal(taskName, signalName, ""));

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

/* This class is not inteded to test mtsCollector::AddSignalCleanup() method. 
   That is, there is no such method in mtsCollector.
   This is for cleaning up a temporary task generated and registered at
   mtsCollectorTest::AddSignal() where mtsTaskManager::RemoveTask() cannot be called
   because an exception is throwed. */
void mtsCollectorTest::TestAddSignalCleanUp(void)
{	
	const std::string taskName = "Task_TestAddSignal";
	const std::string signalName = "Data_TestAddSignal";	

	mtsCollector collector("collector", 10 * cmn_ms);

	mtsTaskManager * taskManager = mtsTaskManager::GetInstance();
	CPPUNIT_ASSERT(taskManager);

    mtsCollectorTestTask * task = 
        dynamic_cast<mtsCollectorTestTask*>(taskManager->GetTask(taskName));
    CPPUNIT_ASSERT(task);

    // Remove a task not to cause potential side-effect during the unit-test process
    CPPUNIT_ASSERT(taskManager->RemoveTask(task));

    // Don't forget
    delete task;
}

void mtsCollectorTest::TestRemoveSignal(void)
{
	const std::string taskName = "Task_TestRemoveSignal";
	const std::string signalName = "Data_TestRemoveSignal";	

	mtsCollector collector("collector", 10 * cmn_ms);
	mtsCollectorTestTask TaskA(taskName, 10 * cmn_ms );

	// 1. Try removing a signal from the empty signal list
    CPPUNIT_ASSERT(!collector.RemoveSignal(taskName, signalName));

	// Add a signal
	mtsTaskManager * taskManager = mtsTaskManager::GetInstance();
	CPPUNIT_ASSERT(taskManager);
	CPPUNIT_ASSERT(taskManager->AddTask(&TaskA));	
	CPPUNIT_ASSERT(collector.AddSignal(taskName, signalName, ""));

	// 2. Try removing a signal with incorrect task name
	CPPUNIT_ASSERT(!collector.RemoveSignal(taskName + "1234", signalName));

	// 3. Try removing a signal with incorrect signal name
	CPPUNIT_ASSERT(!collector.RemoveSignal(taskName, signalName + "1234"));

	// 4. Try removing a signal in a correct way
	CPPUNIT_ASSERT(collector.RemoveSignal(taskName, signalName));

    // Remove a task not to cause potential side-effect during the unit-test process
    CPPUNIT_ASSERT(taskManager->RemoveTask(&TaskA));	
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
	CPPUNIT_ASSERT(collector.AddSignal(taskName, signalName, ""));
	
	// return false if finding a nonregistered task or signal
	CPPUNIT_ASSERT(!collector.FindSignal(taskName, "nonregistered_signal"));
	CPPUNIT_ASSERT(!collector.FindSignal("nonregistered_task", signalName));
	
	// return true if finding a correct one
	CPPUNIT_ASSERT(collector.FindSignal(taskName, signalName));

    // Remove a task not to cause potential side-effect during the unit-test process
    CPPUNIT_ASSERT(taskManager->RemoveTask(&TaskA));
	
}

void mtsCollectorTest::TestCleanup(void)
{
	mtsCollector collector("collector", 10 * cmn_ms);	

    mtsTaskManager * taskManager = mtsTaskManager::GetInstance();
	CPPUNIT_ASSERT(taskManager);

    // Put tasks for this test under the control of the taskManager so that
    // mtsCollector::AddSignal() works correctly.
    mtsCollectorTestTask task1("task1", 10 * cmn_ms );
    mtsCollectorTestTask task2("task2", 10 * cmn_ms );
    mtsCollectorTestTask task3("task3", 10 * cmn_ms );
    CPPUNIT_ASSERT(taskManager->AddTask(&task1));
    CPPUNIT_ASSERT(taskManager->AddTask(&task2));
    CPPUNIT_ASSERT(taskManager->AddTask(&task3));
    {
        CPPUNIT_ASSERT(collector.AddSignal("task1", "signal1-1", ""));
        CPPUNIT_ASSERT(collector.AddSignal("task1", "signal1-2", ""));
        CPPUNIT_ASSERT(collector.AddSignal("task2", "signal2", ""));
        CPPUNIT_ASSERT(collector.AddSignal("task3", "signal3", ""));

        collector.ClearTaskMap();
        CPPUNIT_ASSERT(collector.taskMap.size() == 0);
    }
    // Remove a task not to cause potential side-effect during the unit-test process
    CPPUNIT_ASSERT(taskManager->RemoveTask(&task1));
    CPPUNIT_ASSERT(taskManager->RemoveTask(&task2));
    CPPUNIT_ASSERT(taskManager->RemoveTask(&task3));
}

void mtsCollectorTest::TestSetTimeBaseDouble(void)
{
    const double defaultPeriod = (double) 10 * cmn_ms;
    const double newPeriod = (double) 20 * cmn_ms;

    mtsCollector collector("collector", defaultPeriod);       

    mtsTaskManager * taskManager = mtsTaskManager::GetInstance();
    CPPUNIT_ASSERT(taskManager);

    mtsCollectorTestTask task1("task1", defaultPeriod);
    CPPUNIT_ASSERT(taskManager->AddTask(&task1));
    CPPUNIT_ASSERT(collector.AddSignal("task1", "signal1", ""));
    {
        collector.SetTimeBase(newPeriod, true);
        CPPUNIT_ASSERT_EQUAL(newPeriod, collector.CollectingPeriod);
        CPPUNIT_ASSERT(true == collector.TimeOffsetToZero);

        collector.SetTimeBase(newPeriod, false);
        CPPUNIT_ASSERT(false == collector.TimeOffsetToZero);
    }
    // Remove a task not to cause potential side-effect during the unit-test process
    CPPUNIT_ASSERT(taskManager->RemoveTask(&task1));
}

void mtsCollectorTest::TestSetTimeBaseInt(void)
{
}

//-----------------------------------------------------------------------------
//	Tests for private variables and methods
//
//	Be sure that _OPEN_PRIVATE_FOR_UNIT_TEST_ macro is enabled at mtsCollector.h
//-----------------------------------------------------------------------------
void mtsCollectorTest::TestInit(void)
{
	mtsCollector collector("collector", 10 * cmn_ms);
	
    mtsTaskManager * taskManager = mtsTaskManager::GetInstance();
	CPPUNIT_ASSERT(taskManager);

    // Put tasks for this test under the control of the taskManager so that
    // mtsCollector::AddSignal() works correctly.
    mtsCollectorTestTask task1("taskA", 10 * cmn_ms );
    mtsCollectorTestTask task2("taskB", 10 * cmn_ms );
    mtsCollectorTestTask task3("taskC", 10 * cmn_ms );
    CPPUNIT_ASSERT(taskManager->AddTask(&task1));
    CPPUNIT_ASSERT(taskManager->AddTask(&task2));
    CPPUNIT_ASSERT(taskManager->AddTask(&task3));
    {
        CPPUNIT_ASSERT(collector.AddSignal("taskA", "signal1-1", ""));
        CPPUNIT_ASSERT(collector.AddSignal("taskA", "signal1-2", ""));
        CPPUNIT_ASSERT(collector.AddSignal("taskB", "signal2", ""));
        CPPUNIT_ASSERT(collector.AddSignal("taskC", "signal3", ""));

        collector.Init();
		
        CPPUNIT_ASSERT(collector.taskMap.size() == 0);
		CPPUNIT_ASSERT(false == collector.TimeOffsetToZero);
        CPPUNIT_ASSERT_EQUAL(0.0, collector.CollectingPeriod);
    }
    // Remove a task not to cause potential side-effect during the unit-test process
    CPPUNIT_ASSERT(taskManager->RemoveTask(&task1));
    CPPUNIT_ASSERT(taskManager->RemoveTask(&task2));
    CPPUNIT_ASSERT(taskManager->RemoveTask(&task3));
}

void mtsCollectorTest::TestClearTaskMap(void)
{
	mtsCollector collector("collector", 10 * cmn_ms);
	
    mtsTaskManager * taskManager = mtsTaskManager::GetInstance();
	CPPUNIT_ASSERT(taskManager);

    // Put tasks for this test under the control of the taskManager so that
    // mtsCollector::AddSignal() works correctly.
    mtsCollectorTestTask task1("task1", 10 * cmn_ms );
    mtsCollectorTestTask task2("task2", 10 * cmn_ms );
    mtsCollectorTestTask task3("task3", 10 * cmn_ms );

    CPPUNIT_ASSERT(taskManager->AddTask(&task1));
    CPPUNIT_ASSERT(taskManager->AddTask(&task2));
    CPPUNIT_ASSERT(taskManager->AddTask(&task3));
    {
        CPPUNIT_ASSERT(collector.AddSignal("task1", "signal1-1", ""));
        CPPUNIT_ASSERT(collector.AddSignal("task1", "signal1-2", ""));
        CPPUNIT_ASSERT(collector.AddSignal("task2", "signal2", ""));
        CPPUNIT_ASSERT(collector.AddSignal("task3", "signal3", ""));

        collector.ClearTaskMap();
        CPPUNIT_ASSERT(collector.taskMap.size() == 0);
    }
    // Remove a task not to cause potential side-effect during the unit-test process
    CPPUNIT_ASSERT(taskManager->RemoveTask(&task1));
    CPPUNIT_ASSERT(taskManager->RemoveTask(&task2));
    CPPUNIT_ASSERT(taskManager->RemoveTask(&task3));
}

CPPUNIT_TEST_SUITE_REGISTRATION(mtsCollectorTest);
