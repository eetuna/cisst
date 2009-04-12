/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsTaskManagerTest.cpp 2009-03-05 mjung5 $
  
  Author(s):  Min Yang Jung
  Created on: 2009-03-05
  
  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstCommon/cmnUnits.h>
#include <cisstMultiTask/mtsTask.h>
#include <cisstMultiTask/mtsTaskManager.h>

#include "mtsTaskManagerTest.h"

#include <string>

CMN_IMPLEMENT_SERVICES(mtsTaskManagerTestTask);

//-----------------------------------------------------------------------------
mtsTaskManagerTestTask::mtsTaskManagerTestTask(const std::string & collectorName, 
										   double period) :
	mtsTaskPeriodic(collectorName, period, false, 5000)
{
}

//-----------------------------------------------------------------------------
//	Tests for public variables and methods
//-----------------------------------------------------------------------------
void mtsTaskManagerTest::TestConstructor(void)
{
    mtsTaskManager * taskManager = mtsTaskManager::GetInstance();
    CPPUNIT_ASSERT(taskManager);

    //CPPUNIT_ASSERT(taskManager->IceCommunicatorPtr == NULL);
    //CPPUNIT_ASSERT(taskManager->TaskManagerTypeMember == mtsTaskManager::TASK_MANAGER_LOCAL);

    //taskManager->SetTaskManagerMode(mtsTaskManager::TASK_MANAGER_SERVER);
}

void mtsTaskManagerTest::TestAddTask(void)
{
	mtsTaskManagerTestTask task1("task1", 10 * cmn_ms), task2("task2", 10 * cmn_ms);
	//mtsTaskManager TaskManager;
	//
	//CPPUNIT_ASSERT(0 == TaskManager.TaskMap.GetCount());
	//	
	//CPPUNIT_ASSERT(TaskManager.AddTask(&task1));
	//CPPUNIT_ASSERT(1 == TaskManager.TaskMap.GetCount());
	//
	//CPPUNIT_ASSERT(!TaskManager.AddTask(&task1));
	//CPPUNIT_ASSERT(TaskManager.AddTask(&task2));
	//CPPUNIT_ASSERT(2 == TaskManager.TaskMap.GetCount());
	//
	//CPPUNIT_ASSERT(!TaskManager.AddTask(&task1));
	//CPPUNIT_ASSERT(!TaskManager.AddTask(&task2));
	//CPPUNIT_ASSERT(2 == TaskManager.TaskMap.GetCount());
}

CPPUNIT_TEST_SUITE_REGISTRATION(mtsTaskManagerTest);
