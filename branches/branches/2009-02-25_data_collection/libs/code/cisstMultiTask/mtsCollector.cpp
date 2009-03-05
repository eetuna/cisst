/*
  $Id: mtsCollector.cpp 2009-03-02 mjung5

  Author(s):  Min Yang Jung
  Created on: 2009-02-25

  (C) Copyright 2008-2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#include <cisstMultiTask/mtsCollector.h>

CMN_IMPLEMENT_SERVICES(mtsCollector)

unsigned int mtsCollector::CollectorCount = 0;
mtsTaskManager * mtsCollector::taskManager = NULL;

//-------------------------------------------------------
//	Constructor, Destructor, and Initializer
//-------------------------------------------------------
mtsCollector::mtsCollector(const std::string & collectorName, double period) : 	
	mtsTaskPeriodic(collectorName, period, false)
{
	++CollectorCount;

	if (taskManager == NULL) {
		taskManager = mtsTaskManager::GetInstance();
	}
}

mtsCollector::~mtsCollector()
{
	--CollectorCount;

	CMN_LOG_CLASS(5) << "Collector " << GetName() << " ends." << std::endl;
}

void mtsCollector::Init()
{	
	SignalCollection.clear();		
}

//-------------------------------------------------------
//	Thread management functions (called internally)
//-------------------------------------------------------
void mtsCollector::Startup(void)
{
	// initialization
	Init();
}

void mtsCollector::Run(void)
{
	// periodic execution
	// for test 
	static int i = 0;
	CMN_LOG_CLASS(5) << "----- [ " << GetName() << " ] " << ++i << std::endl;
}

void mtsCollector::Cleanup(void)
{
	// clean up
}

//-------------------------------------------------------
//	Signal Management
//-------------------------------------------------------
bool mtsCollector::AddSignal(const std::string & taskName, 
							 const std::string & signalName, 
							 const std::string & format)
{	
	CMN_ASSERT(taskManager);

	// Ensure that the specified task is under the control of the task manager.
	//
	//	TODO: Should the task be an instance of mtsTaskPeriodic???
	//
	mtsTask * task = taskManager->GetTask(taskName);
	if (task == NULL) {
		CMN_LOG_CLASS(5) << "Unknown task: " << taskName << std::endl;
		return false;
	}

	// Prevent duplicate signal registration
	if (FindSignal(taskName, signalName)) {
		CMN_LOG_CLASS(5) << "Already registered signal: " << taskName 
			<< ", " << signalName << std::endl;
		return false;
	}

	// Check if the specified signal has been added to the state table.
	//if (task->FindStateVectorDataName(signalName)) {
	//	//
	//	// TODO: Throw an exception if already collecting
	//	// TODO: test unit: how to test throwing an exception? (add a test unit)
	//	//
	//	CMN_LOG_CLASS(5) << "Already registered signal: " << signalName << std::endl;
	//	return false;
	//}
	
	// Add a signal
	SignalCollection.insert(make_pair(taskName, signalName));
	CMN_LOG_CLASS(5) << "Signal added: " << taskName << ", " << signalName << std::endl;

	// Adaptively control a period of the collector task in order to be sure that 
	// the collector task has always the minimum period among all tasks added.

	return true;
}

bool mtsCollector::RemoveSignal(const std::string & taskName, const std::string & signalName)
{
	// If no signal data is being collected.
	if (SignalCollection.empty()) {
		CMN_LOG_CLASS(5) << "No signal data is being collected." << std::endl;
		return false;
	}

	// If finding a nonregistered task or signal
	if (!FindSignal(taskName, signalName)) {
		CMN_LOG_CLASS(5) << "Unknown task and/or signal: " << taskName 
			<< ", " << signalName << std::endl;
		return false;
	}

	// Remove a signal from the list
	SignalMap::const_iterator itr = SignalCollection.find(taskName);
	CMN_ASSERT(itr != SignalCollection.end());

	// The specified task is found. Now try to find a signal with the given signal name
	SignalMap::const_iterator itrOnePastTheLast;
	itrOnePastTheLast = SignalCollection.upper_bound(taskName);
	for (; itr != itrOnePastTheLast; ++itr) {
		if (itr->second == signalName) {
			SignalCollection.erase(itr);
			CMN_LOG_CLASS(5) << "Signal removed: " << taskName 
				<< ", " << signalName << std::endl;

			return true;
		}
	}
	
	// This should not occur.
	CMN_ASSERT(false);

	return false;
}

bool mtsCollector::FindSignal(const std::string & taskName, const std::string & signalName)
{
	if (SignalCollection.empty()) return false;

	// First, try to find a signal with the given task name
	SignalMap::const_iterator itr = SignalCollection.find(taskName);
	if (itr == SignalCollection.end()) {
		return false;	// no task found
	}

	// The specified task is found. Now try to find a signal with the given signal name
	SignalMap::const_iterator itrOnePastTheLast;
	itrOnePastTheLast = SignalCollection.upper_bound(taskName);
	for (; itr != itrOnePastTheLast; ++itr) {
		if (itr->second == signalName) {
			return true;	// found the specified task
		}
	}

	// no signal found
	return false;
}

//
//	TODO: Write unit-tests for this method!!!
//
void mtsCollector::AdjustPeriod(const double newPeriod)
{
	CMN_ASSERT(taskManager);

	double minPeriod = GetAveragePeriod();
	double period = 0.0;

	mtsTaskPeriodic * taskPeriodic = NULL;
	SignalMap::const_iterator itrTask = SignalCollection.begin();

	//SignalMap::const_iterator itrSignal, itrLast;
	for (; itrTask != SignalCollection.end(); ++itrTask ) {
		//
		//	TODO: 1 signal per 1 task? or N signal per 1 task??
		//
		//itrLast = SignalCollection.find(itrTask->first);
		//for (; itrSignal != itrLast; ++itrSignal ) {
		//}
		taskPeriodic = dynamic_cast<mtsTaskPeriodic *>(taskManager->GetTask(itrTask->first));
		if (taskPeriodic) {
			period = taskPeriodic->GetPeriodicity();
		} else {
			period = taskPeriodic->GetAveragePeriod();
		}		

		if (minPeriod > period) {
			minPeriod = period;
		}
	}

	minPeriod *= 0.5;
	
	//
	//	TODO: How to control the period of the collector task automatically?
	//

	CMN_LOG_CLASS(5) << "Signal period updated: " << minPeriod << std::endl;
}