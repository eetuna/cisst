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
    TimeOffsetToZero(false),
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
    TimeOffsetToZero = false;
    CollectingPeriod = 0.0; // collect nothing at start-up

    ClearTaskMap();    
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
    ClearTaskMap();
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
    mtsTask * task = taskManager->GetTask(taskName);
    if (task == NULL) {
        CMN_LOG_CLASS(5) << "Unknown task: " << taskName << std::endl;
        return false;
    }

    // Prevent duplicate signal registration
    if (FindSignal(taskName, signalName)) {
        CMN_LOG_CLASS(5) << "Already registered signal: " << taskName 
            << ", " << signalName << std::endl;

        //return false;
        throw mtsCollector::mtsCollectorException(
            "Already collecting signal: " + taskName + ", " + signalName);
    }

    // Double check: check if the specified signal has been already added to the state table.
    if (task->GetStateVectorID(signalName) >= 0) {  // 0: Toc, 1: Tic, 2: Period, >3: user
        CMN_LOG_CLASS(5) << "Already registered signal: " << taskName
            << ", " << signalName << std::endl;

        //return false;
        throw mtsCollector::mtsCollectorException(
            "Already collecting signal: " + taskName + ", " + signalName);
    }

    // Add a signal    
    TaskMap::iterator itr = taskMap.find(taskName);
    if (itr == taskMap.end()) {  // If the task is new
        // Create a new instance of SignalMap
        SignalMap * signalMap = new SignalMap();
        CMN_ASSERT(signalMap);

        signalMap->insert(make_pair(signalName, task));
        taskMap.insert(make_pair(taskName, signalMap));
    } else {    // If the task has one or more signals being collected
        itr->second->insert(make_pair(signalName, task));
    }

    CMN_LOG_CLASS(5) << "Signal added: " << taskName << ", " << signalName << std::endl;

    return true;
}

bool mtsCollector::RemoveSignal(const std::string & taskName, const std::string & signalName)
{
    // If no signal data is being collected.
    if (taskMap.empty()) {
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
    TaskMap::iterator itr = taskMap.find(taskName);
    CMN_ASSERT(itr != taskMap.end());

    SignalMap * signalMap = itr->second;
    SignalMap::iterator _itr = signalMap->find(signalName);
    CMN_ASSERT(_itr != signalMap->end());
    signalMap->erase(_itr);

    // Clean-up
    if (signalMap->empty()) {
        delete signalMap;
        taskMap.erase(itr);
    }

    CMN_LOG_CLASS(5) << "Signal removed: " << taskName << ", " << signalName << std::endl;

    return true;
}

bool mtsCollector::FindSignal(const std::string & taskName, const std::string & signalName)
{
    if (taskMap.empty()) return false;	

    TaskMap::const_iterator itr = taskMap.find(taskName);
    if (itr == taskMap.end()) {
        return false;
    } else {
        SignalMap * signalMap = itr->second;
        SignalMap::const_iterator _itr = signalMap->find(signalName);
        if (_itr == signalMap->end()) {
            return false;
        } else {
            return true;
        }
    }
}

//-------------------------------------------------------
//	Collecting Data
//-------------------------------------------------------
void mtsCollector::SetTimeBase(const double deltaT, const bool offsetToZero)
{
    // Ensure that there is a task being collected.
    if (taskMap.empty() || CollectingPeriod == 0.0) {
        return;
    }

    CollectingPeriod = deltaT;
    TimeOffsetToZero = offsetToZero;

    //
    // TODO: CONTINUE TO IMPLEMENT HERE...    
    //
}

void mtsCollector::SetTimeBase(const int deltaStride, const bool offsetToZero)
{
    // Check if the number of task being collected is one.
    //if (!taskMap.count() != 1) {
    //    return -1;
    //}
}

//-------------------------------------------------------
//	Miscellaneous Functions
//-------------------------------------------------------
void mtsCollector::ClearTaskMap(void)
{
    if (!taskMap.empty()) {        
        TaskMap::iterator itr = taskMap.begin();
        SignalMap::iterator _itr;
        for (; itr != taskMap.end(); ++itr) {
            itr->second->clear();
            delete itr->second;
        }

        taskMap.clear();
    }
}