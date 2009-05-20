/*
  $Id: mtsCollectorBase.cpp 2009-03-02 mjung5

  Author(s):  Min Yang Jung
  Created on: 2009-03-20

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#include <cisstMultiTask/mtsCollectorBase.h>

CMN_IMPLEMENT_SERVICES(mtsCollectorBase)

unsigned int mtsCollectorBase::CollectorCount;
mtsTaskManager * mtsCollectorBase::taskManager;

//-------------------------------------------------------
//	Constructor, Destructor, and Initializer
//-------------------------------------------------------
mtsCollectorBase::mtsCollectorBase(const std::string & collectorName, 
                                   const CollectorLogFormat logFormat) : 
    TimeOffsetToZero(false),    
    IsRunnable(false),
    LogFormat(logFormat),
    mtsTaskContinuous(collectorName)
{
    ++CollectorCount;

    if (taskManager == NULL) {
        taskManager = mtsTaskManager::GetInstance();
    }

    Init();
}

mtsCollectorBase::~mtsCollectorBase()
{
    --CollectorCount;

    CMN_LOG_CLASS(5) << "Collector " << GetName() << " ends." << std::endl;
}

void mtsCollectorBase::Init()
{
    TimeOffsetToZero = false;

    Status = COLLECTOR_STOP;
    DelayedStart = 0.0;
    DelayedStop = 0.0;
    //SamplingInterval = Period;
    //SamplingStride = 0;

    ClearTaskMap();
}

//-------------------------------------------------------
//	Thread management functions (called internally)
//-------------------------------------------------------
void mtsCollectorBase::Run()
{
    //if (!IsAnySignalRegistered()) return;
    if (Status == COLLECTOR_STOP) return;

    // Check for the state transition
    switch (Status) {
        case COLLECTOR_WAIT_START:
            if (Stopwatch.IsRunning()) {
                if (Stopwatch.GetElapsedTime() < DelayedStart) {
                    return;
                } else {
                    // Start collecting
                    DelayedStart = 0.0;
                    Status = COLLECTOR_COLLECTING;
                    IsRunnable = true;
                    Stopwatch.Stop();

                    // Call Collect() method to activate the data collection feature 
                    // of all registered tasks. Normally, Collect() is called by
                    // an event generated from another task of which data is being
                    // collected.
                    Collect();
                }
            } else {
                return;
            }
            break;

        case COLLECTOR_WAIT_STOP:
            if (Stopwatch.IsRunning()) {
                if (Stopwatch.GetElapsedTime() < DelayedStop) {
                    return;
                } else {
                    // Stop collecting
                    DelayedStop = 0.0;
                    Status = COLLECTOR_STOP;
                    IsRunnable = false;
                    Stopwatch.Stop();
                }
            } else {
                return;
            }
            break;
    }

    CMN_ASSERT(Status == COLLECTOR_COLLECTING ||
               Status == COLLECTOR_STOP);

    if (Status == COLLECTOR_STOP) {
        DelayedStop = 0.0;
        CMN_LOG_CLASS(3) << "The collector stopped." << std::endl;
        return;
    }

    // Replaced with command pattern.
    //Collect();
}

void mtsCollectorBase::Cleanup(void)
{
    // clean up
    ClearTaskMap();
}

bool mtsCollectorBase::RemoveSignal(const std::string & taskName, const std::string & signalName)
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

bool mtsCollectorBase::FindSignal(const std::string & taskName, const std::string & signalName)
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
/*
void mtsCollectorBase::SetTimeBase(const double samplingIntervalInSeconds, const bool offsetToZero)
{
    // Ensure that there is a task being collected.
    //if (taskMap.empty()) {
    //    return;
    //}

    // deltaT should be positive.
    if (samplingIntervalInSeconds <= 0.0) {
        return;
    }

    SamplingInterval = samplingIntervalInSeconds;
    SamplingStride = 0;
    TimeOffsetToZero = offsetToZero;
}

void mtsCollectorBase::SetTimeBase(const unsigned int samplingStride, const bool offsetToZero)
{
    // Ensure that there is a task being collected.
    //if (taskMap.empty()) {
    //    return;
    //}

    // deltaStride should be positive.
    if (samplingStride <= 0) {
        return;
    }

    SamplingInterval = 0.0;
    SamplingStride = samplingStride;
    TimeOffsetToZero = offsetToZero;
}
*/

void mtsCollectorBase::Start(const double delayedStartInSecond)
{    
    // Check for state transition
    switch (Status) {
        case COLLECTOR_WAIT_START:
            CMN_LOG_CLASS(5) << "Waiting for the collector to start." << std::endl;
            return;

        case COLLECTOR_WAIT_STOP:
            CMN_LOG_CLASS(5) << "Waiting for the collector to stop." << std::endl;
            return;

        case COLLECTOR_COLLECTING:
            CMN_LOG_CLASS(5) << "The collector is now running." << std::endl;
            return;
    }

    DelayedStart = delayedStartInSecond;
    Status = COLLECTOR_WAIT_START;

    Stopwatch.Reset();
    Stopwatch.Start();
}

void mtsCollectorBase::Stop(const double delayedStopInSecond)
{
    // Check for state transition
    switch (Status) {
        case COLLECTOR_WAIT_START:
            CMN_LOG_CLASS(5) << "Waiting for the collector to start." << std::endl;
            return;

        case COLLECTOR_WAIT_STOP:
            CMN_LOG_CLASS(5) << "Waiting for the collector to stop." << std::endl;
            return;

        case COLLECTOR_STOP:
            CMN_LOG_CLASS(5) << "The collector is not running." << std::endl;
            return;
    }

    DelayedStop = delayedStopInSecond;
    Status = COLLECTOR_WAIT_STOP;

    Stopwatch.Reset();
    Stopwatch.Start();
}

//-------------------------------------------------------
//	Miscellaneous Functions
//-------------------------------------------------------
void mtsCollectorBase::ClearTaskMap(void)
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