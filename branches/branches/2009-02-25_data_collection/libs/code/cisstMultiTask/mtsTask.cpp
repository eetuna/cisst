/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Ankur Kapoor, Peter Kazanzides, Min Yang Jung
  Created on: 2004-04-30

  (C) Copyright 2004-2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstCommon/cmnExport.h>
#include <cisstCommon/cmnPortability.h>
#include <cisstOSAbstraction/osaThread.h>
#include <cisstOSAbstraction/osaSleep.h>
#include <cisstOSAbstraction/osaGetTime.h>
#include <cisstMultiTask/mtsTask.h>
#include <cisstMultiTask/mtsTaskInterface.h>
#include <cisstMultiTask/mtsCollectorDump.h>

#include <iostream>
#include <string>


CMN_IMPLEMENT_SERVICES(mtsTask)

/********************* Methods to connect interfaces  *****************/

bool mtsTask::ConnectRequiredInterface(const std::string & requiredInterfaceName, mtsDeviceInterface * providedInterface)
{
    mtsRequiredInterface *requiredInterface = RequiredInterfaces.GetItem(requiredInterfaceName, 1);
    if (requiredInterface) {
        requiredInterface->ConnectTo(providedInterface);
        CMN_LOG_CLASS(3) << "ConnectRequiredInterface: required interface " << requiredInterfaceName
                         << " successfuly connected to provided interface " << providedInterface->GetName() << std::endl;
        return true;
    }
    return false;            
}


/********************* Methods that call user methods *****************/

void mtsTask::DoRunInternal(void)
{
	StateTable.Start();
	this->Run();
    StateTable.Advance();

    // Check if the event for data collection should be triggered.
    if (CollectData) {
        if (TriggerEnabled) {
            if (++DataCollectionInfo.NewDataCount >= DataCollectionInfo.EventTriggeringLimit) {
                //
                //  To-be-fetched data should not be overwritten while data collector
                //  is fetching them. (e.g., if there are quite many column vectors (=signals)
                //  in StateTable, logging takes quite a long time which might result in
                //  data overwritten. 
                //  In order to prevent this case, 'eventTriggeringRatio' (see constructor)
                //  has to be defined appropriately.
                //
                DataCollectionInfo.EventData = DataCollectionInfo.NewDataCount;
                DataCollectionInfo.TriggerEvent(DataCollectionInfo.EventData);

                DataCollectionInfo.NewDataCount = 0;
                TriggerEnabled = false;
            }
        }
    }
}
  
void mtsTask::StartupInternal(void) {
    CMN_LOG_CLASS(3) << "Starting StartupInternal for " << Name << std::endl;

    bool success = true;
    // Loop through the required interfaces and bind all commands and events
    RequiredInterfacesMapType::MapType::const_iterator requiredIterator = RequiredInterfaces.GetMap().begin();
    mtsDeviceInterface * connectedInterface;
    for (;
         requiredIterator != RequiredInterfaces.GetMap().end();
         requiredIterator++) {
        connectedInterface = requiredIterator->second->GetConnectedInterface();
        if (connectedInterface) {
            CMN_LOG_CLASS(3) << "StartupInternal: ask " << connectedInterface->GetName() 
                             << " to allocate resources for " << this->GetName() << std::endl;
            connectedInterface->AllocateResourcesForCurrentThread();
            CMN_LOG_CLASS(3) << "StartupInternal: binding commands and events" << std::endl;
            success &= requiredIterator->second->BindCommandsAndEvents();
        } else {
            CMN_LOG_CLASS(2) << "StartupInternal: void pointer to required interface (required not connected to provided)" << std::endl;
            success = false;
        }
    }
    // Call user-supplied startup function
    this->Startup();
    // StateChange should already be locked
    if (success) {
       TaskState = READY;
       TriggerEnabled = false;
    }
    else
        CMN_LOG_CLASS(1) << "ERROR: Task " << GetName() << " cannot be started." << std::endl;
    StateChange.Unlock();
    CMN_LOG_CLASS(3) << "Ending StartupInternal for " << Name << std::endl;
}

void mtsTask::CleanupInternal() {
    // Call user-supplied cleanup function
	this->Cleanup();
    // Perform Cleanup on all interfaces provided
    ProvidedInterfaces.Cleanup();
    // StateChange should be locked by Kill().
	TaskState = FINISHED;
    StateChange.Unlock();
	CMN_LOG_CLASS(3) << "Done base class CleanupInternal " << Name << std::endl;
}


/********************* Methods to process queues  *********************/

// Execute all commands in the mailbox.  This is just a temporary implementation, where
// all commands in a mailbox are executed before moving on the next mailbox.  The final
// implementation will probably look at timestamps.  We may also want to pass in a
// parameter (enum) to set the mailbox processing policy.
unsigned int mtsTask::ProcessMailBoxes(ProvidedInterfacesMapType & interfaces)
{
    unsigned int numberOfCommands = 0;
    ProvidedInterfacesMapType::MapType::iterator iterator = interfaces.GetMap().begin();
    const ProvidedInterfacesMapType::MapType::iterator end = interfaces.GetMap().end();
    for (;
         iterator != end;
         ++iterator) {
        numberOfCommands += iterator->second->ProcessMailBoxes();
    }
    return numberOfCommands;
}


unsigned int mtsTask::ProcessQueuedEvents(void) {
    RequiredInterfacesMapType::MapType::iterator iterator = RequiredInterfaces.GetMap().begin();
    const RequiredInterfacesMapType::MapType::iterator end = RequiredInterfaces.GetMap().end();
    unsigned int numberOfEvents = 0;
    for (;
         iterator != end;
         iterator++) {
        numberOfEvents += iterator->second->ProcessMailBoxes();
    }
    return numberOfEvents;
}


/**************** Methods for managing task timing ********************/

void mtsTask::Sleep(double timeInSeconds)
{
    if (Thread.IsValid())
        Thread.Sleep(timeInSeconds);
    else
        osaSleep(timeInSeconds);
}


/********************* Task constructor and destructor *****************/

mtsTask::mtsTask(const std::string & name, 
                 mtsCollectorBase * dataCollector,
                 unsigned int sizeStateTable) :
    mtsDevice(name),
    Thread(),
    TaskState(CONSTRUCTED),
    StateChange(),
	StateTable(sizeStateTable),
    OverranPeriod(false),
    ThreadStartData(0),
    retValue(0),
    RequiredInterfaces("RequiredInterfaces")    
{    
    CollectData = (dataCollector == NULL ? false : true);

    // If the data of this task is to be collected, create a provided interface
    // dedicated for that purpose.
    if (CollectData) {
        DataCollectionInfo.Collector = dataCollector;

        DataCollectionInfo.ProvidedInterface 
            = AddProvidedInterface(GetDataCollectorProvidedInterfaceName());
        if (DataCollectionInfo.ProvidedInterface) {
            // Trigerring reset command registration
            DataCollectionInfo.ProvidedInterface->AddCommandVoid(
                &mtsTask::ResetDataCollectionTrigger, this, 
                mtsCollectorDump::GetDataCollectorResetEventName());

            // Data collection event registration
            DataCollectionInfo.TriggerEvent.Bind(
                DataCollectionInfo.ProvidedInterface->AddEventWrite(
                    mtsCollectorDump::GetDataCollectorEventName(), 
                    DataCollectionInfo.EventData));
        }

        // Determine the ratio value for event triggering.        
        //
        // TODO: an adaptive scaling feature according to 'sizeStateTable' might be useful.
        //
        const double eventTriggeringRatio = mtsCollectorDump::GetEventTriggeringRatio();

        DataCollectionInfo.EventTriggeringLimit = 
            (unsigned int) (sizeStateTable * eventTriggeringRatio);
    }
}

mtsTask::~mtsTask()
{
    CMN_LOG_CLASS(5) << "mtsTask destructor: deleting task " << Name << std::endl;
    if (!IsTerminated()) {
        //It is safe to call CleanupInternal() more than once.
        //Should we call the user-supplied Cleanup()?
        CleanupInternal();
    }
}


/********************* Methods to change task state ******************/

void mtsTask::Kill(void)
{
    CMN_LOG_CLASS(7) << "Kill " << Name << std::endl;
    StateChange.Lock();
    // If we get here, we cannot be in the INITIALIZING or FINISHING
    // states because we are holding the StateChange Mutex. 
    if (TaskState == FINISHED)
        StateChange.Unlock();
    else if (TaskState == CONSTRUCTED) {
        TaskState = FINISHED;
        StateChange.Unlock();
    }
    else {
        TaskState = FINISHING;
        // Unlock StateChange in RunInternal
    }
}


/********************* Methods to query the task state ****************/

const char *mtsTask::TaskStateName(TaskStateType state) const
{
    static char *taskStateNames[] = { "constructed", "initializing", "ready", "active", "finishing", "finished" };
    if ((state < CONSTRUCTED) || (state > FINISHED))
        return "unknown";
    else
        return taskStateNames[state];
}

/********************* Methods to manage interfaces *******************/
	
mtsDeviceInterface * mtsTask::AddProvidedInterface(const std::string & newInterfaceName) {
    mtsTaskInterface * newInterface = new mtsTaskInterface(newInterfaceName, this);
    if (newInterface) {
        if (ProvidedInterfaces.AddItem(newInterfaceName, newInterface)) {
            return newInterface;
        }
        CMN_LOG_CLASS(1) << "AddProvidedInterface: unable to add interface \""
                         << newInterfaceName << "\"" << std::endl;
        delete newInterface;
        return 0;
    }
    CMN_LOG_CLASS(1) << "AddProvidedInterface: unable to create interface \""
                     << newInterfaceName << "\"" << std::endl;
    return 0;
}


mtsRequiredInterface * mtsTask::AddRequiredInterface(const std::string & requiredInterfaceName,
                                                    mtsRequiredInterface *requiredInterface) {
    return RequiredInterfaces.AddItem(requiredInterfaceName, requiredInterface)?requiredInterface:0;
}

mtsRequiredInterface * mtsTask::AddRequiredInterface(const std::string & requiredInterfaceName) {
    // PK: move DEFAULT_EVENT_QUEUE_LEN somewhere else (not in mtsTaskInterface)
    mtsMailBox * mbox = new mtsMailBox(requiredInterfaceName + "Events", mtsTaskInterface::DEFAULT_EVENT_QUEUE_LEN);
    mtsRequiredInterface * requiredInterface = new mtsRequiredInterface(requiredInterfaceName, mbox);
    if (mbox && requiredInterface) {
        if (RequiredInterfaces.AddItem(requiredInterfaceName, requiredInterface)) {
            return requiredInterface;
        }
        CMN_LOG_CLASS(1) << "AddRequiredInterface: unable to add interface \""
                         << requiredInterfaceName << "\"" << std::endl;
        delete requiredInterface;
        return 0;
    }
    CMN_LOG_CLASS(1) << "AddRequiredInterface: unable to create interface or mailbox for \""
                     << requiredInterfaceName << "\"" << std::endl;
    return 0;
}


std::vector<std::string> mtsTask::GetNamesOfRequiredInterfaces(void) const {
    return RequiredInterfaces.GetNames();
}


bool mtsTask::AddObserverToRequiredInterface(const std::string & CMN_UNUSED(requiredInterfaceName),
                                             const std::string & CMN_UNUSED(eventName),
                                             const std::string & CMN_UNUSED(handlerName))
{
    CMN_LOG_CLASS(1) << "AddObserverToRequiredInterface now obsolete" << std::endl;
    return false;
}

	
/********************* Methods to manage event handlers *******************/

mtsCommandWriteBase * mtsTask::GetEventHandlerWrite(const std::string & requiredInterfaceName,
                                                    const std::string & commandName)
{
    mtsRequiredInterface * requiredInterface = GetRequiredInterface(requiredInterfaceName);
    if (requiredInterface) {
        return requiredInterface->GetEventHandlerWrite(commandName);
    }
    return 0;
}


mtsCommandVoidBase * mtsTask::GetEventHandlerVoid(const std::string & requiredInterfaceName,
                                                  const std::string & commandName)
{
    mtsRequiredInterface * requiredInterface = GetRequiredInterface(requiredInterfaceName);
    if (requiredInterface) {
        return requiredInterface->GetEventHandlerVoid(commandName);
    }
    return 0;
}

/********************* Methods for task synchronization ***************/

bool mtsTask::WaitToStart(double timeout)
{
    if (TaskState == INITIALIZING) {
        CMN_LOG_CLASS(5) << "Waiting for task " << Name << " to start." << std::endl;
        // PK: Following doesn't work because WaitToStart is generally called from same thread
        // as Create, which is where the Lock was done.
        //StateChange.Lock();  // Should use TryLock with timeout
        // For now, we just use a Sleep and hope it is long enough
        osaSleep(timeout);
        if (TaskState != READY)
            CMN_LOG_CLASS(1) << "Task " << Name << " did not start properly, state = " << TaskStateName(TaskState) << std::endl;
        StateChange.Unlock();
    }
    return (TaskState >= READY);
}

bool mtsTask::WaitToTerminate(double timeout)
{
	CMN_LOG_CLASS(5) << "WaitToTerminate " << Name << std::endl;
    if (TaskState < FINISHING)
        return false;
    if (TaskState == FINISHING) {
        CMN_LOG_CLASS(5) << "Waiting for task " << Name << " to finish." << std::endl;
        StateChange.Lock();  // Should use TryLock with timeout
        if (TaskState != FINISHED)
            CMN_LOG_CLASS(1) << "Task " << Name << " did not finish properly, state = " << GetTaskStateName() << std::endl;
        StateChange.Unlock();
    }
    // If task state is finished, we wait for the thread to be destroyed
    if ((TaskState == FINISHED) && Thread.IsValid()) {
        CMN_LOG_CLASS(5) << "Waiting for task " << Name << " thread to exit." << std::endl;
        Thread.Wait();
    }
	return true;
}

int mtsTask::GetStateVectorID(const std::string & dataName) const
{	
    return StateTable.GetStateVectorID(dataName);
}

void mtsTask::ResetDataCollectionTrigger(void)
{ 
    TriggerEnabled = true;
}

void mtsTask::GetStateTableHistory(mtsHistoryBase * history,
                                   const unsigned int signalIndex,
                                   const unsigned int lastFetchIndex)
{
    //StateTable.GetStateTableHistory(history, signalIndex, lastFetchIndex);

    //mtsStateTable::AccessorBase * acc = StateTable.GetAccessor(signalName);
    //CMN_ASSERT(acc);

    //mtsStateTable::Accessor<_appropriate_type> * accessor = 
    //   dynamic_cast<Accessor<_appropriate_type> *>(acc);

    //accessor->GetHistory(StateTable.GetIndexReader(), mtsVector_container_with_appropriate_type);

    // Get an object from StateArray
    //StateTable.GetStateDataElement(0))

    //history = new mtsHistory<
}