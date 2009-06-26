/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Ankur Kapoor, Peter Kazanzides
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
#include <cisstMultiTask/mtsTaskManager.h>

#include <cisstMultiTask/mtsDeviceInterfaceProxyServer.h>
#include <cisstMultiTask/mtsDeviceInterfaceProxyClient.h>

#include <iostream>
#include <string>


CMN_IMPLEMENT_SERVICES(mtsTask)

/********************* Methods that call user methods *****************/

void mtsTask::DoRunInternal(void)
{
	StateTable.Start();
	this->Run();
    StateTable.Advance();
}
  
void mtsTask::StartupInternal(void) {
    CMN_LOG_CLASS_INIT_VERBOSE << "Starting StartupInternal for " << Name << std::endl;

    bool success = true;
    unsigned int userId;
    // Loop through the required interfaces and bind all commands and events
    RequiredInterfacesMapType::MapType::const_iterator requiredIterator = RequiredInterfaces.GetMap().begin();
    mtsDeviceInterface * connectedInterface;
    for (;
         requiredIterator != RequiredInterfaces.GetMap().end();
         requiredIterator++) {
        connectedInterface = requiredIterator->second->GetConnectedInterface();
        if (connectedInterface) {
            CMN_LOG_CLASS_INIT_VERBOSE << "StartupInternal: ask " << connectedInterface->GetName() 
                                       << " to allocate resources for " << this->GetName() << std::endl;
            userId = connectedInterface->AllocateResources(this->GetName());
            CMN_LOG_CLASS_INIT_VERBOSE << "StartupInternal: binding commands and events with user Id "
                                       << userId << std::endl;
            success &= requiredIterator->second->BindCommandsAndEvents(userId);
        } else {
            CMN_LOG_CLASS_INIT_WARNING << "StartupInternal: void pointer to required interface (required not connected to provided)" << std::endl;
            success = false;
        }
    }
    // Call user-supplied startup function
    this->Startup();
    // StateChange should already be locked
    if (success)
       TaskState = READY;
    else
        CMN_LOG_CLASS_INIT_ERROR << "ERROR: Task " << GetName() << " cannot be started." << std::endl;
    StateChange.Unlock();
    CMN_LOG_CLASS_INIT_VERBOSE << "Ending StartupInternal for " << Name << std::endl;
}

void mtsTask::CleanupInternal() {
    // Call user-supplied cleanup function
	this->Cleanup();
    // Perform Cleanup on all interfaces provided
    ProvidedInterfaces.ForEachVoid(&mtsDeviceInterface::Cleanup);
    // StateChange should be locked by Kill().
	TaskState = FINISHED;
    StateChange.Unlock();
	CMN_LOG_CLASS_INIT_VERBOSE << "Done base class CleanupInternal " << Name << std::endl;
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

mtsTask::mtsTask(const std::string & name, unsigned int sizeStateTable) :
    mtsDevice(name),
    Thread(),
    TaskState(CONSTRUCTED),
    StateChange(),
	StateTable(sizeStateTable),
    OverranPeriod(false),
    ThreadStartData(0),
    ReturnValue(0),
    TaskInterfaceCommunicatorID("TaskInterfaceServerSender")
{
}

mtsTask::~mtsTask()
{
    CMN_LOG_CLASS_RUN_ERROR << "mtsTask destructor: deleting task " << Name << std::endl;
    if (!IsTerminated()) {
        //It is safe to call CleanupInternal() more than once.
        //Should we call the user-supplied Cleanup()?
        CleanupInternal();
    }

    // Stop all provided interface proxies running.
    ProvidedInterfaceProxyMapType::MapType::iterator it1 =
        ProvidedInterfaceProxies.GetMap().begin();
    for (; it1 != ProvidedInterfaceProxies.GetMap().end(); ++it1) {
        it1->second->Stop();
    }

    // Stop all required interface proxies running.
    RequiredInterfaceProxyMapType::MapType::iterator it2 =
        RequiredInterfaceProxies.GetMap().begin();
    for (; it2 != RequiredInterfaceProxies.GetMap().end(); ++it2) {
        it2->second->Stop();
    }

    osaSleep(500 * cmn_ms);

    // Deallocation
    ProvidedInterfaceProxies.DeleteAll();
    RequiredInterfaceProxies.DeleteAll();
}


/********************* Methods to change task state ******************/

void mtsTask::Kill(void)
{
    CMN_LOG_CLASS_RUN_WARNING << "Kill " << Name << std::endl;
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
    static const char * taskStateNames[] = { "constructed", "initializing", "ready", "active", "finishing", "finished" };
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
        CMN_LOG_CLASS_INIT_ERROR << "AddProvidedInterface: unable to add interface \""
                                 << newInterfaceName << "\"" << std::endl;
        delete newInterface;
        return 0;
    }
    CMN_LOG_CLASS_INIT_ERROR << "AddProvidedInterface: unable to create interface \""
                             << newInterfaceName << "\"" << std::endl;
    return 0;
}


bool mtsTask::AddObserverToRequiredInterface(const std::string & CMN_UNUSED(requiredInterfaceName),
                                             const std::string & CMN_UNUSED(eventName),
                                             const std::string & CMN_UNUSED(handlerName))
{
    CMN_LOG_CLASS_INIT_ERROR << "AddObserverToRequiredInterface now obsolete" << std::endl;
    return false;
}


// deprecated
mtsCommandWriteBase * mtsTask::GetEventHandlerWrite(const std::string & requiredInterfaceName,
                                                    const std::string & commandName)
{
    mtsRequiredInterface * requiredInterface = GetRequiredInterface(requiredInterfaceName);
    if (requiredInterface) {
        return requiredInterface->GetEventHandlerWrite(commandName);
    }
    return 0;
}

// deprecated
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
        CMN_LOG_CLASS_RUN_ERROR << "Waiting for task " << Name << " to start." << std::endl;
        // PK: Following doesn't work because WaitToStart is generally called from same thread
        // as Create, which is where the Lock was done.
        //StateChange.Lock();  // Should use TryLock with timeout
        // For now, we just use a Sleep and hope it is long enough
        osaSleep(timeout);
        if (TaskState != READY)
            CMN_LOG_CLASS_INIT_ERROR << "Task " << Name << " did not start properly, state = " << TaskStateName(TaskState) << std::endl;
        StateChange.Unlock();
    }
    return (TaskState >= READY);
}

bool mtsTask::WaitToTerminate(double timeout)
{
	CMN_LOG_CLASS_RUN_ERROR << "WaitToTerminate " << Name << std::endl;
    if (TaskState < FINISHING)
        return false;
    if (TaskState == FINISHING) {
        CMN_LOG_CLASS_RUN_ERROR << "Waiting for task " << Name << " to finish." << std::endl;
        StateChange.Lock();  // Should use TryLock with timeout
        if (TaskState != FINISHED)
            CMN_LOG_CLASS_INIT_ERROR << "Task " << Name << " did not finish properly, state = " << GetTaskStateName() << std::endl;
        StateChange.Unlock();
    }
    // If task state is finished, we wait for the thread to be destroyed
    if ((TaskState == FINISHED) && Thread.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "Waiting for task " << Name << " thread to exit." << std::endl;
        Thread.Wait();
    }
	return true;
}


void mtsTask::ToStream(std::ostream & outputStream) const
{
    outputStream << "Task name: " << Name << std::endl;
    ProvidedInterfaces.ToStream(outputStream);
    RequiredInterfaces.ToStream(outputStream);
}

//-----------------------------------------------------------------------------
//  Processing Methods
//-----------------------------------------------------------------------------
void mtsTask::StartProvidedInterfaceProxies(const std::string & serverTaskIP)
{
    mtsTaskManager * TaskManager = mtsTaskManager::GetInstance();
    if (TaskManager->GetTaskManagerType() == mtsTaskManager::TASK_MANAGER_LOCAL) {
        CMN_LOG_CLASS_RUN_ERROR << "StartProvidedInterfaceProxies failed: " 
            << "This task manager works only locally." << std::endl;
        return;
    }

    if (ProvidedInterfaces.GetCount() <= 0) {
        CMN_LOG_CLASS_RUN_ERROR << "StartProvidedInterfaceProxies failed: " 
            << "No provided interface exists." << std::endl;
        return;
    }

    const std::string adapterNameBase = "TaskInterfaceServerAdapter";
    const std::string endpointInfoBase = "tcp -p ";
    const std::string communicatorId = TaskInterfaceCommunicatorID;
    
    std::string providedInterfaceName;
    std::string adapterName, endpointInfo;
    std::string portNumber;
    std::string endpointInfoForClient;

    // Run device interface proxy server objects of which type are 
    // mtsDeviceInterfaceProxyServer. A mtsDeviceInterfaceProxyServer object
    // is created to serve one provided interface.
    mtsDeviceInterfaceProxyServer * newProvidedInterfaceProxy = NULL;
    ProvidedInterfacesMapType::MapType::const_iterator it = 
        ProvidedInterfaces.GetMap().begin();
    ProvidedInterfacesMapType::MapType::const_iterator itEnd = 
        ProvidedInterfaces.GetMap().end();
    for (; it != itEnd; ++it) {
        providedInterfaceName = it->first;
        adapterName = adapterNameBase + "_" + providedInterfaceName;
        // Assign a new port number for a new proxy object.
        portNumber = GetPortNumberString(ProvidedInterfaceProxies.GetCount());
        endpointInfo = endpointInfoBase + portNumber;
        endpointInfoForClient = ":default -h " +
                                serverTaskIP + " " +
                                "-p " + portNumber;
        //
        // TODO: Replace hard-coded proxy definition with property files.
        //
        newProvidedInterfaceProxy = new mtsDeviceInterfaceProxyServer(
            adapterName, endpointInfo, communicatorId);

        if (!ProvidedInterfaceProxies.AddItem(providedInterfaceName, newProvidedInterfaceProxy)) {
            CMN_LOG_CLASS_RUN_ERROR << "StartProvidedInterfaceProxies failed: "
                << "cannot add a new provided interface proxy object: "
                << providedInterfaceName << std::endl;
            continue;
        }

        newProvidedInterfaceProxy->Start(this);
        newProvidedInterfaceProxy->GetLogger()->trace(
            "mtsTask", "Provided interface proxy starts: " + providedInterfaceName);
        
        // Inform the global task manager of the existence of the newly created 
        // provided interface proxy with the access information for clients.
        //if (!TaskManager->AddProvidedInterface(
        //    providedInterfaceName, adapterName, endpointInfoForClient, communicatorId, Name)) 
        //{
        //    newProvidedInterfaceProxy->GetLogger()->error(
        //        "failed to register a new provided interface: " + providedInterfaceName);
        //    continue;
        //} else {
        //    newProvidedInterfaceProxy->GetLogger()->trace(
        //        "mtsTask", "registered a new provided interface: " + providedInterfaceName);
        //}
    }
}

void mtsTask::StartRequiredInterfaceProxies(
    const std::string & endpointInfo, const std::string & communicatorID)
{
    mtsTaskManager * TaskManager = mtsTaskManager::GetInstance();
    if (TaskManager->GetTaskManagerType() == mtsTaskManager::TASK_MANAGER_LOCAL) {
        CMN_LOG_CLASS_RUN_ERROR << "StartRequiredInterfaceProxies failed: " 
            << "This task manager works only locally." << std::endl;
        return;
    }

    if (RequiredInterfaces.GetCount() <= 0) {
        CMN_LOG_CLASS_RUN_ERROR << "StartRequiredInterfaceProxies failed: " 
            << "No required interface exists." << std::endl;
        return;
    }

    std::string requiredInterfaceName;

    // Run device interface proxy client objects of which type are 
    // mtsDeviceInterfaceProxyClient. A mtsDeviceInterfaceProxyClient object
    // is created to serve one required interface.
    mtsDeviceInterfaceProxyClient * newRequiredInterfaceProxy = NULL;
    RequiredInterfacesMapType::MapType::const_iterator it = 
        RequiredInterfaces.GetMap().begin();
    RequiredInterfacesMapType::MapType::const_iterator itEnd = 
        RequiredInterfaces.GetMap().end();
    for (; it != itEnd; ++it) {
        requiredInterfaceName = it->first;
        //
        // TODO: Replace hard-coded proxy definition with property files.
        //
        newRequiredInterfaceProxy = new mtsDeviceInterfaceProxyClient(
            endpointInfo, communicatorID);

        if (!RequiredInterfaceProxies.AddItem(requiredInterfaceName, newRequiredInterfaceProxy)) {
            CMN_LOG_CLASS_RUN_ERROR << "StartRequiredInterfaceProxies failed: "
                << "cannot add a new required interface proxy object: "
                << requiredInterfaceName << std::endl;
            continue;
        }

        newRequiredInterfaceProxy->Start(this);
        newRequiredInterfaceProxy->GetLogger()->trace(
            "mtsTask", "Required interface proxy starts: " + requiredInterfaceName);
        
        // Inform the global task manager of the existence of the newly created 
        // required interface proxy object.
        //if (!TaskManager->AddRequiredInterface(requiredInterfaceName, Name)) {
        //    newRequiredInterfaceProxy->GetLogger()->error(
        //        "failed to register a new required interface: " + requiredInterfaceName);
        //    continue;
        //} else {
        //    newRequiredInterfaceProxy->GetLogger()->trace(
        //        "mtsTask", "registered a new required interface: " + requiredInterfaceName);
        //}
    }
}

mtsDeviceInterfaceProxyServer * mtsTask::GetProvidedInterfaceProxy(
    const std::string & providedInterfaceName) const
{
    mtsDeviceInterfaceProxyServer * providedIntertfaceProxy = 
        ProvidedInterfaceProxies.GetItem(providedInterfaceName);
    if (!providedIntertfaceProxy) {
        CMN_LOG_CLASS_RUN_ERROR << "GetProvidedInterfaceProxy: " 
            << "Can't find a provided interface proxy: " << providedInterfaceName << std::endl;
        return NULL;
    }
}

mtsDeviceInterfaceProxyClient * mtsTask::GetRequiredInterfaceProxy(
    const std::string & requiredInterfaceName) const
{
    mtsDeviceInterfaceProxyClient * requiredIntertfaceProxy = 
        RequiredInterfaceProxies.GetItem(requiredInterfaceName);
    if (!requiredIntertfaceProxy) {
        CMN_LOG_CLASS_RUN_ERROR << "GetRequiredInterfaceProxy: " 
            << "Can't find a required interface proxy: " << requiredInterfaceName << std::endl;
        return NULL;
    }
}

/*
void mtsTask::StartInterfaceProxyServer(const std::string & ServerTaskIP)
{
    //mtsTaskManager * TaskManager = mtsTaskManager::GetInstance();
    //if (TaskManager->GetTaskManagerType() == mtsTaskManager::TASK_MANAGER_LOCAL) {
    //    return;
    //}

    //if (ProvidedInterfaces.GetCount() <= 0) {
    //    CMN_LOG_CLASS_RUN_ERROR << "No provided interface exists. Proxy server wasn't created." << std::endl;
    //    return;
    //}

    // Run device interface proxy server objects of which type is 
    // mtsDeviceInterfaceProxyServer. One instance of mtsDeviceInterfaceProxyServer
    // is created for one provided interface.
    ProvidedInterfacesMapType::MapType::const_iterator itBegin = 
        ProvidedInterfaces.GetMap().begin();
    ProvidedInterfacesMapType::MapType::const_iterator itEnd = 
        ProvidedInterfaces.GetMap().end();
    for (; itBegin != itEnd; ++itBegin) {
        const std::string adapterName = "TaskInterfaceServerAdapter";
        //
        // TODO: avoid using hard-coded proxy access information
        //
        const std::string endpointInfo = "tcp -p 11705";
        const std::string endpointInfoForClient = ":default -h " + ServerTaskIP + " -p 11705";
        const std::string communicatorID = TaskInterfaceCommunicatorID;

        Proxy = new mtsDeviceInterfaceProxyServer(adapterName, endpointInfo, communicatorID);
        Proxy->Start(this);
        Proxy->GetLogger()->trace("mtsTask", "Provided interface proxy starts.");
        ProxyServer = dynamic_cast<mtsDeviceInterfaceProxyServer *>(Proxy);

        // Inform the global task manager of the existence of a newly created 
        // provided interface with the access information.
        if (!TaskManager->AddProvidedInterface(
            iterator->first, adapterName, endpointInfoForClient, communicatorID, Name)) 
        {
            Proxy->GetLogger()->error("Failed to add provided interface: " + iterator->first);
            return;
        } else {
            Proxy->GetLogger()->trace("mtsTask", "Registered provided interface: " + iterator->first);
        }
    }
}

void mtsTask::StartProxyClient(const std::string & endpointInfo, 
                               const std::string & communicatorID)
{
    mtsTaskManager * TaskManager = mtsTaskManager::GetInstance();
    if (TaskManager->GetTaskManagerType() == mtsTaskManager::TASK_MANAGER_LOCAL) {
        return;
    }

    if (RequiredInterfaces.GetCount() <= 0) {
        CMN_LOG_CLASS_RUN_ERROR << "No required interface exists. Proxy client wasn't created." << std::endl;
        return;
    }

    // Start a required interface proxy (proxy client, mtsDeviceInterfaceProxyClient)    
    RequiredInterfacesMapType::MapType::iterator iterator = 
        RequiredInterfaces.GetMap().begin();
    //
    // TODO: I assume there is only one provided interface and one required interface
    //
    {
        Proxy = new mtsDeviceInterfaceProxyClient(endpointInfo, communicatorID);
        Proxy->Start(this);
        Proxy->GetLogger()->trace("mtsTask", "Required interface proxy starts.");
        ProxyClient = dynamic_cast<mtsDeviceInterfaceProxyClient *>(Proxy);

        // Inform the global task manager of the existence on a newly created 
        // required interface.
        if (!TaskManager->AddRequiredInterface(iterator->first, Name)) 
        {
            Proxy->GetLogger()->error("Failed to add required interface: " + iterator->first);
        } else {
            Proxy->GetLogger()->trace("mtsTask", "Registered required interface: " + iterator->first);
        }
    }
}
*/

//
//  For a client task
//
/*
const bool mtsTask::GetProvidedInterfaces(
    mtsDeviceInterfaceProxy::ProvidedInterfaceSequence & providedInterfaces)
{
    CMN_ASSERT(ProxyClient);

    return ProxyClient->SendGetProvidedInterfaces(providedInterfaces);
}

bool mtsTask::SendConnectServerSide(
    const std::string & userTaskName, const std::string & requiredInterfaceName,
    const std::string & resourceTaskName, const std::string & providedInterfaceName)
{
    CMN_ASSERT(ProxyClient);

    return ProxyClient->SendConnectServerSide(
        userTaskName, requiredInterfaceName, resourceTaskName, providedInterfaceName);
}

void mtsTask::SendGetCommandId(
    mtsDeviceInterfaceProxy::FunctionProxySet & functionProxies)
{
    CMN_ASSERT(ProxyClient);

    ProxyClient->SendGetCommandId(functionProxies);
}
*/

const std::string mtsTask::GetPortNumberString(const unsigned int id)
{
    static unsigned int defaultTaskPortNumber = 10705;

    unsigned int portNumberAllocated = defaultTaskPortNumber + (id * 5);

    std::stringstream portNumberString;
    portNumberString << portNumberAllocated;

    return portNumberString.str();
}
