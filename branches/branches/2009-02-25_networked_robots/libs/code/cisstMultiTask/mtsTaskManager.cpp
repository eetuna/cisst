/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Ankur Kapoor, Peter Kazanzides, Min Yang Jung
  Created on: 2004-04-30

  (C) Copyright 2004-2008 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstOSAbstraction/osaSleep.h>
#include <cisstMultiTask/mtsTaskManager.h>
#include <cisstMultiTask/mtsDevice.h>
#include <cisstMultiTask/mtsDeviceInterface.h>
#include <cisstMultiTask/mtsTask.h>
#include <cisstMultiTask/mtsTaskInterface.h>

#include <cisstMultiTask/mtsDeviceProxy.h>
#include <cisstMultiTask/mtsTaskManagerProxyServer.h>
#include <cisstMultiTask/mtsTaskManagerProxyClient.h>

#include <cisstMultiTask/mtsCommandVoidProxy.h>
#include <cisstMultiTask/mtsCommandWriteProxy.h>
#include <cisstMultiTask/mtsCommandReadProxy.h>
#include <cisstMultiTask/mtsCommandQualifiedReadProxy.h>
#include <cisstMultiTask/mtsMulticastCommandWriteProxy.h>

CMN_IMPLEMENT_SERVICES(mtsTaskManager);


mtsTaskManager::mtsTaskManager():
    TaskMap("Tasks"),
    DeviceMap("Devices"),
    ProxyServer(0),
    ProxyClient(0),
    TaskManagerTypeMember(TASK_MANAGER_LOCAL),
    TaskManagerCommunicatorID("TaskManagerServerSender"),
    Proxy(0)
{
    __os_init();
    TaskMap.SetOwner(*this);
    DeviceMap.SetOwner(*this);
    TimeServer.SetTimeOrigin();
}


mtsTaskManager::~mtsTaskManager(){
    this->Kill();

    if (Proxy) {
        delete Proxy;
    }
}


mtsTaskManager* mtsTaskManager::GetInstance(void) {
    static mtsTaskManager instance;
    return &instance;
}


bool mtsTaskManager::AddTask(mtsTask * task) {
    bool ret = TaskMap.AddItem(task->GetName(), task, CMN_LOG_LOD_INIT_ERROR);
    if (ret)
        CMN_LOG_CLASS_INIT_VERBOSE << "AddTask: added task named "
                                   << task->GetName() << std::endl;
    return ret;
}


bool mtsTaskManager::RemoveTask(mtsTask * task) {
    //bool ret = TaskMap.RemoveItem(task->GetName(), 1);
    //if (ret)
    //   CMN_LOG_CLASS(3) << "RemoveTask: removed task named "
    //                    << task->GetName() << std::endl;
    //return ret;
    return true;
}


bool mtsTaskManager::AddDevice(mtsDevice * device) {
    mtsTask * task = dynamic_cast<mtsTask *>(device);
    if (task) {
        CMN_LOG_CLASS_INIT_ERROR << "AddDevice: Attempt to add " << task->GetName() << "as a device (use AddTask instead)."
                                 << std::endl;
        return false;
    }
    bool ret = DeviceMap.AddItem(device->GetName(), device, CMN_LOG_LOD_INIT_ERROR);
    if (ret)
        CMN_LOG_CLASS_INIT_VERBOSE << "AddDevice: added device named "
                                   << device->GetName() << std::endl;
    return ret;
}

bool mtsTaskManager::Add(mtsDevice * device) {
    mtsTask * task = dynamic_cast<mtsTask *>(device);
    return task?AddTask(task):AddDevice(device);
}

std::vector<std::string> mtsTaskManager::GetNamesOfDevices(void) const {
    return DeviceMap.GetNames();
}


std::vector<std::string> mtsTaskManager::GetNamesOfTasks(void) const {
    return TaskMap.GetNames();
}

void mtsTaskManager::GetNamesOfTasks(std::vector<std::string>& taskNameContainer) const {
    TaskMap.GetNames(taskNameContainer);
}


mtsDevice * mtsTaskManager::GetDevice(const std::string & deviceName) {
    return DeviceMap.GetItem(deviceName, CMN_LOG_LOD_INIT_ERROR);
}


mtsTask * mtsTaskManager::GetTask(const std::string & taskName) {
    return TaskMap.GetItem(taskName, CMN_LOG_LOD_INIT_ERROR);
}
    

void mtsTaskManager::ToStream(std::ostream & outputStream) const {
    TaskMapType::MapType::const_iterator taskIterator = TaskMap.GetMap().begin();
    const TaskMapType::MapType::const_iterator taskEndIterator = TaskMap.GetMap().end();
    outputStream << "List of tasks: name and address" << std::endl;
    for (; taskIterator != taskEndIterator; ++taskIterator) {
        outputStream << "  Task: " << taskIterator->first << ", address: " << taskIterator->second << std::endl;
    }
    DeviceMapType::MapType::const_iterator deviceIterator = DeviceMap.GetMap().begin();
    const DeviceMapType::MapType::const_iterator deviceEndIterator = DeviceMap.GetMap().end();
    outputStream << "List of devices: name and address" << std::endl;
    for (; deviceIterator != deviceEndIterator; ++deviceIterator) {
        outputStream << "  Device: " << deviceIterator->first << ", adress: " << deviceIterator->second << std::endl;
    }
    AssociationSetType::const_iterator associationIterator = AssociationSet.begin();
    const AssociationSetType::const_iterator associationEndIterator = AssociationSet.end();
    outputStream << "Associations: task::requiredInterface associated to device/task::requiredInterface" << std::endl;
    for (; associationIterator != associationEndIterator; ++associationIterator) {
        outputStream << "  " << associationIterator->first.first << "::" << associationIterator->first.second << std::endl
                     << "  -> " << associationIterator->second.first << "::" << associationIterator->second.second << std::endl;
    }
}


void mtsTaskManager::CreateAll(void) {
    TaskMapType::MapType::const_iterator taskIterator = TaskMap.GetMap().begin();
    const TaskMapType::MapType::const_iterator taskEndIterator = TaskMap.GetMap().end();
    for (; taskIterator != taskEndIterator; ++taskIterator) {
        taskIterator->second->Create();
    }
}


void mtsTaskManager::StartAll(void) {
    // Get the current thread id so that we can check if any task will use the current thread.
    // If so, start that task last because its Start method will not return.
    const osaThreadId threadId = osaGetCurrentThreadId();
    TaskMapType::MapType::const_iterator lastTask = TaskMap.GetMap().end();

    // Loop through all tasks.
    TaskMapType::MapType::const_iterator taskIterator = TaskMap.GetMap().begin();
    const TaskMapType::MapType::const_iterator taskEndIterator = TaskMap.GetMap().end();
    for (; taskIterator != taskEndIterator; ++taskIterator) {
        // Check if the task will use the current thread.
        if (taskIterator->second->Thread.GetId() == threadId) {
            CMN_LOG_CLASS_RUN_ERROR << "StartAll: task " << taskIterator->first << " uses current thread, will start last." << std::endl;
            if (lastTask != TaskMap.GetMap().end())
                CMN_LOG_CLASS_INIT_ERROR << "WARNING: multiple tasks using current thread (only first will be started)." << std::endl;
            else
                lastTask = taskIterator;
        }
        else
            taskIterator->second->Start();  // If task will not use current thread, start it.
    }
    // If there is a task that uses the current thread, start it.
    if (lastTask != TaskMap.GetMap().end())
        lastTask->second->Start();
}


void mtsTaskManager::KillAll(void) {
    // It is not necessary to have any special handling of a task using the current thread.
    TaskMapType::MapType::const_iterator taskIterator = TaskMap.GetMap().begin();
    const TaskMapType::MapType::const_iterator taskEndIterator = TaskMap.GetMap().end();
    for (; taskIterator != taskEndIterator; ++taskIterator) {
        taskIterator->second->Kill();
    }
}


void mtsTaskManager::ToStreamDot(std::ostream & outputStream) const {
    std::vector<std::string> providedInterfacesAvailable, requiredInterfacesAvailable;
    std::vector<std::string>::const_iterator stringIterator;
    unsigned int clusterNumber = 0;
    // dot header
    outputStream << "/* Automatically generated by cisstMultiTask, mtsTaskManager::ToStreamDot.\n"
                 << "   Use Graphviz utility \"dot\" to generate a graph of tasks/devices interactions. */"
                 << std::endl;
    outputStream << "digraph mtsTaskManager {" << std::endl;
    // create all nodes for tasks
    TaskMapType::MapType::const_iterator taskIterator = TaskMap.GetMap().begin();
    const TaskMapType::MapType::const_iterator taskEndIterator = TaskMap.GetMap().end();
    for (; taskIterator != taskEndIterator; ++taskIterator) {
        outputStream << "subgraph cluster" << clusterNumber << "{" << std::endl
                     << "node[style=filled,color=white,shape=box];" << std::endl
                     << "style=filled;" << std::endl
                     << "color=lightgrey;" << std::endl; 
        clusterNumber++;
        outputStream << taskIterator->first
                     << " [label=\"Task:\\n" << taskIterator->first << "\"];" << std::endl;
        providedInterfacesAvailable = taskIterator->second->GetNamesOfProvidedInterfaces();
        for (stringIterator = providedInterfacesAvailable.begin();
             stringIterator != providedInterfacesAvailable.end();
             stringIterator++) {
            outputStream << taskIterator->first << "providedInterface" << *stringIterator
                         << " [label=\"Provided interface:\\n" << *stringIterator << "\"];" << std::endl;
            outputStream << taskIterator->first << "providedInterface" << *stringIterator
                         << "->" << taskIterator->first << ";" << std::endl;
        }
        requiredInterfacesAvailable = taskIterator->second->GetNamesOfRequiredInterfaces();
        for (stringIterator = requiredInterfacesAvailable.begin();
             stringIterator != requiredInterfacesAvailable.end();
             stringIterator++) {
            outputStream << taskIterator->first << "requiredInterface" << *stringIterator
                         << " [label=\"Required interface:\\n" << *stringIterator << "\"];" << std::endl;
            outputStream << taskIterator->first << "->"
                         << taskIterator->first << "requiredInterface" << *stringIterator << ";" << std::endl;
        }
        outputStream << "}" << std::endl;
    }
    // create all nodes for devices
    DeviceMapType::MapType::const_iterator deviceIterator = DeviceMap.GetMap().begin();
    const DeviceMapType::MapType::const_iterator deviceEndIterator = DeviceMap.GetMap().end();
    for (; deviceIterator != deviceEndIterator; ++deviceIterator) {
        outputStream << "subgraph cluster" << clusterNumber << "{" << std::endl
                     << "node[style=filled,color=white,shape=box];" << std::endl
                     << "style=filled;" << std::endl
                     << "color=lightgrey;" << std::endl; 
        clusterNumber++;
        outputStream << deviceIterator->first
                     << " [label=\"Device:\\n" << deviceIterator->first << "\"];" << std::endl;
        providedInterfacesAvailable = deviceIterator->second->GetNamesOfProvidedInterfaces();
        for (stringIterator = providedInterfacesAvailable.begin();
             stringIterator != providedInterfacesAvailable.end();
             stringIterator++) {
            outputStream << deviceIterator->first << "providedInterface" << *stringIterator
                         << " [label=\"Provided interface:\\n" << *stringIterator << "\"];" << std::endl;
            outputStream << deviceIterator->first << "providedInterface" << *stringIterator
                         << "->" << deviceIterator->first << ";" << std::endl;
        }
        outputStream << "}" << std::endl;
    }
    // create edges
    AssociationSetType::const_iterator associationIterator = AssociationSet.begin();
    const AssociationSetType::const_iterator associationEndIterator = AssociationSet.end();
    for (; associationIterator != associationEndIterator; ++associationIterator) {
        outputStream << associationIterator->first.first << "requiredInterface" << associationIterator->first.second
                     << "->"
                     << associationIterator->second.first << "providedInterface" << associationIterator->second.second
                     << ";" << std::endl;
    }
    // end of file
    outputStream << "}" << std::endl;
}


bool mtsTaskManager::Connect(const std::string & userTaskName, const std::string & requiredInterfaceName,
                             const std::string & resourceTaskName, const std::string & providedInterfaceName)
{
    // True if the resource task specified is provided by a remote task
    bool requestServerSideConnect = false;

    const UserType fullUserName(userTaskName, requiredInterfaceName);
    const ResourceType fullResourceName(resourceTaskName, providedInterfaceName);
    const AssociationType association(fullUserName, fullResourceName);
    // check if this connection has already been established
    AssociationSetType::const_iterator associationIterator = AssociationSet.find(association);
    if (associationIterator != AssociationSet.end()) {
        CMN_LOG_CLASS_INIT_ERROR << "Connect: " << userTaskName << "::" << requiredInterfaceName
                                 << " is already connected to " << resourceTaskName << "::" << providedInterfaceName << std::endl;
        return false;
    }
    // check that names are not the same
    if (userTaskName == resourceTaskName) {
        CMN_LOG_CLASS_INIT_ERROR << "Connect: can not connect two tasks/devices with the same name" << std::endl;
        return false;
    }
    // check if the user name corresponds to an existing task
    mtsDevice * userTask = TaskMap.GetItem(userTaskName);
    if (!userTask) {
        userTask = DeviceMap.GetItem(userTaskName);
        if (!userTask) {
            CMN_LOG_CLASS_INIT_ERROR << "Connect: can not find a user task or device named " << userTaskName << std::endl;
            return false;
        }
    }
    // check if the resource name corresponds to an existing task or device
    mtsDevice* resourceDevice = DeviceMap.GetItem(resourceTaskName);
    if (!resourceDevice) {        
        resourceDevice = TaskMap.GetItem(resourceTaskName);
    }
    // find the interface pointer from the local resource first
    //
    // TODO: remove the following variable after creating a server task proxy at client side.
    //
    mtsTask * userTaskTemp = NULL;
    mtsDeviceInterface * resourceInterface;
    if (resourceDevice) {
        resourceInterface = resourceDevice->GetProvidedInterface(providedInterfaceName);
    } else {
        // If we cannot find, the resource interface should be at remote or doesn't exist.
        if (GetTaskManagerType() == TASK_MANAGER_LOCAL) {
            CMN_LOG_CLASS_INIT_ERROR << "Connect: Cannot find a task or device named " << resourceTaskName << std::endl;
            return false;
        } else {
            userTaskTemp = dynamic_cast<mtsTask*>(userTask);
            CMN_ASSERT(userTaskTemp);

            resourceInterface = GetResourceInterface(resourceTaskName, providedInterfaceName, 
                                    userTaskName, requiredInterfaceName, 
                                    userTaskTemp);
            if (!resourceInterface) {
                CMN_LOG_CLASS_INIT_ERROR << "Connect through networks: Cannot find the task or device named " << resourceTaskName << std::endl;
                return false;                
            }

            requestServerSideConnect = true;
        }
    }

    // check the interface pointer we got
    if (resourceInterface == 0) {
        CMN_LOG_CLASS_INIT_ERROR << "Connect: interface pointer for "
                                 << resourceTaskName << "::" << providedInterfaceName << " is null" << std::endl;
        return false;
    }
    // attempt to connect 
    if (!(userTask->ConnectRequiredInterface(requiredInterfaceName, resourceInterface))) {
        CMN_LOG_CLASS_INIT_ERROR << "Connect: connection failed, does " << requiredInterfaceName << " exist?" << std::endl;
        return false;
    }

    // connected, add to the map of connections
    AssociationSet.insert(association);
    CMN_LOG_CLASS_INIT_VERBOSE << "Connect: " << userTaskName << "::" << requiredInterfaceName
                               << " successfully connected to " << resourceTaskName << "::" << providedInterfaceName << std::endl;

    // If the connection between the required interface with the provided interface proxy
    // at client side is established successfully, inform the global task manager of this 
    // fact.
    // 'requestServerSideConnect' is true only if this task manager is at client side and 
    // all the connection processing above are successful.
    if (requestServerSideConnect) {
        CMN_ASSERT(!ProxyServer);   // This is not a global task manager.
        CMN_ASSERT(ProxyClient);

        if (!SendConnectServerSide(userTaskTemp, userTaskName, requiredInterfaceName,
                          resourceTaskName, providedInterfaceName)) 
        {
            CMN_LOG_CLASS_INIT_ERROR << "Connect: server side connection failed." << std::endl;
            return false;
        }

        //
        //  TODO: FIX!! UGLY!!
        //
        mtsDeviceInterfaceProxy::FunctionProxySet functionProxies;
        userTaskTemp->SendGetCommandId(functionProxies);

        functionProxies.ServerTaskProxyName = mtsDeviceProxy::GetServerTaskProxyName(
            resourceTaskName, providedInterfaceName, userTaskName, requiredInterfaceName);
        functionProxies.ProvidedInterfaceProxyName = providedInterfaceName;

        UpdateCommandId(functionProxies);
    }

    return true;
}

mtsDeviceInterface * mtsTaskManager::GetResourceInterface(
    const std::string & resourceTaskName, const std::string & providedInterfaceName,
    const std::string & userTaskName, const std::strinAg & requiredInterfaceName,
    mtsTask * userTask)
{
    mtsDeviceInterface * resourceInterface = NULL;

    // For the use of consistent notation
    mtsTask * clientTask = userTask;

    // Ask the global task manager (TMServer) if the specific task specified 
    // the specific provided interface has been registered.
    if (!IsRegisteredProvidedInterface(resourceTaskName, providedInterfaceName)) {
        CMN_LOG_CLASS_RUN_ERROR << "Connect across networks: '" << providedInterfaceName << "' has not been registered." << resourceTaskName << ", " << std::endl;
        return NULL;
    }

    // If (task, provided interface) exists,
    // 1) Retrieve information from the global task manager to connect
    //    the requested provided interface (mtsDeviceInterfaceProxyServer).                
    mtsTaskManagerProxy::ProvidedInterfaceInfo info;
    if (!GetProvidedInterfaceInfo(resourceTaskName, providedInterfaceName, info)) {
        CMN_LOG_CLASS_RUN_ERROR << "Connect across networks: failed to retrieve proxy information: " << resourceTaskName << ", " << providedInterfaceName << std::endl;
        return NULL;
    }

    // 2) Using the information, start a proxy client (=server proxy, mtsDeviceInterfaceProxyClient object).
    clientTask->StartProxyClient(info.endpointInfo, info.communicatorID);

    //
    // TODO: Does ICE allow a user to register a callback function? (e.g. OnConnect())
    //       If it does, we can remove the following line.
    //
    osaSleep(1*cmn_s);

    // 3) From the server task, get the complete information on the provided 
    //    interface as a set of string.
    mtsDeviceInterfaceProxy::ProvidedInterfaceSequence providedInterfaces;
    if (!clientTask->GetProvidedInterfaces(providedInterfaces)) {
        CMN_LOG_CLASS_RUN_ERROR << "Connect across networks: failed to retrieve provided interface specification: " << resourceTaskName << ", " << providedInterfaceName << std::endl;
        return NULL;
    }

    // 4) Create a server task proxy that has a provided interface proxy.
    // 
    // TODO: MJUNG: this loop has to be refactored to remove duplicity.
    // (see mtsDeviceInterfaceProxyClient::ReceiveConnectServerSide())
    //    
    std::vector<mtsDeviceInterfaceProxy::ProvidedInterface>::const_iterator it
        = providedInterfaces.begin();
    for (; it != providedInterfaces.end(); ++it) {
        //
        //!!!!!!!!!!!!!!!!
        //
        //CMN_ASSERT(providedInterfaceName == it->InterfaceName);
        if (providedInterfaceName != it->InterfaceName) continue;

        // Create a server task proxy of which name follows the naming rule above.
        // (see mtsDeviceProxy.h as to why serverTaskProxy is of mtsDevice type, not
        // of mtsTask.)
        std::string serverTaskProxyName = mtsDeviceProxy::GetServerTaskProxyName(
            resourceTaskName, providedInterfaceName, userTaskName, requiredInterfaceName);
        mtsDeviceProxy * serverTaskProxy = new mtsDeviceProxy(serverTaskProxyName);

        // Create a provided interface proxy using the information received from the 
        // server task.
        if (!CreateProvidedInterfaceProxy(*it, serverTaskProxy, clientTask)) {
            CMN_LOG_CLASS_RUN_ERROR << "Connect across networks: failed to create a server task proxy: " << serverTaskProxyName << std::endl;
            return NULL;
        }

        // Add the proxy task to the local task manager
        if (!AddDevice(serverTaskProxy)) {
            CMN_LOG_CLASS_RUN_ERROR << "Connect across networks: failed to add a server task proxy: " << serverTaskProxyName << std::endl;
            return NULL;
        }

        // Return a pointer to the provided interface proxy as if the interface was initially
        // created in client's local memory space.
        resourceInterface = serverTaskProxy->GetProvidedInterface(providedInterfaceName);

        //
        // TODO: Currently, it is assumed that there is only one provided interface.
        //
        return resourceInterface;
    }

    // The following line should not be reached.
    CMN_ASSERT(false);

    return NULL;
}

bool mtsTaskManager::Disconnect(const std::string & userTaskName, const std::string & requiredInterfaceName,
                                const std::string & resourceTaskName, const std::string & providedInterfaceName)
{
    CMN_LOG_CLASS_RUN_ERROR << "Disconnect not implemented!!!" << std::endl;
    return true;
}

//
// TODO: Move this method to mtsDeviceInterfaceProxyServer class.
//
bool mtsTaskManager::CreateProvidedInterfaceProxy(
    const mtsDeviceInterfaceProxy::ProvidedInterface & providedInterface,
    mtsDevice * serverTaskProxy, mtsTask * clientTask)
{
    // 1) Create a local provided interface (a provided interface proxy).
    mtsDeviceInterface * providedInterfaceProxy = serverTaskProxy->AddProvidedInterface(providedInterface.InterfaceName);
    if (!providedInterfaceProxy) {
        CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: Could not add provided interface: " 
                                << providedInterface.InterfaceName << std::endl;
        return false;
    }

    // 2) Create command proxies.
    // CommandId is initially set to zero meaning that it needs to be updated.
    // An actual value will be assigned later when UpdateCommandId() is executed.
    int commandId = NULL;
    std::string commandName, eventName;

#define ADD_COMMANDS_BEGIN(_commandType) \
    {\
        mtsCommand##_commandType##Proxy * newCommand##_commandType = NULL;\
        mtsDeviceInterfaceProxy::Command##_commandType##Sequence::const_iterator it\
            = providedInterface.Commands##_commandType.begin();\
        for (; it != providedInterface.Commands##_commandType.end(); ++it) {\
            commandName = it->Name;
#define ADD_COMMANDS_END \
        }\
    }

    // 2-1) Void
    ADD_COMMANDS_BEGIN(Void)
        newCommandVoid = new mtsCommandVoidProxy(
            commandId, clientTask->GetProxyClient(), commandName);
        CMN_ASSERT(newCommandVoid);
        providedInterfaceProxy->GetCommandVoidMap().AddItem(commandName, newCommandVoid);
    ADD_COMMANDS_END

    // 2-2) Write
    ADD_COMMANDS_BEGIN(Write)
        newCommandWrite = new mtsCommandWriteProxy(
            commandId, clientTask->GetProxyClient(), commandName);
        CMN_ASSERT(newCommandWrite);
        providedInterfaceProxy->GetCommandWriteMap().AddItem(commandName, newCommandWrite);
    ADD_COMMANDS_END

    // 2-3) Read
    ADD_COMMANDS_BEGIN(Read)
        newCommandRead = new mtsCommandReadProxy(
            commandId, clientTask->GetProxyClient(), commandName);
        CMN_ASSERT(newCommandRead);
        providedInterfaceProxy->GetCommandReadMap().AddItem(commandName, newCommandRead);
    ADD_COMMANDS_END

    // 2-4) QualifiedRead
    ADD_COMMANDS_BEGIN(QualifiedRead)
        newCommandQualifiedRead = new mtsCommandQualifiedReadProxy(
            commandId, clientTask->GetProxyClient(), commandName);
        CMN_ASSERT(newCommandQualifiedRead);
        providedInterfaceProxy->GetCommandQualifiedReadMap().AddItem(commandName, newCommandQualifiedRead);
    ADD_COMMANDS_END

    //{
    //    mtsFunctionVoid * newEventVoidGenerator = NULL;
    //    mtsDeviceInterfaceProxy::EventVoidSequence::const_iterator it =
    //        providedInterface.EventsVoid.begin();
    //    for (; it != providedInterface.EventsVoid.end(); ++it) {
    //        eventName = it->Name;            
    //        newEventVoidGenerator = new mtsFunctionVoid();
    //        newEventVoidGenerator->Bind(providedInterfaceProxy->AddEventVoid(eventName));            
    //    }
    //}
#define ADD_EVENTS_BEGIN(_eventType)\
    {\
        mtsFunction##_eventType * newEvent##_eventType##Generator = NULL;\
        mtsDeviceInterfaceProxy::Event##_eventType##Sequence::const_iterator it =\
        providedInterface.Events##_eventType.begin();\
        for (; it != providedInterface.Events##_eventType.end(); ++it) {\
            eventName = it->Name;
#define ADD_EVENTS_END \
        }\
    }

    // 3) Create event generator proxies.
    ADD_EVENTS_BEGIN(Void);
        newEventVoidGenerator = new mtsFunctionVoid();
        newEventVoidGenerator->Bind(providedInterfaceProxy->AddEventVoid(eventName));
    ADD_EVENTS_END;
    
    mtsMulticastCommandWriteProxy * newMulticastCommandWriteProxy = NULL;
    ADD_EVENTS_BEGIN(Write);
        newEventWriteGenerator = new mtsFunctionWrite();
        newMulticastCommandWriteProxy = new mtsMulticastCommandWriteProxy(
            it->Name, it->ArgumentTypeName);
        CMN_ASSERT(providedInterfaceProxy->AddEvent(it->Name, newMulticastCommandWriteProxy));
        CMN_ASSERT(newEventWriteGenerator->Bind(newMulticastCommandWriteProxy));
    ADD_EVENTS_END;

#undef ADD_COMMANDS_BEGIN
#undef ADD_COMMANDS_END
#undef ADD_EVENTS_BEGIN
#undef ADD_EVENTS_END

    return true;
}

void mtsTaskManager::UpdateCommandId(mtsDeviceInterfaceProxy::FunctionProxySet functionProxies)
{
    const std::string serverTaskProxyName = functionProxies.ServerTaskProxyName;
    mtsDevice * serverTaskProxy = GetDevice(serverTaskProxyName);
    CMN_ASSERT(serverTaskProxy);

    mtsProvidedInterface * providedInterfaceProxy = 
        serverTaskProxy->GetProvidedInterface(functionProxies.ProvidedInterfaceProxyName);
    CMN_ASSERT(providedInterfaceProxy);

    //mtsCommandVoidProxy * commandVoid = NULL;
    //mtsDeviceInterfaceProxy::FunctionProxySequence::const_iterator it = 
    //    functionProxies.FunctionVoidProxies.begin();
    //for (; it != functionProxies.FunctionVoidProxies.end(); ++it) {
    //    commandVoid = dynamic_cast<mtsCommandVoidProxy*>(
    //        providedInterfaceProxy->GetCommandVoid(it->Name));
    //    CMN_ASSERT(commandVoid);
    //    commandVoid->SetCommandId(it->FunctionProxyPointer);
    //}

    // Replace a command id value with an actual pointer to the function
    // pointer at server side (this resolves thread synchronization issue).
#define REPLACE_COMMAND_ID(_commandType)\
    mtsCommand##_commandType##Proxy * command##_commandType = NULL;\
    mtsDeviceInterfaceProxy::FunctionProxySequence::const_iterator it##_commandType = \
        functionProxies.Function##_commandType##Proxies.begin();\
    for (; it##_commandType != functionProxies.Function##_commandType##Proxies.end(); ++it##_commandType) {\
        command##_commandType = dynamic_cast<mtsCommand##_commandType##Proxy*>(\
            providedInterfaceProxy->GetCommand##_commandType(it##_commandType->Name));\
        if (command##_commandType)\
            command##_commandType->SetCommandId(it##_commandType->FunctionProxyPointer);\
    }

    REPLACE_COMMAND_ID(Void);
    REPLACE_COMMAND_ID(Write);
    REPLACE_COMMAND_ID(Read);
    REPLACE_COMMAND_ID(QualifiedRead);
}

void mtsTaskManager::StartProxies()
{
    // Start the task manager proxy
    if (TaskManagerTypeMember == TASK_MANAGER_SERVER) {
        Proxy = new mtsTaskManagerProxyServer(
            "TaskManagerServerAdapter", "tcp -p 10705", TaskManagerCommunicatorID);
        ProxyServer = dynamic_cast<mtsTaskManagerProxyServer *>(Proxy);
        Proxy->Start(this);
    } else {
        CMN_LOG_CLASS_INIT_DEBUG << "GlobalTaskManagerIP: " << GlobalTaskManagerIP << std::endl;
        CMN_LOG_CLASS_INIT_DEBUG << "ServerTaskIP: " << ServerTaskIP << std::endl;

        Proxy = new mtsTaskManagerProxyClient(":default -h " + GlobalTaskManagerIP + " -p 10705", TaskManagerCommunicatorID);
        ProxyClient = dynamic_cast<mtsTaskManagerProxyClient *>(Proxy);
        Proxy->Start(this);

        //FIX:OPI
        // Start a task interface proxy. Currently it is assumed that there is only
        // one provided interface and one required interface.
        TaskMapType::MapType::const_iterator taskIterator = TaskMap.GetMap().begin();
        const TaskMapType::MapType::const_iterator taskEndIterator = TaskMap.GetMap().end();
        for (; taskIterator != taskEndIterator; ++taskIterator) {
            //taskIterator->second->StartInterfaceProxyServer(ServerTaskIP);
            taskIterator->second->StartProvidedInterfaceProxies(ServerTaskIP);
        }
    }
}

//-----------------------------------------------------------------------------
//  Task Manager Layer Processing
//-----------------------------------------------------------------------------
const bool mtsTaskManager::AddProvidedInterface(
    const std::string & newProvidedInterfaceName,
    const std::string & adapterName,
    const std::string & endpointInfo,
    const std::string & communicatorID,
    const std::string & taskName)
{
    return ProxyClient->SendAddProvidedInterface(
        newProvidedInterfaceName, adapterName, endpointInfo, communicatorID, taskName);
}

const bool mtsTaskManager::AddRequiredInterface(
    const std::string & newRequiredInterfaceName,const std::string & taskName)
{
    return ProxyClient->SendAddRequiredInterface(newRequiredInterfaceName, taskName);
}

const bool mtsTaskManager::IsRegisteredProvidedInterface(
    const std::string & taskName, const std::string & providedInterfaceName)
{
    return ProxyClient->SendIsRegisteredProvidedInterface(taskName, providedInterfaceName);
}

const bool mtsTaskManager::GetProvidedInterfaceInfo(
    const ::std::string & taskName,
    const std::string & providedInterfaceName,
    ::mtsTaskManagerProxy::ProvidedInterfaceInfo & info) const
{
    return ProxyClient->SendGetProvidedInterfaceInfo(taskName, providedInterfaceName, info);
}

//void mtsTaskManager::InvokeNotifyInterfaceConnectionResult(
//    const bool isServerTask, const bool isSuccess,
//    const std::string & userTaskName,     const std::string & requiredInterfaceName,
//    const std::string & resourceTaskName, const std::string & providedInterfaceName)
//{
//    ProxyClient->SendNotifyInterfaceConnectionResult(
//        isServerTask, isSuccess,
//        userTaskName, requiredInterfaceName,
//        resourceTaskName, providedInterfaceName);
//}

//-----------------------------------------------------------------------------
//  Task Layer Processing
//-----------------------------------------------------------------------------
bool mtsTaskManager::SendConnectServerSide(mtsTask * clientTask,
    const std::string & userTaskName, const std::string & requiredInterfaceName,
    const std::string & resourceTaskName, const std::string & providedInterfaceName)
{
    return clientTask->SendConnectServerSide(
        userTaskName, requiredInterfaceName, resourceTaskName, providedInterfaceName);
}