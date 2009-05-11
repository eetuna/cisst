/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Ankur Kapoor, Peter Kazanzides
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
#include <cisstMultiTask/mtsCommandQueuedWriteProxy.h>

CMN_IMPLEMENT_SERVICES(mtsTaskManager);

mtsTaskManager::mtsTaskManager(void) : 
    TaskMap("Task"), DeviceMap("Device"),
    TaskManagerCommunicatorID("TaskManagerServerSender"),
    TaskManagerTypeMember(TASK_MANAGER_LOCAL),
    Proxy(0), ProxyServer(0), ProxyClient(0)
{
    __os_init();
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
    bool ret = TaskMap.AddItem(task->GetName(), task, 1);
    if (ret)
       CMN_LOG_CLASS(3) << "AddTask: added task named "
                        << task->GetName() << std::endl;
    return ret;
}


bool mtsTaskManager::RemoveTask(mtsTask * task) {
    // MJUNG: TODO: This very simple implementation considers TaskMap only.
    // There are much more things to be done if we want RemoveTask() to work
    // correctly because removing a task is tightly coupled with the CISST
    // multithreaded and object-oriented architecture.
    bool ret = TaskMap.RemoveItem(task->GetName(), 1);
    if (ret)
       CMN_LOG_CLASS(3) << "RemoveTask: removed task named "
                        << task->GetName() << std::endl;
    return ret;
}


bool mtsTaskManager::AddDevice(mtsDevice * device) {
    mtsTask * task = dynamic_cast<mtsTask *>(device);
    if (task) {
        CMN_LOG_CLASS(1) << "AddDevice: Attempt to add " << task->GetName() << "as a device (use AddTask instead)."
                         << std::endl;
        return false;
    }
    bool ret = DeviceMap.AddItem(device->GetName(), device, 1);
    if (ret)
        CMN_LOG_CLASS(3) << "AddDevice: added device named "
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

void mtsTaskManager::GetNamesOfTasks(std::vector<std::string>& taskNameContainer) const
{
    return TaskMap.GetNames(taskNameContainer);
}


mtsDevice * mtsTaskManager::GetDevice(const std::string & deviceName) {
    return DeviceMap.GetItem(deviceName, 1);
}


mtsTask * mtsTaskManager::GetTask(const std::string & taskName) {
    return TaskMap.GetItem(taskName, 1);
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
            CMN_LOG_CLASS(5) << "StartAll: task " << taskIterator->first << " uses current thread, will start last." << std::endl;
            if (lastTask != TaskMap.GetMap().end())
                CMN_LOG_CLASS(1) << "WARNING: multiple tasks using current thread (only first will be started)." << std::endl;
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


bool mtsTaskManager::Connect(const std::string & userTaskName, const std::string & interfaceRequiredName,
                             const std::string & resourceTaskName, const std::string & providedInterfaceName)
{
    // True if the resource task specified is provided by a remote task
    bool RemoteConnect = false;

    const UserType fullUserName(userTaskName, interfaceRequiredName);
    const ResourceType fullResourceName(resourceTaskName, providedInterfaceName);
    const AssociationType association(fullUserName, fullResourceName);
    // check if this connection has already been established
    AssociationSetType::const_iterator associationIterator = AssociationSet.find(association);
    if (associationIterator != AssociationSet.end()) {
        CMN_LOG_CLASS(1) << "Connect: " << userTaskName << "::" << interfaceRequiredName
                         << " is already connected to " << resourceTaskName << "::" << providedInterfaceName << std::endl;
        return false;
    }
    // check that names are not the same
    if (userTaskName == resourceTaskName) {
        CMN_LOG_CLASS(1) << "Connect: can not connect two tasks/devices with the same name" << std::endl;
        return false;
    }
    // check if the user name corresponds to an existing task
    mtsTask* userTask = TaskMap.GetItem(userTaskName);
    if (!userTask) {
        CMN_LOG_CLASS(1) << "Connect: can not find a task named " << userTaskName << std::endl;
        return false;
    }
    // check if the resource name corresponds to an existing task or device
    mtsDevice* resourceDevice = DeviceMap.GetItem(resourceTaskName);
    if (!resourceDevice) {        
        resourceDevice = TaskMap.GetItem(resourceTaskName);
    }
    // find the interface pointer from the resource
    mtsDeviceInterface * resourceInterface;
    if (resourceDevice)
        resourceInterface = resourceDevice->GetProvidedInterface(providedInterfaceName);
    else {
        if (GetTaskManagerType() == TASK_MANAGER_LOCAL) {
            CMN_LOG_CLASS(1) << "Connect: can not find a task or device named " << resourceTaskName << std::endl;
            return false;
        } else {            
            resourceInterface = GetResourceInterface(resourceTaskName, providedInterfaceName, 
                                    userTaskName, interfaceRequiredName, 
                                    userTask);
            if (!resourceInterface)
            {
                CMN_LOG_CLASS(1) << "Connect through networks: can not find a task or device named " << resourceTaskName << std::endl;
                return false;                
            }

            RemoteConnect = true;
        }
    }

    // check the interface pointer we got
    if (resourceInterface == 0) {
        CMN_LOG_CLASS(1) << "Connect: interface pointer for "
                         << resourceTaskName << "::" << providedInterfaceName << " is null" << std::endl;
        return false;
    }
    // attempt to connect 
    if (!(userTask->ConnectRequiredInterface(interfaceRequiredName, resourceInterface))) {
        CMN_LOG_CLASS(1) << "Connect: connection failed, does " << interfaceRequiredName << " exist?" << std::endl;
        return false;
    }

    // After creating a required interface proxy at the server task, try to connect it with 
    // a provided interface at server side.
    if (RemoteConnect) {
        if (!ConnectAtServerSide(interfaceRequiredName, providedInterfaceName)) {
            CMN_LOG_CLASS(1) << "Connect: connection at server side failed: " 
                << interfaceRequiredName << "(req) - " 
                << resourceInterface << "(prv)" << std::endl;
            return false;
        }
    }

    // connected, add to the map of connections
    AssociationSet.insert(association);
    CMN_LOG_CLASS(3) << "Connect: " << userTaskName << "::" << interfaceRequiredName
                     << " successfully connected to " << resourceTaskName << "::" << providedInterfaceName << std::endl;
    return true;
}

mtsDeviceInterface * mtsTaskManager::GetResourceInterface(
    const std::string & resourceTaskName, const std::string & providedInterfaceName,
    const std::string & userTaskName, const std::string & interfaceRequiredName,
    mtsTask * userTask)
{
    mtsDeviceInterface * resourceInterface = NULL;

    // For the use of consistent notation
    mtsTask * clientTask = userTask;

    // Ask the global task manager (TMServer) if the specified task providing
    // the specific provided interface has been registered.
    if (InvokeIsRegisteredProvidedInterface(resourceTaskName, providedInterfaceName))            
    {
        // If (task, provided interface) exists,
        // 1) Retrieve information from the global task manager to connect
        //    the requested provided interface (mtsDeviceInterfaceProxyServer).                
        mtsTaskManagerProxy::ProvidedInterfaceInfo info;
        if (!InvokeGetProvidedInterfaceInfo(resourceTaskName, providedInterfaceName, info)) {
            CMN_LOG_CLASS(1) << "Connect over networks: failed to retrieve proxy information: " << resourceTaskName << ", " << providedInterfaceName << std::endl;
            return NULL;
        }

        // 2) Using the information, start a proxy client (=server proxy, mtsDeviceInterfaceProxyClient object).
        clientTask->StartProxyClient(info.endpointInfo, info.communicatorID);

        // 3) From the interface proxy server, get the complete information on the provided 
        //    interface as a set of string.
        mtsDeviceInterfaceProxy::ProvidedInterfaceSpecificationSeq specs;
        if (!clientTask->GetProvidedInterfaceSpecification(specs)) {
            CMN_LOG_CLASS(1) << "Connect over networks: failed to retrieve provided interface specification: " << resourceTaskName << ", " << providedInterfaceName << std::endl;
            return NULL;
        }

        // 4) Extract and present the complete information on this provided interface 
        // as a set of string.
        std::string serverProxyName;
        std::vector<mtsDeviceInterfaceProxy::ProvidedInterfaceSpecification>::const_iterator it
            = specs.begin();
        for (; it != specs.end(); ++it) {
            //
            // TODO: handle a case that there are multiple provided interfaces at server task.
            //
            CMN_ASSERT(providedInterfaceName == it->interfaceName);

            /* ServerTaskProxy naming rule:
                
                TS:PI-Network-TC:RI

               where TS: the name of the server task
                     PI: the name of the provided interface
                     TC: the name of the client task
                     RI: the name of the required interface
            */
            serverProxyName = resourceTaskName;
            //serverProxyName = resourceTaskName + ":"          // TS
            //                      it->interfaceName + "-Network-" // PI
            //                      userTaskName + ":"              // TC
            //                      interfaceRequiredName;          // RI

            mtsDeviceProxy * serverTaskProxy = new mtsDeviceProxy(serverProxyName);
            CMN_ASSERT(serverTaskProxy);

            if (!CreateProvidedInterfaceProxy(*it, serverTaskProxy, clientTask)) {
                CMN_LOG_CLASS(1) << "CreateProvidedInterfaceProxy FAILED: " << serverProxyName << std::endl;
                return NULL;
            }

            // Add this proxy task to the task manager
            //if (!AddTask(serverTaskProxy)) {
            if (!AddDevice(serverTaskProxy)) {
                CMN_LOG_CLASS(1) << "CreateProvidedInterfaceProxy: Adding task failed: " << serverProxyName << std::endl;
                return NULL;
            }

            resourceInterface = serverTaskProxy->GetProvidedInterface(providedInterfaceName);

            //
            // TODO: Currently, it is assumed that there is only one provided interface.
            //
            return resourceInterface;
        }
    }

    return NULL;
}

bool mtsTaskManager::Disconnect(const std::string & userTaskName, const std::string & requiredInterfaceName,
                                const std::string & resourceTaskName, const std::string & providedInterfaceName) {
    CMN_LOG_CLASS(1) << "Disconnect not implemented!!!" << std::endl;
    return true;
}

bool mtsTaskManager::CreateProvidedInterfaceProxy(
    const mtsDeviceInterfaceProxy::ProvidedInterfaceSpecification & spec,
    mtsDevice * serverTaskProxy, mtsTask * clientTask)
{
    // 1) Create a local provided interface (a provided interface proxy)
    if (!serverTaskProxy->AddProvidedInterface(spec.interfaceName)) {
        CMN_LOG_CLASS(1) << "CreateProvidedInterfaceProxy: Could not add provided interface." << std::endl;
        return false;
    }

    // 2) Restore Commands by creating command proxies specified by the provided interface
    mtsDeviceInterface * providedInterface = serverTaskProxy->GetProvidedInterface(spec.interfaceName);
    CMN_ASSERT(providedInterface);

    std::string commandName;
    int commandSID;

#define ITERATE_INTERFACE_BEGIN( _commandType ) \
    {\
        mtsDeviceInterfaceProxy::Command##_commandType##Seq::const_iterator it \
            = spec.commands##_commandType##.begin();\
        for (; it != spec.commands##_commandType##.end(); ++it) {\
            commandName = it->Name;\
            commandSID = it->CommandSID;

#define ITERATE_INTERFACE_END \
        }\
    }

    // 2-1) Void
    ITERATE_INTERFACE_BEGIN(Void)
        mtsCommandVoidProxy * newCommandVoid = new mtsCommandVoidProxy(
            commandSID, clientTask->GetProxyClient(), commandName);
        CMN_ASSERT(newCommandVoid);
        providedInterface->GetCommandVoidMap().AddItem(it->Name, newCommandVoid, 1);
    ITERATE_INTERFACE_END

    // 2-2) Write
    ITERATE_INTERFACE_BEGIN(Write)
        //cmnGenericObject * prototype = cmnClassRegister::Create(it->ArgumentTypeName);
        mtsCommandWriteProxy * newCommandWrite = new mtsCommandWriteProxy(
            commandSID, clientTask->GetProxyClient(), commandName);
        CMN_ASSERT(newCommandWrite);
        providedInterface->GetCommandWriteMap().AddItem(it->Name, newCommandWrite, 1);
    ITERATE_INTERFACE_END

    // 2-3) Read
    ITERATE_INTERFACE_BEGIN(Read)
        //cmnGenericObject * prototype = cmnClassRegister::Create(it->ArgumentTypeName);
        mtsCommandReadProxy * newCommandRead = new mtsCommandReadProxy(
            commandSID, clientTask->GetProxyClient(), commandName);
        CMN_ASSERT(newCommandRead);
        providedInterface->GetCommandReadMap().AddItem(it->Name, newCommandRead, 1);
    ITERATE_INTERFACE_END

    // 2-4) QualifiedRead
    //ITERATE_INTERFACE_BEGIN(QualifiedRead)
    //    cmnGenericObject * prototype1 = cmnClassRegister::Create(it->Argument1TypeName);
    //    cmnGenericObject * prototype2 = cmnClassRegister::Create(it->Argument2TypeName);
    //    mtsCommandQualifiedReadProxy * newCommandQualifiedRead = 
    //        new mtsCommandQualifiedReadProxy(commandName, prototype1, prototype2);
    //    CMN_ASSERT(newCommandQualifiedRead);
    //    providedInterface->CommandsQualifiedRead.AddItem(commandName, newCommandQualifiedRead);
    //ITERATE_INTERFACE_END

#undef ITERATE_INTERFACE_BEGIN
#undef ITERATE_INTERFACE_END

    // TODO:
    //
    // 3) Restore Events
    //

    return true;
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
        Proxy = new mtsTaskManagerProxyClient(":default -h " + GlobalTaskManagerIP + " -p 10705", TaskManagerCommunicatorID);
        ProxyClient = dynamic_cast<mtsTaskManagerProxyClient *>(Proxy);
        Proxy->Start(this);

        osaSleep(500 * cmn_ms);

        // Start a task interface proxy. Currently it is assumed that there is only
        // one provided interface and one required interface.
        TaskMapType::MapType::const_iterator taskIterator = TaskMap.GetMap().begin();
        const TaskMapType::MapType::const_iterator taskEndIterator = TaskMap.GetMap().end();
        for (; taskIterator != taskEndIterator; ++taskIterator) {
            taskIterator->second->StartInterfaceProxyServer(ServerTaskIP);
        }
    }
}

const bool mtsTaskManager::ConnectAtServerSide(const std::string requiredInterfaceName,
                                               const std::string providedInterfaceName)
{
    // 1. Create a required interface proxy at server side.
    // 2. Connect the required interface proxy with the provided interface specified.
    // 3. Inform the global task manager of the fact that two interfaces are connected.

    //
    //  TODO: IMPLEMENT ME
    //
    return true;
}

//
//  Task Manager Layer Processing
//
const bool mtsTaskManager::InvokeAddProvidedInterface(
    const std::string & newProvidedInterfaceName,
    const std::string & adapterName,
    const std::string & endpointInfo,
    const std::string & communicatorID,
    const std::string & taskName)
{
    return ProxyClient->AddProvidedInterface(
        newProvidedInterfaceName, adapterName, endpointInfo, communicatorID, taskName);
}

const bool mtsTaskManager::InvokeAddRequiredInterface(
    const std::string & newRequiredInterfaceName,const std::string & taskName)
{
    return ProxyClient->AddRequiredInterface(newRequiredInterfaceName, taskName);
}

const bool mtsTaskManager::InvokeIsRegisteredProvidedInterface(
    const std::string & taskName, const std::string & providedInterfaceName)
{
    return ProxyClient->IsRegisteredProvidedInterface(taskName, providedInterfaceName);
}

const bool mtsTaskManager::InvokeGetProvidedInterfaceInfo(
    const ::std::string & taskName,
    const std::string & providedInterfaceName,
    ::mtsTaskManagerProxy::ProvidedInterfaceInfo & info) const
{
    return ProxyClient->GetProvidedInterfaceInfo(taskName, providedInterfaceName, info);
}

//
//  Task Layer Processing
//
