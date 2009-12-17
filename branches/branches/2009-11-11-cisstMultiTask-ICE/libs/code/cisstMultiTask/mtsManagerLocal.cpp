/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerLocal.cpp 978 2009-11-22 03:02:48Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-12-07

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstCommon/cmnSerializer.h>
#include <cisstOSAbstraction/osaSleep.h>
#include <cisstMultiTask/mtsManagerLocal.h>
#include <cisstMultiTask/mtsDevice.h>
//#include <cisstMultiTask/mtsDeviceInterface.h>
#include <cisstMultiTask/mtsTask.h>
#include <cisstMultiTask/mtsTaskContinuous.h>
#include <cisstMultiTask/mtsTaskPeriodic.h>
#include <cisstMultiTask/mtsTaskFromCallback.h>
//#include <cisstMultiTask/mtsTaskInterface.h>
#include <cisstMultiTask/mtsManagerGlobal.h>

#include <cisstMultiTask/mtsCommandVoidProxy.h>
#include <cisstMultiTask/mtsCommandWriteProxy.h>
#include <cisstMultiTask/mtsCommandReadProxy.h>
#include <cisstMultiTask/mtsCommandQualifiedReadProxy.h>
#include <cisstMultiTask/mtsMulticastCommandVoid.h>
#include <cisstMultiTask/mtsMulticastCommandWriteProxy.h>

#include <cisstMultiTask/mtsFunctionVoid.h>
#include <cisstMultiTask/mtsFunctionReadOrWrite.h>
#include <cisstMultiTask/mtsFunctionQualifiedReadOrWrite.h>

//#if CISST_MTS_HAS_ICE
//#include <cisstMultiTask/mtsDeviceProxy.h>
//#include <cisstMultiTask/mtsManagerLocalProxyServer.h>
//#include <cisstMultiTask/mtsManagerLocalProxyClient.h>
//#endif // CISST_MTS_HAS_ICE

CMN_IMPLEMENT_SERVICES(mtsManagerLocal);

mtsManagerLocal * mtsManagerLocal::Instance;

mtsManagerLocal::mtsManagerLocal(const std::string & thisProcessName, 
                                 const std::string & thisProcessIP) :
    ComponentMap("Components"),
    ManagerGlobal(NULL),
//    JGraphSocket(osaSocket::TCP),
//#if CISST_MTS_HAS_ICE
//    TaskManagerCommunicatorID("TaskManagerServerSender"),
//#endif
    ProcessName(thisProcessName),
    ProcessIP(thisProcessIP)
{
    __os_init();
    ComponentMap.SetOwner(*this);
    
    // TODO: The following line is commented out to speed up unit-tests.
    // (if it is enabled, then every time mtsManagerLocal object is created,
    // it runs again and again and again).
    //
    //  Don't Forget uncomment the following line after finishing implementation!!!
    //
    //TimeServer.SetTimeOrigin();

    //JGraphSocketConnected = false;

//#if CISST_MTS_HAS_ICE
//    TaskManagerTypeMember = TASK_MANAGER_LOCAL;    
//    ProxyGlobalTaskManager = 0;
//    ProxyTaskManagerClient = 0;
//#endif

    /*
    // Try to connect to the JGraph application software (Java program).
    // Note that the JGraph application also sends event messages back via the socket,
    // though we don't currently read them. To do this, it would be best to implement
    // the TaskManager as a periodic task.
    JGraphSocketConnected = JGraphSocket.Connect("127.0.0.1", 4444);
    if (JGraphSocketConnected) {
        osaSleep(1.0 * cmn_s);  // need to wait or JGraph server will not start properly
    } else {
        CMN_LOG_CLASS_INIT_WARNING << "Failed to connect to JGraph server" << std::endl;
    }
    */

    //
    // TODO: handle multiple network interfaces (how does a user choose which network 
    // interface is used?)
    //

    // If both arguments are provided, run this task manager in the network mode.
    // That is, an instance of mtsManagerGlobal acts as a proxy for the global 
    // manager and it connects to the global manager over a network.
    if ((thisProcessName != "") && (thisProcessIP != "")) {
        CMN_LOG_CLASS_INIT_VERBOSE << "Local component manager running in NETWORK mode" << std::endl;
        // TODO: create a global manager proxy
        // TODO: connect to the global manager
        //ManagerGlobal = new mtsManagerGlobalProxyClient;
    } 
    // If one of the arguments is missing, run this component manager in standalone 
    // mode. In this case, an instance of mtsManagerGlobalInterface becomes the 
    // global component manager.
    else {
        CMN_LOG_CLASS_INIT_VERBOSE << "Local component manager running in STANDALONE mode. " << std::endl;

        ManagerGlobal = new mtsManagerGlobal;

        if (!ManagerGlobal->AddProcess(this)) {
            CMN_LOG_CLASS_INIT_ERROR << "failed in registering default process" << std::endl;
        }
    }
}

mtsManagerLocal::~mtsManagerLocal()
{
    // If ManagerGlobal is not NULL, it means Cleanup() has not been called 
    // before. Thus, it needs to be called here to terminate safely and cleanly.
    if (ManagerGlobal) {
        Cleanup();
    }
}


void mtsManagerLocal::Cleanup(void)
{
    /*
#if CISST_MTS_HAS_ICE    // Clean up resources allocated for proxy objects.
    if (ProxyGlobalTaskManager) {
        ProxyGlobalTaskManager->Stop();
        osaSleep(200 * cmn_ms);
        delete ProxyGlobalTaskManager;
    }

    if (ProxyTaskManagerClient) {
        ProxyTaskManagerClient->Stop();
        osaSleep(200 * cmn_ms);
        delete ProxyTaskManagerClient;
    }
#endif

    JGraphSocket.Close();
    JGraphSocketConnected = false;
    */
    
    // Release global component manager
    if (ManagerGlobal) {
        // TODO: Add proxy (network) clean-up before delete
        delete ManagerGlobal;
        ManagerGlobal = NULL;
    }

    Kill();
}

mtsManagerLocal * mtsManagerLocal::GetInstance(
    const std::string & thisProcessName, const std::string & thisProcessIP)
{
    if (!Instance) {
        Instance = new mtsManagerLocal(thisProcessName, thisProcessIP);
    }

    return Instance;
}

bool mtsManagerLocal::AddComponent(mtsDevice * component) 
{
    if (!component) {
        CMN_LOG_CLASS_RUN_ERROR << "added component: " << "invalid argument" << std::endl;
        return false;
    }

    std::string componentName = component->GetName();

    // Try to register new component to the global component manager first.
    if (!ManagerGlobal->AddComponent(ProcessName, componentName)) {
        CMN_LOG_CLASS_RUN_ERROR << "failed to add component: " << componentName << std::endl;
        return false;
    }

    // Register all the existing required interfaces and provided interfaces to 
    // the global component manager.
    std::vector<std::string> interfaceNames = component->GetNamesOfRequiredInterfaces();
    for (unsigned int i = 0; i < interfaceNames.size(); ++i) {
        if (!ManagerGlobal->AddRequiredInterface(ProcessName, componentName, interfaceNames[i]))
        {
            CMN_LOG_CLASS_RUN_ERROR << "failed to add required interface: " 
                << componentName << ":" << interfaceNames[i] << std::endl;
            return false;
        }
    }

    interfaceNames = component->GetNamesOfProvidedInterfaces();
    for (unsigned int i = 0; i < interfaceNames.size(); ++i) {
        if (!ManagerGlobal->AddProvidedInterface(ProcessName, componentName, interfaceNames[i]))
        {
            CMN_LOG_CLASS_RUN_ERROR << "failed to add provided interface: " 
                << componentName << ":" << interfaceNames[i] << std::endl;
            return false;
        }
    }

    CMN_LOG_CLASS_RUN_VERBOSE << "Global component manager: added component: " << component->GetName() << std::endl;

    ComponentMapChange.Lock();
    bool result = ComponentMap.AddItem(component->GetName(), component, CMN_LOG_LOD_RUN_ERROR);
    if (result) {
        CMN_LOG_CLASS_INIT_VERBOSE << "added component: "
                                   << component->GetName() << std::endl;
        //if (JGraphSocketConnected) {
        //    std::string buffer = task->ToGraphFormat();
        //    CMN_LOG_CLASS_INIT_VERBOSE << "Sending " << buffer << std::endl;
        //    JGraphSocket.Send(buffer);
        //}
    }
    ComponentMapChange.Unlock();

    return result;
}

bool CISST_DEPRECATED mtsManagerLocal::AddTask(mtsTask * component)
{
    return AddComponent(component);
}

bool CISST_DEPRECATED mtsManagerLocal::AddDevice(mtsDevice * component)
{
    return AddComponent(component);
}

bool mtsManagerLocal::RemoveComponent(mtsDevice * component)
{
    if (!component) {
        CMN_LOG_CLASS_RUN_ERROR << "failed to remove component: invalid argument" << std::endl;
        return false;
    }

    return RemoveComponent(component->GetName());
}

bool mtsManagerLocal::RemoveComponent(const std::string & componentName)
{
    // Try to remove this component from the global component manager first.
    if (!ManagerGlobal->RemoveComponent(ProcessName, componentName)) {
        CMN_LOG_CLASS_RUN_ERROR << "failed to remove component: " << componentName << std::endl;
        return false;
    }

    ComponentMapChange.Lock();
    bool result = ComponentMap.RemoveItem(componentName, CMN_LOG_LOD_RUN_ERROR);
    if (result) {
        CMN_LOG_CLASS_INIT_VERBOSE << "removed component: " << componentName << std::endl;
    }
    ComponentMapChange.Unlock();

    return result;
}

std::vector<std::string> mtsManagerLocal::GetNamesOfComponents(void) const 
{
    return ComponentMap.GetNames();
}

std::vector<std::string> CISST_DEPRECATED mtsManagerLocal::GetNamesOfTasks(void) const 
{
    mtsDevice * component;
    std::vector<std::string> namesOfTasks;

    ComponentMapType::const_iterator it;
    const ComponentMapType::const_iterator itEnd = ComponentMap.end();
    for (; it != itEnd; ++it) {
        component = dynamic_cast<mtsTask*>(it->second);
        if (component) {
            namesOfTasks.push_back(it->first);
        }
    }

    return namesOfTasks;
}

std::vector<std::string> CISST_DEPRECATED mtsManagerLocal::GetNamesOfDevices(void) const 
{
    mtsDevice * component;
    std::vector<std::string> namesOfDevices;

    ComponentMapType::const_iterator it;
    const ComponentMapType::const_iterator itEnd = ComponentMap.end();
    for (; it != itEnd; ++it) {
        component = dynamic_cast<mtsTask*>(it->second);
        if (!component) {
            namesOfDevices.push_back(it->first);
        }
    }

    return namesOfDevices;
}

void mtsManagerLocal::GetNamesOfComponents(std::vector<std::string> & namesOfComponents) const 
{
    ComponentMap.GetNames(namesOfComponents);
}

void CISST_DEPRECATED mtsManagerLocal::GetNamesOfDevices(std::vector<std::string>& namesOfDevices) const
{
    mtsDevice * component;

    ComponentMapType::const_iterator it;
    const ComponentMapType::const_iterator itEnd = ComponentMap.end();
    for (; it != itEnd; ++it) {
        component = dynamic_cast<mtsTask*>(it->second);
        if (!component) {
            namesOfDevices.push_back(it->first);
        }
    }
}

void CISST_DEPRECATED mtsManagerLocal::GetNamesOfTasks(std::vector<std::string>& namesOfTasks) const
{
    mtsDevice * component;

    ComponentMapType::const_iterator it;
    const ComponentMapType::const_iterator itEnd = ComponentMap.end();
    for (; it != itEnd; ++it) {
        component = dynamic_cast<mtsTask*>(it->second);
        if (component) {
            namesOfTasks.push_back(it->first);
        }
    }
}

mtsDevice * mtsManagerLocal::GetComponent(const std::string & componentName) const
{
    return ComponentMap.GetItem(componentName, CMN_LOG_LOD_RUN_ERROR);
}

mtsTask CISST_DEPRECATED * mtsManagerLocal::GetTask(const std::string & taskName)
{
    mtsTask * componentTask = NULL;

    mtsDevice * component = ComponentMap.GetItem(taskName);
    if (component) {
        componentTask = dynamic_cast<mtsTask*>(component);
    }

    return componentTask;
}

mtsDevice CISST_DEPRECATED * mtsManagerLocal::GetDevice(const std::string & deviceName)
{
    return ComponentMap.GetItem(deviceName);
}
   
void mtsManagerLocal::ToStream(std::ostream & outputStream) const
{
    CMN_LOG_CLASS_RUN_VERBOSE << "ToStream() is not yet implemented" << std::endl;
}

void mtsManagerLocal::CreateAll(void) 
{
    mtsTask * componentTask;

    ComponentMapChange.Lock();

    ComponentMapType::const_iterator it = ComponentMap.begin();
    const ComponentMapType::const_iterator itEnd = ComponentMap.end();

    for (; it != itEnd; ++it) {
        // Skip components of mtsDevice type
        componentTask = dynamic_cast<mtsTask*>(it->second);
        if (!componentTask) continue;

        // Note that the order of dynamic casts does matter for figuring out 
        // the original task type because there are multiple inheritance 
        // relationships between task type components.

        // mtsTaskPeriodic type component
        componentTask = dynamic_cast<mtsTaskPeriodic*>(it->second);
        if (componentTask) {
            componentTask->Create();
            continue;
        }

        // mtsTaskContinuous type component
        componentTask = dynamic_cast<mtsTaskContinuous*>(it->second);
        if (componentTask) {
            componentTask->Create();
            continue;
        }

        // mtsTaskFromCallback type component
        componentTask = dynamic_cast<mtsTaskFromCallback*>(it->second);
        if (componentTask) {
            componentTask->Create();
            continue;
        }
    }

    ComponentMapChange.Unlock();
}

void mtsManagerLocal::StartAll(void) 
{
    // Get the current thread id so that we can check if any task will use the current thread.
    // If so, start that task last because its Start method will not return.
    const osaThreadId threadId = osaGetCurrentThreadId();
    
    mtsTask * componentTask, * componentTaskTemp;

    ComponentMapChange.Lock();

    ComponentMapType::const_iterator it = ComponentMap.begin();
    const ComponentMapType::const_iterator itEnd = ComponentMap.end();
    ComponentMapType::const_iterator itLastTask = ComponentMap.end();

    for (; it != ComponentMap.end(); ++it) {
        // Skip components of mtsDevice type
        componentTaskTemp = dynamic_cast<mtsTask*>(it->second);
        if (!componentTaskTemp) continue;

        // Note that the order of dynamic casts does matter for figuring out 
        // the original task type because there are multiple inheritance 
        // relationships between task type components.
        
        // mtsTaskPeriodic type component
        componentTaskTemp = dynamic_cast<mtsTaskPeriodic*>(it->second);
        if (componentTaskTemp) {
            componentTask = componentTaskTemp;
        } else {
            // mtsTaskContinuous type component
            componentTaskTemp = dynamic_cast<mtsTaskContinuous*>(it->second);            
            if (componentTaskTemp) {
                componentTask = componentTaskTemp;
            } else {
                // mtsTaskFromCallback type component
                componentTaskTemp = dynamic_cast<mtsTaskFromCallback*>(it->second);
                if (componentTaskTemp) {
                    componentTask = componentTaskTemp;
                } else {
                    componentTask = NULL;
                    CMN_LOG_CLASS_RUN_ERROR << "StartAll: invalid component: unknown mtsTask type" << std::endl;
                    continue;
                }
            }
        }

        // Check if the task will use the current thread.
        if (componentTask->Thread.GetId() == threadId) {
            CMN_LOG_CLASS_INIT_WARNING << "StartAll: component \"" << it->first << "\" uses current thread, will start last." << std::endl;
            if (itLastTask != ComponentMap.end()) {
                CMN_LOG_CLASS_RUN_ERROR << "StartAll: multiple tasks using current thread (only first will be started)." << std::endl;
            } else {
                itLastTask = it;
            }
        } else {
            CMN_LOG_CLASS_INIT_DEBUG << "StartAll: starting task \"" << it->first << "\"" << std::endl;
            componentTask->Start();  // If task will not use current thread, start it immediately.
        }
    }

    if (itLastTask != ComponentMap.end()) {
        // mtsTaskPeriodic type component
        componentTaskTemp = dynamic_cast<mtsTaskPeriodic*>(itLastTask->second);
        if (componentTaskTemp) {
            componentTask = componentTaskTemp;
        } else {
            // mtsTaskContinuous type component
            componentTaskTemp = dynamic_cast<mtsTaskContinuous*>(itLastTask->second);            
            if (componentTaskTemp) {
                componentTask = componentTaskTemp;
            } else {
                // mtsTaskFromCallback type component
                componentTaskTemp = dynamic_cast<mtsTaskFromCallback*>(itLastTask->second);
                if (componentTaskTemp) {
                    componentTask = componentTaskTemp;
                } else {
                    componentTask = NULL;
                    CMN_LOG_CLASS_RUN_ERROR << "StartAll: invalid component: unknown mtsTask type (last component)" << std::endl;
                }
            }
        }

        if (componentTask) {
            componentTask->Start();
        }
    }

    ComponentMapChange.Unlock();
}

void mtsManagerLocal::KillAll(void) 
{
    // It is not necessary to have any special handling of a task using the current thread.
    mtsTask *componentTask, *componentTaskTemp;
    
    ComponentMapChange.Lock();

    ComponentMapType::const_iterator it = ComponentMap.begin();
    const ComponentMapType::const_iterator itEnd = ComponentMap.end();
    for (; it != itEnd; ++it) {
        // mtsTaskPeriodic type component
        componentTaskTemp = dynamic_cast<mtsTaskPeriodic*>(it->second);
        if (componentTaskTemp) {
            componentTask = componentTaskTemp;
        } else {
            // mtsTaskContinuous type component
            componentTaskTemp = dynamic_cast<mtsTaskContinuous*>(it->second);            
            if (componentTaskTemp) {
                componentTask = componentTaskTemp;
            } else {
                // mtsTaskFromCallback type component
                componentTaskTemp = dynamic_cast<mtsTaskFromCallback*>(it->second);
                if (componentTaskTemp) {
                    componentTask = componentTaskTemp;
                } else {
                    componentTask = NULL;
                    CMN_LOG_CLASS_RUN_ERROR << "KillAll: invalid component: unknown mtsTask type" << std::endl;
                    continue;
                }
            }
        }
        componentTask->Kill();
    }

    ComponentMapChange.Unlock();
}

void mtsManagerLocal::ToStreamDot(std::ostream & outputStream) const {
    // NOP
}

//
//  TODO: Currently, we assume that at most one component (either a server or client) 
//  is missing. However, as noted in the project wiki, there can be a very special and
//  interesting case that a component with no local interfaces needs to use external 
//  interfaces. 
//  (see https://trac.lcsr.jhu.edu/cisst/wiki/Private/cisstMultiTaskNetwork)
//
//  Should this be possible, Connect() has to be updated to be able to handle that case.
//
bool mtsManagerLocal::Connect(const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
                              const std::string & serverComponentName, const std::string & serverProvidedInterfaceName)
{
    // Try to connect two local components
    unsigned int connectionID = ManagerGlobal->Connect(ProcessName,
        ProcessName, clientComponentName, clientRequiredInterfaceName,
        ProcessName, serverComponentName, serverProvidedInterfaceName);

    if (connectionID != static_cast<unsigned int>(mtsManagerGlobalInterface::CONNECT_LOCAL)) {
        CMN_LOG_CLASS_RUN_ERROR << "Connect: Global Component Manager failed to reserve connection: "
            << clientComponentName << ":" << clientRequiredInterfaceName << " - "
            << serverComponentName << ":" << serverProvidedInterfaceName << std::endl;
        return false;
    }

    return ConnectLocally(clientComponentName, clientRequiredInterfaceName, 
                          serverComponentName, serverProvidedInterfaceName);
}

bool mtsManagerLocal::Connect(
    const std::string & clientProcessName,
    const std::string & clientComponentName,
    const std::string & clientRequiredInterfaceName,
    const std::string & serverProcessName,
    const std::string & serverComponentName,
    const std::string & serverProvidedInterfaceName)
{
    unsigned int connectionID = ManagerGlobal->Connect(ProcessName,
        clientProcessName, clientComponentName, clientRequiredInterfaceName,
        serverProcessName, serverComponentName, serverProvidedInterfaceName);

    if (connectionID == static_cast<unsigned int>(mtsManagerGlobalInterface::CONNECT_ERROR)) {
        CMN_LOG_CLASS_RUN_ERROR << "Connect: Global Manager failed to reserve connection: "
            << clientProcessName << ":" << clientComponentName << ":" << clientRequiredInterfaceName << " - "
            << serverProcessName << ":" << serverComponentName << ":" << serverProvidedInterfaceName << std::endl;
        return false;
    }

    // TODO: before proceeding to the ConnectLocally(), proxy components should be
    // created by the GCM.

    return ConnectLocally(clientComponentName, clientRequiredInterfaceName, 
                          serverComponentName, serverProvidedInterfaceName, connectionID);
}

bool mtsManagerLocal::ConnectLocally(
    const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
    const std::string & serverComponentName, const std::string & serverProvidedInterfaceName,
    const unsigned int connectionID)
{
    // At this point, the connection can be established without validity check
    // because it is assumed that this method is called only after the global 
    // component manager has successfully confirmed the validity and existence 
    // of components and interfaces specified.
    mtsDevice * client = GetComponent(clientComponentName);    
    if (!client) {
        CMN_LOG_CLASS_RUN_ERROR << "Connect: failed to find client component: " << clientComponentName << std::endl;
        return false;
    }

    mtsDevice * server = GetComponent(serverComponentName);
    if (!server) {
        CMN_LOG_CLASS_RUN_ERROR << "Connect: failed to find server component: " << serverComponentName << std::endl;
        return false;
    }

    // Get the client component and the provided interface object.
    mtsProvidedInterface * serverProvidedInterface = server->GetProvidedInterface(serverProvidedInterfaceName);
    if (!serverProvidedInterface) {
        CMN_LOG_CLASS_RUN_ERROR << "Connect: failed to find provided interface: " 
            << serverComponentName << ":" << serverProvidedInterface << std::endl;
        return false;
    }

    // Connect two interfaces
    if (!client->ConnectRequiredInterface(clientRequiredInterfaceName, serverProvidedInterface)) {
        CMN_LOG_CLASS_RUN_ERROR << "Connect: failed to connect interfaces: " 
            << clientComponentName << ":" << clientRequiredInterfaceName << " - "
            << serverComponentName << ":" << serverProvidedInterfaceName << std::endl;
        return false;
    }

    CMN_LOG_CLASS_RUN_VERBOSE << "Connect: successfully connected: " 
            << clientComponentName << ":" << clientRequiredInterfaceName << " - "
            << serverComponentName << ":" << serverProvidedInterfaceName << std::endl;

    // Inform the global component manager of that the connection is successfully
    // established (otherwise, the global component manager disconnects this
    // connection after timeout).
    if (connectionID != static_cast<unsigned int>(mtsManagerGlobalInterface::CONNECT_LOCAL)) {
        if (!ManagerGlobal->ConnectConfirm(connectionID)) {
            CMN_LOG_CLASS_RUN_ERROR << "Connect: failed to confirm connection" << std::endl;
            return false;
        }
        CMN_LOG_CLASS_RUN_VERBOSE << "Connect: succeeded to confirm connection" << std::endl;
    }

    return true;
}

void mtsManagerLocal::Disconnect(const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
                                 const std::string & serverComponentName, const std::string & serverProvidedInterfaceName)
{
    ManagerGlobal->Disconnect(
        ProcessName, clientComponentName, clientRequiredInterfaceName,
        ProcessName, serverComponentName, serverProvidedInterfaceName);
}

void mtsManagerLocal::Disconnect(
    const std::string & clientProcessName,
    const std::string & clientComponentName,
    const std::string & clientRequiredInterfaceName,
    const std::string & serverProcessName,
    const std::string & serverComponentName,
    const std::string & serverProvidedInterfaceName)
{
    //
    // TODO: IMPLEMENT THIS
    //
}

bool mtsManagerLocal::GetProvidedInterfaceDescription(
    const std::string & componentName, const std::string & providedInterfaceName, 
    ProvidedInterfaceDescription & providedInterfaceDescription) const
{
    // Get the component instance specified
    mtsDevice * component = GetComponent(componentName);
    if (!component) {
        CMN_LOG_CLASS_RUN_ERROR << "GetProvidedInterfaceDescription: no component \""
            << componentName << "\" found in local component manager \"" << ProcessName << "\"" << std::endl;
        return false;
    }

    // Get the provided interface specified
    mtsDeviceInterface * providedInterface = component->GetProvidedInterface(providedInterfaceName);
    if (!providedInterface) {
        CMN_LOG_CLASS_RUN_ERROR << "GetProvidedInterfaceDescription: no provided interface \""
            << providedInterfaceName << "\" found in component \"" << componentName << "\"" << std::endl;
        return false;
    }

    // Extract all the information of the command objects or events registered.
    // Note that argument prototypes are returned serialized.
    providedInterfaceDescription.ProvidedInterfaceName = providedInterface->GetName();

    // Serializer initialization
    std::stringstream streamBuffer;
    cmnSerializer serializer(streamBuffer);

    // Extract void commands
    mtsDeviceInterface::CommandVoidMapType::MapType::const_iterator itVoid = 
        providedInterface->CommandsVoid.begin();
    mtsDeviceInterface::CommandVoidMapType::MapType::const_iterator itVoidEnd = 
        providedInterface->CommandsVoid.end();
    for (; itVoid != itVoidEnd; ++itVoid) {
        CommandVoidElement element;
        element.Name = itVoid->second->GetName();
        providedInterfaceDescription.CommandsVoid.push_back(element);
    }

    // Extract write commands
    mtsDeviceInterface::CommandWriteMapType::MapType::const_iterator itWrite = 
        providedInterface->CommandsWrite.begin();
    mtsDeviceInterface::CommandWriteMapType::MapType::const_iterator itWriteEnd = 
        providedInterface->CommandsWrite.end();
    for (; itWrite != itWriteEnd; ++itWrite) {
        CommandWriteElement element;
        element.Name = itWrite->second->GetName();
        // argument serialization
        streamBuffer.str("");
        serializer.Serialize(*(itWrite->second->GetArgumentPrototype()));
        element.ArgumentPrototypeSerialized = streamBuffer.str();
        providedInterfaceDescription.CommandsWrite.push_back(element);
    }

    // Extract read commands
    mtsDeviceInterface::CommandReadMapType::MapType::const_iterator itRead = 
        providedInterface->CommandsRead.begin();
    mtsDeviceInterface::CommandReadMapType::MapType::const_iterator itReadEnd = 
        providedInterface->CommandsRead.end();
    for (; itRead != itReadEnd; ++itRead) {
        CommandReadElement element;
        element.Name = itRead->second->GetName();
        // argument serialization
        streamBuffer.str("");
        serializer.Serialize(*(itRead->second->GetArgumentPrototype()));
        element.ArgumentPrototypeSerialized = streamBuffer.str();
        providedInterfaceDescription.CommandsRead.push_back(element);
    }

    // Extract qualified read commands
    mtsDeviceInterface::CommandQualifiedReadMapType::MapType::const_iterator itQualifiedRead = 
        providedInterface->CommandsQualifiedRead.begin();
    mtsDeviceInterface::CommandQualifiedReadMapType::MapType::const_iterator itQualifiedReadEnd = 
        providedInterface->CommandsQualifiedRead.end();
    for (; itQualifiedRead != itQualifiedReadEnd; ++itQualifiedRead) {
        CommandQualifiedReadElement element;
        element.Name = itQualifiedRead->second->GetName();
        // argument1 serialization
        streamBuffer.str("");
        serializer.Serialize(*(itQualifiedRead->second->GetArgument1Prototype()));
        element.Argument1PrototypeSerialized = streamBuffer.str();
        // argument2 serialization
        streamBuffer.str("");
        serializer.Serialize(*(itQualifiedRead->second->GetArgument2Prototype()));
        element.Argument2PrototypeSerialized = streamBuffer.str();
        providedInterfaceDescription.CommandsQualifiedRead.push_back(element);
    }

    // Extract void events
    mtsDeviceInterface::EventVoidMapType::MapType::const_iterator itEventVoid = 
        providedInterface->EventVoidGenerators.begin();
    mtsDeviceInterface::EventVoidMapType::MapType::const_iterator itEventVoidEnd = 
        providedInterface->EventVoidGenerators.end();
    for (; itEventVoid != itEventVoidEnd; ++itEventVoid) {
        EventVoidElement element;
        element.Name = itEventVoid->second->GetName();
        providedInterfaceDescription.EventsVoid.push_back(element);
    }

    // Extract write events
    mtsDeviceInterface::EventWriteMapType::MapType::const_iterator itEventWrite = 
        providedInterface->EventWriteGenerators.begin();
    mtsDeviceInterface::EventWriteMapType::MapType::const_iterator itEventWriteEnd = 
        providedInterface->EventWriteGenerators.end();
    for (; itEventWrite != itEventWriteEnd; ++itEventWrite) {
        EventWriteElement element;
        element.Name = itEventWrite->second->GetName();
        // argument serialization
        streamBuffer.str("");
        serializer.Serialize(*(itEventWrite->second->GetArgumentPrototype()));
        element.ArgumentPrototypeSerialized = streamBuffer.str();
        providedInterfaceDescription.EventsWrite.push_back(element);
    }

    return true;
}

bool mtsManagerLocal::GetRequiredInterfaceDescription(
    const std::string & componentName, const std::string & requiredInterfaceName, 
    RequiredInterfaceDescription & requiredInterfaceDescription) const
{
    // Get the component instance specified
    mtsDevice * component = GetComponent(componentName);
    if (!component) {
        CMN_LOG_CLASS_RUN_ERROR << "GetRequiredInterfaceDescription: no component \""
            << componentName << "\" found in local component manager \"" << ProcessName << "\"" << std::endl;
        return false;
    }

    // Get the provided interface specified
    mtsRequiredInterface * requiredInterface = component->GetRequiredInterface(requiredInterfaceName);
    if (!requiredInterface) {
        CMN_LOG_CLASS_RUN_ERROR << "GetRequiredInterfaceDescription: no provided interface \""
            << requiredInterfaceName << "\" found in component \"" << componentName << "\"" << std::endl;
        return false;
    }

    // Serializer initialization
    std::stringstream streamBuffer;
    cmnSerializer serializer(streamBuffer);

    // Extract void functions
    requiredInterfaceDescription.FunctionVoidNames = requiredInterface->GetNamesOfCommandPointersVoid();
    // Extract write functions
    requiredInterfaceDescription.FunctionWriteNames = requiredInterface->GetNamesOfCommandPointersWrite();
    // Extract read functions
    requiredInterfaceDescription.FunctionReadNames = requiredInterface->GetNamesOfCommandPointersRead();
    // Extract qualified read functions
    requiredInterfaceDescription.FunctionQualifiedReadNames = requiredInterface->GetNamesOfCommandPointersQualifiedRead();

    // Extract void event handlers
    mtsRequiredInterface::EventHandlerVoidMapType::MapType::const_iterator itVoid =
        requiredInterface->EventHandlersVoid.begin();
    mtsRequiredInterface::EventHandlerVoidMapType::MapType::const_iterator itVoidEnd =
        requiredInterface->EventHandlersVoid.end();
    for (; itVoid != itVoidEnd; ++itVoid) {
        CommandVoidElement element;
        element.Name = itVoid->second->GetName();
        requiredInterfaceDescription.EventHandlersVoid.push_back(element);
    }

    // Extract write event handlers
    mtsRequiredInterface::EventHandlerWriteMapType::MapType::const_iterator itWrite =
        requiredInterface->EventHandlersWrite.begin();
    mtsRequiredInterface::EventHandlerWriteMapType::MapType::const_iterator itWriteEnd =
        requiredInterface->EventHandlersWrite.end();
    for (; itWrite != itWriteEnd; ++itWrite) {
        CommandWriteElement element;
        element.Name = itWrite->second->GetName();
        // argument serialization
        streamBuffer.str("");
        serializer.Serialize(*(itWrite->second->GetArgumentPrototype()));
        element.ArgumentPrototypeSerialized = streamBuffer.str();
        requiredInterfaceDescription.EventHandlersWrite.push_back(element);
    }

    return true;
}

bool mtsManagerLocal::CreateProvidedInterfaceProxy(
    const std::string & componentName,
    ProvidedInterfaceDescription & providedInterfaceDescription) const
{
    // Get the component instance specified
    mtsDevice * component = GetComponent(componentName);
    if (!component) {
        CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: no component \""
            << componentName << "\" found in local component manager \"" << ProcessName << "\"" << std::endl;
        return false;
    }

    // Check if the name of the provided interface proxy is unique
    mtsProvidedInterface * providedInterface = 
        component->GetProvidedInterface(providedInterfaceDescription.ProvidedInterfaceName);
    if (providedInterface) {
        CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: can't create provided interface proxy: "
            << "duplicate name: " << providedInterfaceDescription.ProvidedInterfaceName << std::endl;
        return false;
    }

    //
    // TODO: If an original component is mtsTask, should a component proxy be always of mtsTask type? 
    //       Currently, it is of mtsDevice type because we assumed that there is only one 
    //
    // Create a local provided interface (a provided interface proxy)
    mtsProvidedInterface * providedInterfaceProxy = NULL;
    //    AddProvidedInterface(providedInterfaceInfo.InterfaceName);
    //if (!providedInterfaceProxy) {
    //    CMN_LOG_RUN_ERROR << "CreateProvidedInterfaceProxy: AddProvidedInterface failed." << std::endl;
    //    return NULL;
    //}

    // Create command proxies using the given description on the original
    // provided interface.
    // CommandId is initially set to zero and will be updated later by 
    // GetCommandId() for thread-safety.

    //
    //  TODO: GetCommandId() should be updated (or renamed)
    //

    // Note that argument prototypes passed in the description have been
    // serialized so it should be deserialized to recover and use orignial 
    // argument prototypes.
    std::string commandName;
    mtsGenericObject * argumentPrototype = NULL,
                     * argument1Prototype = NULL, 
                     * argument2Prototype = NULL;

    std::stringstream streamBuffer;
    cmnDeSerializer deserializer(streamBuffer);

    // Create void command proxies
    mtsCommandVoidProxy * newCommandVoid = NULL;
    CommandVoidVector::const_iterator itVoid = providedInterfaceDescription.CommandsVoid.begin();
    const CommandVoidVector::const_iterator itVoidEnd = providedInterfaceDescription.CommandsVoid.end();
    for (; itVoid != itVoidEnd; ++itVoid) {
        commandName = itVoid->Name;
        newCommandVoid = new mtsCommandVoidProxy(0, (mtsDeviceInterfaceProxyClient*) NULL /* TODO: UPDATE THIS PROXY POINTER!!! */, commandName);
        if (!providedInterfaceProxy->GetCommandVoidMap().AddItem(commandName, newCommandVoid)) {
            CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: failed to add void command proxy: " << commandName << std::endl;
            //
            // TODO: providedInterfaceProxy should be removed from 'component' because 
            // the integrity of the provided interface proxy is corrupted.
            //
            return false;
        }
    }

    // Create write command proxies
    mtsCommandWriteProxy * newCommandWrite = NULL;
    CommandWriteVector::const_iterator itWrite = providedInterfaceDescription.CommandsWrite.begin();
    const CommandWriteVector::const_iterator itWriteEnd = providedInterfaceDescription.CommandsWrite.end();
    for (; itWrite != itWriteEnd; ++itWrite) {
        commandName = itWrite->Name;
        newCommandWrite = new mtsCommandWriteProxy(0, (mtsDeviceInterfaceProxyClient*) NULL /* TODO: UPDATE THIS PROXY POINTER!!! */, commandName);
        if (!providedInterfaceProxy->GetCommandWriteMap().AddItem(commandName, newCommandWrite)) {
            CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: failed to add write command proxy: " << commandName << std::endl;
            //
            // TODO: providedInterfaceProxy should be removed from 'component' because 
            // the integrity of the provided interface proxy is corrupted.
            //
            return false;
        }

        // argument deserialization
        streamBuffer.str("");
        streamBuffer << itWrite->ArgumentPrototypeSerialized;
        argumentPrototype = dynamic_cast<mtsGenericObject *>(deserializer.DeSerialize());
        if (!argumentPrototype) {
            CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: failed to create write command proxy: " << commandName << std::endl;
            //
            // TODO: providedInterfaceProxy should be removed from 'component' because 
            // the integrity of the provided interface proxy is corrupted.
            //
            return false;
        }
        newCommandWrite->SetArgumentPrototype(argumentPrototype);
    }

    // Create read command proxies
    mtsCommandReadProxy * newCommandRead = NULL;
    CommandReadVector::const_iterator itRead = providedInterfaceDescription.CommandsRead.begin();
    const CommandReadVector::const_iterator itReadEnd = providedInterfaceDescription.CommandsRead.end();
    for (; itRead != itReadEnd; ++itRead) {
        commandName = itRead->Name;
        newCommandRead = new mtsCommandReadProxy(0, (mtsDeviceInterfaceProxyClient*) NULL, commandName);
        if (!providedInterfaceProxy->GetCommandReadMap().AddItem(commandName, newCommandRead)) {
            CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: failed to add read command proxy: " << commandName << std::endl;
            //
            // TODO: providedInterfaceProxy should be removed from 'component' because 
            // the integrity of the provided interface proxy is corrupted.
            //
            return false;
        }

        // argument deserialization
        streamBuffer.str("");
        streamBuffer << itRead->ArgumentPrototypeSerialized;
        argumentPrototype = dynamic_cast<mtsGenericObject *>(deserializer.DeSerialize());
        if (!argumentPrototype) {
            CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: failed to create read command proxy: " << commandName << std::endl;
            //
            // TODO: providedInterfaceProxy should be removed from 'component' because 
            // the integrity of the provided interface proxy is corrupted.
            //
            return false;
        }
        newCommandRead->SetArgumentPrototype(argumentPrototype);
    }

    // Create qualified read command proxies
    mtsCommandQualifiedReadProxy * newCommandQualifiedRead = NULL;
    CommandQualifiedReadVector::const_iterator itQualifiedRead = providedInterfaceDescription.CommandsQualifiedRead.begin();
    const CommandQualifiedReadVector::const_iterator itQualifiedReadEnd = providedInterfaceDescription.CommandsQualifiedRead.end();
    for (; itQualifiedRead != itQualifiedReadEnd; ++itQualifiedRead) {
        commandName = itQualifiedRead->Name;
        newCommandQualifiedRead = new mtsCommandQualifiedReadProxy(0, (mtsDeviceInterfaceProxyClient*) NULL, commandName);
        if (!providedInterfaceProxy->GetCommandQualifiedReadMap().AddItem(commandName, newCommandQualifiedRead)) {
            CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: failed to add qualified read command proxy: " << commandName << std::endl;
            //
            // TODO: providedInterfaceProxy should be removed from 'component' because 
            // the integrity of the provided interface proxy is corrupted.
            //
            return false;
        }

        // argument1 deserialization
        streamBuffer.str("");
        streamBuffer << itQualifiedRead->Argument1PrototypeSerialized;
        argument1Prototype = dynamic_cast<mtsGenericObject *>(deserializer.DeSerialize());        
        // argument2 deserialization
        streamBuffer.str("");
        streamBuffer << itQualifiedRead->Argument2PrototypeSerialized;
        argument2Prototype = dynamic_cast<mtsGenericObject *>(deserializer.DeSerialize());        
        if (!argument1Prototype || !argument2Prototype) {
            CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: failed to create qualified read command proxy: " << commandName << std::endl;
            //
            // TODO: providedInterfaceProxy should be removed from 'component' because 
            // the integrity of the provided interface proxy is corrupted.
            //
            return false;
        }
        newCommandQualifiedRead->SetArgumentPrototype(argument1Prototype, argument2Prototype);
    }

    // Create event generator proxies
    std::string eventName;

    // Create void event generator proxies
    mtsFunctionVoid * eventVoidGeneratorProxy = NULL;
    EventVoidVector::const_iterator itEventVoid = providedInterfaceDescription.EventsVoid.begin();
    const EventVoidVector::const_iterator itEventVoidEnd = providedInterfaceDescription.EventsVoid.end();
    for (; itEventVoid != itEventVoidEnd; ++itEventVoid) {
        eventName = itEventVoid->Name;
        eventVoidGeneratorProxy = new mtsFunctionVoid();
        //if (!EventVoidGeneratorProxyMap.AddItem(eventName, eventVoidGeneratorProxy)) {
        //    CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: failed to add void event proxy: " << eventName << std::endl;
        //    //
        //    // TODO: providedInterfaceProxy should be removed from 'component' because 
        //    // the integrity of the provided interface proxy is corrupted.
        //    //
        //    return false;
        //}
        
        if (!eventVoidGeneratorProxy->Bind(providedInterfaceProxy->AddEventVoid(eventName))) {
            CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: failed to bind with void event proxy: " << eventName << std::endl;
            //
            // TODO: providedInterfaceProxy should be removed from 'component' because 
            // the integrity of the provided interface proxy is corrupted.
            //
            return false;
        }
    }

    // Create write event generator proxies
    mtsFunctionWrite * eventWriteGeneratorProxy = NULL;
    mtsMulticastCommandWriteProxy * eventMulticastCommandProxy = NULL;

    EventWriteVector::const_iterator itEventWrite = providedInterfaceDescription.EventsWrite.begin();
    const EventWriteVector::const_iterator itEventWriteEnd = providedInterfaceDescription.EventsWrite.end();
    for (; itEventWrite != itEventWriteEnd; ++itEventWrite) {
        eventName = itEventWrite->Name;
        eventWriteGeneratorProxy = new mtsFunctionWrite();
        //if (!EventWriteGeneratorProxyMap.AddItem(eventName, eventWriteGeneratorProxy)) {
        //    CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: failed to add write event generator proxy: " << eventName << std::endl;
        //    //
        //    // TODO: providedInterfaceProxy should be removed from 'component' because 
        //    // the integrity of the provided interface proxy is corrupted.
        //    //
        //    return false;
        //}
        //
        eventMulticastCommandProxy = new mtsMulticastCommandWriteProxy(eventName);

        // event argument deserialization
        streamBuffer.str("");
        streamBuffer << itEventWrite->ArgumentPrototypeSerialized;
        argumentPrototype = dynamic_cast<mtsGenericObject *>(deserializer.DeSerialize());
        if (!argumentPrototype) {
            CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: failed to create write event proxy: " << commandName << std::endl;
            //
            // TODO: providedInterfaceProxy should be removed from 'component' because 
            // the integrity of the provided interface proxy is corrupted.
            //
            return false;
        }
        eventMulticastCommandProxy->SetArgumentPrototype(argumentPrototype);

        if (!providedInterfaceProxy->AddEvent(eventName, eventMulticastCommandProxy)) {
            CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: failed to add write event proxy: " << eventName << std::endl;
            //
            // TODO: providedInterfaceProxy should be removed from 'component' because 
            // the integrity of the provided interface proxy is corrupted.
            //
            return false;
        }
        if (!eventWriteGeneratorProxy->Bind(eventMulticastCommandProxy)) {
            CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: failed to bind with write event proxy: " << eventName << std::endl;
            //
            // TODO: providedInterfaceProxy should be removed from 'component' because 
            // the integrity of the provided interface proxy is corrupted.
            //
            return false;
        }
    }

    return false;
}

bool mtsManagerLocal::CreateRequiredInterfaceProxy(
    const std::string & componentName,
    RequiredInterfaceDescription & requiredInterfaceDescription) const
{
    // Get the component instance specified
    mtsDevice * component = GetComponent(componentName);
    if (!component) {
        CMN_LOG_CLASS_RUN_ERROR << "CreateRequiredInterfaceProxy: no component \""
            << componentName << "\" found in local component manager \"" << ProcessName << "\"" << std::endl;
        return false;
    }

    // Check if the name of the required interface proxy is unique
    mtsRequiredInterface * requiredInterface = 
        component->GetRequiredInterface(requiredInterfaceDescription.RequiredInterfaceName);
    if (requiredInterface) {
        CMN_LOG_CLASS_RUN_ERROR << "CreateRequiredInterfaceProxy: can't create required interface proxy: "
            << "duplicate name: " << requiredInterfaceDescription.RequiredInterfaceName << std::endl;
        return false;
    }

    // Create a local required interface (a required interface proxy)
    mtsRequiredInterface * requiredInterfaceProxy = NULL;
    //AddRequiredInterface(requiredInterfaceName);
    //if (!requiredInterfaceProxy) {
    //    CMN_LOG_RUN_ERROR << "CreateRequiredInterfaceProxy: Cannot add required interface: "
    //        << requiredInterfaceName << std::endl;
    //    return NULL;
    //}

    //
    // TODO: Update the following line
    //
    mtsProvidedInterface * providedInterface = NULL;

    // Populate the new required interface
    mtsFunctionVoid  * functionVoidProxy = NULL;
    mtsFunctionWrite * functionWriteProxy = NULL;
    mtsFunctionRead  * functionReadProxy = NULL;
    mtsFunctionQualifiedRead * functionQualifiedReadProxy = NULL;

    // Create void function proxies
    const std::vector<std::string> namesOfFunctionVoid = requiredInterfaceDescription.FunctionVoidNames;
    for (unsigned int i = 0; i < namesOfFunctionVoid.size(); ++i) {
        functionVoidProxy = new mtsFunctionVoid(providedInterface, namesOfFunctionVoid[i]);
        //
        // TODO: How to/where to define FunctionVoidProxyMap to store function pointers waiting for
        // being updated by the server task??
        //
        //CMN_ASSERT(FunctionVoidProxyMap.AddItem(namesOfFunctionVoid[i], functionVoidProxy)); 
        if (!requiredInterfaceProxy->AddFunction(namesOfFunctionVoid[i], *functionVoidProxy)) {
            CMN_LOG_CLASS_RUN_ERROR << "CreateRequiredInterfaceProxy: failed to add void function proxy: " << namesOfFunctionVoid[i] << std::endl;
            return false;
        }
    }

    // Create write function proxies
    const std::vector<std::string> namesOfFunctionWrite = requiredInterfaceDescription.FunctionWriteNames;
    for (unsigned int i = 0; i < namesOfFunctionWrite.size(); ++i) {
        functionWriteProxy = new mtsFunctionWrite(providedInterface, namesOfFunctionWrite[i]);
        //
        // TODO: How to/where to define FunctionWriteProxyMap to store function pointers waiting for
        // being updated by the server task??
        //
        //CMN_ASSERT(FunctionWriteProxyMap.AddItem(namesOfFunctionWrite[i], functionWriteProxy)); 
        if (!requiredInterfaceProxy->AddFunction(namesOfFunctionWrite[i], *functionWriteProxy)) {
            CMN_LOG_CLASS_RUN_ERROR << "CreateRequiredInterfaceProxy: failed to add write function proxy: " << namesOfFunctionWrite[i] << std::endl;
            return false;
        }
    }

    // Create read function proxies
    const std::vector<std::string> namesOfFunctionRead = requiredInterfaceDescription.FunctionReadNames;
    for (unsigned int i = 0; i < namesOfFunctionRead.size(); ++i) {
        functionReadProxy = new mtsFunctionRead(providedInterface, namesOfFunctionRead[i]);
        //
        // TODO: How to/where to define FunctionReadProxyMap to store function pointers waiting for
        // being updated by the server task??
        //
        //CMN_ASSERT(FunctionReadProxyMap.AddItem(namesOfFunctionRead[i], functionReadProxy)); 
        if (!requiredInterfaceProxy->AddFunction(namesOfFunctionRead[i], *functionReadProxy)) {
            CMN_LOG_CLASS_RUN_ERROR << "CreateRequiredInterfaceProxy: failed to add read function proxy: " << namesOfFunctionRead[i] << std::endl;
            return false;
        }
    }

    // Create QualifiedRead function proxies
    const std::vector<std::string> namesOfFunctionQualifiedRead = requiredInterfaceDescription.FunctionQualifiedReadNames;
    for (unsigned int i = 0; i < namesOfFunctionQualifiedRead.size(); ++i) {
        functionQualifiedReadProxy = new mtsFunctionQualifiedRead(providedInterface, namesOfFunctionQualifiedRead[i]);
        //
        // TODO: How to/where to define FunctionQualifiedReadProxyMap to store function pointers waiting for
        // being updated by the server task??
        //
        //CMN_ASSERT(FunctionQualifiedReadProxyMap.AddItem(namesOfFunctionQualifiedRead[i], functionQualifiedReadProxy)); 
        if (!requiredInterfaceProxy->AddFunction(namesOfFunctionQualifiedRead[i], *functionQualifiedReadProxy)) {
            CMN_LOG_CLASS_RUN_ERROR << "CreateRequiredInterfaceProxy: failed to add qualified read function proxy: " << namesOfFunctionQualifiedRead[i] << std::endl;
            return false;
        }
    }

    // Create event handler proxies
    std::string eventName;

    // Create event handler proxies with CommandId set to zero which will be 
    // updated later by UpdateEventHandlerId().
    //
    // TODO: CHECK THE FOLLOWING
    //
    // Note that all events created are disabled by default. Commands that are
    // actually bounded and used at the client will only be enabled by the
    // execution of UpdateEventHandlerId() method.

    // Create void event handler proxy
    mtsCommandVoidProxy * actualEventVoidCommandProxy = NULL;
    std::vector<std::string> namesOfEventsVoid = providedInterface->GetNamesOfEventsVoid();
    for (unsigned int i = 0; i < namesOfEventsVoid.size(); ++i) {
        eventName = namesOfEventsVoid[i];
        actualEventVoidCommandProxy = new mtsCommandVoidProxy(0, /* TODO: UPDATE */ (mtsDeviceInterfaceProxyServer*) NULL, eventName);
        actualEventVoidCommandProxy->Disable();

        if (!requiredInterfaceProxy->EventHandlersVoid.AddItem(eventName, actualEventVoidCommandProxy)) {
            CMN_LOG_CLASS_RUN_ERROR << "CreateRequiredInterfaceProxy: failed to add void event handler proxy: " << eventName << std::endl;
            return false;
        }
        //
        // TODO: How to handle/Where to define EventHandlerVoidProxyMap???
        //
        //CMN_ASSERT(EventHandlerVoidProxyMap.AddItem(eventName, actualEventVoidCommandProxy));
    }

    // Create write event handler proxy
    mtsCommandWriteProxy * actualEventWriteCommandProxy = NULL;    
    std::vector<std::string> namesOfEventsWrite = providedInterface->GetNamesOfEventsWrite();
    for (unsigned int i = 0; i < namesOfEventsWrite.size(); ++i) {
        eventName = namesOfEventsWrite[i];
        actualEventWriteCommandProxy = new mtsCommandWriteProxy(0, /* TODO: UPDATE */ (mtsDeviceInterfaceProxyServer*) NULL, eventName);
        actualEventWriteCommandProxy->Disable();

        if (!requiredInterfaceProxy->EventHandlersWrite.AddItem(eventName, actualEventWriteCommandProxy)) {
            CMN_LOG_CLASS_RUN_ERROR << "CreateRequiredInterfaceProxy: failed to add write event handler proxy: " << eventName << std::endl;
            return false;
        }

        //
        // TODO: How to handle/Where to define EventHandlerVoidProxyMap???
        //
        //CMN_ASSERT(EventHandlerWriteProxyMap.AddItem(eventName, actualEventWriteCommandProxy));
    }

    // Using AllocateResources(), get pointers which have been allocated for this 
    // required interface and are thread-safe to use.
    unsigned int userId;
    std::string userName = requiredInterfaceProxy->GetName() + "Proxy";
    userId = providedInterface->AllocateResources(userName);

    // Connect to the original device or task that provides allocated resources.
    requiredInterfaceProxy->ConnectTo(providedInterface);
    if (!requiredInterfaceProxy->BindCommandsAndEvents(userId)) {
        CMN_LOG_RUN_ERROR << "CreateRequiredInterfaceProxy: BindCommandsAndEvents failed: userName="
            << userName << ", userId=" << userId << std::endl;
        return false;
    }

    return true;
}