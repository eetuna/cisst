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

#include <cisstOSAbstraction/osaSleep.h>
#include <cisstMultiTask/mtsManagerLocal.h>
#include <cisstMultiTask/mtsManagerGlobal.h>
#include <cisstMultiTask/mtsComponentProxy.h>
#include <cisstMultiTask/mtsTaskContinuous.h>
#include <cisstMultiTask/mtsTaskPeriodic.h>
#include <cisstMultiTask/mtsTaskFromCallback.h>

#include <cisstMultiTask/mtsCommandVoidProxy.h>
#include <cisstMultiTask/mtsCommandWriteProxy.h>
#include <cisstMultiTask/mtsCommandReadProxy.h>
#include <cisstMultiTask/mtsCommandQualifiedReadProxy.h>
#include <cisstMultiTask/mtsMulticastCommandVoid.h>
#include <cisstMultiTask/mtsMulticastCommandWriteProxy.h>

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

mtsManagerLocal::mtsManagerLocal(const std::string & thisProcessName) :
    ComponentMap("Components"),
    ManagerGlobal(NULL),
    ProcessName(thisProcessName),
    ProcessIP("")
{
    __os_init();
    ComponentMap.SetOwner(*this);
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

    ComponentMap.DeleteAll();

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
        CMN_LOG_CLASS_RUN_ERROR << "AddComponent: invalid component" << std::endl;
        return false;
    }

    const std::string componentName = component->GetName();

    // Try to register new component to the global component manager first.
    if (!ManagerGlobal->AddComponent(ProcessName, componentName)) {
        CMN_LOG_CLASS_RUN_ERROR << "AddComponent: failed to add component: " << componentName << std::endl;
        return false;
    }

    // Register all the existing required interfaces and provided interfaces to 
    // the global component manager.
    std::vector<std::string> interfaceNames = component->GetNamesOfRequiredInterfaces();
    for (unsigned int i = 0; i < interfaceNames.size(); ++i) {
        if (!ManagerGlobal->AddRequiredInterface(ProcessName, componentName, interfaceNames[i], false))
        {
            CMN_LOG_CLASS_RUN_ERROR << "AddComponent: failed to add required interface: " 
                << componentName << ":" << interfaceNames[i] << std::endl;
            return false;
        }
    }
 
    interfaceNames = component->GetNamesOfProvidedInterfaces();
    for (unsigned int i = 0; i < interfaceNames.size(); ++i) {
        if (!ManagerGlobal->AddProvidedInterface(ProcessName, componentName, interfaceNames[i], false))
        {
            CMN_LOG_CLASS_RUN_ERROR << "AddComponent: failed to add provided interface: " 
                << componentName << ":" << interfaceNames[i] << std::endl;
            return false;
        }
    }

    CMN_LOG_CLASS_RUN_VERBOSE << "AddComponent: "
        << "successfully added component to the global component manager: " << componentName << std::endl;

    bool success;
    ComponentMapChange.Lock();
    success = ComponentMap.AddItem(componentName, component);
    ComponentMapChange.Unlock();

    if (!success) {
        CMN_LOG_CLASS_RUN_ERROR << "AddComponent: "
            << "failed to add component to local component manager: " << componentName << std::endl;
        return false;
    }

    CMN_LOG_CLASS_RUN_VERBOSE << "AddComponent: "
        << "successfully added component to local component manager: " << componentName << std::endl;

    return true;
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
        CMN_LOG_CLASS_RUN_ERROR << "RemoveComponent: invalid argument" << std::endl;
        return false;
    }

    return RemoveComponent(component->GetName());
}

bool mtsManagerLocal::RemoveComponent(const std::string & componentName)
{
    // Notify the global component manager of the removal of this component
    if (!ManagerGlobal->RemoveComponent(ProcessName, componentName)) {
        CMN_LOG_CLASS_RUN_ERROR << "RemoveComponent: failed to remove component at global component manager: " << componentName << std::endl;
        return false;
    }

    bool success;
    ComponentMapChange.Lock();
    success = ComponentMap.RemoveItem(componentName);
    ComponentMapChange.Unlock();

    if (!success) {
        CMN_LOG_CLASS_RUN_ERROR << "RemoveComponent: failed to removed component: " << componentName << std::endl;
        return false;
    }
    
    CMN_LOG_CLASS_RUN_ERROR << "RemoveComponent: removed component: " << componentName << std::endl;

    return true;
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

const bool mtsManagerLocal::FindComponent(const std::string & componentName) const
{
    return (GetComponent(componentName) != NULL);
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
    unsigned int connectionID = ManagerGlobal->Connect(
        ProcessName, clientComponentName, clientRequiredInterfaceName,
        ProcessName, serverComponentName, serverProvidedInterfaceName);

    if (connectionID != static_cast<unsigned int>(mtsManagerGlobalInterface::CONNECT_LOCAL)) {
        CMN_LOG_CLASS_RUN_ERROR << "Connect: Global Component Manager failed to reserve connection: "
            << clientComponentName << ":" << clientRequiredInterfaceName << " - "
            << serverComponentName << ":" << serverProvidedInterfaceName << std::endl;
        return false;
    }

    return ConnectLocally(ProcessName, clientComponentName, clientRequiredInterfaceName, 
                          ProcessName, serverComponentName, serverProvidedInterfaceName);
}

bool mtsManagerLocal::Connect(
    const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
    const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName)
{
    // Within mtsManagerGlobal::Connect() method, all the proxy components with
    // proxy interfaces are created internally.
    unsigned int connectionID = ManagerGlobal->Connect(
        clientProcessName, clientComponentName, clientRequiredInterfaceName,
        serverProcessName, serverComponentName, serverProvidedInterfaceName);

    if (connectionID == static_cast<unsigned int>(mtsManagerGlobalInterface::CONNECT_ERROR)) {
        CMN_LOG_CLASS_RUN_ERROR << "Connect: Global Manager failed to reserve connection: "
            << clientProcessName << ":" << clientComponentName << ":" << clientRequiredInterfaceName << " - "
            << serverProcessName << ":" << serverComponentName << ":" << serverProvidedInterfaceName << std::endl;
        return false;
    }

    return ConnectLocally(clientProcessName, clientComponentName, clientRequiredInterfaceName, 
                          serverProcessName, serverComponentName, serverProvidedInterfaceName, connectionID);
}

bool mtsManagerLocal::ConnectLocally(
    const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
    const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName,
    const unsigned int connectionID)
{
    // At this point, it is guaranteed that all components and interfaces exist
    // in the same process because the global component manager has already 
    // checked and created proxy objects as needed.
    mtsDevice * clientComponent = GetComponent(clientComponentName);
    if (!clientComponent) {
        clientComponent = GetComponent(mtsManagerGlobal::GetComponentProxyName(clientProcessName, clientComponentName));
        if (!clientComponent) {
            CMN_LOG_CLASS_RUN_ERROR << "Connect: failed to find client component: " << clientComponentName << std::endl;
            return false;
        }
    }

    mtsDevice * serverComponent = GetComponent(serverComponentName);
    if (!serverComponent) {
        serverComponent = GetComponent(mtsManagerGlobal::GetComponentProxyName(serverProcessName, serverComponentName));
        if (!serverComponent) {
            CMN_LOG_CLASS_RUN_ERROR << "Connect: failed to find server component: " << serverComponentName << std::endl;
            return false;
        }
    }

    // Get the client component and the provided interface object.
    mtsProvidedInterface * serverProvidedInterface = serverComponent->GetProvidedInterface(serverProvidedInterfaceName);
    if (!serverProvidedInterface) {
        CMN_LOG_CLASS_RUN_ERROR << "Connect: failed to find provided interface: " 
            << serverComponentName << ":" << serverProvidedInterface << std::endl;
        return false;
    }

    // Connect two interfaces
    if (!clientComponent->ConnectRequiredInterface(clientRequiredInterfaceName, serverProvidedInterface)) {
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
        CMN_LOG_CLASS_RUN_VERBOSE << "Connect: successfully confirmed remote connection" << std::endl;
    }

    return true;
}

bool mtsManagerLocal::Disconnect(const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
                                 const std::string & serverComponentName, const std::string & serverProvidedInterfaceName)
{
    bool success = ManagerGlobal->Disconnect(
        ProcessName, clientComponentName, clientRequiredInterfaceName,
        ProcessName, serverComponentName, serverProvidedInterfaceName);

    if (!success) {
        CMN_LOG_CLASS_RUN_ERROR << "Disconnect: disconnection failed." << std::endl;
        return false;
    }

    CMN_LOG_CLASS_RUN_ERROR << "Disconnect: successfully disconnected." << std::endl;
    return true;
}

bool mtsManagerLocal::Disconnect(
    const std::string & clientProcessName,
    const std::string & clientComponentName,
    const std::string & clientRequiredInterfaceName,
    const std::string & serverProcessName,
    const std::string & serverComponentName,
    const std::string & serverProvidedInterfaceName)
{
    bool success = ManagerGlobal->Disconnect(
        clientProcessName, clientComponentName, clientRequiredInterfaceName,
        serverProcessName, serverComponentName, serverProvidedInterfaceName);

    if (!success) {
        CMN_LOG_CLASS_RUN_ERROR << "Disconnect: disconnection failed." << std::endl;
        return false;
    }

    CMN_LOG_CLASS_RUN_ERROR << "Disconnect: successfully disconnected." << std::endl;
    return true;
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

    return providedInterface->GetProvidedInterfaceDescription(providedInterfaceDescription);
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

    return requiredInterface->GetRequiredInterfaceDescription(requiredInterfaceDescription);
}

bool mtsManagerLocal::CreateComponentProxy(const std::string & componentProxyName)
{
    // Create a component proxy
    mtsDevice * newComponent = new mtsComponentProxy(componentProxyName);

    bool success = AddComponent(newComponent);
    if (!success) {
        delete newComponent;
        return false;
    }

    return true;
}

bool mtsManagerLocal::RemoveComponentProxy(const std::string & componentProxyName)
{
    return RemoveComponent(componentProxyName);
}

bool mtsManagerLocal::CreateProvidedInterfaceProxy(
    const std::string & serverComponentProxyName,
    ProvidedInterfaceDescription & providedInterfaceDescription)
{
    const std::string providedInterfaceName = providedInterfaceDescription.ProvidedInterfaceName;

    // Get current component proxy. If none, returns false because a component
    // proxy should be created before an interface proxy is created.
    mtsDevice * serverComponent = GetComponent(serverComponentProxyName);
    if (!serverComponent) {
        CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: "
            << "no component proxy found: " << serverComponentProxyName << std::endl;
        return false;
    }

    // Convert the component into its original type
    mtsComponentProxy * serverComponentProxy = dynamic_cast<mtsComponentProxy*>(serverComponent);
    if (!serverComponentProxy) {
        CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: "
            << "invalid component proxy: " << serverComponentProxyName << std::endl;
        return false;
    }

    // Create provided interface proxy.
    if (!serverComponentProxy->CreateProvidedInterfaceProxy(providedInterfaceDescription)) {
        CMN_LOG_CLASS_RUN_VERBOSE << "CreateProvidedInterfaceProxy: "
            << "failed to create Provided interface proxy: " << serverComponentProxyName << ":" 
            << providedInterfaceName << std::endl;
        return false;
    }

    // Notify the global component manager of the creation of provided interface proxy
    if (!ManagerGlobal->AddProvidedInterface(ProcessName, serverComponentProxyName, providedInterfaceName, true))
    {
        CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: "
            << "failed to add provided interface proxy to global component manager: "
            << ProcessName << ":" << serverComponentProxyName << ":" << providedInterfaceName << std::endl;
        return false;
    }

    CMN_LOG_CLASS_RUN_VERBOSE << "CreateProvidedInterfaceProxy: "
        << "successfully created Provided interface proxy: " << serverComponentProxyName << ":" 
        << providedInterfaceName << std::endl;
    return true;
}

bool mtsManagerLocal::CreateRequiredInterfaceProxy(
    const std::string & clientComponentProxyName, RequiredInterfaceDescription & requiredInterfaceDescription)
{
    const std::string requiredInterfaceName = requiredInterfaceDescription.RequiredInterfaceName;

    // Get current component proxy. If none, returns false because a component
    // proxy should be created before an interface proxy is created.
    mtsDevice * clientComponent = GetComponent(clientComponentProxyName);
    if (!clientComponent) {
        CMN_LOG_CLASS_RUN_ERROR << "CreateRequiredInterfaceProxy: "
            << "no component proxy found: " << clientComponentProxyName << std::endl;
        return false;
    }

    // Convert the component into its orginal type
    mtsComponentProxy * clientComponentProxy = dynamic_cast<mtsComponentProxy*>(clientComponent);
    if (!clientComponentProxy) {
        CMN_LOG_CLASS_RUN_ERROR << "CreateRequiredInterfaceProxy: "
            << "invalid component proxy: " << clientComponentProxyName << std::endl;
        return false;
    }

    // Create required interface proxy
    if (!clientComponentProxy->CreateRequiredInterfaceProxy(requiredInterfaceDescription)) {
        CMN_LOG_CLASS_RUN_VERBOSE << "CreateRequiredInterfaceProxy: "
            << "failed to create required interface proxy: " << clientComponentProxyName << ":" 
            << requiredInterfaceName << std::endl;
        return false;
    }

    // Notify the global component manager of the creation of provided interface proxy
    if (!ManagerGlobal->AddRequiredInterface(ProcessName, clientComponentProxyName, requiredInterfaceName, true))
    {
        CMN_LOG_CLASS_RUN_ERROR << "CreateRequiredInterfaceProxy: "
            << "failed to add required interface proxy to global component manager: "
            << ProcessName << ":" << clientComponentProxyName << ":" << requiredInterfaceName << std::endl;
        return false;
    }

    CMN_LOG_CLASS_RUN_VERBOSE << "CreateRequiredInterfaceProxy: "
        << "successfully created required interface proxy: " << clientComponentProxyName << ":" 
        << requiredInterfaceName << std::endl;
    return true;
}

bool mtsManagerLocal::RemoveProvidedInterfaceProxy(
    const std::string & clientComponentProxyName, const std::string & providedInterfaceProxyName)
{
    mtsDevice * clientComponent = GetComponent(clientComponentProxyName);
    if (!clientComponent) {
        CMN_LOG_CLASS_RUN_ERROR << "RemoveProvidedInterfaceProxy: can't find client component: " << clientComponentProxyName << std::endl;
        return false;
    }

    mtsComponentProxy * clientComponentProxy = dynamic_cast<mtsComponentProxy*>(clientComponent);
    if (!clientComponentProxy) {
        CMN_LOG_CLASS_RUN_ERROR << "RemoveProvidedInterfaceProxy: client component is not a proxy: " << clientComponentProxyName << std::endl;
        return false;
    }

    // Check the number of required interfaces using (connecting to) this provided interface.
    mtsProvidedInterface * providedInterfaceProxy = clientComponentProxy->GetProvidedInterface(providedInterfaceProxyName);
    if (!providedInterfaceProxy) {
        CMN_LOG_CLASS_RUN_ERROR << "RemoveProvidedInterfaceProxy: can't get provided interface proxy.: " << providedInterfaceProxyName << std::endl;
        return false;
    }

    // Remove provided interface proxy only when user counter is zero.
    if (--providedInterfaceProxy->UserCounter == 0) {
        // Remove provided interface from component proxy.
        if (!clientComponentProxy->RemoveProvidedInterfaceProxy(providedInterfaceProxyName)) {
            CMN_LOG_CLASS_RUN_ERROR << "RemoveProvidedInterfaceProxy: failed to remove provided interface proxy: " << providedInterfaceProxyName << std::endl;
            return false;
        }

        CMN_LOG_CLASS_RUN_VERBOSE << "RemoveProvidedInterfaceProxy: removed provided interface: " 
            << clientComponentProxyName << ":" << providedInterfaceProxyName << std::endl;
    } else {
        CMN_LOG_CLASS_RUN_VERBOSE << "RemoveProvidedInterfaceProxy: decreased active user counter. current counter: " 
            << providedInterfaceProxy->UserCounter << std::endl;
    }

    return true;
}

bool mtsManagerLocal::RemoveRequiredInterfaceProxy(
    const std::string & serverComponentProxyName, const std::string & requiredInterfaceProxyName)
{
    mtsDevice * serverComponent = GetComponent(serverComponentProxyName);
    if (!serverComponent) {
        CMN_LOG_CLASS_RUN_ERROR << "RemoveRequiredInterfaceProxy: can't find server component: " << serverComponentProxyName << std::endl;
        return false;
    }

    mtsComponentProxy * serverComponentProxy = dynamic_cast<mtsComponentProxy*>(serverComponent);
    if (!serverComponentProxy) {
        CMN_LOG_CLASS_RUN_ERROR << "RemoveRequiredInterfaceProxy: server component is not a proxy: " << serverComponentProxyName << std::endl;
        return false;
    }

    // Remove required interface from component proxy.
    if (!serverComponentProxy->RemoveRequiredInterfaceProxy(requiredInterfaceProxyName)) {
        CMN_LOG_CLASS_RUN_ERROR << "RemoveRequiredInterfaceProxy: failed to remove required interface proxy: " << requiredInterfaceProxyName << std::endl;
        return false;
    }

    CMN_LOG_CLASS_RUN_VERBOSE << "RemoveRequiredInterfaceProxy: removed required interface: " 
        << serverComponentProxyName << ":" << requiredInterfaceProxyName << std::endl;

    return true;
}

const int mtsManagerLocal::GetCurrentInterfaceCount(const std::string & componentName) const
{
    // Check if the component specified exists
    mtsDevice * component = GetComponent(componentName);
    if (!component) {
        CMN_LOG_CLASS_RUN_ERROR << "GetCurrentInterfaceCount: no component found: " << componentName << " on " << ProcessName << std::endl;
        return -1;
    }

    const unsigned int numOfProvidedInterfaces = component->ProvidedInterfaces.size();
    const unsigned int numOfRequiredInterfaces = component->RequiredInterfaces.size();

    return (const int) (numOfProvidedInterfaces + numOfRequiredInterfaces);
}