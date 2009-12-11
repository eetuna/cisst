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
#include <cisstMultiTask/mtsDevice.h>
//#include <cisstMultiTask/mtsDeviceInterface.h>
#include <cisstMultiTask/mtsTask.h>
//#include <cisstMultiTask/mtsTaskInterface.h>
#include <cisstMultiTask/mtsManagerGlobal.h>

//#if CISST_MTS_HAS_ICE
//#include <cisstMultiTask/mtsDeviceProxy.h>
//#include <cisstMultiTask/mtsManagerLocalProxyServer.h>
//#include <cisstMultiTask/mtsManagerLocalProxyClient.h>
//#endif // CISST_MTS_HAS_ICE

CMN_IMPLEMENT_SERVICES(mtsManagerLocal);

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
    this->Kill();

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
    
    if (ManagerGlobal) {
        // TODO: Add proxy (network) clean-up before delete
        delete ManagerGlobal;
        ManagerGlobal = NULL;
    }
}

mtsManagerLocal * mtsManagerLocal::GetInstance(
    const std::string & thisProcessName, const std::string & thisProcessIP)
{
    static mtsManagerLocal instance(thisProcessName, thisProcessIP);
    return &instance;
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

bool mtsManagerLocal::RemoveTask(mtsDevice * component) 
{
    // Try to remove this component from the global component manager first.
    if (!ManagerGlobal->RemoveComponent(ProcessName, component->GetName())) {
        CMN_LOG_CLASS_RUN_ERROR << "failed to remove component: " << component->GetName() << std::endl;
        return false;
    }

    CMN_LOG_CLASS_RUN_VERBOSE << "Global component manager: removed component: " << component->GetName() << std::endl;

    ComponentMapChange.Lock();
    bool result = ComponentMap.RemoveItem(component->GetName(), CMN_LOG_LOD_RUN_ERROR);
    if (result) {
        CMN_LOG_CLASS_INIT_VERBOSE << "removed component: "
                                   << component->GetName() << std::endl;
    }
    ComponentMapChange.Unlock();
    return result;
}

std::vector<std::string> mtsManagerLocal::GetNamesOfProcesses(void) const
{
    //
    // TODO:
    // 1. add data structure for process list management
    // 2. update the following line
    //
    return ComponentMap.GetNames();
}

std::vector<std::string> mtsManagerLocal::GetNamesOfComponents(void) const 
{
    return ComponentMap.GetNames();
}

void mtsManagerLocal::GetNamesOfComponents(std::vector<std::string> & namesOfComponents) const 
{
    ComponentMap.GetNames(namesOfComponents);
}

mtsDevice * mtsManagerLocal::GetComponent(const std::string & componentName) 
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
    
void mtsManagerLocal::ToStream(std::ostream & outputStream) const {
    // NOP
}

void mtsManagerLocal::CreateAll(void) 
{
    mtsTask * componentTask;

    ComponentMapChange.Lock();

    ComponentMapType::const_iterator it = ComponentMap.begin();
    const ComponentMapType::const_iterator itEnd = ComponentMap.end();

    for (; it != itEnd; ++it) {
        componentTask = dynamic_cast<mtsTask*>(it->second);
        if (!componentTask) continue;

        componentTask->Create();
    }

    ComponentMapChange.Unlock();
}

void mtsManagerLocal::StartAll(void) 
{
    // Get the current thread id so that we can check if any task will use the current thread.
    // If so, start that task last because its Start method will not return.
    const osaThreadId threadId = osaGetCurrentThreadId();
    
    mtsTask * componentTask;

    ComponentMapChange.Lock();

    ComponentMapType::const_iterator it = ComponentMap.begin();
    const ComponentMapType::const_iterator itEnd = ComponentMap.end();
    ComponentMapType::const_iterator itLastTask = ComponentMap.end();

    for (; it != ComponentMap.end(); ++it) {
        componentTask = dynamic_cast<mtsTask*>(it->second);
        if (!componentTask) continue;   // Start components that are of type mtsTask

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
        componentTask = dynamic_cast<mtsTask*>(itLastTask->second);
        CMN_ASSERT(componentTask);
        componentTask->Start();
    }

    ComponentMapChange.Unlock();
}

void mtsManagerLocal::KillAll(void) 
{
    // It is not necessary to have any special handling of a task using the current thread.
    mtsTask * componentTask;
    
    ComponentMapChange.Lock();

    ComponentMapType::const_iterator it = ComponentMap.begin();
    const ComponentMapType::const_iterator itEnd = ComponentMap.end();
    for (; it != itEnd; ++it) {
        componentTask = dynamic_cast<mtsTask*>(it->second);
        if (!componentTask) continue;

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
//  interesting case that a component needs to use external components. That is, the
//  component contains no component while it connects an external client component to
//  an external server component.
//  (see https://trac.lcsr.jhu.edu/cisst/wiki/Private/cisstMultiTaskNetwork)
//
//  Should this be possible, Connect() has to be updated to handle that case.
//
bool mtsManagerLocal::Connect(const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
                              const std::string & serverComponentName, const std::string & serverProvidedInterfaceName)
{
    if (!ManagerGlobal->Connect(ProcessName, clientComponentName, clientRequiredInterfaceName,
                                ProcessName, serverComponentName, serverProvidedInterfaceName))
    {
        CMN_LOG_CLASS_RUN_ERROR << "Connect: Global Manager failed to connect two interfaces: "
            << clientComponentName << ":" << clientRequiredInterfaceName << " - "
            << serverComponentName << ":" << serverProvidedInterfaceName << std::endl;
        return false;
    }

    // Since Global Manager have successfully verified the validity and existence 
    // of components and interfaces specified, the connection can be established.
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

    return true;
}

void mtsManagerLocal::Disconnect(const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
                                 const std::string & serverComponentName, const std::string & serverProvidedInterfaceName)
{
    return ManagerGlobal->Disconnect(
        ProcessName, clientComponentName, clientRequiredInterfaceName,
        ProcessName, serverComponentName, serverProvidedInterfaceName);
}
