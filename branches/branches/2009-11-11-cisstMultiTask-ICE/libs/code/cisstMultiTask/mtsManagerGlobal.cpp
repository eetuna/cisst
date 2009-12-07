/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerGlobal.h 794 2009-09-01 21:43:56Z pkazanz1 $

  Author(s):  Min Yang Jung
  Created on: 2009-11-12

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstMultiTask/mtsManagerGlobal.h>

CMN_IMPLEMENT_SERVICES(mtsManagerGlobal);

#define CHECK_DELETE(_var) \
    {\
        AllocatedPointerType::iterator _it = AllocatedPointers.find(reinterpret_cast<unsigned int>(_var));\
        CMN_ASSERT(_it != AllocatedPointers.end());\
        AllocatedPointers.erase(_it);\
    }

mtsManagerGlobal::mtsManagerGlobal()
{
}

mtsManagerGlobal::~mtsManagerGlobal()
{
    CleanUp();
}

//-------------------------------------------------------------------------
//  Processing Methods
//-------------------------------------------------------------------------
bool mtsManagerGlobal::CleanUp(void)
{    
    bool ret = true;

    // Remove all processes safely
    ProcessMapType::iterator it = ProcessMap.GetMap().begin();
    while (it != ProcessMap.GetMap().end()) {        
        ret &= RemoveProcess(it->first);
        it = ProcessMap.GetMap().begin();
    }

    // Debugging code
    if (AllocatedPointers.size() != 0) {
        AllocatedPointerType::const_iterator it = AllocatedPointers.begin();
        for (; it != AllocatedPointers.end(); ++it) {
            std::cout << "############# LINE: " << it->second << std::endl;
        }
    }

    return ret;
}

//-------------------------------------------------------------------------
//  Process Management
//-------------------------------------------------------------------------
bool mtsManagerGlobal::AddProcess(const std::string & processName)
{
    // AddProcess() doesn't need to be called to check duplicate process registration
    // since cmnNamedMap::AddItem() internally checks duplicity before adding an item.

    bool success = ProcessMap.AddItem(processName, NULL);
    if (!success ) {
        CMN_LOG_CLASS_RUN_ERROR << "Can't add process: " << processName << std::endl;
    }

    return success;
}

bool mtsManagerGlobal::FindProcess(const std::string & processName) const
{
    return ProcessMap.FindItem(processName);
}

bool mtsManagerGlobal::RemoveProcess(const std::string & processName)
{
    // TODO: To remove a process safely, the following things should be handled first:
    //
    // 1) Notifying connected processes (if any)
    // 2) Terminate session
    // 3) Proxy termination

    // 4) Remove from map
    // 5) Memory release

    if (!ProcessMap.FindItem(processName)) return false;

    bool ret = true;
    
    ComponentMapType * componentMap = ProcessMap.GetItem(processName);

    // When componentMap is not NULL, all components that the process manages 
    // should be removed first.
    if (componentMap) {
        ComponentMapType::iterator it = componentMap->begin();
        while (it != componentMap->end()) {
            ret &= RemoveComponent(processName, it->first);
            it = componentMap->begin();
        }
        
        CHECK_DELETE(componentMap);
    }

    // Remove the process from process map
    ret &= ProcessMap.RemoveItem(processName);

    return ret;
}

//-------------------------------------------------------------------------
//  Component Management
//-------------------------------------------------------------------------
bool mtsManagerGlobal::AddComponent(const std::string & processName, const std::string & componentName)
{
    // AddComponent() doesn't need to be called to check duplicate process registration
    // since cmnNamedMap::AddItem() internally checks duplicity before adding an item.

    if (!FindProcess(processName)) {
        CMN_LOG_CLASS_RUN_ERROR << "Can't find registered process: " << processName << std::endl;
        return false;
    }

    ComponentMapType * componentMap = ProcessMap.GetItem(processName);

    // If the process did not register before
    if (componentMap == NULL) {
        componentMap = new ComponentMapType(processName);
        (ProcessMap.GetMap())[processName] = componentMap;

        // TOOD: REMOVE THIS
        AllocatedPointers[reinterpret_cast<unsigned int>(componentMap)] = __LINE__;
    }

    bool ret = componentMap->AddItem(componentName, NULL);
    if (!ret) {
        CMN_LOG_CLASS_RUN_ERROR << "Can't add a component: " 
            << "\"" << processName << "\" - \"" << componentName << "\"" << std::endl;
    }

    return ret;
}

bool mtsManagerGlobal::FindComponent(const std::string & processName, const std::string & componentName) const
{
    if (!FindProcess(processName)) {
        CMN_LOG_CLASS_RUN_ERROR << "Can't find a registered process: " << processName << std::endl;
        return false;
    }

    ComponentMapType * componentMap = ProcessMap.GetItem(processName);
    if (!componentMap) {
        return false;
    }

    return componentMap->FindItem(componentName);
}

bool mtsManagerGlobal::RemoveComponent(const std::string & processName, const std::string & componentName)
{
    // TODO: Before removing an element from relevent maps, Disconnect() should be 
    // called first for a case that there is any active connection related to the element.

    // Check existence of the process
    if (!ProcessMap.FindItem(processName)) return false;

    bool ret = true;

    ComponentMapType * componentMap = ProcessMap.GetItem(processName);
    if (componentMap) {
        // Check existence of the component
        if (!componentMap->FindItem(componentName)) return false;

        // When interfaceMapType is not NULL, all interfaces that the component manages
        // should be removed first.
        InterfaceMapType * interfaceMap = componentMap->GetItem(componentName);
        if (interfaceMap) {
            ConnectedInterfaceMapType::iterator it;

            // Remove all required interfaces that the process manage.
            it = interfaceMap->RequiredInterfaceMap.GetMap().begin();
            while (it != interfaceMap->RequiredInterfaceMap.GetMap().end()) {
                ret &= RemoveRequiredInterface(processName, componentName, it->first);
                it = interfaceMap->RequiredInterfaceMap.GetMap().begin();
            }

            // Remove all provided interfaces that the process manage.
            it = interfaceMap->ProvidedInterfaceMap.GetMap().begin();
            while (it != interfaceMap->ProvidedInterfaceMap.GetMap().end()) {
                ret &= RemoveProvidedInterface(processName, componentName, it->first);
                it = interfaceMap->ProvidedInterfaceMap.GetMap().begin();
            }

            CHECK_DELETE(interfaceMap);
        }
    }

    // Remove the component from component map
    ret &= componentMap->RemoveItem(componentName);

    return ret;
}

//-------------------------------------------------------------------------
//  Interface Management
//-------------------------------------------------------------------------
bool mtsManagerGlobal::AddProvidedInterface(
    const std::string & processName, const std::string & componentName, const std::string & interfaceName)
{
    // AddProvidedInterface() doesn't need to be called to check duplicate process registration
    // since cmnNamedMap::AddItem() internally checks duplicity before adding an item.

    if (!FindComponent(processName, componentName)) {
        CMN_LOG_CLASS_RUN_ERROR << "Can't find a registered component: " 
            << "\"" << processName << "\" - \"" << componentName << "\"" << std::endl;
        return false;
    }

    ComponentMapType * componentMap = ProcessMap.GetItem(processName);
    InterfaceMapType * interfaceMap = componentMap->GetItem(componentName);

    // If the component did not registered before
    if (interfaceMap == NULL) {
        interfaceMap = new InterfaceMapType;
        (componentMap->GetMap())[componentName] = interfaceMap;

        // TODO: Remove this
        AllocatedPointers[reinterpret_cast<unsigned int>(interfaceMap)] = __LINE__;
    }

    bool ret = interfaceMap->ProvidedInterfaceMap.AddItem(interfaceName, NULL);
    if (!ret) {
        CMN_LOG_CLASS_RUN_ERROR << "Can't add a provided interface: " << interfaceName << std::endl;
    }

    return ret;
}

bool mtsManagerGlobal::AddRequiredInterface(
    const std::string & processName, const std::string & componentName, const std::string & interfaceName)
{
    // AddRequiredInterface() doesn't need to be called to check duplicate process registration
    // since cmnNamedMap::AddItem() internally checks duplicity before adding an item.

    if (!FindComponent(processName, componentName)) {
        CMN_LOG_CLASS_RUN_ERROR << "Can't find a registered component: " 
            << "\"" << processName << "\" - \"" << componentName << "\"" << std::endl;
        return false;
    }

    ComponentMapType * componentMap = ProcessMap.GetItem(processName);
    InterfaceMapType * interfaceMap = componentMap->GetItem(componentName);

    // If the component did not registered before
    if (interfaceMap == NULL) {
        interfaceMap = new InterfaceMapType;
        (componentMap->GetMap())[componentName] = interfaceMap;

        // TODO: Remove this
        AllocatedPointers[reinterpret_cast<unsigned int>(interfaceMap)] = __LINE__;
    }

    bool ret = interfaceMap->RequiredInterfaceMap.AddItem(interfaceName, NULL);
    if (!ret) {
        CMN_LOG_CLASS_RUN_ERROR << "Can't add a required interface: " << interfaceName << std::endl;
    }

    return ret;
}

bool mtsManagerGlobal::FindProvidedInterface(
    const std::string & processName, const std::string & componentName, const std::string & interfaceName) const
{
    if (!FindComponent(processName, componentName)) {
        CMN_LOG_CLASS_RUN_ERROR << "Can't find a registered component: " 
            << "\"" << processName << "\" - \"" << componentName << "\"" << std::endl;
        return false;
    }
    
    ComponentMapType * componentMap = ProcessMap.GetItem(processName);
    if (!componentMap) return false;

    InterfaceMapType * interfaceMap = componentMap->GetItem(componentName);
    if (!interfaceMap) return false;

    return interfaceMap->ProvidedInterfaceMap.FindItem(interfaceName);
}

bool mtsManagerGlobal::FindRequiredInterface(
    const std::string & processName, const std::string & componentName, const std::string & interfaceName) const
{
    if (!FindComponent(processName, componentName)) {
        CMN_LOG_CLASS_RUN_ERROR << "Can't find a registered component: " 
            << "\"" << processName << "\" - \"" << componentName << "\"" << std::endl;
        return false;
    }
    
    ComponentMapType * componentMap = ProcessMap.GetItem(processName);
    if (!componentMap) return false;

    InterfaceMapType * interfaceMap = componentMap->GetItem(componentName);
    if (!interfaceMap) return false;

    return interfaceMap->RequiredInterfaceMap.FindItem(interfaceName);
}

bool mtsManagerGlobal::RemoveProvidedInterface(
    const std::string & processName, const std::string & componentName, const std::string & interfaceName)
{
    // Check existence of the process
    if (!ProcessMap.FindItem(processName)) return false;

    // Check existence of the component
    ComponentMapType * componentMap = ProcessMap.GetItem(processName);
    if (!componentMap) return false;
    if (!componentMap->FindItem(componentName)) return false;

    // Check existence of the provided interface
    InterfaceMapType * interfaceMap = componentMap->GetItem(componentName);
    if (!interfaceMap) return false;
    if (!interfaceMap->ProvidedInterfaceMap.FindItem(interfaceName)) return false;

    // When connectionMap is not NULL, all connection information that the provided interface
    // has should be removed first.
    bool ret = true;
    ConnectionMapType * connectionMap = interfaceMap->ProvidedInterfaceMap.GetItem(interfaceName);
    if (connectionMap) {
        ConnectionMapType::iterator it = connectionMap->begin();
        while (it != connectionMap->end()) {
            Disconnect(it->second->GetProcessName(), it->second->GetComponentName(), it->second->GetInterfaceName(),
                processName, componentName, interfaceName);
            it = connectionMap->begin();
        }
        CHECK_DELETE(connectionMap);
    }

    // Remove the provided interface from provided interface map
    ret &= interfaceMap->ProvidedInterfaceMap.RemoveItem(interfaceName);

    return ret;
}

bool mtsManagerGlobal::RemoveRequiredInterface(
    const std::string & processName, const std::string & componentName, const std::string & interfaceName)
{
    // Check existence of the process
    if (!ProcessMap.FindItem(processName)) return false;

    // Check existence of the component
    ComponentMapType * componentMap = ProcessMap.GetItem(processName);
    if (!componentMap) return false;
    if (!componentMap->FindItem(componentName)) return false;

    // Check existence of the provided interface
    InterfaceMapType * interfaceMap = componentMap->GetItem(componentName);
    if (!interfaceMap) return false;
    if (!interfaceMap->RequiredInterfaceMap.FindItem(interfaceName)) return false;

    // When connectionMap is not NULL, all connection information that the provided interface
    // has should be removed first.
    bool ret = true;
    ConnectionMapType * connectionMap = interfaceMap->RequiredInterfaceMap.GetItem(interfaceName);
    if (connectionMap) {
        ConnectionMapType::iterator it = connectionMap->begin();
        while (it != connectionMap->end()) {
            Disconnect(processName, componentName, interfaceName, 
                it->second->GetProcessName(), it->second->GetComponentName(), it->second->GetInterfaceName());
            it = connectionMap->begin();
        }
        CHECK_DELETE(connectionMap);
    }

    // Remove the required interface from provided interface map
    ret &= interfaceMap->RequiredInterfaceMap.RemoveItem(interfaceName);

    return ret;
}

//-------------------------------------------------------------------------
//  Connection Management
//-------------------------------------------------------------------------
bool mtsManagerGlobal::Connect(
    const std::string & clientProcessName,
    const std::string & clientComponentName,
    const std::string & clientRequiredInterfaceName,
    const std::string & serverProcessName,
    const std::string & serverComponentName,
    const std::string & serverProvidedInterfaceName)
{
    // Check if the required interface specified actually exist.
    if (!FindRequiredInterface(clientProcessName, clientComponentName, clientRequiredInterfaceName)) {
        CMN_LOG_CLASS_RUN_VERBOSE << "Can't connect: required interface does not exist: "
            << GetInterfaceUID(clientProcessName, clientComponentName, clientRequiredInterfaceName)
            << std::endl;
        return false;
    }

    // Check if the provided interface specified actually exist.
    if (!FindProvidedInterface(serverProcessName, serverComponentName, serverProvidedInterfaceName)) {
        CMN_LOG_CLASS_RUN_VERBOSE << "Can't connect: provided interface does not exist: "
            << GetInterfaceUID(serverProcessName, serverComponentName, serverProvidedInterfaceName)
            << std::endl;
        return false;
    }

    // Check if the two interfaces are already connected.
    int ret = IsAlreadyConnected(clientProcessName,
                                 clientComponentName,
                                 clientRequiredInterfaceName,
                                 serverProcessName,
                                 serverComponentName,
                                 serverProvidedInterfaceName);
    // When error occurs (because of non-existing components, etc)
    if (ret < 0) {
        CMN_LOG_CLASS_RUN_VERBOSE << "Can't connect: "
            << "one or more processes, components, or interfaces are missing." << std::endl;
        return false;
    }
    // When interfaces have been connected to each other
    else if (ret > 0) {
        CMN_LOG_CLASS_RUN_VERBOSE << "Can't connect: Two interfaces are already connected: "
            << GetInterfaceUID(clientProcessName, clientComponentName, clientRequiredInterfaceName)
            << " and "
            << GetInterfaceUID(serverProcessName, serverComponentName, serverProvidedInterfaceName)
            << std::endl;
        return false;
    }


    // Step 1. Make serverProcess create a proxy for clientComponent in its local memory space.

    // Step 2. Make clientProcess create a proxy for serverComponent in its local memory space.

    // Step 3. If both proxies are created successfully, let clientComponentProxy connect to 
    //         serverComponentProxy across a network.

    // Step 4. If the connection is established well, then make both client process and
    //         server process establish local connection. That is,
    //
    //         for client process: connect requiredInterface to providedInterfaceProxy locally
    //         for server process: connect requiredInterfaceProxy to providedInterface locally

    // Step 5. After confirming that both local connections are established well,
    //         update Global Manager's connection data structure.

    InterfaceMapType * interfaceMap;
    ConnectionMapType * connectionMap;    
    std::string interfaceUID;

    // Connect client's required interface with server's provided interface.
    connectionMap = GetConnectionsOfRequiredInterface(clientProcessName,
                                                      clientComponentName,
                                                      clientRequiredInterfaceName,
                                                      &interfaceMap);
    // If the required interface has never connected to other provided interfaces
    if (connectionMap == NULL) {
        // Create a connection map for the required interface
        connectionMap = new ConnectionMapType(clientRequiredInterfaceName);
        (interfaceMap->RequiredInterfaceMap.GetMap())[clientRequiredInterfaceName] = connectionMap;

        // TODO: Remove this
        AllocatedPointers[reinterpret_cast<unsigned int>(connectionMap)] = __LINE__;
    }

    // Add an element containing information about the connected provided interface
    if (!AddConnectedInterface(connectionMap, serverProcessName, serverComponentName, serverProvidedInterfaceName)) {
        CMN_LOG_CLASS_RUN_ERROR << "Can't connect: "
            << "failed to add information about connected provided interface." << std::endl;
        return false;
    }

    // Connect server's provided interface with client's required interface.
    connectionMap = GetConnectionsOfProvidedInterface(serverProcessName,
                                                      serverComponentName,
                                                      serverProvidedInterfaceName,
                                                      &interfaceMap);
    // If the provided interface has never been connected with other required interfaces
    if (connectionMap == NULL) {
        // Create a connection map for the provided interface
        connectionMap = new ConnectionMapType(serverProvidedInterfaceName);
        (interfaceMap->ProvidedInterfaceMap.GetMap())[serverProvidedInterfaceName] = connectionMap;

        // TODO: Remove this
        AllocatedPointers[reinterpret_cast<unsigned int>(connectionMap)] = __LINE__;
    }

    // Add an element containing information about the connected provided interface
    if (!AddConnectedInterface(connectionMap, clientProcessName, clientComponentName, clientRequiredInterfaceName)) {
        CMN_LOG_CLASS_RUN_ERROR << "Can't connect: "
            << "failed to add information about connected required interface." << std::endl;

        // Before returning false, should clean up required interface's connection information
        Disconnect(clientProcessName, clientComponentName, clientRequiredInterfaceName,
                   serverProcessName, serverComponentName, serverProvidedInterfaceName);
        return false;
    }

    //
    // TODO: create task proxy and connect original task with the proxy (physically)
    //
    
    return true;
}

bool mtsManagerGlobal::IsAlreadyConnected(
    const std::string & clientProcessName,
    const std::string & clientComponentName,
    const std::string & clientRequiredInterfaceName,
    const std::string & serverProcessName,
    const std::string & serverComponentName,
    const std::string & serverProvidedInterfaceName)
{
    // It is assumed that the existence of interfaces has already been checked before
    // calling this method.

    // Check if the required interface is connected to the provided interface
    ConnectionMapType * connectionMap = GetConnectionsOfRequiredInterface(
        clientProcessName, clientComponentName, clientRequiredInterfaceName);
    if (connectionMap) {
        if (connectionMap->FindItem(GetInterfaceUID(serverProcessName, serverComponentName, serverProvidedInterfaceName)))
        {
            CMN_LOG_CLASS_RUN_VERBOSE << "Already established connection" << std::endl;
            return true;
        }
    }

    // Check if the provided interface is connected to the required interface
    connectionMap = GetConnectionsOfProvidedInterface(
        serverProcessName, serverComponentName, serverProvidedInterfaceName);
    if (connectionMap) {
        if (connectionMap->FindItem(GetInterfaceUID(clientProcessName, clientComponentName, clientProcessName)))
        {
            CMN_LOG_CLASS_RUN_VERBOSE << "Already established connection" << std::endl;
            return true;
        }
    }

    return false;
}

void mtsManagerGlobal::Disconnect(
    const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
    const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName)
{
    InterfaceMapType * interfaceMap;
    std::string interfaceUID;

    // Update required interface's connection information first
    ConnectionMapType * connectionMap = GetConnectionsOfRequiredInterface(clientProcessName,
                                                                          clientComponentName,
                                                                          clientRequiredInterfaceName, 
                                                                          &interfaceMap);
    if (connectionMap == NULL) return;

    // Get connection information element
    interfaceUID = GetInterfaceUID(serverProcessName, serverComponentName, serverProvidedInterfaceName);
    ConnectedInterfaceInfo * info = connectionMap->GetItem(interfaceUID);
    // Release allocated memory
    if (info) {
        CHECK_DELETE(info);
    }
    // Remove connection information
    connectionMap->RemoveItem(interfaceUID);

    // Update provided interface's connection information later
    connectionMap = GetConnectionsOfProvidedInterface(serverProcessName,
                                                      serverComponentName,
                                                      serverProvidedInterfaceName,
                                                      &interfaceMap);
    if (connectionMap == NULL) return;

    // Get connection information element
    interfaceUID = GetInterfaceUID(clientProcessName, clientComponentName, clientRequiredInterfaceName);
    info = connectionMap->GetItem(interfaceUID);
    // Release allocated memory
    if (info) {
        CHECK_DELETE(info);
    }
    // Remove connection information
    connectionMap->RemoveItem(interfaceUID);
}

mtsManagerGlobal::ConnectionMapType * mtsManagerGlobal::GetConnectionsOfProvidedInterface(
    const std::string & severProcessName, const std::string & serverComponentName, 
    const std::string & providedInterfaceName, InterfaceMapType ** interfaceMap)
{
    ComponentMapType * componentMap = ProcessMap.GetItem(severProcessName);
    if (componentMap == NULL) return NULL;

    *interfaceMap = componentMap->GetItem(serverComponentName);
    if (*interfaceMap == NULL) return NULL;

    ConnectionMapType * connectionMap = (*interfaceMap)->ProvidedInterfaceMap.GetItem(providedInterfaceName);

    return connectionMap;
}

mtsManagerGlobal::ConnectionMapType * mtsManagerGlobal::GetConnectionsOfProvidedInterface(
    const std::string & severProcessName, const std::string & serverComponentName, 
    const std::string & providedInterfaceName) const
{
    ComponentMapType * componentMap = ProcessMap.GetItem(severProcessName);
    if (componentMap == NULL) return NULL;

    InterfaceMapType * interfaceMap = componentMap->GetItem(serverComponentName);
    if (interfaceMap == NULL) return NULL;

    ConnectionMapType * connectionMap = interfaceMap->ProvidedInterfaceMap.GetItem(providedInterfaceName);

    return connectionMap;
}

mtsManagerGlobal::ConnectionMapType * mtsManagerGlobal::GetConnectionsOfRequiredInterface(
    const std::string & clientProcessName, const std::string & clientComponentName, 
    const std::string & requiredInterfaceName, InterfaceMapType ** interfaceMap)
{
    ComponentMapType * componentMap = ProcessMap.GetItem(clientProcessName);
    if (componentMap == NULL) return NULL;

    *interfaceMap = componentMap->GetItem(clientComponentName);
    if (*interfaceMap == NULL) return NULL;

    ConnectionMapType * connectionMap = (*interfaceMap)->RequiredInterfaceMap.GetItem(requiredInterfaceName);

    return connectionMap;
}

mtsManagerGlobal::ConnectionMapType * mtsManagerGlobal::GetConnectionsOfRequiredInterface(
    const std::string & clientProcessName, const std::string & clientComponentName, 
    const std::string & requiredInterfaceName) const
{
    ComponentMapType * componentMap = ProcessMap.GetItem(clientProcessName);
    if (componentMap == NULL) return NULL;

    InterfaceMapType * interfaceMap = componentMap->GetItem(clientComponentName);
    if (interfaceMap == NULL) return NULL;

    ConnectionMapType * connectionMap = interfaceMap->RequiredInterfaceMap.GetItem(requiredInterfaceName);

    return connectionMap;
}

bool mtsManagerGlobal::AddConnectedInterface(ConnectionMapType * connectionMap,
    const std::string & processName, const std::string & componentName,
    const std::string & interfaceName)
{
    if (!connectionMap) return false;

    ConnectedInterfaceInfo * connectedInterfaceInfo = 
        new ConnectedInterfaceInfo(processName, componentName, interfaceName);

    // TOOD: REMOVE THIS
    AllocatedPointers[reinterpret_cast<unsigned int>(connectedInterfaceInfo)] = __LINE__;

    std::string interfaceUID = GetInterfaceUID(processName, componentName, interfaceName);
    if (!connectionMap->AddItem(interfaceUID, connectedInterfaceInfo)) {
        CMN_LOG_CLASS_RUN_ERROR << "Cannot add peer interface's information: " 
            << GetInterfaceUID(processName, componentName, interfaceName) << std::endl;
        return false;
    }

    return true;
}