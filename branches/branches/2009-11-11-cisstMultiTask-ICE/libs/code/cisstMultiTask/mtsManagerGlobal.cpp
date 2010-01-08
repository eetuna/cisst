/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerGlobal.h 794 2009-09-01 21:43:56Z pkazanz1 $

  Author(s):  Min Yang Jung
  Created on: 2009-11-12

  (C) Copyright 2009-2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstMultiTask/mtsManagerGlobal.h>

CMN_IMPLEMENT_SERVICES(mtsManagerGlobal);

// TODO: Debug code: remove later!
#define CHECK_DELETE(_var) \
    {\
        AllocatedPointerType::iterator _it = AllocatedPointers.find(reinterpret_cast<unsigned int>(_var));\
        CMN_ASSERT(_it != AllocatedPointers.end());\
        AllocatedPointers.erase(_it);\
    }

mtsManagerGlobal::mtsManagerGlobal()
{
    ConnectionID = static_cast<unsigned int>(mtsManagerGlobalInterface::CONNECT_REMOTE_BASE);
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
bool mtsManagerGlobal::AddProcess(mtsManagerLocalInterface * localManager)
{
    if (!localManager) {
        CMN_LOG_CLASS_RUN_ERROR << "AddProcess: invalid local manager" << std::endl;
        return false;
    }

    // Check if the local component manager has already been registered.
    if (FindProcess(localManager->GetProcessName())) {
        CMN_LOG_CLASS_RUN_ERROR << "AddProcess: already registered process: " << localManager->GetProcessName() << std::endl;
        return false;
    }
    
    const std::string processName = localManager->GetProcessName();
    bool success = ProcessMap.AddItem(processName, NULL);

    if (success) {
        LocalManagerMapChange.Lock();
        success = LocalManagerMap.AddItem(processName, localManager);
        LocalManagerMapChange.Unlock();

        if (!success) {
            CMN_LOG_CLASS_RUN_ERROR << "Failed to add process in local component manager map: " << processName << std::endl;
        }
    } else {
        CMN_LOG_CLASS_RUN_ERROR << "Failed to add process in process map: " << processName << std::endl;
    }

    return success;
}

bool mtsManagerGlobal::FindProcess(const std::string & processName) const
{
    return ProcessMap.FindItem(processName);
}

mtsManagerLocalInterface * mtsManagerGlobal::GetProcessObject(const std::string & processName)
{
    if (!ProcessMap.FindItem(processName)) {
        CMN_LOG_CLASS_RUN_ERROR << "GetProcessObject: Can't find registered process: " << processName << std::endl;
        return false;
    }

    return LocalManagerMap.GetItem(processName);
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

    if (!FindProcess(processName)) {
        CMN_LOG_CLASS_RUN_ERROR << "RemoveProcess: Can't find registered process: " << processName << std::endl;
        return false;
    }

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

    LocalManagerMapChange.Lock();
    ret &= LocalManagerMap.RemoveItem(processName);
    LocalManagerMapChange.Unlock();

    // Remove the process from process map
    ret &= ProcessMap.RemoveItem(processName);

    return ret;
}

//-------------------------------------------------------------------------
//  Component Management
//-------------------------------------------------------------------------
bool mtsManagerGlobal::AddComponent(const std::string & processName, const std::string & componentName)
{
    if (!FindProcess(processName)) {
        CMN_LOG_CLASS_RUN_ERROR << "AddComponent: Can't find registered process: " << processName << std::endl;
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
        CMN_LOG_CLASS_RUN_ERROR << "AddComponent: Can't add a component: " 
            << "\"" << processName << "\" - \"" << componentName << "\"" << std::endl;
    }

    return ret;
}

bool mtsManagerGlobal::FindComponent(const std::string & processName, const std::string & componentName) const
{
    if (!FindProcess(processName)) {
        CMN_LOG_CLASS_RUN_ERROR << "FindComponent: Can't find a registered process: " << processName << std::endl;
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
    // Check if the component has been registered
    if (!FindComponent(processName, componentName)) {
        CMN_LOG_CLASS_RUN_ERROR << "RemoveComponent: Can't find component: " 
            << processName << ":" << componentName << std::endl;
        return false;
    }

    bool ret = true;

    ComponentMapType * componentMap = ProcessMap.GetItem(processName);
    CMN_ASSERT(componentMap);

    // When interfaceMapType is not NULL, all interfaces that the component manages
    // should be removed first.
    InterfaceMapType * interfaceMap = componentMap->GetItem(componentName);
    if (interfaceMap) {
        ConnectedInterfaceMapType::iterator it;

        // Remove all the required interfaces that the process manage.
        it = interfaceMap->RequiredInterfaceMap.GetMap().begin();
        while (it != interfaceMap->RequiredInterfaceMap.GetMap().end()) {
            ret &= RemoveRequiredInterface(processName, componentName, it->first);
            it = interfaceMap->RequiredInterfaceMap.GetMap().begin();
        }

        // Remove all the provided interfaces that the process manage.
        it = interfaceMap->ProvidedInterfaceMap.GetMap().begin();
        while (it != interfaceMap->ProvidedInterfaceMap.GetMap().end()) {
            ret &= RemoveProvidedInterface(processName, componentName, it->first);
            it = interfaceMap->ProvidedInterfaceMap.GetMap().begin();
        }

        CHECK_DELETE(interfaceMap);
    }

    // Remove the component from component map
    ret &= componentMap->RemoveItem(componentName);

    return ret;
}

//-------------------------------------------------------------------------
//  Interface Management
//-------------------------------------------------------------------------
bool mtsManagerGlobal::AddProvidedInterface(
    const std::string & processName, const std::string & componentName, const std::string & interfaceName, const bool isProxyInterface)
{
    if (!FindComponent(processName, componentName)) {
        CMN_LOG_CLASS_RUN_ERROR << "AddProvidedInterface: Can't find a registered component: " 
            << "\"" << processName << "\" - \"" << componentName << "\"" << std::endl;
        return false;
    }

    ComponentMapType * componentMap = ProcessMap.GetItem(processName);
    InterfaceMapType * interfaceMap = componentMap->GetItem(componentName);

    // If the component did not have any interface before
    if (interfaceMap == NULL) {
        interfaceMap = new InterfaceMapType;
        (componentMap->GetMap())[componentName] = interfaceMap;

        // TODO: Remove this
        AllocatedPointers[reinterpret_cast<unsigned int>(interfaceMap)] = __LINE__;
    }

    // Add the interface    
    if (!interfaceMap->ProvidedInterfaceMap.AddItem(interfaceName, NULL)) {
        CMN_LOG_CLASS_RUN_ERROR << "AddProvidedInterface: Can't add a provided interface: " << interfaceName << std::endl;
        return false;
    }

    interfaceMap->ProvidedInterfaceTypeMap[interfaceName] = isProxyInterface;

    return true;
}

bool mtsManagerGlobal::AddRequiredInterface(
    const std::string & processName, const std::string & componentName, const std::string & interfaceName, const bool isProxyInterface)
{
    if (!FindComponent(processName, componentName)) {
        CMN_LOG_CLASS_RUN_ERROR << "AddRequiredInterface: Can't find a registered component: " 
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

    // Add the interface    
    if (!interfaceMap->RequiredInterfaceMap.AddItem(interfaceName, NULL)) {
        CMN_LOG_CLASS_RUN_ERROR << "AddRequiredInterface: Can't add a required interface: " << interfaceName << std::endl;
        return false;
    }

    interfaceMap->RequiredInterfaceTypeMap[interfaceName] = isProxyInterface;

    return true;
}

bool mtsManagerGlobal::FindProvidedInterface(
    const std::string & processName, const std::string & componentName, const std::string & interfaceName) const
{
    if (!FindComponent(processName, componentName)) {
        CMN_LOG_CLASS_RUN_ERROR << "FindProvidedInterface: Can't find a registered component: " 
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
        CMN_LOG_CLASS_RUN_ERROR << "FindRequiredInterface: Can't find a registered component: " 
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
    if (!ProcessMap.FindItem(processName)) {
        CMN_LOG_CLASS_RUN_ERROR << "RemoveProvidedInterface: Can't find registered process: " << processName << std::endl;
        return false;
    }

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
    // Check if the process exists
    if (!ProcessMap.FindItem(processName)) {
        CMN_LOG_CLASS_RUN_ERROR << "RemoveRequiredInterface: Can't find registered process: " << processName << std::endl;
        return false;
    }

    // Check if the component exists
    ComponentMapType * componentMap = ProcessMap.GetItem(processName);
    CMN_ASSERT(componentMap);
    if (!componentMap->FindItem(componentName)) return false;

    // Check if the provided interface exists
    InterfaceMapType * interfaceMap = componentMap->GetItem(componentName);
    if (!interfaceMap) return false;
    if (!interfaceMap->RequiredInterfaceMap.FindItem(interfaceName)) return false;

    // When connectionMap is not NULL, all the connections that the provided 
    // interface is connected to should be removed.
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
unsigned int mtsManagerGlobal::Connect(
    const std::string & clientProcessName,
    const std::string & clientComponentName,
    const std::string & clientRequiredInterfaceName,
    const std::string & serverProcessName,
    const std::string & serverComponentName,
    const std::string & serverProvidedInterfaceName)
{
    // Check if the required interface specified actually exist.
    if (!FindRequiredInterface(clientProcessName, clientComponentName, clientRequiredInterfaceName)) {
        CMN_LOG_CLASS_RUN_VERBOSE << "Connect: required interface does not exist: "
            << GetInterfaceUID(clientProcessName, clientComponentName, clientRequiredInterfaceName)
            << std::endl;
        return static_cast<unsigned int>(mtsManagerGlobalInterface::CONNECT_ERROR);
    }

    // Check if the provided interface specified actually exist.
    if (!FindProvidedInterface(serverProcessName, serverComponentName, serverProvidedInterfaceName)) {
        CMN_LOG_CLASS_RUN_VERBOSE << "Connect: provided interface does not exist: "
            << GetInterfaceUID(serverProcessName, serverComponentName, serverProvidedInterfaceName)
            << std::endl;
        return static_cast<unsigned int>(mtsManagerGlobalInterface::CONNECT_ERROR);
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
        CMN_LOG_CLASS_RUN_VERBOSE << "Connect: "
            << "one or more processes, components, or interfaces are missing." << std::endl;
        return static_cast<unsigned int>(mtsManagerGlobalInterface::CONNECT_ERROR);
    }
    // When interfaces have already been connected to each other
    else if (ret > 0) {
        CMN_LOG_CLASS_RUN_VERBOSE << "Connect: Two interfaces are already connected: "
            << GetInterfaceUID(clientProcessName, clientComponentName, clientRequiredInterfaceName)
            << " and "
            << GetInterfaceUID(serverProcessName, serverComponentName, serverProvidedInterfaceName)
            << std::endl;
        return static_cast<unsigned int>(mtsManagerGlobalInterface::CONNECT_ERROR);
    }

    // Determine the type of connection: local vs. remote
    unsigned int thisConnectionID;
    bool isRemoteConnection = false;

    // In case of local connection
    if (clientProcessName == serverProcessName) {
        thisConnectionID = static_cast<unsigned int>(mtsManagerGlobalInterface::CONNECT_LOCAL);        
    }
    // In case of remote connection
    else {
        // Term definitions
        // - Server manager : local component manager that manages server components
        // - Client manager : local component manager that manages client components
        
        // Get local component managers that manages the client and the server component.
        mtsManagerLocalInterface * localManagerClient = GetProcessObject(clientProcessName);
        if (!localManagerClient) {
            CMN_LOG_CLASS_RUN_ERROR << "Connect: cannot find local component manager with client process: " << clientProcessName << std::endl;
            return static_cast<unsigned int>(mtsManagerGlobalInterface::CONNECT_ERROR);
        }

        mtsManagerLocalInterface * localManagerServer = GetProcessObject(serverProcessName);
        if (!localManagerServer) {
            CMN_LOG_CLASS_RUN_ERROR << "Connect: cannot find local component manager with server process: " << serverProcessName << std::endl;
            return static_cast<unsigned int>(mtsManagerGlobalInterface::CONNECT_ERROR);
        }

        // Check if the server manager has the client component proxy. If not,
        // create one.
        const std::string clientComponentProxyName = GetComponentProxyName(clientProcessName, clientComponentName);
        if (!FindComponent(serverProcessName, clientComponentProxyName)) {
            if (!localManagerServer->CreateComponentProxy(clientComponentProxyName)) {
                CMN_LOG_CLASS_RUN_ERROR << "Connect: failed to create client component proxy" << std::endl;
                return false;
            }
            CMN_LOG_CLASS_RUN_VERBOSE << "Connect: client component proxy is created: " 
                << clientComponentProxyName << " on " << localManagerServer->GetProcessName() << std::endl;
        }

        // Check if the client manager has the client component proxy. If not,
        // create one.
        const std::string serverComponentProxyName = GetComponentProxyName(serverProcessName, serverComponentName);
        if (!FindComponent(clientProcessName, serverComponentProxyName)) {
            if (!localManagerClient->CreateComponentProxy(serverComponentProxyName)) {
                CMN_LOG_CLASS_RUN_ERROR << "Connect: failed to create server component proxy" << std::endl;
                return false;
            }
            CMN_LOG_CLASS_RUN_VERBOSE << "Connect: server component proxy is created: " 
                << serverComponentProxyName << " on " << localManagerClient->GetProcessName() << std::endl;
        }

        // Check if the specified interfaces exist in each process. If not,
        // create an interface proxy.
        // Note that, under the current design, a required interface can connect
        // to multiple provided interfaces whereas a required interface connects
        // to only one provided interface.
        // Thus, a required interface proxy is created whenever a server component
        // doesn't have it while a provided interface proxy is generated only at 
        // the first time when a client component doesn't have it.
        
        // Check if provided interface proxy already exists at the client side.
        bool foundProvidedInterfaceProxy = FindProvidedInterface(clientProcessName, serverComponentProxyName, serverProvidedInterfaceName);

        // Check if required interface proxy already exists at the server side.
        bool foundRequiredInterfaceProxy = FindRequiredInterface(serverProcessName, clientComponentProxyName, clientRequiredInterfaceName);

        // Create an interface proxy (or proxies) as needed.
        //
        // From server and client managers, extract the information about the 
        // two interfaces specified. The global component manager will
        // deliver this information to peer local component managers so that 
        // they can create proxy components.

        // Create provided interface proxy
        if (!foundProvidedInterfaceProxy) {
            // Extract provided interface description
            ProvidedInterfaceDescription providedInterfaceDescription;
            if (!localManagerServer->GetProvidedInterfaceDescription(
                serverComponentName, serverProvidedInterfaceName, providedInterfaceDescription)) 
            {
                CMN_LOG_CLASS_RUN_ERROR << "Connect: failed to get provided interface description: "
                    << serverProcessName << ":" << serverComponentName << ":" << serverProvidedInterfaceName << std::endl;
                return false;
            }

            // Create provided interface proxy at the client side
            if (!localManagerClient->CreateProvidedInterfaceProxy(serverComponentProxyName, providedInterfaceDescription)) {
                CMN_LOG_CLASS_RUN_ERROR << "Connect: failed to create provided interface proxy: "
                    << serverComponentProxyName << " in " << clientProcessName << std::endl;
                return false;
            }
        }
        
        // Create required interface proxy
        if (!foundRequiredInterfaceProxy) {
            // Extract required interface description
            RequiredInterfaceDescription requiredInterfaceDescription;
            if (!localManagerClient->GetRequiredInterfaceDescription(
                clientComponentName, clientRequiredInterfaceName, requiredInterfaceDescription)) 
            {
                CMN_LOG_CLASS_RUN_ERROR << "Connect: failed to get required interface description: "
                    << clientProcessName << ":" << clientComponentName << ":" << clientRequiredInterfaceName << std::endl;
                return false;
            }

            // Create required interface proxy at the server side
            if (!localManagerServer->CreateRequiredInterfaceProxy(clientComponentProxyName, requiredInterfaceDescription)) {
                CMN_LOG_CLASS_RUN_ERROR << "Connect: failed to create required interface proxy: "
                    << clientComponentProxyName << " in " << serverProcessName << std::endl;
                return false;
            }
        }

        // TODO: implement here
        //
        // 1. create connection information structure element
        // 2. assign connection id for the element
        // 3. enqueue the element with timer set
        // 4. let two LCMs create proxy components
        // 5. return connectionID back to LCM while iterating timers, if any
        thisConnectionID = ConnectionID++;
        isRemoteConnection = true;
    }


    // Step 1

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
    if (!AddConnectedInterface(connectionMap, serverProcessName, serverComponentName, serverProvidedInterfaceName, isRemoteConnection)) {
        CMN_LOG_CLASS_RUN_ERROR << "Connect: "
            << "failed to add information about connected provided interface." << std::endl;
        return static_cast<unsigned int>(mtsManagerGlobalInterface::CONNECT_ERROR);
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
    if (!AddConnectedInterface(connectionMap, clientProcessName, clientComponentName, clientRequiredInterfaceName, isRemoteConnection)) {
        CMN_LOG_CLASS_RUN_ERROR << "Connect: "
            << "failed to add information about connected required interface." << std::endl;

        // Before returning false, should clean up required interface's connection information
        Disconnect(clientProcessName, clientComponentName, clientRequiredInterfaceName,
                   serverProcessName, serverComponentName, serverProvidedInterfaceName);
        return static_cast<unsigned int>(mtsManagerGlobalInterface::CONNECT_ERROR);
    }

    CMN_LOG_CLASS_RUN_VERBOSE << "Connect: successfully connected: " 
        << GetInterfaceUID(clientProcessName, clientComponentName, clientRequiredInterfaceName) << " - "
        << GetInterfaceUID(serverProcessName, serverComponentName, serverProvidedInterfaceName) << std::endl;

    return thisConnectionID;
}

bool mtsManagerGlobal::ConnectConfirm(unsigned int connectionSessionID)
{
    //
    // TODO: handle connectionSessionID
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

bool mtsManagerGlobal::Disconnect(
    const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
    const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName)
{
    bool success = false;

    // Get connection information
    ConnectionMapType * connectionMapOfRequiredInterface = GetConnectionsOfRequiredInterface(
        clientProcessName, clientComponentName, clientRequiredInterfaceName);
    if (!connectionMapOfRequiredInterface) {
        CMN_LOG_CLASS_RUN_ERROR << "Disconnect: failed to disconnect. Required interface has no connection: "
            << GetInterfaceUID(clientProcessName, clientComponentName, clientRequiredInterfaceName) << std::endl;
        return false;
    }

    ConnectionMapType * connectionMapOfProvidedInterface = GetConnectionsOfProvidedInterface(
        serverProcessName, serverComponentName, serverProvidedInterfaceName);
    if (!connectionMapOfProvidedInterface) {
        CMN_LOG_CLASS_RUN_ERROR << "Disconnect: failed to disconnect. Provided interface has no connection: "
            << GetInterfaceUID(serverProcessName, serverComponentName, serverProvidedInterfaceName) << std::endl;
        return false;
    }
    
    bool remoteConnection = false;  // true if the connection to be disconnected is a remote connection.
    std::string interfaceUID;
    ConnectedInterfaceInfo * connectionInfo;

    // Remove required interfaces' connection information
    if (connectionMapOfRequiredInterface->size()) {
        // Get an element that contains connection information
        interfaceUID = GetInterfaceUID(serverProcessName, serverComponentName, serverProvidedInterfaceName);
        connectionInfo = connectionMapOfRequiredInterface->GetItem(interfaceUID);        

        // Release allocated memory
        if (connectionInfo) {
            remoteConnection = connectionInfo->IsRemoteConnection();
            CHECK_DELETE(connectionInfo);
        }

        // Remove connection information
        if (connectionMapOfRequiredInterface->FindItem(interfaceUID)) {
            if (!connectionMapOfRequiredInterface->RemoveItem(interfaceUID)) {
                CMN_LOG_CLASS_RUN_ERROR << "Disconnect: failed to update connection map at server side" << std::endl;
                return false;
            }

            // If the required interface is a proxy object (not an original interface), 
            // it should be removed when the connection is disconnected.
            if (remoteConnection) {
                mtsManagerLocalInterface * localManagerServer = GetProcessObject(serverProcessName);
                const std::string clientComponentProxyName = GetComponentProxyName(clientProcessName, clientComponentName);
                if (!localManagerServer->RemoveRequiredInterfaceProxy(clientComponentProxyName, clientRequiredInterfaceName)) 
                {
                    CMN_LOG_CLASS_RUN_ERROR << "Disconnect: failed to update local component manager at server side" << std::endl;
                    return false;
                }

                // Check if the component proxy should be removed (because no interface exists on it)
                if (localManagerServer->GetCurrentInterfaceCount(clientComponentProxyName) == 0) {
                    CMN_LOG_CLASS_RUN_VERBOSE <<"Disconnect: remove client component proxy with no active interface: " << clientComponentProxyName << std::endl;
                    if (!localManagerServer->RemoveComponentProxy(clientComponentProxyName)) {
                        CMN_LOG_CLASS_RUN_ERROR << "Disconnect: failed to remove client component proxy: " 
                            << clientComponentProxyName << " on " << serverProcessName << std::endl;
                        return false;
                    }
                }
            }

            success = true;
        }
    }

    // Remove provided interfaces' connection information
    if (connectionMapOfProvidedInterface->size()) {
        // Get an element that contains connection information
        interfaceUID = GetInterfaceUID(clientProcessName, clientComponentName, clientRequiredInterfaceName);
        connectionInfo = connectionMapOfProvidedInterface->GetItem(interfaceUID);        

        // Release allocated memory
        if (connectionInfo) {
            remoteConnection = connectionInfo->IsRemoteConnection();
            CHECK_DELETE(connectionInfo);
        }

        // Remove connection information
        if (connectionMapOfProvidedInterface->FindItem(interfaceUID)) {
            if (!connectionMapOfProvidedInterface->RemoveItem(interfaceUID)) {
                CMN_LOG_CLASS_RUN_ERROR << "Disconnect: failed to update connection map at client side" << std::endl;
                return false;
            }

            // If the required interface is a proxy object (not an original interface), 
            // it should be removed when the connection is disconnected.
            if (remoteConnection) {
                mtsManagerLocalInterface * localManagerClient = GetProcessObject(clientProcessName);
                const std::string serverComponentProxyName = GetComponentProxyName(serverProcessName, serverComponentName);
                if (!localManagerClient->RemoveProvidedInterfaceProxy(serverComponentProxyName, serverProvidedInterfaceName))
                {
                    CMN_LOG_CLASS_RUN_ERROR << "Disconnect: failed to update local component manager at client side" << std::endl;
                    return false;
                }

                // Check if the component proxy should be removed (because no interface exists on it)
                if (localManagerClient->GetCurrentInterfaceCount(serverComponentProxyName) == 0) {
                    CMN_LOG_CLASS_RUN_VERBOSE <<"Disconnect: remove server component proxy with no active interface: " << serverComponentProxyName << std::endl;
                    if (!localManagerClient->RemoveComponentProxy(serverComponentProxyName)) {
                        CMN_LOG_CLASS_RUN_ERROR << "Disconnect: failed to remove server component proxy: " 
                            << serverComponentProxyName << " on " << clientProcessName << std::endl;
                        return false;
                    }
                }
            }

            success = true;
        }
    }

    if (success) {
        CMN_LOG_CLASS_RUN_VERBOSE << "Disconnect: successfully disconnected: "
            << GetInterfaceUID(clientProcessName, clientComponentName, clientRequiredInterfaceName) << " - "
            << GetInterfaceUID(serverProcessName, serverComponentName, serverProvidedInterfaceName) << std::endl;
    }

    return success;
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
    const std::string & interfaceName, const bool isRemoteConnection)
{
    if (!connectionMap) return false;

    ConnectedInterfaceInfo * connectedInterfaceInfo = 
        new ConnectedInterfaceInfo(processName, componentName, interfaceName, isRemoteConnection);

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