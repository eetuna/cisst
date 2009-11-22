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
void mtsManagerGlobal::CleanUp(void)
{    
    // Remove all processes safely
    for (ProcessMapType::iterator it = ProcessMap.GetMap().begin(); it != ProcessMap.end(); ++it)
    {
        RemoveProcess(it->first);
    }
}

/*
mtsManagerGlobal::ComponentMapType * mtsManagerGlobal::GetComponentMap(
    const std::string & processName)
{
    if (!FindProcess(processName)) {
        CMN_LOG_CLASS_RUN_ERROR << "Cannot get component map: no process \"" << processName
            << "\" is found." << std::endl;
        return false;
    }

    return ProcessMap.GetItem(processName, CMN_LOG_LOD_VERY_VERBOSE);
}

mtsManagerGlobal::ConnectedInterfaceMapType * mtsManagerGlobal::GetProvidedInterfaceMap(
    const std::string & processName, const std::string & componentName)
{
    if (!FindComponent(processName, componentName)) {
        CMN_LOG_CLASS_RUN_ERROR << "Cannot get provided interface map: no process \"" << processName
            << "\" with component \"" << componentName << "\" is found." << std::endl;
        return false;
    }

    //mtsManagerGlobal::ComponentMapType * componentMap = GetComponentMap(processName);

    //if (componentMap == NULL) {

    //GetComponentMap
    return NULL;
}

mtsManagerGlobal::ConnectedInterfaceMapType * mtsManagerGlobal::GetRequiredInterfaceMap(
    const std::string & processName, const std::string & componentName)
{
    return NULL;
}

mtsManagerGlobal::ConnectionMapType * mtsManagerGlobal::GetProvidedInterfaceConnectionMap(
    const std::string & processName, const std::string & componentName, const std::string & providedInterfaceName)
{
    return NULL;
}

mtsManagerGlobal::ConnectionMapType * mtsManagerGlobal::GetRequiredInterfaceConnectionMap(
    const std::string & processName, const std::string & componentName, const std::string & requiredInterfaceName)
{
    return NULL;
}
*/

//-------------------------------------------------------------------------
//  Process Management
//-------------------------------------------------------------------------
bool mtsManagerGlobal::AddProcess(const std::string & processName)
{
    // AddProcess() doesn't need to be called to check duplicate process registration
    // since cmnNamedMap::AddItem() internally checks duplicity before adding an item.

    bool ret = ProcessMap.AddItem(processName, NULL);
    if (!ret) {
        CMN_LOG_CLASS_RUN_ERROR << "Can't add process: " << processName << std::endl;
    }

    return ret;
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
    bool ret = true;

    ComponentMapType * componentMap = ProcessMap.GetItem(processName);
    if (componentMap) {
        for (ComponentMapType::iterator it = componentMap->GetMap().begin(); it != componentMap->GetMap().end(); ++it) 
        {
            ret &= RemoveComponent(processName, it->first);
        }
        delete componentMap;
    }    
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

    // If the process did not registered before
    if (componentMap == NULL) {
        componentMap = new ComponentMapType(processName);
        
        if (!ProcessMap.AddItem(processName, componentMap)) {
            CMN_LOG_CLASS_RUN_ERROR << "Faild to add a process: " 
                << "\"" << processName << "\"" << std::endl;
            return false;
        }
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

/*! Remove a component from the global manager. */
bool mtsManagerGlobal::RemoveComponent(
    const std::string & processName, const std::string & componentName)
{
    // TODO: Before removing an element from relevent maps, Disconnect() should be 
    // called first for a case that there is any active connection related to the element.

    if (!FindComponent(processName, componentName)) {
        return false;
    }

    ComponentMapType * componentMap = ProcessMap.GetItem(processName);    
    // If there is no component registered
    if (componentMap == NULL) {
        return true;
    }

    bool ret = true;

    InterfaceMapType * interfaceMap = componentMap->GetItem(componentName);
    if (interfaceMap) {
        ConnectedInterfaceMapType::iterator it;

        // Remove provided interfaces
        for (it = interfaceMap->ProvidedInterfaceMap.GetMap().begin(); 
             it != interfaceMap->ProvidedInterfaceMap.GetMap().end(); ++it)
        {
            ret &= RemoveProvidedInterface(processName, componentName, it->first);
        }
        // Remove required interfaces
        for (it = interfaceMap->RequiredInterfaceMap.GetMap().begin(); 
             it != interfaceMap->RequiredInterfaceMap.GetMap().end(); ++it)
        {
            ret &= RemoveRequiredInterface(processName, componentName, it->first);
        }
        delete interfaceMap;
    }    
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
        
        if (!componentMap->AddItem(componentName, interfaceMap)) {
            CMN_LOG_CLASS_RUN_ERROR << "Failed to add a component: " 
                << "\"" << processName << "\" - \"" << componentName << "\"" << std::endl;
            return false;
        }
    }

    bool ret = interfaceMap->ProvidedInterfaceMap.AddItem(interfaceName, NULL);
    if (!ret) {
        CMN_LOG_CLASS_RUN_ERROR << "Can't add a provided interface: " 
            << "\"" << processName << "\" - \"" << componentName << "\" - \"" << interfaceName << "\"" << std::endl;
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
    InterfaceMapType * interfaceMap = componentMap->GetItem(componentName);
    if (!interfaceMap) {
        return false;
    }

    return interfaceMap->ProvidedInterfaceMap.FindItem(interfaceName);
}

bool mtsManagerGlobal::RemoveProvidedInterface(
    const std::string & processName, const std::string & componentName, const std::string & interfaceName)
{
    if (!FindProvidedInterface(processName, componentName, interfaceName)) {
        return false;
    }
    
    // If there is no component registered
    ComponentMapType * componentMap = ProcessMap.GetItem(processName);
    if (componentMap == NULL) {
        // NOP
        return true;
    }

    // If there is no interface registered
    InterfaceMapType * interfaceMap = componentMap->GetItem(componentName);
    if (interfaceMap == NULL) {
        // NOP
        return true;
    }
    
    bool ret = true;

    ConnectionMapType * connectionMap = interfaceMap->ProvidedInterfaceMap.GetItem(interfaceName);
    if (connectionMap) {
        ConnectionMapType::iterator it;

        for (it = connectionMap->GetMap().begin(); 
             it != connectionMap->GetMap().end(); ++it)
        {
            ret &= Disconnect(it->second->GetProcessName(),
                              it->second->GetComponentName(),
                              it->second->GetInterfaceName(),
                              processName,
                              componentName,
                              interfaceName);
        }
        delete connectionMap;
    }
    ret &= interfaceMap->ProvidedInterfaceMap.RemoveItem(interfaceName);

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
        CMN_LOG_CLASS_RUN_ERROR << "Can't connect: "
            << "required interface does not exist: "
            << "\"" << clientProcessName << "\" - \"" << clientComponentName 
            << "\"" << clientRequiredInterfaceName << "\"" << std::endl;
        return false;
    }

    // Check if the provided interface specified actually exist.
    if (!FindProvidedInterface(serverProcessName, serverComponentName, serverProvidedInterfaceName)) {
        CMN_LOG_CLASS_RUN_ERROR << "Can't connect: "
            << "provided interface does not exist: "
            << "\"" << serverProcessName << "\" - \"" << serverComponentName 
            << "\"" << serverProvidedInterfaceName << "\"" << std::endl;
        return false;
    }

    //
    // TODO: define mtsComponentManagerInterface first.
    //

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
    ConnectedInterfaceInfo * connectedInterfaceInfo;

    //
    // Connect client's required interface with server's provided interface.
    //
    connectionMap = GetRequiredInterfaceConnectionMap(clientProcessName,
                                                      clientComponentName,
                                                      clientRequiredInterfaceName,
                                                      &interfaceMap);
    // If this is the first time for the required interface to connect to other interfaces
    if (connectionMap == NULL) {
        connectedInterfaceInfo = new ConnectedInterfaceInfo(serverProcessName,
                                                            serverComponentName,
                                                            serverProvidedInterfaceName);
    }

    if (!interfaceMap->RequiredInterfaceMap.AddItem(serverProvidedInterfaceName, connectionMap)) {
        CMN_LOG_CLASS_RUN_ERROR << "Can't connect: "
            << "failed to add required interface's connection information" << std::endl;
        return false;
    }

    //
    // Connect server's provided interface with client's required interface.
    //
    connectionMap = GetProvidedInterfaceConnectionMap(serverProcessName,
                                                      serverComponentName,
                                                      serverProvidedInterfaceName,
                                                      &interfaceMap);
    // If this is the first time for the provided interface to connect to other interfaces
    if (connectionMap == NULL) {
        connectedInterfaceInfo = new ConnectedInterfaceInfo(clientProcessName,
                                                            clientComponentName,
                                                            clientRequiredInterfaceName);
    }

    if (!interfaceMap->ProvidedInterfaceMap.AddItem(clientRequiredInterfaceName, connectionMap)) {
        CMN_LOG_CLASS_RUN_ERROR << "Can't connect: "
            << "failed to add provided interface's connection information" << std::endl;

        // If provided interface's connection information failed to be updated,
        // required interface's connection information should be removed.
        if (!Disconnect(clientProcessName,
                        clientComponentName,
                        clientRequiredInterfaceName,
                        serverProcessName,
                        serverComponentName,
                        serverProvidedInterfaceName)) 
        {
            CMN_LOG_CLASS_RUN_ERROR << "SERIOUS ERROR: Disconnect() failed: "
                << "Connection information may corrupted!!!" << std::endl;
        }
        return false;
    }

    return true;
}

bool mtsManagerGlobal::Disconnect(
    const std::string & clientProcessName,
    const std::string & clientComponentName,
    const std::string & clientRequiredInterfaceName,
    const std::string & serverProcessName,
    const std::string & serverComponentName,
    const std::string & serverProvidedInterfaceName)
{
    // TODO: implement this
    return true;
}

mtsManagerGlobal::ConnectionMapType * mtsManagerGlobal::GetProvidedInterfaceConnectionMap(
    const std::string & severProcessName, const std::string & serverComponentName, 
    const std::string & providedInterfaceName, InterfaceMapType ** interfaceMap)
{
    ComponentMapType * componentMap = ProcessMap.GetItem(severProcessName);
    *interfaceMap = componentMap->GetItem(serverComponentName);
    ConnectionMapType * connectionMap = (*interfaceMap)->ProvidedInterfaceMap.GetItem(providedInterfaceName);

    return connectionMap;
}

mtsManagerGlobal::ConnectionMapType * mtsManagerGlobal::GetRequiredInterfaceConnectionMap(
    const std::string & clientProcessName, const std::string & clientComponentName, 
    const std::string & requiredInterfaceName, InterfaceMapType ** interfaceMap)
{
    ComponentMapType * componentMap = ProcessMap.GetItem(clientProcessName);
    *interfaceMap = componentMap->GetItem(clientComponentName);
    ConnectionMapType * connectionMap = (*interfaceMap)->RequiredInterfaceMap.GetItem(requiredInterfaceName);

    return connectionMap;
}