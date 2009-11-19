/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsGlobalManager.h 794 2009-09-01 21:43:56Z pkazanz1 $

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

#include <cisstMultiTask/mtsGlobalManager.h>

CMN_IMPLEMENT_SERVICES(mtsGlobalManager);

mtsGlobalManager::mtsGlobalManager()
{
}

mtsGlobalManager::~mtsGlobalManager()
{
    CleanUp();
}

void mtsGlobalManager::CleanUp(void)
{
    // TODO: add ProcessProxyMap clean-up
    ProcessMapType::iterator itProcess;
    ComponentMapType::iterator itComponent;

    for (itProcess = ProcessMap.GetMap().begin(); itProcess != ProcessMap.end(); ++itProcess) 
    {
        for (itComponent = itProcess->second->GetMap().begin();
             itComponent != itProcess->second->GetMap().end();
             ++itComponent)
        {
            delete itComponent->second;            
        }
        delete itProcess->second;
    }
}

//-----------------------------------------------------
//  Component Management
//-----------------------------------------------------
bool mtsGlobalManager::AddComponent(const std::string & processName, const std::string & componentName)
{
    if (FindComponent(processName, componentName)) {
        CMN_LOG_CLASS_RUN_WARNING << "Already added component (process: " << processName
            << ", component: " << componentName << ")" << std::endl;
        return false;
    }

    // If this process is not registered before, create a new process element first.
    ComponentMapType * componentMap = ProcessMap.GetItem(processName);
    if (componentMap == 0) {
        componentMap = new ComponentMapType(processName);
        ConnectionMapType * connectionMap = new ConnectionMapType(componentName);
        componentMap->AddItem(componentName, connectionMap);
        return ProcessMap.AddItem(processName, componentMap);
    }
    else
    {
        ConnectionMapType * connectionMap = new ConnectionMapType(componentName);
        return componentMap->AddItem(componentName, connectionMap);
    }
}

bool mtsGlobalManager::FindComponent(const std::string & processName, const std::string & componentName) const
{
    ComponentMapType * componentMap = ProcessMap.GetItem(processName);
    
    // no process found
    if (componentMap == 0) {
        return false;
    }

    ConnectionMapType * connectionMap = componentMap->GetItem(componentName);
    if (connectionMap == 0) {
        return false;
    }

    return true;
}

/*! Remove a component from the global manager. */
bool mtsGlobalManager::RemoveComponent(
    const std::string & processName, const std::string & componentName)
{
    if (FindComponent(processName, componentName)) {
        CMN_LOG_CLASS_RUN_WARNING << "Can't find registered component (process: " << processName
            << ", component: " << componentName << ")" << std::endl;
        return false;
    }

    ComponentMapType * componentMap = ProcessMap.GetItem(processName);
    CMN_ASSERT(componentMap);
    
    ConnectionMapType * connectionMap = componentMap->GetItem(componentName);
    CMN_ASSERT(connectionMap);

    // TODO: Before removing an element from relevent maps, Disconnect() should be 
    // called first if there is any active connection established before.

    return true;
}

//-----------------------------------------------------
//  Interface Management
//-----------------------------------------------------
bool AddInterface(
    const std::string & processName, const std::string & componentName,
    const std::string & interfaceName, const bool isProvidedInterface)
{
    return true;
}

bool mtsGlobalManager::FindInterface(
    const std::string & processName, const std::string & componentName,
    const std::string & interfaceName) const
{
    ComponentMapType * componentMap = ProcessMap.GetItem(processName);

    // no process found
    if (componentMap == 0) {
        return false;
    }

    ConnectionMapType * connectionMap = componentMap->GetItem(componentName);
    
    // no component found
    if (connectionMap == 0) {
        return false;
    }

    return (connectionMap->GetItem(interfaceName) != 0);
}

bool RemoveInterface(
    const std::string & processName, const std::string & componentName,
    const std::string & interfaceName, const bool isProvidedInterface)
{
    return true;
}

//-----------------------------------------------------
//  Connection Management
//-----------------------------------------------------
bool mtsGlobalManager::Connect(
    const std::string & clientProcessName,
    const std::string & clientComponentName,
    const std::string & clientRequiredInterfaceName,
    const std::string & serverProcessName,
    const std::string & serverComponentName,
    const std::string & serverProvidedInterfaceName)
{
    // Check argument validity
    //if (!FindComponent(clientProcessName, clientComponentName)

    return true;
}

bool mtsGlobalManager::Disconnect()
{
    // TODO: implement this
    return true;
}
