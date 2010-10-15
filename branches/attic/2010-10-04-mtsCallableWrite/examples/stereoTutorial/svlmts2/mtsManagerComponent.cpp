/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
 $Id: $

 Author(s):  Anton Deguet
 Created on: 2010

 (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
 Reserved.

 --- begin cisst license - do not edit ---

 This software is provided "as is" under an open source license, with
 no warranty.  The complete license can be found in license.txt and
 http://www.cisst.org/cisst/license.txt.

 --- end cisst license ---

 */

#include "mtsManagerComponent.h"
#include <cisstMultiTask/mtsInterfaceProvided.h>
#include <cisstMultiTask/mtsInterfaceRequired.h>

CMN_IMPLEMENT_SERVICES(mtsDescriptionNewComponent);
CMN_IMPLEMENT_SERVICES(mtsDescriptionConnection);
CMN_IMPLEMENT_SERVICES(mtsManagerComponent);

mtsManagerComponent::mtsManagerComponent(const std::string & componentName):
    mtsTaskFromSignal(componentName, 50)
{
    UseSeparateLogFileDefault();
    mtsInterfaceProvided * interfaceProvided;

    // interface provided for local components
    interfaceProvided = this->AddInterfaceProvided("ForComponents");
    interfaceProvided->AddCommandWrite(&mtsManagerComponent::CreateComponent, this, "CreateComponent");
    interfaceProvided->AddCommandWrite(&mtsManagerComponent::Connect, this, "Connect");

    // interface provided for remote component managers
    interfaceProvided = this->AddInterfaceProvided("ForManagers");
    interfaceProvided->AddCommandWrite(&mtsManagerComponent::CreateComponentLocally, this, "CreateComponent");
    interfaceProvided->AddCommandWrite(&mtsManagerComponent::ConnectLocally, this, "Connect");
}


void mtsManagerComponent::Startup(void)
{
}


void mtsManagerComponent::Run(void)
{
    ProcessQueuedCommands();
    ProcessQueuedEvents();
}


void mtsManagerComponent::ConnectToRemoteManager(const std::string & processName)
{
    // some error checks should be added
    OtherManager * otherManager = new OtherManager;
    OtherManagers.AddItem(processName, otherManager);
    otherManager->RequiredInterface = this->AddInterfaceRequired(processName);
    otherManager->RequiredInterface->AddFunction("CreateComponent", otherManager->CreateComponent);
    otherManager->RequiredInterface->AddFunction("Connect", otherManager->Connect);

    mtsComponentManager::GetInstance()->Connect(mtsComponentManager::GetInstance()->GetProcessName(),
                                                "Manager",
                                                processName,
                                                processName,
                                                "Manager",
                                                "ForManagers");
}


void mtsManagerComponent::CreateComponent(const mtsDescriptionNewComponent & component)
{
    CMN_LOG_CLASS_RUN_VERBOSE << "CreateComponent: called to created \"" << component << "\"" << std::endl; 
    if (component.ProcessName == mtsComponentManager::GetInstance()->GetProcessName()) {
        CreateComponentLocally(component);
    } else {
        OtherManager * otherManager = OtherManagers.GetItem(component.ProcessName);
        otherManager->CreateComponent(component);
    }
}


void mtsManagerComponent::CreateComponentLocally(const mtsDescriptionNewComponent & component)
{
    CMN_LOG_CLASS_RUN_VERBOSE << "CreateComponentLocally: called to created \"" << component << "\"" << std::endl; 
    // looking in class register to create this component
    cmnGenericObject * basePointer = cmnClassRegister::Create(component.ClassName);
    if (!basePointer) {
        CMN_LOG_CLASS_INIT_ERROR << "CreateComponentLocally: unable to create component of type \""
                                 << component.ClassName << "\"" << std::endl;
        return;
    }
    // make sure this is an mtsComponent
    mtsComponent * componentPointer = dynamic_cast<mtsComponent *>(basePointer);
    if (!componentPointer) {
        CMN_LOG_CLASS_INIT_ERROR << "CreateComponentLocally: class \"" << component.ClassName
                                 << "\" is not derived from mtsComponent" << std::endl;
        delete basePointer;
        return;
    }
    // rename the component
    componentPointer->SetName(component.ComponentName);
    mtsManagerLocal::GetInstance()->AddComponent(componentPointer);
}


void mtsManagerComponent::Connect(const mtsDescriptionConnection & connection)
{
    CMN_LOG_CLASS_RUN_VERBOSE << "Connect: called to connect \"" << connection << "\"" << std::endl; 
    if ((connection.Client.ProcessName == mtsComponentManager::GetInstance()->GetProcessName())
        || (connection.Server.ProcessName == mtsComponentManager::GetInstance()->GetProcessName())) {
        ConnectLocally(connection);
    } else {
        OtherManager * otherManager = OtherManagers.GetItem(connection.Client.ProcessName);
        otherManager->Connect(connection);
    }
}


void mtsManagerComponent::ConnectLocally(const mtsDescriptionConnection & connection)
{
    CMN_LOG_CLASS_RUN_VERBOSE << "ConnectLocally: called to connect \"" << connection << "\"" << std::endl; 
    mtsManagerLocal::GetInstance()->Connect(connection.Client.ProcessName,
                                            connection.Client.ComponentName,
                                            connection.Client.InterfaceName,
                                            connection.Server.ProcessName,
                                            connection.Server.ComponentName,
                                            connection.Server.InterfaceName);
}
