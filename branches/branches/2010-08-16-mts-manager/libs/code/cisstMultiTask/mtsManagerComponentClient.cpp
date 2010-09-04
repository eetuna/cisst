/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerComponentClient.cpp 1726 2010-08-30 05:07:54Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2010-08-29

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstMultiTask/mtsManagerComponentClient.h>
#include <cisstMultiTask/mtsInterfaceProvided.h>
#include <cisstMultiTask/mtsInterfaceRequired.h>
#include <cisstOSAbstraction/osaGetTime.h>

CMN_IMPLEMENT_SERVICES(mtsManagerComponentClient);

std::string mtsManagerComponentClient::NameOfInterfaceComponentProvided = "InterfaceComponentProvided";
std::string mtsManagerComponentClient::NameOfInterfaceComponentRequired = "InterfaceComponentRequired";
std::string mtsManagerComponentClient::NameOfInterfaceLCMProvided       = "InterfaceLCMProvided";
std::string mtsManagerComponentClient::NameOfInterfaceLCMRequired       = "InterfaceLCMRequired";

const std::string SuffixManagerComponentClient = "_MNGR_COMP";

mtsManagerComponentClient::mtsManagerComponentClient(const std::string & componentName)
    : mtsManagerComponentBase(componentName)
{
}

mtsManagerComponentClient::~mtsManagerComponentClient()
{
}

std::string mtsManagerComponentClient::GetNameOfManagerComponentClient(const std::string & processName)
{
    std::string name(processName);
    name += SuffixManagerComponentClient;

    return name;
}

void mtsManagerComponentClient::Startup(void)
{
   CMN_LOG_CLASS_INIT_VERBOSE << "Manager component CLIENT starts" << std::endl;
}

void mtsManagerComponentClient::Run(void)
{
    mtsManagerComponentBase::Run();
}

void mtsManagerComponentClient::Cleanup(void)
{
}

bool mtsManagerComponentClient::CreateAndAddNewComponent(const std::string & className, const std::string & componentName)
{
    // Try to create component as requested
    mtsManagerLocal * LCM = mtsManagerLocal::GetInstance();

    mtsComponent * newComponent = LCM->CreateComponent(className, componentName);
    if (!newComponent) {
        CMN_LOG_CLASS_RUN_ERROR << "CreateAndAddNewComponent: failed to create component: " 
            << "\"" << componentName << "\" of type \"" << className << "\"" << std::endl;
        return false;
    }

    if (!LCM->AddComponent(newComponent)) {
        CMN_LOG_CLASS_RUN_ERROR << "CreateAndAddNewComponent: failed to add component: "
            << "\"" << componentName << "\" of type \"" << className << "\"" << std::endl;
        return false;
    }

    CMN_LOG_CLASS_RUN_VERBOSE << "CreateAndAddNewComponent: successfully added component: "
        << "\"" << componentName << "\" of type \"" << className << "\"" << std::endl;

    return true;
}

bool mtsManagerComponentClient::AddInterfaceComponent(void)
{
    // InterfaceComponent's required interface is not create here but is created
    // when a user component with internal interfaces connects to the manager 
    // component client.  
    // See mtsManagerComponentClient::CreateInterfaceComponentFunctionSet()
    // for the creation of required interfaces.

    // Add provided interface to which InterfaceInternal's required interface connects.
    std::string interfaceName = mtsManagerComponentClient::NameOfInterfaceComponentProvided;
    mtsInterfaceProvided * provided = AddInterfaceProvided(interfaceName);
    if (!provided) {
        CMN_LOG_CLASS_INIT_ERROR << "AddInterfaceComponent: failed to add \"Component\" provided interface: " << interfaceName << std::endl;
        return false;
    }

    provided->AddCommandWrite(&mtsManagerComponentClient::InterfaceComponentCommands_ComponentCreate,
                              this, mtsManagerComponentBase::CommandNames::ComponentCreate);
    provided->AddCommandWrite(&mtsManagerComponentClient::InterfaceComponentCommands_ComponentConnect,
                              this, mtsManagerComponentBase::CommandNames::ComponentConnect);
    provided->AddCommandRead(&mtsManagerComponentClient::InterfaceComponentCommands_GetNamesOfProcesses,
                              this, mtsManagerComponentBase::CommandNames::GetNamesOfProcesses);
    provided->AddCommandQualifiedRead(&mtsManagerComponentClient::InterfaceComponentCommands_GetNamesOfComponents,
                              this, mtsManagerComponentBase::CommandNames::GetNamesOfComponents);
    provided->AddCommandQualifiedRead(&mtsManagerComponentClient::InterfaceComponentCommands_GetNamesOfInterfaces,
                              this, mtsManagerComponentBase::CommandNames::GetNamesOfInterfaces);
    provided->AddCommandRead(&mtsManagerComponentClient::InterfaceComponentCommands_GetListOfConnections,
                              this, mtsManagerComponentBase::CommandNames::GetListOfConnections);
    
    CMN_LOG_CLASS_INIT_VERBOSE << "AddInterfaceComponent: successfully added \"Component\" interfaces" << std::endl;

    return true;
}

bool mtsManagerComponentClient::AddInterfaceLCM(void)
{
    // Add required interface
    std::string interfaceName = mtsManagerComponentClient::NameOfInterfaceLCMRequired;
    mtsInterfaceRequired * required = AddInterfaceRequired(interfaceName);
    if (!required) {
        CMN_LOG_CLASS_INIT_ERROR << "AddInterfaceLCM: failed to add \"LCM\" required interface: " << interfaceName << std::endl;
        return false;
    }
    required->AddFunction(mtsManagerComponentBase::CommandNames::ComponentCreate,
                          InterfaceLCMFunction.ComponentCreate);
    required->AddFunction(mtsManagerComponentBase::CommandNames::ComponentConnect,
                          InterfaceLCMFunction.ComponentConnect);
    required->AddFunction(mtsManagerComponentBase::CommandNames::GetNamesOfProcesses,
                          InterfaceLCMFunction.GetNamesOfProcesses);
    required->AddFunction(mtsManagerComponentBase::CommandNames::GetNamesOfComponents,
                          InterfaceLCMFunction.GetNamesOfComponents);
    required->AddFunction(mtsManagerComponentBase::CommandNames::GetNamesOfInterfaces,
                          InterfaceLCMFunction.GetNamesOfInterfaces);
    required->AddFunction(mtsManagerComponentBase::CommandNames::GetListOfConnections,
                          InterfaceLCMFunction.GetListOfConnections);

    // Add provided interface
    interfaceName = mtsManagerComponentClient::NameOfInterfaceLCMProvided;
    mtsInterfaceProvided * provided = AddInterfaceProvided(interfaceName);
    if (!provided) {
        CMN_LOG_CLASS_INIT_ERROR << "AddInterfaceLCM: failed to add \"LCM\" required interface: " << interfaceName << std::endl;
        return false;
    }
    provided->AddCommandWrite(&mtsManagerComponentClient::InterfaceLCMCommands_ComponentCreate,
                             this, mtsManagerComponentBase::CommandNames::ComponentCreate);
    provided->AddCommandWrite(&mtsManagerComponentClient::InterfaceLCMCommands_ComponentConnect, 
                             this, mtsManagerComponentBase::CommandNames::ComponentConnect);

    CMN_LOG_CLASS_INIT_VERBOSE << "AddInterfaceLCM: successfully added \"LCM\" interfaces" << std::endl;

    return true;
}

bool mtsManagerComponentClient::CreateInterfaceComponentFunctionSet(const std::string & clientComponentName)
{
    if (InterfaceComponentFunctionMap.FindItem(clientComponentName)) {
        CMN_LOG_CLASS_INIT_VERBOSE << "CreateInterfaceComponentFunctionSet: component is already known: " 
                                   << clientComponentName << std::endl;
        return true;
    }

    // Create a new set of function objects
    InterfaceComponentFunctionType * newFunctionSet = new InterfaceComponentFunctionType;

    std::string interfaceName = mtsManagerComponentClient::NameOfInterfaceComponentRequired;
    interfaceName += "For";
    interfaceName += clientComponentName;
    mtsInterfaceRequired * required = AddInterfaceRequired(interfaceName);
    if (!required) {
        CMN_LOG_CLASS_INIT_ERROR << "CreateInterfaceComponentFunctionSet: failed to create \"Component\" required interface: " << interfaceName << std::endl;
        return false;
    }
    required->AddFunction(mtsManagerComponentBase::CommandNames::ComponentStart,
                          newFunctionSet->ComponentStart);
    required->AddFunction(mtsManagerComponentBase::CommandNames::ComponentStop,
                          newFunctionSet->ComponentStop);
    required->AddFunction(mtsManagerComponentBase::CommandNames::ComponentResume,
                          newFunctionSet->ComponentResume);

    // Add a required interface (InterfaceComponent's required interface) to connect
    // to the provided interface (InterfaceInternal's provided interface) of the 
    // connecting component.
    if (!InterfaceComponentFunctionMap.AddItem(clientComponentName, newFunctionSet)) {
        CMN_LOG_CLASS_INIT_ERROR << "CreateInterfaceComponentFunctionSet: failed to add \"Component\" required interface: " 
            << "\"" << clientComponentName << "\", " << interfaceName << std::endl;
        return false;
    }

    // Connect InterfaceGCM's required interface to InterfaceLCM's provided interface
    mtsManagerLocal * LCM = mtsManagerLocal::GetInstance();
    if (!LCM->Connect(this->GetName(), interfaceName,
                      clientComponentName, mtsComponent::NameOfInterfaceInternalProvided))
    {
        CMN_LOG_CLASS_INIT_ERROR << "CreateInterfaceComponentFunctionSet: failed to connect: " 
            << this->GetName() << ":" << interfaceName
            << " - "
            << clientComponentName << ":" << mtsComponent::NameOfInterfaceInternalProvided
            << std::endl;
        return false;
    }

    CMN_LOG_CLASS_INIT_VERBOSE << "CreateInterfaceComponentFunctionSet: creation and connection success" << std::endl;

    return true;
}

void mtsManagerComponentClient::InterfaceComponentCommands_ComponentCreate(const mtsDescriptionComponent & arg)
{
    mtsManagerLocal * thisLCM = mtsManagerLocal::GetInstance();
    if (thisLCM->GetProcessName() == arg.ProcessName) {
        if (!CreateAndAddNewComponent(arg.ClassName, arg.ProcessName)) {
            CMN_LOG_CLASS_RUN_ERROR << "InterfaceComponentCommands_ComponentCreate: failed to create component dynamically" << std::endl;
            // MJ TEMP
            cmnThrow(std::runtime_error("InterfaceComponentCommands_ComponentCreate: failed to execute \"ComponentCreate\""));
        }
        return;
    }

    if (!InterfaceLCMFunction.ComponentCreate.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "InterfaceComponentCommands_ComponentCreate: invalid function - has not been bound to command" << std::endl;
        // MJ TEMP
        cmnThrow(std::runtime_error("InterfaceComponentCommands_ComponentCreate: failed to execute \"Component Create\""));
    }

    InterfaceLCMFunction.ComponentCreate(arg);
}

void mtsManagerComponentClient::InterfaceComponentCommands_ComponentConnect(const mtsDescriptionConnection & arg)
{
    if (!InterfaceLCMFunction.ComponentConnect.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "InterfaceComponentCommands_ComponentConnect: invalid function - has not been bound to command" << std::endl;
        // MJ TEMP
        cmnThrow(std::runtime_error("InterfaceComponentCommands_ComponentConnect: failed to execute \"Component Connect\""));
    }

    InterfaceLCMFunction.ComponentConnect(arg);
}

void mtsManagerComponentClient::InterfaceComponentCommands_GetNamesOfProcesses(mtsStdStringVec & names) const
{
    if (!InterfaceLCMFunction.GetNamesOfProcesses.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "InterfaceComponentCommands_GetNamesOfProcesses: invalid function - has not been bound to command" << std::endl;
        // MJ TEMP
        cmnThrow(std::runtime_error("InterfaceComponentCommands_GetNamesOfProcesses: failed to execute \"GetNamesOfProcesses\""));
    }

    InterfaceLCMFunction.GetNamesOfProcesses(names);
}

void mtsManagerComponentClient::InterfaceComponentCommands_GetNamesOfComponents(const mtsStdString & processName, mtsStdStringVec & names) const
{
    if (!InterfaceLCMFunction.GetNamesOfComponents.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "InterfaceComponentCommands_GetNamesOfComponents: invalid function - has not been bound to command" << std::endl;
        // MJ TEMP
        cmnThrow(std::runtime_error("InterfaceComponentCommands_GetNamesOfComponents: failed to execute \"GetNamesOfComponents\""));
    }

    InterfaceLCMFunction.GetNamesOfComponents(processName, names);
}

void mtsManagerComponentClient::InterfaceComponentCommands_GetNamesOfInterfaces(const mtsStdString & processName, mtsStdStringVec & names) const
{
    if (!InterfaceLCMFunction.GetNamesOfInterfaces.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "InterfaceComponentCommands_GetNamesOfInterfaces: invalid function - has not been bound to command" << std::endl;
        // MJ TEMP
        cmnThrow(std::runtime_error("InterfaceComponentCommands_GetNamesOfInterfaces: failed to execute \"GetNamesOfInterfaces\""));
    }

    InterfaceLCMFunction.GetNamesOfInterfaces(processName, names);
}

void mtsManagerComponentClient::InterfaceComponentCommands_GetListOfConnections(mtsStdStringVec & list) const
{
    if (!InterfaceLCMFunction.GetListOfConnections.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "InterfaceComponentCommands_GetListOfConnections: invalid function - has not been bound to command" << std::endl;
        // MJ TEMP
        cmnThrow(std::runtime_error("InterfaceComponentCommands_GetListOfConnections: failed to execute \"GetListOfConnections\""));
    }

    InterfaceLCMFunction.GetListOfConnections(list);
}

void mtsManagerComponentClient::InterfaceLCMCommands_ComponentCreate(const mtsDescriptionComponent & arg)
{
    if (!CreateAndAddNewComponent(arg.ClassName, arg.ComponentName)) {
        CMN_LOG_CLASS_RUN_ERROR << "InterfaceLCMCommands_ComponentCreate: invalid function - has not been bound to command" << std::endl;
        // MJ TEMP
        cmnThrow(std::runtime_error("InterfaceLCMCommands_ComponentCreate: failed to execute \"ComponentCreate\""));
    }

    CMN_LOG_CLASS_RUN_VERBOSE << "InterfaceLCMCommands_ComponentCreate: successfully created new component: " << arg << std::endl;
}

void mtsManagerComponentClient::InterfaceLCMCommands_ComponentConnect(const mtsDescriptionConnection & arg)
{
    // Try to connect interfaces as requested
    mtsManagerLocal * LCM = mtsManagerLocal::GetInstance()->GetInstance();

#if CISST_MTS_HAS_ICE
    if (!LCM->Connect(arg.Client.ProcessName, arg.Client.ComponentName, arg.Client.InterfaceName,
                      arg.Server.ProcessName, arg.Server.ComponentName, arg.Server.InterfaceName))
#else
    if (!LCM->Connect(arg.Client.ComponentName, arg.Client.InterfaceName,
                      arg.Server.ComponentName, arg.Server.InterfaceName))
#endif
    {
        CMN_LOG_CLASS_RUN_ERROR << "InterfaceLCMCommands_ComponentConnect: failed to connect: " << arg << std::endl;
        // MJ TEMP
        cmnThrow(std::runtime_error("InterfaceLCMCommands_ComponentConnect: failed to execute \"Component Connect\""));
    }

    CMN_LOG_CLASS_RUN_VERBOSE << "InterfaceLCMCommands_ComponentConnect: successfully connected: " << arg << std::endl;
}
