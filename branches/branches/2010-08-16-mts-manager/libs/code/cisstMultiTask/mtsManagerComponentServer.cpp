/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerComponentServer.cpp 1726 2010-08-30 05:07:54Z mjung5 $

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

#include <cisstMultiTask/mtsManagerComponentServer.h>
#include <cisstMultiTask/mtsManagerComponentClient.h>
#include <cisstMultiTask/mtsManagerGlobal.h>

CMN_IMPLEMENT_SERVICES(mtsManagerComponentServer);

std::string mtsManagerComponentServer::NameOfManagerComponentServer = "MNGR-COMP-SERVER";
std::string mtsManagerComponentServer::NameOfInterfaceGCMProvided = "InterfaceGCMProvided";
std::string mtsManagerComponentServer::NameOfInterfaceGCMRequired = "InterfaceGCMRequired";

mtsManagerComponentServer::mtsManagerComponentServer(mtsManagerGlobal * gcm)
    : mtsManagerComponentBase(mtsManagerComponentServer::NameOfManagerComponentServer),
      GCM(gcm)
{
    // Prevent this component from being created more than once
    // MJ: singleton can be implemented instead.
    static int instanceCount = 0;
    if (instanceCount != 0) {
        cmnThrow(std::runtime_error("Error in creating manager component server: it's already created"));
    }
}

mtsManagerComponentServer::~mtsManagerComponentServer()
{
    InterfaceGCMFunctionType * set = 0;
    InterfaceGCMFunctionMapType::iterator it = InterfaceGCMFunctionMap.begin();
    const InterfaceGCMFunctionMapType::iterator itEnd = InterfaceGCMFunctionMap.end();
    for (; it != itEnd; ++it) {
        delete it->second;
    }
}

void mtsManagerComponentServer::Startup(void)
{
    CMN_LOG_CLASS_INIT_VERBOSE << "Manager component SERVER starts" << std::endl;
}

void mtsManagerComponentServer::Run(void)
{
    mtsManagerComponentBase::Run();
}

void mtsManagerComponentServer::Cleanup(void)
{
}

void mtsManagerComponentServer::GetNamesOfProcesses(mtsStdStringVec & stdStringVec) const
{
    std::vector<std::string> namesOfProcesses;
    GCM->GetNamesOfProcesses(namesOfProcesses);

    const size_t n = namesOfProcesses.size();
    stdStringVec.SetSize(n);
    for (unsigned int i = 0; i < n; ++i) {
        stdStringVec(i) = namesOfProcesses[i];
    }
}

bool mtsManagerComponentServer::AddInterfaceGCM(void)
{
    // InterfaceGCM's required interface is not create here but is created
    // when a manager component client connects to the manager component
    // server.  
    // See mtsManagerComponentServer::CreateInterfaceGCMFunctionSet()
    // for the creation of required interfaces.

    // Add provided interface to which InterfaceLCM's required interface connects.
    std::string interfaceName = mtsManagerComponentServer::NameOfInterfaceGCMProvided;
    mtsInterfaceProvided * provided = AddInterfaceProvided(interfaceName);
    if (!provided) {
        CMN_LOG_CLASS_INIT_ERROR << "AddInterfaceGCM: failed to add \"GCM\" provided interface: " << interfaceName << std::endl;
        return false;
    }

    provided->AddCommandWrite(&mtsManagerComponentServer::InterfaceGCMCommands_ComponentCreate,
                              this, mtsManagerComponentBase::CommandNames::ComponentCreate);
    provided->AddCommandWrite(&mtsManagerComponentServer::InterfaceGCMCommands_ComponentConnect,
                              this, mtsManagerComponentBase::CommandNames::ComponentConnect);
    provided->AddCommandRead(&mtsManagerComponentServer::InterfaceGCMCommands_GetNamesOfProcesses,
                              this, mtsManagerComponentBase::CommandNames::GetNamesOfProcesses);
    provided->AddCommandQualifiedRead(&mtsManagerComponentServer::InterfaceGCMCommands_GetNamesOfComponents,
                              this, mtsManagerComponentBase::CommandNames::GetNamesOfComponents);
    provided->AddCommandQualifiedRead(&mtsManagerComponentServer::InterfaceGCMCommands_GetNamesOfInterfaces,
                              this, mtsManagerComponentBase::CommandNames::GetNamesOfInterfaces);
    provided->AddCommandRead(&mtsManagerComponentServer::InterfaceGCMCommands_GetListOfConnections,
                              this, mtsManagerComponentBase::CommandNames::GetListOfConnections);

    CMN_LOG_CLASS_INIT_VERBOSE << "AddInterfaceGCM: successfully added \"GCM\" interfaces" << std::endl;

    return true;
}

bool mtsManagerComponentServer::CreateInterfaceGCMFunctionSet(const std::string & clientProcessName)
{
    if (InterfaceGCMFunctionMap.FindItem(clientProcessName)) {
        CMN_LOG_CLASS_INIT_VERBOSE << "CreateInterfaceGCMFunctionSet: process is already known" << std::endl;
        return true;
    }

    // Create a new set of function objects
    InterfaceGCMFunctionType * newFunctionSet = new InterfaceGCMFunctionType;

    std::string interfaceName = mtsManagerComponentServer::NameOfInterfaceGCMRequired;
    interfaceName += "For";
    interfaceName += clientProcessName;
    mtsInterfaceRequired * required = AddInterfaceRequired(interfaceName);
    if (!required) {
        CMN_LOG_CLASS_INIT_ERROR << "CreateInterfaceGCMFunctionSet: failed to create \"GCM\" required interface: " << interfaceName << std::endl;
        return false;
    }
    required->AddFunction(mtsManagerComponentBase::CommandNames::ComponentCreate,
                          newFunctionSet->ComponentCreate);
    required->AddFunction(mtsManagerComponentBase::CommandNames::ComponentConnect,
                          newFunctionSet->ComponentConnect);

    // Add a required interface (InterfaceGCM's required interface) to connect
    // to the provided interface (InterfaceLCM's provided interface) of the 
    // connecting process.
    if (!InterfaceGCMFunctionMap.AddItem(clientProcessName, newFunctionSet)) {
        CMN_LOG_CLASS_INIT_ERROR << "CreateInterfaceGCMFunctionSet: failed to add \"GCM\" required interface: " 
            << "\"" << clientProcessName << "\", " << interfaceName << std::endl;
        return false;
    }

    // Connect InterfaceGCM's required interface to InterfaceLCM's provided interface
    mtsManagerLocal * LCM = mtsManagerLocal::GetInstance();
#if CISST_MTS_HAS_ICE
    if (!LCM->Connect(LCM->GetProcessName(), this->GetName(), interfaceName,
                     clientProcessName, 
                     mtsManagerComponentClient::GetNameOfManagerComponentClient(clientProcessName),
                     mtsManagerComponentClient::NameOfInterfaceLCMProvided))
    {
        CMN_LOG_CLASS_INIT_ERROR << "CreateInterfaceGCMFunctionSet: failed to connect: " 
            << mtsManagerGlobal::GetInterfaceUID(LCM->GetProcessName(), this->GetName(), interfaceName)
            << " - "
            << mtsManagerGlobal::GetInterfaceUID(clientProcessName,
                    mtsManagerComponentClient::GetNameOfManagerComponentClient(clientProcessName),
                    mtsManagerComponentClient::NameOfInterfaceLCMProvided)
            << std::endl;
        return false;
    }
#else
    if (!LCM->Connect(this->GetName(), interfaceName,
                      mtsManagerComponentClient::GetNameOfManagerComponentClient(clientProcessName),
                      mtsManagerComponentClient::NameOfInterfaceLCMProvided))
    {
        CMN_LOG_CLASS_INIT_ERROR << "CreateInterfaceGCMFunctionSet: failed to connect: " 
            << this->GetName() << ":" << interfaceName
            << " - "
            << mtsManagerComponentClient::GetNameOfManagerComponentClient(clientProcessName) << ":"
            << mtsManagerComponentClient::NameOfInterfaceLCMProvided
            << std::endl;
        return false;
    }
#endif

    CMN_LOG_CLASS_INIT_VERBOSE << "CreateInterfaceGCMFunctionSet: creation and connection success" << std::endl;

    return true;
}

void mtsManagerComponentServer::InterfaceGCMCommands_ComponentCreate(const mtsDescriptionComponent & arg)
{
    // Check if a new component with the name specified can be created
    if (GCM->FindComponent(arg.ProcessName, arg.ComponentName)) {
        CMN_LOG_CLASS_RUN_ERROR << "InterfaceGCMCommands_ComponentCreate: failed to create component: " << arg << std::endl
                                << "InterfaceGCMCommands_ComponentCreate: component already exists" << std::endl;
        return;
    }

    // Get a set of function objects that are bound to the InterfaceLCM's provided
    // interface.
    InterfaceGCMFunctionType * functionSet = InterfaceGCMFunctionMap.GetItem(arg.ProcessName);
    if (!functionSet) {
        CMN_LOG_CLASS_RUN_ERROR << "InterfaceGCMCommands_ComponentCreate: failed to get function set: " << arg << std::endl;
        // MJ TEMP
        cmnThrow(std::runtime_error("InterfaceGCMCommands_ComponentCreate: failed to execute \"Component Create\""));
        return;
    }

    functionSet->ComponentCreate(arg);
}

void mtsManagerComponentServer::InterfaceGCMCommands_ComponentConnect(const mtsDescriptionConnection & arg)
{
    // We don't check argument validity with the GCM at this stage and rely on 
    // the current normal connection procedure (GCM allows connection at the 
    // request of LCM) because the GCM guarantees that arguments are valid.
    // The Connect request is then passed to the manager component client which
    // calls local component manager's Connect() method.

    // Get a set of function objects that are bound to the InterfaceLCM's provided
    // interface.
    InterfaceGCMFunctionType * functionSet = InterfaceGCMFunctionMap.GetItem(arg.Client.ProcessName);
    if (!functionSet) {
        CMN_LOG_CLASS_RUN_ERROR << "InterfaceGCMCommands_ComponentConnect: failed to get function set: " << arg << std::endl;
        // MJ TEMP
        cmnThrow(std::runtime_error("InterfaceGCMCommands_ComponentConnect: failed to execute \"Component Connect\""));
        return;
    }

    functionSet->ComponentConnect(arg);
}

void mtsManagerComponentServer::InterfaceGCMCommands_GetNamesOfProcesses(mtsStdStringVec & names) const
{
    std::vector<std::string> _names;
    GCM->GetNamesOfProcesses(_names);

    names.SetSize(_names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        names(i) = _names[i];
    }
}

void mtsManagerComponentServer::InterfaceGCMCommands_GetNamesOfComponents(const mtsStdString & processName, mtsStdStringVec & names) const
{
    std::vector<std::string> _names;
    GCM->GetNamesOfComponents(processName, _names);

    names.SetSize(_names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        names(i) = _names[i];
    }
}

void mtsManagerComponentServer::InterfaceGCMCommands_GetNamesOfInterfaces(const mtsStdString & processName, mtsStdStringVec & names) const
{
    std::vector<std::string> componentNames, interfaceNames, _names;
    GCM->GetNamesOfComponents(processName, componentNames);

    std::string interfaceName;
    for (size_t i = 0; i < componentNames.size(); ++i) {
        // Extract names of provided interfaces
        interfaceNames.clear();
        GCM->GetNamesOfInterfacesProvidedOrOutput(processName, componentNames[i], interfaceNames);

        for (size_t j = 0; j < interfaceNames.size(); ++j) {
            interfaceName = processName;
            interfaceName += ".";
            interfaceName += componentNames[i];
            interfaceName += ".";
            interfaceName += "(Prv)";
            interfaceName += interfaceNames[j];

            _names.push_back(interfaceName);
        }
        // Extract names of required interfaces
        interfaceNames.clear();
        GCM->GetNamesOfInterfacesRequiredOrInput(processName, componentNames[i], interfaceNames);

        for (size_t j = 0; j < interfaceNames.size(); ++j) {
            interfaceName = processName;
            interfaceName += ".";
            interfaceName += componentNames[i];
            interfaceName += ".";
            interfaceName += "(Req)";
            interfaceName += interfaceNames[j];

            _names.push_back(interfaceName);
        }
    }

    names.SetSize(_names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        names(i) = _names[i];
    }
}

void mtsManagerComponentServer::InterfaceGCMCommands_GetListOfConnections(mtsStdStringVec & list) const
{
    std::vector<mtsManagerGlobalInterface::ConnectionStrings> _list;
    GCM->GetListOfConnections(_list);

    std::string connection;
    list.SetSize(_list.size());
    for (size_t i = 0; i < list.size(); ++i) {
        connection = mtsManagerGlobal::GetInterfaceUID(_list[i].ClientProcessName,
            _list[i].ClientComponentName, _list[i].ClientInterfaceRequiredName);
        connection += " - ";
        connection += mtsManagerGlobal::GetInterfaceUID(_list[i].ServerProcessName,
            _list[i].ServerComponentName, _list[i].ServerInterfaceProvidedName);
        list(i) = connection;
    }
}