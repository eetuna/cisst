/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerComponentBase.cpp 1726 2010-08-30 05:07:54Z mjung5 $

  Author(s):  Anton Deguet, Min Yang Jung
  Created on: 2010-08-29

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstMultiTask/mtsManagerComponentBase.h>
#include <cisstMultiTask/mtsInterfaceProvided.h>
#include <cisstMultiTask/mtsInterfaceRequired.h>

// Names of commands
const std::string mtsManagerComponentBase::CommandNames::ComponentCreate  = "ComponentCreate";
const std::string mtsManagerComponentBase::CommandNames::ComponentConnect = "ComponentConnect";
const std::string mtsManagerComponentBase::CommandNames::ComponentStart   = "ComponentStart";
const std::string mtsManagerComponentBase::CommandNames::ComponentStop    = "ComponentStop";
const std::string mtsManagerComponentBase::CommandNames::ComponentResume  = "ComponentResume";
const std::string mtsManagerComponentBase::CommandNames::GetNamesOfProcesses  = "GetNamesOfProcesses";
const std::string mtsManagerComponentBase::CommandNames::GetNamesOfComponents = "GetNamesOfComponents";
const std::string mtsManagerComponentBase::CommandNames::GetNamesOfInterfaces = "GetNamesOfInterfaces";
const std::string mtsManagerComponentBase::CommandNames::GetListOfConnections = "GetListOfConnections";
// Names of events
const std::string mtsManagerComponentBase::EventNames::AddComponent  = "AddComponentEvent";
const std::string mtsManagerComponentBase::EventNames::AddConnection = "AddConnectionEvent";
const std::string mtsManagerComponentBase::EventNames::ChangeState   = "ChangeState";

CMN_IMPLEMENT_SERVICES(mtsManagerComponentBase);

mtsManagerComponentBase::mtsManagerComponentBase(const std::string & componentName)
    : mtsTaskFromSignal(componentName, 50)
{
    UseSeparateLogFileDefault();
}

mtsManagerComponentBase::~mtsManagerComponentBase()
{
}

void mtsManagerComponentBase::Run(void)
{
    ProcessQueuedCommands();
    ProcessQueuedEvents();
}

void mtsManagerComponentBase::Cleanup(void)
{
}

// PK: following could be in a separate file
CMN_IMPLEMENT_SERVICES(mtsManagerComponentServices)

// Constructor
mtsManagerComponentServices::mtsManagerComponentServices() : required(0)
{
}

bool mtsManagerComponentServices::SetInterfaceRequired(mtsInterfaceRequired *req)
{
    required = req;
    if (required) {
        required->AddFunction(mtsManagerComponentBase::CommandNames::ComponentCreate, ComponentCreate);
        required->AddFunction(mtsManagerComponentBase::CommandNames::ComponentConnect, ComponentConnect);
        required->AddFunction(mtsManagerComponentBase::CommandNames::ComponentStart, ComponentStart);
        required->AddFunction(mtsManagerComponentBase::CommandNames::ComponentStop, ComponentStop);
        required->AddFunction(mtsManagerComponentBase::CommandNames::ComponentResume, ComponentResume);
        required->AddFunction(mtsManagerComponentBase::CommandNames::GetNamesOfProcesses, GetNamesOfProcesses);
        required->AddFunction(mtsManagerComponentBase::CommandNames::GetNamesOfComponents, GetNamesOfComponents);
        required->AddFunction(mtsManagerComponentBase::CommandNames::GetNamesOfInterfaces, GetNamesOfInterfaces);
        required->AddFunction(mtsManagerComponentBase::CommandNames::GetListOfConnections, GetListOfConnections);
    }
    return (required != 0);
}

bool mtsManagerComponentServices::RequestComponentCreate(const std::string & className, const std::string & componentName) const
{
    if (!this->ComponentCreate.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "RequestComponentCreate: invalid function - has not been bound to command" << std::endl;
        return false;
    }

    mtsDescriptionComponent arg;
    arg.ProcessName   = mtsManagerLocal::GetInstance()->GetProcessName();
    arg.ClassName     = className;
    arg.ComponentName = componentName;

    // MJ: TODO: change this with blocking command
    this->ComponentCreate(arg);

    CMN_LOG_CLASS_RUN_VERBOSE << "RequestComponentCreate: requested component creation: " << arg << std::endl;

    return true;
}

bool mtsManagerComponentServices::RequestComponentCreate(
    const std::string& processName, const std::string & className, const std::string & componentName) const
{
    if (!this->ComponentCreate.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "RequestComponentCreate: invalid function - has not been bound to command" << std::endl;
        return false;
    }

    mtsDescriptionComponent arg;
    arg.ProcessName   = processName;
    arg.ClassName     = className;
    arg.ComponentName = componentName;

    // MJ: TODO: change this with blocking command
    this->ComponentCreate(arg);

    CMN_LOG_CLASS_RUN_VERBOSE << "RequestComponentCreate: requested component creation: " << arg << std::endl;

    return true;
}

bool mtsManagerComponentServices::RequestComponentConnect(
    const std::string & clientComponentName, const std::string & clientInterfaceRequiredName,
    const std::string & serverComponentName, const std::string & serverInterfaceProvidedName) const
{
    if (!this->ComponentConnect.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "RequestComponentConnect: invalid function - has not been bound to command" << std::endl;
        return false;
    }

    mtsDescriptionConnection arg;
    const std::string thisProcessName = mtsManagerLocal::GetInstance()->GetProcessName();
    arg.Client.ProcessName   = thisProcessName;
    arg.Client.ComponentName = clientComponentName;
    arg.Client.InterfaceName = clientInterfaceRequiredName;
    arg.Server.ProcessName   = thisProcessName;
    arg.Server.ComponentName = serverComponentName;
    arg.Server.InterfaceName = serverInterfaceProvidedName;
    arg.ConnectionID = -1;  // not yet assigned

    // MJ: TODO: change this with blocking command
    this->ComponentConnect(arg);

    CMN_LOG_CLASS_RUN_VERBOSE << "RequestComponentConnect: requested component connection: " << arg << std::endl;

    return true;
}

bool mtsManagerComponentServices::RequestComponentConnect(
    const std::string & clientProcessName,
    const std::string & clientComponentName, const std::string & clientInterfaceRequiredName,
    const std::string & serverProcessName,
    const std::string & serverComponentName, const std::string & serverInterfaceProvidedName) const
{
    if (!this->ComponentConnect.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "RequestComponentConnect: invalid function - has not been bound to command" << std::endl;
        return false;
    }

    mtsDescriptionConnection arg;
    arg.Client.ProcessName   = clientProcessName;
    arg.Client.ComponentName = clientComponentName;
    arg.Client.InterfaceName = clientInterfaceRequiredName;
    arg.Server.ProcessName   = serverProcessName;
    arg.Server.ComponentName = serverComponentName;
    arg.Server.InterfaceName = serverInterfaceProvidedName;
    arg.ConnectionID = -1;  // not yet assigned

    // MJ: TODO: change this with blocking command
    this->ComponentConnect(arg);

    CMN_LOG_CLASS_RUN_VERBOSE << "RequestComponentConnect: requested component connection: " << arg << std::endl;

    return true;
}

bool mtsManagerComponentServices::RequestComponentStart(const std::string & componentName, const double delayInSecond) const
{
    if (!this->ComponentStart.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "RequestComponentStart: invalid function - has not been bound to command" << std::endl;
        return false;
    }

    mtsComponentStatusControl arg;
    arg.ProcessName   = mtsManagerLocal::GetInstance()->GetProcessName();
    arg.ComponentName = componentName;
    arg.DelayInSecond = delayInSecond;
    arg.Command       = mtsComponentStatusControl::COMPONENT_START;

    // MJ: TODO: change this with blocking command
    this->ComponentStart(arg);

    CMN_LOG_CLASS_RUN_VERBOSE << "RequestComponentStart: requested component start: " << arg << std::endl;

    return true;
}

bool mtsManagerComponentServices::RequestComponentStart(const std::string& processName, const std::string & componentName,
                                         const double delayInSecond) const
{
    if (!this->ComponentStart.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "RequestComponentStart: invalid function - has not been bound to command" << std::endl;
        return false;
    }

    mtsComponentStatusControl arg;
    arg.ProcessName   = processName;
    arg.ComponentName = componentName;
    arg.DelayInSecond = delayInSecond;
    arg.Command       = mtsComponentStatusControl::COMPONENT_START;

    // MJ: TODO: change this with blocking command
    this->ComponentStart(arg);

    CMN_LOG_CLASS_RUN_VERBOSE << "RequestComponentStart: requested component start: " << arg << std::endl;

    return true;
}

bool mtsManagerComponentServices::RequestComponentStop(const std::string & componentName, const double delayInSecond) const
{
    if (!this->ComponentStop.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "RequestComponentStop: invalid function - has not been bound to command" << std::endl;
        return false;
    }

    mtsComponentStatusControl arg;
    arg.ProcessName   = mtsManagerLocal::GetInstance()->GetProcessName();
    arg.ComponentName = componentName;
    arg.DelayInSecond = delayInSecond;
    arg.Command       = mtsComponentStatusControl::COMPONENT_STOP;

    // MJ: TODO: change this with blocking command
    this->ComponentStop(arg);

    CMN_LOG_CLASS_RUN_VERBOSE << "RequestComponentStop: requested component stop: " << arg << std::endl;

    return true;
}

bool mtsManagerComponentServices::RequestComponentStop(const std::string& processName, const std::string & componentName,
                                        const double delayInSecond) const
{
    if (!this->ComponentStop.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "RequestComponentStop: invalid function - has not been bound to command" << std::endl;
        return false;
    }

    mtsComponentStatusControl arg;
    arg.ProcessName   = processName;
    arg.ComponentName = componentName;
    arg.DelayInSecond = delayInSecond;
    arg.Command       = mtsComponentStatusControl::COMPONENT_STOP;

    // MJ: TODO: change this with blocking command
    this->ComponentStop(arg);

    CMN_LOG_CLASS_RUN_VERBOSE << "RequestComponentStop: requested component stop: " << arg << std::endl;

    return true;
}

bool mtsManagerComponentServices::RequestComponentResume(const std::string & componentName, const double delayInSecond) const
{
    if (!this->ComponentResume.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "RequestComponentResume: invalid function - has not been bound to command" << std::endl;
        return false;
    }

    mtsComponentStatusControl arg;
    arg.ProcessName   = mtsManagerLocal::GetInstance()->GetProcessName();
    arg.ComponentName = componentName;
    arg.DelayInSecond = delayInSecond;
    arg.Command       = mtsComponentStatusControl::COMPONENT_RESUME;

    // MJ: TODO: change this with blocking command
    this->ComponentResume(arg);

    CMN_LOG_CLASS_RUN_VERBOSE << "RequestComponentResume: requested component resume: " << arg << std::endl;

    return true;
}

bool mtsManagerComponentServices::RequestComponentResume(const std::string& processName, const std::string & componentName,
                                          const double delayInSecond) const
{
    if (!this->ComponentResume.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "RequestComponentResume: invalid function - has not been bound to command" << std::endl;
        return false;
    }

    mtsComponentStatusControl arg;
    arg.ProcessName   = processName;
    arg.ComponentName = componentName;
    arg.DelayInSecond = delayInSecond;
    arg.Command       = mtsComponentStatusControl::COMPONENT_RESUME;

    // MJ: TODO: change this with blocking command
    this->ComponentResume(arg);

    CMN_LOG_CLASS_RUN_VERBOSE << "RequestComponentResume: requested component resume: " << arg << std::endl;

    return true;
}

bool mtsManagerComponentServices::RequestGetNamesOfProcesses(std::vector<std::string> & namesOfProcesses) const
{
    if (!this->GetNamesOfProcesses.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "RequestGetNamesOfProcesses: invalid function - has not been bound to command" << std::endl;
        return false;
    }

    mtsStdStringVec names;
    this->GetNamesOfProcesses(names);

    mtsParameterTypes::ConvertVectorStringType(names, namesOfProcesses);

    return true;
}

bool mtsManagerComponentServices::RequestGetNamesOfComponents(const std::string & processName, std::vector<std::string> & namesOfComponents) const
{
    if (!this->GetNamesOfComponents.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "RequestGetNamesOfComponents: invalid function - has not been bound to command" << std::endl;
        return false;
    }

    mtsStdStringVec names;
    this->GetNamesOfComponents(mtsStdString(processName), names);

    mtsParameterTypes::ConvertVectorStringType(names, namesOfComponents);

    return true;
}

bool mtsManagerComponentServices::RequestGetNamesOfInterfaces(const std::string & processName,
                                               const std::string & componentName,
                                               std::vector<std::string> & namesOfInterfacesRequired,
                                               std::vector<std::string> & namesOfInterfacesProvided) const
{
    if (!this->GetNamesOfInterfaces.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "RequestGetNamesOfInterfaces: invalid function - has not been bound to command" << std::endl;
        return false;
    }

    // input arg
    mtsDescriptionComponent argIn;
    argIn.ProcessName   = processName;
    argIn.ComponentName = componentName;

    // output arg
    mtsDescriptionInterface argOut;

    this->GetNamesOfInterfaces(argIn, argOut);

    mtsParameterTypes::ConvertVectorStringType(argOut.InterfaceRequiredNames, namesOfInterfacesRequired);
    mtsParameterTypes::ConvertVectorStringType(argOut.InterfaceProvidedNames, namesOfInterfacesProvided);

    return true;
}

bool mtsManagerComponentServices::RequestGetListOfConnections(std::vector<mtsDescriptionConnection> & listOfConnections) const
{
    if (!this->GetListOfConnections.IsValid()) {
        CMN_LOG_CLASS_RUN_ERROR << "RequestGetListOfConnections: invalid function - has not been bound to command" << std::endl;
        return false;
    }

    this->GetListOfConnections(listOfConnections);

    return true;
}
