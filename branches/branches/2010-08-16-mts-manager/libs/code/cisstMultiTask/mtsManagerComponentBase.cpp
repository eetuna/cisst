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
std::string mtsManagerComponentBase::CommandNames::ComponentCreate  = "ComponentCreate";
std::string mtsManagerComponentBase::CommandNames::ComponentConnect = "ComponentConnect";
std::string mtsManagerComponentBase::CommandNames::ComponentStart   = "ComponentStart";
std::string mtsManagerComponentBase::CommandNames::ComponentStop    = "ComponentStop";
std::string mtsManagerComponentBase::CommandNames::ComponentResume  = "ComponentResume";
std::string mtsManagerComponentBase::CommandNames::GetNamesOfProcesses  = "GetNamesOfProcesses";
std::string mtsManagerComponentBase::CommandNames::GetNamesOfComponents = "GetNamesOfComponents";
std::string mtsManagerComponentBase::CommandNames::GetNamesOfInterfaces = "GetNamesOfInterfaces";
std::string mtsManagerComponentBase::CommandNames::GetListOfConnections = "GetListOfConnections";
std::string mtsManagerComponentBase::EventNames::AddComponent = "AddComponentEvent";
std::string mtsManagerComponentBase::EventNames::AddConnection = "AddConnectionEvent";
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
