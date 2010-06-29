/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Ankur Kapoor, Peter Kazanzides, Anton Deguet
  Created on: 2004-04-30

  (C) Copyright 2004-2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstMultiTask/mtsComponent.h>
#include <cisstMultiTask/mtsInterfaceRequired.h>
#include <cisstMultiTask/mtsInterfaceProvided.h>


mtsComponent::mtsComponent(const std::string & componentName):
    Name(componentName),
    InterfacesProvidedOrOutput("InterfacesProvided"),
    InterfacesRequiredOrInput("InterfacesRequiredOrInput")
{
    InterfacesProvidedOrOutput.SetOwner(*this);
    InterfacesRequiredOrInput.SetOwner(*this);
}


const std::string & mtsComponent::GetName(void) const
{
    return this->Name;
}


void mtsComponent::SetName(const std::string & componentName)
{
    this->Name = componentName;
}


void mtsComponent::Start(void)
{
    CMN_LOG_CLASS_INIT_VERBOSE << "Start: default start method for component \""
                               << this->GetName() << "\"" << std::endl;
}


void mtsComponent::Configure(const std::string & CMN_UNUSED(filename))
{
    CMN_LOG_CLASS_INIT_VERBOSE << "Configure: default start method for component \""
                               << this->GetName() << "\"" << std::endl;
}


std::vector<std::string> mtsComponent::GetNamesOfInterfacesProvidedOrOutput(void) const
{
    return InterfacesProvidedOrOutput.GetNames();
}


std::vector<std::string> mtsComponent::GetNamesOfInterfacesProvided(void) const
{
    std::vector<std::string> names;
    InterfacesProvidedListType::const_iterator iterator = InterfacesProvided.begin();
    const InterfacesProvidedListType::const_iterator end = InterfacesProvided.end();
    for (;
         iterator != end;
         ++iterator) {
        names.push_back((*iterator)->GetName());
    }
    return names;
}


mtsInterfaceProvided * mtsComponent::AddInterfaceProvided(const std::string & interfaceProvidedName)
{
    mtsInterfaceProvided * interfaceProvided = new mtsInterfaceProvided(interfaceProvidedName, this,
                                                                        mtsInterfaceProvided::COMMANDS_SHOULD_NOT_BE_QUEUED);
    if (interfaceProvided) {
        if (InterfacesProvidedOrOutput.AddItem(interfaceProvidedName, interfaceProvided, CMN_LOG_LOD_INIT_ERROR)) {
            InterfacesProvided.push_back(interfaceProvided);
            return interfaceProvided;
        }
        CMN_LOG_CLASS_INIT_ERROR << "AddInterfaceProvided: unable to add interface \""
                                 << interfaceProvidedName << "\"" << std::endl;
        delete interfaceProvided;
        return 0;
    }
    CMN_LOG_CLASS_INIT_ERROR << "AddInterfaceProvided: unable to create interface \""
                             << interfaceProvidedName << "\"" << std::endl;
    return 0;
}


mtsInterfaceProvidedOrOutput *
mtsComponent::GetInterfaceProvidedOrOutput(const std::string & interfaceProvidedOrOutputName)
{
    return InterfacesProvidedOrOutput.GetItem(interfaceProvidedOrOutputName,
                                              CMN_LOG_LOD_INIT_ERROR);
}


mtsInterfaceProvided *
mtsComponent::GetInterfaceProvided(const std::string & interfaceProvidedName) const
{
    return dynamic_cast<mtsInterfaceProvided *>(InterfacesProvidedOrOutput.GetItem(interfaceProvidedName,
                                                                                   CMN_LOG_LOD_INIT_ERROR));
}


mtsInterfaceRequiredOrInput *
mtsComponent::GetInterfaceRequiredOrInput(const std::string & interfaceRequiredOrInputName)
{
    return InterfacesRequiredOrInput.GetItem(interfaceRequiredOrInputName,
                                             CMN_LOG_LOD_INIT_ERROR);
}


mtsInterfaceRequired *
mtsComponent::GetInterfaceRequired(const std::string & interfaceRequiredName)
{
    return dynamic_cast<mtsInterfaceRequired *>(InterfacesRequiredOrInput.GetItem(interfaceRequiredName,
                                                                                  CMN_LOG_LOD_INIT_ERROR));
}


mtsInterfaceRequired * mtsComponent::AddInterfaceRequiredExisting(const std::string & interfaceRequiredName,
                                                                  mtsInterfaceRequired * interfaceRequired) {
    if (InterfacesRequiredOrInput.AddItem(interfaceRequiredName, interfaceRequired)) {
        InterfacesRequired.push_back(interfaceRequired);
        return interfaceRequired;
    }
    return 0;
}


mtsInterfaceRequired * mtsComponent::AddInterfaceRequiredUsingMailbox(const std::string & interfaceRequiredName,
                                                                      mtsMailBox * mailBox)
{
    mtsInterfaceRequired * interfaceRequired = new mtsInterfaceRequired(interfaceRequiredName, mailBox);
    if (interfaceRequired) {
        if (InterfacesRequiredOrInput.AddItem(interfaceRequiredName, interfaceRequired)) {
            InterfacesRequired.push_back(interfaceRequired);
            return interfaceRequired;
        }
        CMN_LOG_CLASS_INIT_ERROR << "AddInterfaceRequired: unable to add interface \""
                                 << interfaceRequiredName << "\"" << std::endl;
        if (interfaceRequired) {
            delete interfaceRequired;
        }
    } else {
        CMN_LOG_CLASS_INIT_ERROR << "AddInterfaceRequired: unable to create interface for \""
                                 << interfaceRequiredName << "\"" << std::endl;
    }
    return 0;
}


mtsInterfaceRequired * mtsComponent::AddInterfaceRequired(const std::string & interfaceRequiredName) {
    // by default, no mailbox for base component, events are not queued
    return this->AddInterfaceRequiredUsingMailbox(interfaceRequiredName, 0);
}


std::vector<std::string> mtsComponent::GetNamesOfInterfacesRequiredOrInput(void) const {
    return InterfacesRequiredOrInput.GetNames();
}


const mtsInterfaceProvidedOrOutput * mtsComponent::GetInterfaceProvidedOrOutputFor(const std::string & interfaceRequiredOrInputName) {
    mtsInterfaceRequiredOrInput * interfaceRequiredOrInput =
        InterfacesRequiredOrInput.GetItem(interfaceRequiredOrInputName, CMN_LOG_LOD_INIT_WARNING);
    return interfaceRequiredOrInput ? interfaceRequiredOrInput->GetConnectedInterface() : 0;
}


bool mtsComponent::ConnectInterfaceRequiredOrInput(const std::string & interfaceRequiredOrInputName,
                                                   mtsInterfaceProvidedOrOutput * interfaceProvidedOrOutput)
{
    mtsInterfaceRequiredOrInput * interfaceRequiredOrInput =
        InterfacesRequiredOrInput.GetItem(interfaceRequiredOrInputName, CMN_LOG_LOD_INIT_ERROR);
    if (interfaceRequiredOrInput) {
        if (interfaceRequiredOrInput->ConnectTo(interfaceProvidedOrOutput)) {
            CMN_LOG_CLASS_INIT_VERBOSE << "ConnectInterfaceRequiredOrInput: component \""
                                       << this->GetName()
                                       << "\" required/input interface \"" << interfaceRequiredOrInputName
                                       << "\" successfully connected to provided/output interface \""
                                       << interfaceProvidedOrOutput->GetName() << "\"" << std::endl;
            return true;
        } else {
            CMN_LOG_CLASS_INIT_ERROR << "ConnectInterfaceRequiredOrInput: component \""
                                     << this->GetName()
                                     << "\" required/input interface \"" << interfaceRequiredOrInputName
                                     << "\" failed to connect to provided/output interface \""
                                     << interfaceProvidedOrOutput->GetName() << "\"" << std::endl;
        }
    } else {
        CMN_LOG_CLASS_INIT_ERROR << "ConnectInterfaceRequiredOrInput: component \""
                                 << this->GetName()
                                 << "\" doesn't have required/input interface \""
                                 << interfaceRequiredOrInputName << "\"" << std::endl;
    }
    return false;
}


void mtsComponent::ToStream(std::ostream & outputStream) const
{
    outputStream << "Component name: " << Name << std::endl;
    InterfacesProvidedOrOutput.ToStream(outputStream);
    InterfacesRequiredOrInput.ToStream(outputStream);
}


std::string mtsComponent::ToGraphFormat(void) const
{
    std::string buffer("add taska [[");
    buffer = "add taska [[" + Name + "],[";
    InterfacesRequiredOrInputMapType::const_iterator reqit = InterfacesRequiredOrInput.begin();
    while (reqit != InterfacesRequiredOrInput.end()) {
        buffer += reqit->first;
        reqit++;
        if (reqit != InterfacesRequiredOrInput.end())
            buffer += ",";
    }
    buffer += "],[";
    InterfacesProvidedOrOutputMapType::const_iterator provit = InterfacesProvidedOrOutput.begin();
    while (provit != InterfacesProvidedOrOutput.end()) {
        buffer += provit->first;
        provit++;
        if (provit != InterfacesProvidedOrOutput.end())
            buffer += ",";
    }
    buffer += "]]\n";
    return buffer;
}
