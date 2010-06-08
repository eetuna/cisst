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
#include <cisstMultiTask/mtsTaskInterface.h>


mtsComponent::mtsComponent(const std::string & componentName):
    Name(componentName),
    ProvidedInterfaces("ProvidedInterfaces"),
    InterfacesRequiredOrInput("InterfacesRequiredOrInput")
{
    ProvidedInterfaces.SetOwner(*this);
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


std::vector<std::string> mtsComponent::GetNamesOfProvidedInterfaces(void) const
{
    return ProvidedInterfaces.GetNames();
}


mtsDeviceInterface * mtsComponent::AddProvidedInterface(const std::string & newInterfaceName)
{
    mtsDeviceInterface * newInterface = new mtsDeviceInterface(newInterfaceName, this);
    if (newInterface) {
        if (ProvidedInterfaces.AddItem(newInterfaceName, newInterface, CMN_LOG_LOD_INIT_ERROR)) {
            return newInterface;
        }
        CMN_LOG_CLASS_INIT_ERROR << "AddProvidedInterface: unable to add interface \""
                                 << newInterfaceName << "\"" << std::endl;
        delete newInterface;
        return 0;
    }
    CMN_LOG_CLASS_INIT_ERROR << "AddProvidedInterface: unable to create interface \""
                             << newInterfaceName << "\"" << std::endl;
    return 0;
}


mtsDeviceInterface * mtsComponent::GetProvidedInterface(const std::string & interfaceName) const
{
    return ProvidedInterfaces.GetItem(interfaceName, CMN_LOG_LOD_INIT_ERROR);
}


mtsInterfaceRequiredOrInput * mtsComponent::GetInterfaceRequiredOrInput(const std::string & interfaceRequiredOrInputName)
{
    return InterfacesRequiredOrInput.GetItem(interfaceRequiredOrInputName);
}


mtsInterfaceRequired * mtsComponent::GetInterfaceRequired(const std::string & interfaceRequiredName)
{
    return dynamic_cast<mtsInterfaceRequired *>(InterfacesRequiredOrInput.GetItem(interfaceRequiredName));
}


//#if 0 // adeguet1, is this needed, dangerous now when using mtsTaskFromSignal ...
mtsInterfaceRequired * mtsComponent::AddInterfaceRequired(const std::string & interfaceRequiredName,
                                                          mtsInterfaceRequired * interfaceRequired) {
    return InterfacesRequiredOrInput.AddItem(interfaceRequiredName, interfaceRequired) ? interfaceRequired : 0;
}
//#endif

mtsInterfaceRequired * mtsComponent::AddInterfaceRequired(const std::string & interfaceRequiredName) {
    // PK: move DEFAULT_EVENT_QUEUE_LEN somewhere else (not in mtsTaskInterface)
    mtsMailBox * mailBox = new mtsMailBox(interfaceRequiredName + "Events", mtsTaskInterface::DEFAULT_EVENT_QUEUE_LEN);
    mtsInterfaceRequired * interfaceRequired = new mtsInterfaceRequired(interfaceRequiredName, mailBox);
    if (mailBox && interfaceRequired) {
        if (InterfacesRequiredOrInput.AddItem(interfaceRequiredName, interfaceRequired)) {
            return interfaceRequired;
        }
        CMN_LOG_CLASS_INIT_ERROR << "AddInterfaceRequired: unable to add interface \""
                                 << interfaceRequiredName << "\"" << std::endl;
        if (interfaceRequired) {
            delete interfaceRequired;
        }
        if (mailBox) {
            delete mailBox;
        }
        return 0;
    }
    CMN_LOG_CLASS_INIT_ERROR << "AddInterfaceRequired: unable to create interface or mailbox for \""
                             << interfaceRequiredName << "\"" << std::endl;
    return 0;
}


std::vector<std::string> mtsComponent::GetNamesOfInterfacesRequiredOrInput(void) const {
    return InterfacesRequiredOrInput.GetNames();
}


const mtsDeviceInterface * mtsComponent::GetProvidedInterfaceFor(const std::string & interfaceRequiredName) {
    mtsInterfaceRequiredOrInput * interfaceRequiredOrInput =
        InterfacesRequiredOrInput.GetItem(interfaceRequiredName, CMN_LOG_LOD_INIT_WARNING);
    return interfaceRequiredOrInput ? interfaceRequiredOrInput->GetConnectedInterface() : 0;
}


bool mtsComponent::ConnectInterfaceRequiredOrInput(const std::string & interfaceRequiredOrInputName,
                                                   mtsDeviceInterface * providedInterface)
{
    mtsInterfaceRequiredOrInput * interfaceRequiredOrInput =
        InterfacesRequiredOrInput.GetItem(interfaceRequiredOrInputName, CMN_LOG_LOD_INIT_ERROR);
    if (interfaceRequiredOrInput) {
        if (interfaceRequiredOrInput->ConnectTo(providedInterface)) {
            CMN_LOG_CLASS_INIT_VERBOSE << "ConnectInterfaceRequiredOrInput: component \""
                                       << this->GetName()
                                       << "\" required/input interface \"" << interfaceRequiredOrInputName
                                       << "\" successfully connected to provided/output interface \"" << providedInterface->GetName() << "\"" << std::endl;
            return true;
        } else {
            CMN_LOG_CLASS_INIT_ERROR << "ConnectInterfaceRequiredOrInput: component \""
                                     << this->GetName()
                                     << "\" required/input interface \"" << interfaceRequiredOrInputName
                                     << "\" failt to connect to provided/output interface \"" << providedInterface->GetName() << "\"" << std::endl;
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
    ProvidedInterfaces.ToStream(outputStream);
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
    ProvidedInterfacesMapType::const_iterator provit = ProvidedInterfaces.begin();
    while (provit != ProvidedInterfaces.end()) {
        buffer += provit->first;
        provit++;
        if (provit != ProvidedInterfaces.end())
            buffer += ",";
    }
    buffer += "]]\n";
    return buffer;
}
