/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Peter Kazanzides, Anton Deguet
  Created on: 2008-11-13

  (C) Copyright 2008-2010 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstMultiTask/mtsInterfaceRequired.h>
#include <cisstCommon/cmnSerializer.h>


mtsInterfaceRequired::mtsInterfaceRequired(const std::string & interfaceName, mtsMailBox * mailBox) :
    mtsInterfaceRequiredOrInput(interfaceName),
    MailBox(mailBox),
    FunctionsVoid("FunctionsVoid"),
    FunctionsRead("FunctionsRead"),
    FunctionsWrite("FunctionsWrite"),
    FunctionsQualifiedRead("FunctionsQualifiedRead"),
    EventHandlersVoid("EventHandlersVoid"),
    EventHandlersWrite("EventHandlersWrite")
{
    FunctionsVoid.SetOwner(*this);
    FunctionsRead.SetOwner(*this);
    FunctionsWrite.SetOwner(*this);
    FunctionsQualifiedRead.SetOwner(*this);
    EventHandlersVoid.SetOwner(*this);
    EventHandlersWrite.SetOwner(*this);
}


mtsInterfaceRequired::~mtsInterfaceRequired()
{
    FunctionsVoid.DeleteAll();
    FunctionsRead.DeleteAll();
    FunctionsWrite.DeleteAll();
    FunctionsQualifiedRead.DeleteAll();
}


std::vector<std::string> mtsInterfaceRequired::GetNamesOfFunctions(void) const {
    std::vector<std::string> commands = GetNamesOfFunctionsVoid();
    std::vector<std::string> tmp = GetNamesOfFunctionsRead();
    commands.insert(commands.begin(), tmp.begin(), tmp.end());
    tmp.clear();
    tmp = GetNamesOfFunctionsWrite();
    commands.insert(commands.begin(), tmp.begin(), tmp.end());
    tmp.clear();
    tmp = GetNamesOfFunctionsQualifiedRead();
    commands.insert(commands.begin(), tmp.begin(), tmp.end());
    return commands;
}


std::vector<std::string> mtsInterfaceRequired::GetNamesOfFunctionsVoid(void) const {
    return FunctionsVoid.GetNames();
}


std::vector<std::string> mtsInterfaceRequired::GetNamesOfFunctionsRead(void) const {
    return FunctionsRead.GetNames();
}


std::vector<std::string> mtsInterfaceRequired::GetNamesOfFunctionsWrite(void) const {
    return FunctionsWrite.GetNames();
}


std::vector<std::string> mtsInterfaceRequired::GetNamesOfFunctionsQualifiedRead(void) const {
    return FunctionsQualifiedRead.GetNames();
}


std::vector<std::string> mtsInterfaceRequired::GetNamesOfEventHandlersVoid(void) const {
    return EventHandlersVoid.GetNames();
}


std::vector<std::string> mtsInterfaceRequired::GetNamesOfEventHandlersWrite(void) const {
    return EventHandlersWrite.GetNames();
}


mtsCommandVoidBase * mtsInterfaceRequired::GetEventHandlerVoid(const std::string & eventName) const {
    return EventHandlersVoid.GetItem(eventName);
}


mtsCommandWriteBase * mtsInterfaceRequired::GetEventHandlerWrite(const std::string & eventName) const {
    return EventHandlersWrite.GetItem(eventName);
}


bool mtsInterfaceRequired::ConnectTo(mtsDeviceInterface * interfaceProvided) {
    this->ProvidedInterface = interfaceProvided;
    unsigned int newUserId;
    newUserId = interfaceProvided->AllocateResources(this->GetName());
    CMN_LOG_CLASS_INIT_VERBOSE << "ConnectInterfaceRequiredOrInput: binding commands and events with user Id \"" << newUserId << "\"" << std::endl;
    return this->BindCommandsAndEvents(newUserId);
}


bool mtsInterfaceRequired::Disconnect(void)
{
    // First, do the command pointers.  In the future, it may be better to set the pointers to NOPVoid, NOPRead, etc.,
    // which can be static members of the corresponding command classes.
    FunctionInfoMapType::iterator iter;
    for (iter = FunctionsVoid.begin(); iter != FunctionsVoid.end(); iter++)
        iter->second->Detach();
    for (iter = FunctionsRead.begin(); iter != FunctionsRead.end(); iter++)
        iter->second->Detach();
    for (iter = FunctionsWrite.begin(); iter != FunctionsWrite.end(); iter++)
        iter->second->Detach();
    for (iter = FunctionsQualifiedRead.begin(); iter != FunctionsQualifiedRead.end(); iter++)
        iter->second->Detach();
#if 0
    // Now, do the event handlers.  Still need to implement RemoveObserver
    EventHandlerVoidMapType::iterator iterEventVoid;
    for (iterEventVoid = EventHandlersVoid.begin(); iterEventVoid != EventHandlersVoid.end(); iterEventVoid++)
        ProvidedInterface->RemoveObserver(iterEventVoid->first, iterEventVoid->second);
    EventHandlerWriteMapType::iterator iterEventWrite;
    for (iterEventWrite = EventHandlersWrite.begin(); iterEventWrite != EventHandlersWrite.end(); iterEventWrite++)
        ProvidedInterface->RemoveObserver(iterEventWrite->first, iterEventWrite->second);
#endif
    return true;
}


bool mtsInterfaceRequired::BindCommandsAndEvents(unsigned int userId)
{
    bool success = true;
    bool result;
    // First, do the command pointers
    FunctionInfoMapType::iterator iter;
    mtsFunctionVoid * functionVoid;
    for (iter = FunctionsVoid.begin();
         iter != FunctionsVoid.end();
         iter++) {
        functionVoid = dynamic_cast<mtsFunctionVoid *>(iter->second->FunctionPointer);
        result = functionVoid->Bind(ProvidedInterface->GetCommandVoid(iter->first, userId));
        if (!result) {
            CMN_LOG_CLASS_INIT_WARNING << "BindCommandsAndEvents: failed for void command \""
                                       << iter->first << "\" (connecting \""
                                       << this->GetName() << "\" to \""
                                       << this->ProvidedInterface->GetName() << "\")"<< std::endl;
        } else {
            CMN_LOG_CLASS_INIT_DEBUG << "BindCommandsAndEvents: succeeded for void command \""
                                     << iter->first << "\" (connecting \""
                                     << this->GetName() << "\" to \""
                                     << this->ProvidedInterface->GetName() << "\")"<< std::endl;
        }
        if (!iter->second->IsRequired) {
            result = true;
        }
        success &= result;
    }
    mtsFunctionRead * functionRead;
    for (iter = FunctionsRead.begin();
         iter != FunctionsRead.end();
         iter++) {
        functionRead = dynamic_cast<mtsFunctionRead *>(iter->second->FunctionPointer);
        result = functionRead->Bind(ProvidedInterface->GetCommandRead(iter->first));
        if (!result) {
            CMN_LOG_CLASS_INIT_WARNING << "BindCommandsAndEvents: failed for read command \""
                                       << iter->first << "\" (connecting \""
                                       << this->GetName() << "\" to \""
                                       << this->ProvidedInterface->GetName() << "\")"<< std::endl;
        } else {
            CMN_LOG_CLASS_INIT_DEBUG << "BindCommandsAndEvents: succeeded for read command \""
                                     << iter->first  << "\" (connecting \""
                                     << this->GetName() << "\" to \""
                                     << this->ProvidedInterface->GetName() << "\")"<< std::endl;
        }
        if (!iter->second->IsRequired) {
            result = true;
        }
        success &= result;
    }
    mtsFunctionWrite * functionWrite;
    for (iter = FunctionsWrite.begin();
         iter != FunctionsWrite.end();
         iter++) {
        functionWrite = dynamic_cast<mtsFunctionWrite *>(iter->second->FunctionPointer);
        result = functionWrite->Bind(ProvidedInterface->GetCommandWrite(iter->first, userId));
        if (!result) {
            CMN_LOG_CLASS_INIT_WARNING << "BindCommandsAndEvents: failed for write command \""
                                       << iter->first << "\" (connecting \""
                                       << this->GetName() << "\" to \""
                                       << this->ProvidedInterface->GetName() << "\")"<< std::endl;
        } else {
            CMN_LOG_CLASS_INIT_DEBUG << "BindCommandsAndEvents: succeeded for write command \""
                                     << iter->first << "\" (connecting \""
                                     << this->GetName() << "\" to \""
                                     << this->ProvidedInterface->GetName() << "\")"<< std::endl;
        }
        if (!iter->second->IsRequired) {
            result = true;
        }
        success &= result;
    }
    mtsFunctionQualifiedRead * functionQualifiedRead;
    for (iter = FunctionsQualifiedRead.begin();
         iter != FunctionsQualifiedRead.end();
         iter++) {
        functionQualifiedRead = dynamic_cast<mtsFunctionQualifiedRead *>(iter->second->FunctionPointer);
        result = functionQualifiedRead->Bind(ProvidedInterface->GetCommandQualifiedRead(iter->first));
        if (!result) {
            CMN_LOG_CLASS_INIT_WARNING << "BindCommandsAndEvents: failed for qualified read command \""
                                       << iter->first << "\" (connecting \""
                                       << this->GetName() << "\" to \""
                                       << this->ProvidedInterface->GetName() << "\")"<< std::endl;
        } else {
            CMN_LOG_CLASS_INIT_DEBUG << "BindCommandsAndEvents: succeeded for qualified read command \""
                                     << iter->first << "\" (connecting \""
                                     << this->GetName() << "\" to \""
                                     << this->ProvidedInterface->GetName() << "\")"<< std::endl;
        }
        if (!iter->second->IsRequired) {
            result = true;
        }
        success &= result;
    }

    if (!success) {
        CMN_LOG_CLASS_INIT_ERROR << "BindCommandsAndEvents: required commands missing (connecting \""
                                 << this->GetName() << "\" to \""
                                 << this->ProvidedInterface->GetName() << "\")"<< std::endl;
    }

    // Now, do the event handlers
    EventHandlerVoidMapType::iterator iterEventVoid;
    for (iterEventVoid = EventHandlersVoid.begin();
         iterEventVoid != EventHandlersVoid.end();
         iterEventVoid++) {
        result = ProvidedInterface->AddObserver(iterEventVoid->first, iterEventVoid->second);
        if (!result) {
            CMN_LOG_CLASS_INIT_WARNING << "BindCommandsAndEvents: failed to add observer for void event \""
                                       << iterEventVoid->first << "\" (connecting \""
                                       << this->GetName() << "\" to \""
                                       << this->ProvidedInterface->GetName() << "\")"<< std::endl;
        } else {
            CMN_LOG_CLASS_INIT_DEBUG << "BindCommandsAndEvents: succeeded to add observer for void event \""
                                     << iterEventVoid->first << "\" (connecting \""
                                     << this->GetName() << "\" to \""
                                     << this->ProvidedInterface->GetName() << "\")"<< std::endl;
        }
    }

    EventHandlerWriteMapType::iterator iterEventWrite;
    for (iterEventWrite = EventHandlersWrite.begin();
         iterEventWrite != EventHandlersWrite.end();
         iterEventWrite++) {
        result = ProvidedInterface->AddObserver(iterEventWrite->first, iterEventWrite->second);
        if (!result) {
            CMN_LOG_CLASS_INIT_WARNING << "BindCommandsAndEvents: failed to add observer for write event \""
                                       << iterEventWrite->first << "\" (connecting \""
                                       << this->GetName() << "\" to \""
                                       << this->ProvidedInterface->GetName() << "\")"<< std::endl;
        } else {
            CMN_LOG_CLASS_INIT_DEBUG << "BindCommandsAndEvents: succeeded to add observer for write event \""
                                     << iterEventWrite->first << "\" (connecting \""
                                     << this->GetName() << "\" to \""
                                     << this->ProvidedInterface->GetName() << "\")"<< std::endl;
        }
    }

    return success;
}


void mtsInterfaceRequired::DisableAllEvents(void) {
    EventHandlersVoid.ForEachVoid(&mtsCommandBase::Disable);
    EventHandlersWrite.ForEachVoid(&mtsCommandBase::Disable);
}


void mtsInterfaceRequired::EnableAllEvents(void) {
    EventHandlersVoid.ForEachVoid(&mtsCommandBase::Enable);
    EventHandlersWrite.ForEachVoid(&mtsCommandBase::Enable);
}


unsigned int mtsInterfaceRequired::ProcessMailBoxes(void)
{
    unsigned int numberOfCommands = 0;
    while (MailBox->ExecuteNext()) {
        numberOfCommands++;
    }
    return numberOfCommands;
}


void mtsInterfaceRequired::ToStream(std::ostream & outputStream) const
{
    outputStream << "Required Interface name: " << Name << std::endl;
    FunctionsVoid.ToStream(outputStream);
    FunctionsRead.ToStream(outputStream);
    FunctionsWrite.ToStream(outputStream);
    FunctionsQualifiedRead.ToStream(outputStream);
    EventHandlersVoid.ToStream(outputStream);
    EventHandlersWrite.ToStream(outputStream);
}


bool mtsInterfaceRequired::AddFunction(const std::string & functionName, mtsFunctionVoid & function, bool required)
{
    return FunctionsVoid.AddItem(functionName, new FunctionInfo(function, required));
}


bool mtsInterfaceRequired::AddFunction(const std::string & functionName, mtsFunctionRead & function, bool required)
{
    return FunctionsRead.AddItem(functionName, new FunctionInfo(function, required));
}


bool mtsInterfaceRequired::AddFunction(const std::string & functionName, mtsFunctionWrite & function, bool required)
{
    return FunctionsWrite.AddItem(functionName, new FunctionInfo(function, required));
}


bool mtsInterfaceRequired::AddFunction(const std::string & functionName, mtsFunctionQualifiedRead & function, bool required)
{
    return FunctionsQualifiedRead.AddItem(functionName, new FunctionInfo(function, required));
}
