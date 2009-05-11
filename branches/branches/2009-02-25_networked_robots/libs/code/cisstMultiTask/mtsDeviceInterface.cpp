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

#include <cisstCommon/cmnGenericObjectProxy.h>
#include <cisstMultiTask/mtsDeviceInterface.h>

CMN_IMPLEMENT_SERVICES(mtsDeviceInterface)


mtsCommandVoidBase * mtsDeviceInterface::GetCommandVoid(const std::string & commandName) const {
    return CommandsVoid.GetItem(commandName, 1);
}

mtsCommandReadBase * mtsDeviceInterface::GetCommandRead(const std::string & commandName) const {
    return CommandsRead.GetItem(commandName, 1);
}

mtsCommandWriteBase * mtsDeviceInterface::GetCommandWrite(const std::string & commandName) const {
    return CommandsWrite.GetItem(commandName, 1);
}

mtsCommandQualifiedReadBase * mtsDeviceInterface::GetCommandQualifiedRead(const std::string & commandName) const {
    return CommandsQualifiedRead.GetItem(commandName, 1);
}

std::vector<std::string> mtsDeviceInterface::GetNamesOfCommands(void) const {
    std::vector<std::string> commands = GetNamesOfCommandsVoid();
    std::vector<std::string> tmp = GetNamesOfCommandsRead();
    commands.insert(commands.begin(), tmp.begin(), tmp.end());
    tmp.clear();
    tmp = GetNamesOfCommandsWrite();
    commands.insert(commands.begin(), tmp.begin(), tmp.end());
    tmp.clear();
    tmp = GetNamesOfCommandsQualifiedRead();
    commands.insert(commands.begin(), tmp.begin(), tmp.end());
    return commands;
}


std::vector<std::string> mtsDeviceInterface::GetNamesOfCommandsVoid(void) const {
    return CommandsVoid.GetNames();
}


std::vector<std::string> mtsDeviceInterface::GetNamesOfCommandsRead(void) const {
    return CommandsRead.GetNames();
}


std::vector<std::string> mtsDeviceInterface::GetNamesOfCommandsWrite(void) const {
    return CommandsWrite.GetNames();
}


std::vector<std::string> mtsDeviceInterface::GetNamesOfCommandsQualifiedRead(void) const {
    return CommandsQualifiedRead.GetNames();
}


mtsDeviceInterface::CommandVoidMapType & mtsDeviceInterface::GetCommandVoidMap(void) {
    return CommandsVoid;
}


mtsDeviceInterface::CommandReadMapType & mtsDeviceInterface::GetCommandReadMap(void) {
    return CommandsRead;
}


mtsDeviceInterface::CommandWriteMapType & mtsDeviceInterface::GetCommandWriteMap(void) {
    return CommandsWrite;
}


mtsDeviceInterface::CommandQualifiedReadMapType & mtsDeviceInterface::GetCommandQualifiedReadMap(void) {
    return CommandsQualifiedRead;
}


mtsCommandVoidBase * mtsDeviceInterface::AddEventVoid(const std::string & eventName) {
    mtsMulticastCommandVoid * eventMulticastCommand = new mtsMulticastCommandVoid(eventName);
    if (eventMulticastCommand) {
        if (AddEvent(eventName, eventMulticastCommand)) {
            return eventMulticastCommand;
        }
        delete eventMulticastCommand;
        CMN_LOG_CLASS(1) << "AddEventVoid: unable to add event \""
                         << eventName << "\"" << std::endl;
        return 0;
    }
    CMN_LOG_CLASS(0) << "AddEventVoid: unable to create multi-cast command for event \""
                     << eventName << "\"" << std::endl;
    return 0;
}


bool mtsDeviceInterface::AddEvent(const std::string & name, mtsMulticastCommandVoid * generator)
{
    if (EventWriteGenerators.GetItem(name)) {
        // Is this check really needed?
        CMN_LOG_CLASS(3) << "AddEvent (void): event " << name << " already exists as write event, ignored." << std::endl;
        return false;
    }
    return EventVoidGenerators.AddItem(name, generator, 1);
}


bool mtsDeviceInterface::AddEvent(const std::string & name, mtsMulticastCommandWriteBase * generator)
{
    if (EventVoidGenerators.GetItem(name)) {
        // Is this check really needed?
        CMN_LOG_CLASS(3) << "AddEvent (write): event " << name << " already exists as void event, ignored." << std::endl;
        return false;
    }
    return EventWriteGenerators.AddItem(name, generator, 1);
}


std::vector<std::string> mtsDeviceInterface::GetNamesOfEventsVoid(void) const {
    return EventVoidGenerators.GetNames();
}


std::vector<std::string> mtsDeviceInterface::GetNamesOfEventsWrite(void) const {
    return EventWriteGenerators.GetNames();
}


bool mtsDeviceInterface::AddObserver(const std::string & eventName, mtsCommandVoidBase * handler)
{
    mtsMulticastCommandVoid * multicastCommand = EventVoidGenerators.GetItem(eventName);
    if (multicastCommand) {
        // should probably check for duplicates (have AddCommand return bool?)
        multicastCommand->AddCommand(handler);
        return true;
    } else {
        CMN_LOG_CLASS(1) << "AddObserver (void): cannot find event named \"" << eventName << "\"" << std::endl;
        return false;
    }
}


bool mtsDeviceInterface::AddObserver(const std::string & eventName, mtsCommandWriteBase * handler)
{
    mtsMulticastCommandWriteBase * multicastCommand = EventWriteGenerators.GetItem(eventName);
    if (multicastCommand) {
        // should probably check for duplicates (have AddCommand return bool?)
        multicastCommand->AddCommand(handler);
        return true;
    } else {
        CMN_LOG_CLASS(1) << "AddObserver (write): cannot find event named \"" << eventName << "\"" << std::endl;
        return false;
    }
}



unsigned int mtsDeviceInterface::AllocateResourcesForCurrentThread(void)
{
    // no queued commands in this interface, we just keep track of the
    // requests
    const osaThreadId consumerId = osaGetCurrentThreadId();
    ThreadIdCountersType::iterator iterator = ThreadIdCounters.begin();
    bool found = false;
    while (!found && iterator != ThreadIdCounters.end()) {
        if ((iterator->first).Equal(consumerId)) {
            found = true;
        } else {
            iterator++;
        }
    }
    if (!found) {
        CMN_LOG_CLASS(3) << "AllocateResourcesForCurrentThread: new thread Id (" << consumerId << ")" << std::endl;
        ThreadIdCounters.resize(ThreadIdCounters.size() + 1,
                                ThreadIdCounterPairType(consumerId, 1));
        return 1;
    } else {
        CMN_LOG_CLASS(3) << "AllocateResourcesForCurrentThread: already registered thread Id (" << consumerId << ")" << std::endl;
        return (iterator->second)++;
    }
}


void mtsDeviceInterface::ToStream(std::ostream & outputStream) const
{
    outputStream << "Provided Interface name: " << Name << std::endl;
    // Previous version of code numbered all commands in a single series.
    // Now, we restart numbering for each type of command.
    CommandsVoid.ToStream(outputStream);
    CommandsRead.ToStream(outputStream);
    CommandsWrite.ToStream(outputStream);
    CommandsQualifiedRead.ToStream(outputStream);
    // Previous version of code numbered all events in a single series.
    // Now, we restart numbering for each type (e.g., void or write).
    EventVoidGenerators.ToStream(outputStream);
    EventWriteGenerators.ToStream(outputStream);
}

//-------------------------------------------------------------------------
//  Interface Proxy Related
//-------------------------------------------------------------------------
/*
const bool mtsDeviceInterface::AddCommandVoidProxyMapElement(
    const unsigned int commandVoidProxyID, const std::string & commandVoidProxyName)
{
    CommandVoidProxyMapType::const_iterator it = CommandVoidProxyMap.find(commandVoidProxyID);
    if (it != CommandVoidProxyMap.end()) {
        CMN_LOG_CLASS(3) << "AddCommandVoidProxyMapElement: already registered commandVoidProxy: "
            << "(" << Name << ":" << commandVoidProxyName << "," << commandVoidProxyID << ")" << std::endl;
        return false;
    }

    //mtsCommandVoidBase * commandVoid = this->GetCommandVoid(commandVoidProxyName);
    mtsCommandVoidBase * commandVoid = CommandsVoid.GetItem(commandVoidProxyName, 1);

    if (!commandVoid) {
        CMN_LOG_CLASS(3) << "AddCommandVoidProxyMapElement: no such CommandVoid object exists: "
            << "(" << Name << ":" << commandVoidProxyName << ")" << std::endl;
        return false;
    }
    
    CommandVoidProxyMap.insert(std::make_pair(commandVoidProxyID, commandVoid));

    return true;
}

const bool mtsDeviceInterface::AddCommandWriteProxyMapElement(
    const unsigned int commandWriteProxyID, const std::string & commandWriteProxyName)
{
    CommandWriteProxyMapType::const_iterator it = CommandWriteProxyMap.find(commandWriteProxyID);
    if (it != CommandWriteProxyMap.end()) {
        CMN_LOG_CLASS(3) << "AddCommandWriteProxyMapElement: already registered commandWriteProxy: "
            << "(" << Name << ":" << commandWriteProxyName << "," << commandWriteProxyID << ")" << std::endl;
        return false;
    }

    mtsCommandWriteBase * commandWrite = CommandsWrite.GetItem(commandWriteProxyName, 1);
    if (!commandWrite) {
        CMN_LOG_CLASS(3) << "AddCommandWriteProxyMapElement: no such CommandWrite object exists: "
            << "(" << Name << ":" << commandWriteProxyName << ")" << std::endl;
        return false;
    }
    
    CommandWriteProxyMap.insert(std::make_pair(commandWriteProxyID, commandWrite));

    return true;
}

const bool mtsDeviceInterface::AddCommandReadProxyMapElement(
    const unsigned int commandReadProxyID, const std::string & commandReadProxyName)
{
    CommandReadProxyMapType::const_iterator it = CommandReadProxyMap.find(commandReadProxyID);
    if (it != CommandReadProxyMap.end()) {
        CMN_LOG_CLASS(3) << "AddCommandReadProxyMapElement: already registered commandReadProxy: "
            << "(" << Name << ":" << commandReadProxyName << "," << commandReadProxyID << ")" << std::endl;
        return false;
    }

    mtsCommandReadBase * commandRead = CommandsRead.GetItem(commandReadProxyName, 1);
    if (!commandRead) {
        CMN_LOG_CLASS(3) << "AddCommandReadProxyMapElement: no such CommandRead object exists: "
            << "(" << Name << ":" << commandReadProxyName << ")" << std::endl;
        return false;
    }
    
    CommandReadProxyMap.insert(std::make_pair(commandReadProxyID, commandRead));

    return true;
}

const bool mtsDeviceInterface::AddCommandQualifiedReadProxyMapElement(
    const unsigned int commandQualifiedReadProxyID, const std::string & commandQualifiedReadProxyName)
{
    CommandQualifiedReadProxyMapType::const_iterator it = CommandQualifiedReadProxyMap.find(commandQualifiedReadProxyID);
    if (it != CommandQualifiedReadProxyMap.end()) {
        CMN_LOG_CLASS(3) << "AddCommandQualifiedReadProxyMapElement: already registered commandQualifiedReadProxy: "
            << "(" << Name << ":" << commandQualifiedReadProxyName << "," << commandQualifiedReadProxyID << ")" << std::endl;
        return false;
    }

    mtsCommandQualifiedReadBase * commandQualifiedRead = CommandsQualifiedRead.GetItem(commandQualifiedReadProxyName, 1);
    if (!commandQualifiedRead) {
        CMN_LOG_CLASS(3) << "AddCommandQualifiedReadProxyMapElement: no such CommandQualifiedRead object exists: "
            << "(" << Name << ":" << commandQualifiedReadProxyName << ")" << std::endl;
        return false;
    }
    
    CommandQualifiedReadProxyMap.insert(std::make_pair(commandQualifiedReadProxyID, commandQualifiedRead));

    return true;
}

void mtsDeviceInterface::ExecuteCommandVoid(const unsigned int commandID)
{
    CommandVoidProxyMapType::const_iterator it = CommandVoidProxyMap.find(commandID);
    CMN_ASSERT(it != CommandVoidProxyMap.end());

    it->second->Execute();
}
*/