/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Ankur Kapoor, Peter Kazanzides
  Created on: 2004-04-30

  (C) Copyright 2004-2007 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstCommon/cmnExport.h>
#include <cisstCommon/cmnPortability.h>
#include <cisstOSAbstraction/osaThread.h>
#include <cisstOSAbstraction/osaThreadBuddy.h>
#include <cisstMultiTask/mtsInterfaceProvided.h>

#include <iostream>
#include <string>


mtsInterfaceProvided::mtsInterfaceProvided(const std::string & name, mtsComponent * component,
                                           InterfaceQueuingPolicy queuingPolicy, mtsCommandVoidBase * postCommandQueuedCommand):
    BaseType(name, component),
    MailBox(0),
    QueuingPolicy(queuingPolicy),
    OriginalInterface(0),
    UserCounter(0),
    CommandsVoid("CommandsVoid", true),
    CommandsWrite("CommandsWrite", true),
    CommandsRead("CommandsRead", true),
    CommandsQualifiedRead("CommandsQualifiedRead", true),
    CommandsQueuedVoid("CommandsQueuedVoid"),
    CommandsQueuedWrite("CommandsQueuedWrite"),
    EventVoidGenerators("EventVoidGenerators", true),
    EventWriteGenerators("EventWriteGenerators", true),
    CommandsInternal("CommandsInternal", true),
    PostCommandQueuedCommand(postCommandQueuedCommand),
    Mutex()
{
    // by default, if queuing is required we assume this is not an end
    // user interface but a factory.  this setting can only be changed
    // by the factory interface using the private method SetAsEndUser.
    if (queuingPolicy == COMMANDS_SHOULD_BE_QUEUED) {
        EndUserInterface = false;
    } else {
        EndUserInterface = true;
    }
    // consistency check
    if (postCommandQueuedCommand && (queuingPolicy == COMMANDS_SHOULD_NOT_BE_QUEUED)) {
        CMN_LOG_CLASS_INIT_ERROR << "constructor: a post command queued command has been provided while queuing is turned off"
                                 << std::endl;
    }
    // set owner for all maps (to make logs more readable)
    CommandsVoid.SetOwner(*this);
    CommandsWrite.SetOwner(*this);
    CommandsRead.SetOwner(*this);
    CommandsQualifiedRead.SetOwner(*this);
    CommandsQueuedVoid.SetOwner(*this);
    CommandsQueuedWrite.SetOwner(*this);
    EventVoidGenerators.SetOwner(*this);
    EventWriteGenerators.SetOwner(*this);
    CommandsInternal.SetOwner(*this);
}



mtsInterfaceProvided::mtsInterfaceProvided(mtsInterfaceProvided * originalInterface,
                                           const std::string & userName,
                                           size_t mailBoxSize):
    BaseType(originalInterface->GetName() + "For" + userName,
             originalInterface->Component),
    QueuingPolicy(COMMANDS_SHOULD_BE_QUEUED),
    OriginalInterface(originalInterface),
    EndUserInterface(true),
    UserCounter(0),
    CommandsVoid("CommandsVoid", true),
    CommandsWrite("CommandsWrite", true),
    CommandsRead("CommandsRead", true),
    CommandsQualifiedRead("CommandsQualifiedRead", true),
    CommandsQueuedVoid("CommandsQueuedVoid"),
    CommandsQueuedWrite("CommandsQueuedWrite"),
    EventVoidGenerators("EventVoidGenerators", true),
    EventWriteGenerators("EventWriteGenerators", true),
    CommandsInternal("CommandsInternal", true),
    PostCommandQueuedCommand(originalInterface->PostCommandQueuedCommand),
    Mutex()
{
    // set owner for all maps (to make logs more readable)
    CommandsVoid.SetOwner(*this);
    CommandsWrite.SetOwner(*this);
    CommandsRead.SetOwner(*this);
    CommandsQualifiedRead.SetOwner(*this);
    CommandsQueuedVoid.SetOwner(*this);
    CommandsQueuedWrite.SetOwner(*this);
    EventVoidGenerators.SetOwner(*this);
    EventWriteGenerators.SetOwner(*this);
    CommandsInternal.SetOwner(*this);

    // duplicate what needs to be duplicated (i.e. void and write
    // commands)
    MailBox = new mtsMailBox(this->GetName(),
                             mailBoxSize,
                             this->PostCommandQueuedCommand);

    // clone void commands
    CommandQueuedVoidMapType::const_iterator iterVoid = originalInterface->CommandsQueuedVoid.begin();
    const CommandQueuedVoidMapType::const_iterator endVoid = originalInterface->CommandsQueuedVoid.end();
    mtsCommandVoidBase * commandVoid;
    for (;
         iterVoid != endVoid;
         iterVoid++) {
        commandVoid = iterVoid->second->Clone(this->MailBox);
        CommandsVoid.AddItem(iterVoid->first, commandVoid, CMN_LOG_LOD_INIT_ERROR);
        CMN_LOG_CLASS_INIT_VERBOSE << "constructor: cloned void command \"" << iterVoid->first
                                   << "\" for \"" << this->GetName() << "\"" << std::endl;
    }
    // clone write commands
    CommandQueuedWriteMapType::const_iterator iterWrite = originalInterface->CommandsQueuedWrite.begin();
    const CommandQueuedWriteMapType::const_iterator endWrite = originalInterface->CommandsQueuedWrite.end();
    mtsCommandWriteBase * commandWrite;
    for (;
         iterWrite != endWrite;
         iterWrite++) {
        commandWrite = iterWrite->second->Clone(this->MailBox, DEFAULT_ARG_BUFFER_LEN);
        CommandsWrite.AddItem(iterWrite->first, commandWrite, CMN_LOG_LOD_INIT_ERROR);
        CMN_LOG_CLASS_INIT_VERBOSE << "constructor: cloned write command " << iterWrite->first
                                   << "\" for \"" << this->GetName() << "\"" << std::endl;
    }
}


mtsInterfaceProvided::~mtsInterfaceProvided() {
	CMN_LOG_CLASS_INIT_VERBOSE << "Class mtsInterfaceProvided: Class destructor" << std::endl;
    // ADV: Need to add all cleanup, i.e. make sure all mailboxes are
    // properly deleted.
}


void mtsInterfaceProvided::Cleanup() {
    std::cerr << "Need to cleanup all create interfaces ..." << std::endl;
#if 0 // adeguet1, adv
    InterfacesProvidedCreatedType::iterator op;
    for (op = QueuedCommands.begin(); op != QueuedCommands.end(); op++) {
        delete op->second->GetMailBox();
        delete op->second;
    }
    QueuedCommands.erase(QueuedCommands.begin(), QueuedCommands.end());
#endif
	CMN_LOG_CLASS_INIT_VERBOSE << "Done base class Cleanup " << Name << std::endl;
}



// Execute all commands in the mailbox.  This is just a temporary implementation, where
// all commands in a mailbox are executed before moving on the next mailbox.  The final
// implementation will probably look at timestamps.  We may also want to pass in a
// parameter (enum) to set the mailbox processing policy.
size_t mtsInterfaceProvided::ProcessMailBoxes(void)
{
    if (!this->EndUserInterface) {
        size_t numberOfCommands = 0;
        InterfaceProvidedCreatedVectorType::iterator iterator = InterfacesProvidedCreated.begin();
        const InterfaceProvidedCreatedVectorType::iterator end = InterfacesProvidedCreated.end();
        mtsMailBox * mailBox;
        for (;
             iterator != end;
             ++iterator) {
            mailBox = iterator->second->GetMailBox();
            while (mailBox->ExecuteNext()) {
                numberOfCommands++;
            }
        }
        return numberOfCommands;
    }
    CMN_LOG_CLASS_RUN_ERROR << "ProcessMailBoxes: called on end user interface. " << std::endl;
    return 0;
}


void mtsInterfaceProvided::ToStream(std::ostream & outputStream) const
{
    outputStream << "Provided Interface \"" << Name << "\"" << std::endl;
    CommandsVoid.ToStream(outputStream);
    CommandsWrite.ToStream(outputStream);
    CommandsRead.ToStream(outputStream);
    CommandsQualifiedRead.ToStream(outputStream);
    CommandsQueuedVoid.ToStream(outputStream);
    CommandsQueuedWrite.ToStream(outputStream);
    EventVoidGenerators.ToStream(outputStream);
    EventWriteGenerators.ToStream(outputStream);
}



mtsMailBox * mtsInterfaceProvided::GetMailBox(void)
{
    return this->MailBox;
}


mtsCommandVoidBase * mtsInterfaceProvided::AddCommandVoid(mtsCommandVoidBase * command)
{
    mtsCommandQueuedVoidBase * queuedCommand = 0;
    if (command) {
        if (!CommandsVoid.AddItem(command->GetName(), command, CMN_LOG_LOD_INIT_ERROR)) {
            command = 0;
            CMN_LOG_CLASS_INIT_ERROR << "AddCommandVoid: unable to add command \""
                                     << command->GetName() << "\"" << std::endl;
        }
    } else {
        CMN_LOG_CLASS_INIT_ERROR << "AddCommandVoid: unable to create command \""
                                 << command->GetName() << "\"" << std::endl;
    }
    if (command) {
        queuedCommand = new mtsCommandQueuedVoid(0, command);
        if (!CommandsQueuedVoid.AddItem(command->GetName(), queuedCommand, CMN_LOG_LOD_INIT_ERROR)) {
            CommandsVoid.RemoveItem(command->GetName(), CMN_LOG_LOD_INIT_ERROR);
            delete queuedCommand;
            queuedCommand = 0;
            CMN_LOG_CLASS_INIT_ERROR << "AddCommandVoid: unable to add queued command \""
                                     << command->GetName() << "\"" << std::endl;
        }
    }
    return queuedCommand;
}


mtsCommandWriteBase * mtsInterfaceProvided::AddCommandWrite(mtsCommandWriteBase * command)
{
    mtsCommandQueuedWriteBase * queuedCommand = 0;
    if (command) {
        if (!CommandsWrite.AddItem(command->GetName(), command, CMN_LOG_LOD_INIT_ERROR)) {
            command = 0;
            CMN_LOG_CLASS_INIT_ERROR << "AddCommandWrite: unable to add command \""
                                     << command->GetName() << "\"" << std::endl;
        }
    } else {
        CMN_LOG_CLASS_INIT_ERROR << "AddCommandWrite: unable to create command \""
                                 << command->GetName() << "\"" << std::endl;
    }
    if (command) {
        // Create with no mailbox and 0 size argument queue.
        queuedCommand = new mtsCommandQueuedWriteGeneric(0, command, 0);
        if (!CommandsQueuedWrite.AddItem(command->GetName(), queuedCommand, CMN_LOG_LOD_INIT_ERROR)) {
            CommandsVoid.RemoveItem(command->GetName(), CMN_LOG_LOD_INIT_ERROR);
            delete queuedCommand;
            queuedCommand = 0;
            CMN_LOG_CLASS_INIT_ERROR << "AddCommandWrite: unable to add queued command \""
                                     << command->GetName() << "\"" << std::endl;
        }
    }
    return queuedCommand;
}


mtsCommandReadBase * mtsInterfaceProvided::AddCommandRead(mtsCommandReadBase * command)
{
    if (command) {
        if (!CommandsRead.AddItem(command->GetName(), command, CMN_LOG_LOD_INIT_ERROR)) {
            command = 0;
            CMN_LOG_CLASS_INIT_ERROR << "AddCommandRead: unable to add command \""
                                     << command->GetName() << "\"" << std::endl;
        }
    } else {
        CMN_LOG_CLASS_INIT_ERROR << "AddCommandRead: unable to create command \""
                                 << command->GetName() << "\"" << std::endl;
    }
    return command;
}


mtsCommandQualifiedReadBase * mtsInterfaceProvided::AddCommandQualifiedRead(mtsCommandQualifiedReadBase *command)
{
    if (command) {
        if (!CommandsQualifiedRead.AddItem(command->GetName(), command, CMN_LOG_LOD_INIT_ERROR)) {
            command = 0;
            CMN_LOG_CLASS_INIT_ERROR << "AddCommandQualifiedRead: unable to add command \""
                                     << command->GetName() << "\"" << std::endl;
        }
    } else {
        CMN_LOG_CLASS_INIT_ERROR << "AddCommandQualifiedRead: unable to create command \""
                                 << command->GetName() << "\"" << std::endl;
    }
    return command;
}


mtsCommandWriteBase* mtsInterfaceProvided::AddCommandFilteredWrite(mtsCommandQualifiedReadBase *filter, mtsCommandWriteBase *command)
{
    if (filter && command) {
        if (!CommandsInternal.AddItem(filter->GetName(), filter, CMN_LOG_LOD_INIT_ERROR)) {
            CMN_LOG_CLASS_INIT_ERROR << "AddCommandFilteredWrite: unable to add filter \""
                                     << command->GetName() << "\"" << std::endl;
            return 0;
        }
        // The mtsCommandWrite is called commandName because that name will be used by mtsCommandFilteredQueuedWrite.
        //  For clarity, we store it in the internal map under the name commandName+"Write".
        if (!CommandsInternal.AddItem(command->GetName()+"Write", command, CMN_LOG_LOD_INIT_ERROR)) {
            CMN_LOG_CLASS_INIT_ERROR << "AddCommandFilteredWrite: unable to add command \""
                                     << command->GetName() << "\"" << std::endl;
            CommandsInternal.RemoveItem(filter->GetName(), CMN_LOG_LOD_INIT_ERROR);
            return 0;
        }
        mtsCommandQueuedWriteBase * queuedCommand = new mtsCommandFilteredQueuedWrite(filter, command);
        if (CommandsQueuedWrite.AddItem(command->GetName(), queuedCommand, CMN_LOG_LOD_INIT_ERROR)) {
            return queuedCommand;
        } else {
            CommandsInternal.RemoveItem(filter->GetName(), CMN_LOG_LOD_INIT_ERROR);
            CommandsInternal.RemoveItem(command->GetName(), CMN_LOG_LOD_INIT_ERROR);
            delete queuedCommand;
            CMN_LOG_CLASS_INIT_ERROR << "AddCommandFilteredWrite: unable to add queued command \""
                                     << command->GetName() << "\"" << std::endl;
            return 0;
        }
    }
    return 0;
}


mtsInterfaceProvided * mtsInterfaceProvided::GetEndUserInterface(const std::string & userName)
{
    // check if this is already a end user interface
    if (this->EndUserInterface) {
        return this;
    }

    // else we need to duplicate this interface
    this->UserCounter++;
    CMN_LOG_CLASS_INIT_VERBOSE << "GetEndUserInterface: interface \"" << this->Name
                               << "\" creating new copy (#" << this->UserCounter
                               << ") for user \"" << userName << "\"" << std::endl;
    // new end user interface created with default size for mailbox
    mtsInterfaceProvided * interfaceProvided = new mtsInterfaceProvided(this,
                                                                        userName,
                                                                        DEFAULT_ARG_BUFFER_LEN);
    InterfacesProvidedCreated.resize(InterfacesProvidedCreated.size() + 1,
                                     InterfaceProvidedCreatedPairType(this->UserCounter, interfaceProvided));
    return interfaceProvided;
}


mtsCommandVoidBase * mtsInterfaceProvided::AddEventVoid(const std::string & eventName) {
    mtsMulticastCommandVoid * eventMulticastCommand = new mtsMulticastCommandVoid(eventName);
    if (eventMulticastCommand) {
        if (AddEvent(eventName, eventMulticastCommand)) {
            return eventMulticastCommand;
        }
        delete eventMulticastCommand;
        CMN_LOG_CLASS_INIT_ERROR << "AddEventVoid: unable to add event \""
                                 << eventName << "\"" << std::endl;
        return 0;
    }
    CMN_LOG_CLASS_INIT_ERROR << "AddEventVoid: unable to create multi-cast command for event \""
                             << eventName << "\"" << std::endl;
    return 0;
}


bool mtsInterfaceProvided::AddEventVoid(mtsFunctionVoid & eventTrigger,
                                      const std::string eventName) {
    mtsCommandVoidBase * command;
    command = this->AddEventVoid(eventName);
    if (command) {
        eventTrigger.Bind(command);
        return true;
    }
    return false;
}


bool mtsInterfaceProvided::AddEvent(const std::string & name, mtsMulticastCommandVoid * generator)
{
    if (EventWriteGenerators.GetItem(name, CMN_LOG_LOD_NOT_USED)) {
        // Is this check really needed?
        CMN_LOG_CLASS_INIT_VERBOSE << "AddEvent (void): event " << name << " already exists as write event, ignored." << std::endl;
        return false;
    }
    return EventVoidGenerators.AddItem(name, generator, CMN_LOG_LOD_INIT_ERROR);
}


bool mtsInterfaceProvided::AddEvent(const std::string & name, mtsMulticastCommandWriteBase * generator)
{
    if (EventVoidGenerators.GetItem(name, CMN_LOG_LOD_NOT_USED)) {
        // Is this check really needed?
        CMN_LOG_CLASS_INIT_VERBOSE << "AddEvent (write): event " << name << " already exists as void event, ignored." << std::endl;
        return false;
    }
    return EventWriteGenerators.AddItem(name, generator, CMN_LOG_LOD_INIT_ERROR);
}


std::vector<std::string> mtsInterfaceProvided::GetNamesOfCommands(void) const {
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


std::vector<std::string> mtsInterfaceProvided::GetNamesOfCommandsVoid(void) const {
    return CommandsVoid.GetNames();
}


std::vector<std::string> mtsInterfaceProvided::GetNamesOfCommandsWrite(void) const {
    return CommandsWrite.GetNames();
}


std::vector<std::string> mtsInterfaceProvided::GetNamesOfCommandsRead(void) const {
    return CommandsRead.GetNames();
}


std::vector<std::string> mtsInterfaceProvided::GetNamesOfCommandsQualifiedRead(void) const {
    return CommandsQualifiedRead.GetNames();
}


std::vector<std::string> mtsInterfaceProvided::GetNamesOfEventsVoid(void) const {
    return EventVoidGenerators.GetNames();
}


std::vector<std::string> mtsInterfaceProvided::GetNamesOfEventsWrite(void) const {
    return EventWriteGenerators.GetNames();
}


mtsCommandVoidBase * mtsInterfaceProvided::GetCommandVoid(const std::string & commandName,
                                                          unsigned int CMN_UNUSED(userId)) const {
    if (this->EndUserInterface) {
        return CommandsVoid.GetItem(commandName, CMN_LOG_LOD_INIT_ERROR);
    }
    CMN_LOG_CLASS_INIT_ERROR << "GetCommandVoid: can not retrieve a command from \"factory\" interface \""
                             << this->GetName()
                             << "\", you must call GetEndUserInterface to make sure you are using a end-user interface"
                             << std::endl;
    return 0;
}


mtsCommandWriteBase * mtsInterfaceProvided::GetCommandWrite(const std::string & commandName,
                                                            unsigned int CMN_UNUSED(userId)) const {
    if (this->EndUserInterface) {
        return CommandsWrite.GetItem(commandName, CMN_LOG_LOD_INIT_ERROR);
    }
    CMN_LOG_CLASS_INIT_ERROR << "GetCommandWrite: can not retrieve a command from \"factory\" interface \""
                             << this->GetName()
                             << "\", you must call GetEndUserInterface to make sure you are using a end-user interface"
                             << std::endl;
    return 0;
}


mtsCommandReadBase * mtsInterfaceProvided::GetCommandRead(const std::string & commandName) const {
    if (this->OriginalInterface) {
        return this->OriginalInterface->GetCommandRead(commandName);
    }
    return CommandsRead.GetItem(commandName, CMN_LOG_LOD_INIT_ERROR);
}


mtsCommandQualifiedReadBase * mtsInterfaceProvided::GetCommandQualifiedRead(const std::string & commandName) const {
    if (this->OriginalInterface) {
        return this->OriginalInterface->GetCommandQualifiedRead(commandName);
    }
    return CommandsQualifiedRead.GetItem(commandName, CMN_LOG_LOD_INIT_ERROR);
}


bool mtsInterfaceProvided::AddObserver(const std::string & eventName, mtsCommandVoidBase * handler)
{
    if (this->OriginalInterface) {
        return this->OriginalInterface->AddObserver(eventName, handler);
    }
    mtsMulticastCommandVoid * multicastCommand = EventVoidGenerators.GetItem(eventName);
    if (multicastCommand) {
        // should probably check for duplicates (have AddCommand return bool?)
        multicastCommand->AddCommand(handler);
        return true;
    } else {
        CMN_LOG_CLASS_INIT_ERROR << "AddObserver (void): cannot find event named \"" << eventName << "\"" << std::endl;
        return false;
    }
}


bool mtsInterfaceProvided::AddObserver(const std::string & eventName, mtsCommandWriteBase * handler)
{
    if (this->OriginalInterface) {
        return this->OriginalInterface->AddObserver(eventName, handler);
    }
    mtsMulticastCommandWriteBase * multicastCommand = EventWriteGenerators.GetItem(eventName);
    if (multicastCommand) {
        // should probably check for duplicates (have AddCommand return bool?)
        multicastCommand->AddCommand(handler);
        return true;
    } else {
        CMN_LOG_CLASS_INIT_ERROR << "AddObserver (write): cannot find event named \"" << eventName << "\"" << std::endl;
        return false;
    }
}

