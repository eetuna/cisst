/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Ankur Kapoor, Peter Kazanzides, Anton Deguet
  Created on: 2005-05-02

  (C) Copyright 2005-2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


#include <cisstMultiTask/mtsCommandQueuedVoid.h>
#include <cisstMultiTask/mtsCallableVoidBase.h>

mtsCommandQueuedVoid::mtsCommandQueuedVoid(void):
    BaseType(),
    MailBox(0)
{}


mtsCommandQueuedVoid::mtsCommandQueuedVoid(mtsCallableVoidBase * callable,
                                           const std::string & name,
                                           mtsMailBox * mailBox,
                                           size_t size):
    BaseType(callable, name),
    MailBox(mailBox),
    BlockingFlagQueue(size, MTS_NOT_BLOCKING)
{}


mtsCommandQueuedVoid * mtsCommandQueuedVoid::Clone(mtsMailBox * mailBox, size_t size) const
{
    return new mtsCommandQueuedVoid(this->Callable, this->Name,
                                    mailBox, size);
}


mtsExecutionResult mtsCommandQueuedVoid::Execute(mtsBlockingType blocking)
{
    // check if this command is enabled
    if (!this->IsEnabled()) {
        return mtsExecutionResult::COMMAND_DISABLED;
    }
    // check if there is a mailbox (i.e. if the command is associated to an interface
    if (!MailBox) {
        CMN_LOG_RUN_ERROR << "Class mtsCommandQueuedVoid: Execute: no mailbox for \""
                          << this->Name << "\"" << std::endl;
        return mtsExecutionResult::COMMAND_HAS_NO_MAILBOX;
    }
    // copy the blocking flag to the local storage.
    if (!BlockingFlagQueue.Put(blocking)) {
        CMN_LOG_RUN_ERROR << "Class mtsCommandQueuedVoid: Execute: BlockingFlagQueue full for \""
                          << this->Name << "\"" << std::endl;
        return mtsExecutionResult::COMMAND_ARGUMENT_QUEUE_FULL;
    }
    // finally try to queue to mailbox
    if (!MailBox->Write(this)) {
        CMN_LOG_RUN_ERROR << "Class mtsCommandQueuedVoid: Execute: Mailbox full for \""
                          << this->Name << "\"" <<  std::endl;
        BlockingFlagQueue.Get(); // pop blocking flag from local storage
        return mtsExecutionResult::INTERFACE_COMMAND_MAILBOX_FULL;
    }
    return mtsExecutionResult::COMMAND_QUEUED;
}


mtsBlockingType mtsCommandQueuedVoid::BlockingFlagGet(void)
{
    return *(this->BlockingFlagQueue.Get());
}


std::string mtsCommandQueuedVoid::GetMailBoxName(void) const
{
    return this->MailBox ? this->MailBox->GetName() : "null pointer!";
}


void mtsCommandQueuedVoid::ToStream(std::ostream & outputStream) const
{
    outputStream << "mtsCommandQueuedVoid: Mailbox \""
                 << this->GetMailBoxName()
                 << "\" for command(void) using " << *(this->Callable)
                 << " currently " << (this->IsEnabled() ? "enabled" : "disabled");
}
