/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Ankur Kapoor, Peter Kazanzides, Anton Deguet
  Created on: 2005-05-02

  (C) Copyright 2005-2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


#include <cisstMultiTask/mtsCommandQueuedVoidBase.h>


mtsCommandQueuedVoidBase::mtsCommandQueuedVoidBase(void):
    BaseType(),
    MailBox(0),
    ActualCommand(0)
{}


mtsCommandQueuedVoidBase::mtsCommandQueuedVoidBase(mtsMailBox * mailBox,
                                                   mtsCommandVoidBase * actualCommand,
                                                   size_t size):
    BaseType(actualCommand->GetName()),
    MailBox(mailBox),
    ActualCommand(actualCommand),
    BlockingFlagQueue(size, MTS_NOT_BLOCKING)
{}


mtsCommandQueuedVoidBase * mtsCommandQueuedVoidBase::Clone(mtsMailBox * mailBox, size_t size) const
{
    return new mtsCommandQueuedVoidBase(mailBox, this->ActualCommand, size);
}


mtsCommandBase::ReturnType mtsCommandQueuedVoidBase::Execute(mtsBlockingType blocking)
{
    if (this->IsEnabled()) {
        if (!MailBox) {
            CMN_LOG_RUN_ERROR << "Class mtsCommandQueuedVoid: Execute: no mailbox for \""
                              << this->Name << "\"" << std::endl;
            return mtsCommandBase::NO_MAILBOX;
        }
        if (BlockingFlagQueue.Put(blocking)) {
            if (MailBox->Write(this)) {
                if (blocking == MTS_BLOCKING) {
                    MailBox->ThreadSignalWait();
                }
                return mtsCommandBase::DEV_OK;
            } else {
                CMN_LOG_RUN_ERROR << "Class mtsCommandQueuedVoid: Execute: Mailbox full for \""
                                  << this->Name << "\"" <<  std::endl;
                BlockingFlagQueue.Get(); // pop blocking flag from local storage
                return mtsCommandBase::MAILBOX_FULL;
            }
        } else {
            CMN_LOG_RUN_ERROR << "Class mtsCommandQueuedVoid: Execute: BlockingFlagQueue full for \""
                              << this->Name << "\"" << std::endl;
        }
        return mtsCommandBase::MAILBOX_FULL;
    }
    return mtsCommandBase::DISABLED;
}


mtsCommandVoidBase * mtsCommandQueuedVoidBase::GetActualCommand(void)
{
    return this->ActualCommand;
}


mtsBlockingType mtsCommandQueuedVoidBase::BlockingFlagGet(void)
{
    return *(this->BlockingFlagQueue.Get());
}


const std::string mtsCommandQueuedVoidBase::GetMailBoxName(void) const
{
    return this->MailBox ? this->MailBox->GetName() : "NULL";
}


void mtsCommandQueuedVoidBase::ToStream(std::ostream & outputStream) const
{
    outputStream << "mtsCommandQueuedVoid: Mailbox \"";
    if (this->MailBox) {
        outputStream << this->MailBox->GetName();
    } else {
        outputStream << "Undefined";
    }
    outputStream << "\" for command " << *(this->ActualCommand)
                 << " currently " << (this->IsEnabled() ? "enabled" : "disabled");
}
