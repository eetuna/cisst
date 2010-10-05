/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s): Anton Deguet
  Created on: 2010-09-16

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


/*!
  \file
  \brief Define an internal command for cisstMultiTask
*/


#ifndef _mtsCommandQueuedVoidReturn_h
#define _mtsCommandQueuedVoidReturn_h

#include <cisstMultiTask/mtsCommandQueuedVoidReturnBase.h>


/*!
  \ingroup cisstMultiTask

 */

/*! VoidReturn queued command using templated _returnType parameter */
class mtsCommandQueuedVoidReturn: public mtsCommandQueuedVoidReturnBase
{
public:
    typedef mtsCommandQueuedVoidReturnBase BaseType;

    /*! This type. */
    typedef mtsCommandQueuedVoidReturn ThisType;

private:
    /*! Private copy constructor to prevent copies */
    inline mtsCommandQueuedVoidReturn(const ThisType & CMN_UNUSED(other));

public:

    inline mtsCommandQueuedVoidReturn(mtsCommandVoidReturnBase * actualCommand):
        BaseType(0, actualCommand)
    {}


    inline mtsCommandQueuedVoidReturn(mtsMailBox * mailBox, mtsCommandVoidReturnBase * actualCommand):
        BaseType(mailBox, actualCommand)
    {}


    // ReturnsQueue destructor should get called
    inline virtual ~mtsCommandQueuedVoidReturn() {}


    inline virtual mtsCommandQueuedVoidReturnBase * Clone(mtsMailBox * mailBox) const {
        return new mtsCommandQueuedVoidReturn(mailBox, this->ActualCommand);
    }


    virtual mtsExecutionResult Execute(mtsGenericObject & result)
    {
        if (this->IsEnabled()) {
            if (!MailBox) {
                CMN_LOG_RUN_ERROR << "Class mtsCommandQueuedVoidReturn: Execute: no mailbox for \""
                                  << this->Name << "\"" << std::endl;
                return mtsExecutionResult::NO_MAILBOX;
            }
            // preserve address of result and wait to be dequeued
            ResultPointer = &result;
            if (!MailBox->Write(this)) {
                CMN_LOG_RUN_ERROR << "Class mtsCommandQueuedVoidReturn: Execute: mailbox full for \""
                                  << this->Name << "\"" <<  std::endl;
                return mtsExecutionResult::MAILBOX_FULL;
            }
            MailBox->ThreadSignalWait();
            return mtsExecutionResult::DEV_OK;
        }
        return mtsExecutionResult::DISABLED;
    }


    /* documented in base class */
    inline const mtsGenericObject * GetReturnPrototype(void) const {
        return this->ActualCommand->GetReturnPrototype();
    }

    /* documented in base class */
    inline size_t NumberOfArguments(void) const {
        return 0;
    }

    /* documented in base class */
    inline bool Returns(void) const {
        return true;
    }
};


#endif // _mtsCommandQueuedVoidReturn_h

