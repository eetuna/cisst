/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Anton Deguet
  Created on: 2010-09-16

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


/*!
  \file
  \brief Defines base class for a queued write command.
*/

#ifndef _mtsCommandQueuedVoidReturnBase_h
#define _mtsCommandQueuedVoidReturnBase_h

#include <cisstMultiTask/mtsCommandVoidReturnBase.h>
#include <cisstMultiTask/mtsMailBox.h>

// Always include last
#include <cisstMultiTask/mtsExport.h>

class CISST_EXPORT mtsCommandQueuedVoidReturnBase: public mtsCommandVoidReturnBase {
protected:
    typedef mtsCommandVoidReturnBase BaseType;
    mtsMailBox * MailBox;
    mtsCommandVoidReturnBase * ActualCommand;

    /*! Thread signal used for blocking */
    osaThreadSignal ThreadSignal;

protected:
    /*! Pointer on caller provided placeholder for result */
    mtsGenericObject * ResultPointer;

private:
    inline mtsCommandQueuedVoidReturnBase(void):
        BaseType("??"),
        MailBox(0),
        ActualCommand(0),
        ResultPointer(0)
    {}

public:
    inline mtsCommandQueuedVoidReturnBase(mtsMailBox * mailBox, mtsCommandVoidReturnBase * actualCommand):
        BaseType(actualCommand->GetName()),
        MailBox(mailBox),
        ActualCommand(actualCommand),
        ResultPointer(0)
    {
        this->SetReturnPrototype(ActualCommand->GetReturnPrototype());
    }


    inline virtual ~mtsCommandQueuedVoidReturnBase() {}

    inline virtual mtsCommandVoidReturnBase * GetActualCommand(void) {
        return ActualCommand;
    }

    virtual void ToStream(std::ostream & outputStream) const;

    virtual mtsCommandQueuedVoidReturnBase * Clone(mtsMailBox* mailBox) const = 0;

    virtual mtsExecutionResult Execute(mtsGenericObject & result) = 0;

    void ThreadSignalRaise(void);

    inline virtual const std::string GetMailBoxName(void) const {
        return this->MailBox ? this->MailBox->GetName() : "NULL";
    }

    inline mtsGenericObject * GetResultPointer(void) {
        return ResultPointer;
    }
};

#endif // _mtsCommandQueuedVoidReturn_h
