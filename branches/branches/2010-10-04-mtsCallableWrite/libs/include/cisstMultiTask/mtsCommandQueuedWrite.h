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


/*!
  \file
  \brief Define an internal command for cisstMultiTask
*/


#ifndef _mtsCommandQueuedWrite_h
#define _mtsCommandQueuedWrite_h

#include <cisstMultiTask/mtsCommandWrite.h>
#include <cisstMultiTask/mtsQueue.h>
#include <cisstMultiTask/mtsMailBox.h>

// Always include last
#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

 */


/*! Write queued command using mtsGenericObject parameter. This is used for all
    queued write commands.  It is also used to create a generic event
    observer (combined with mtsMulticastCommandWriteBase) that can
    accept any payload (derived from mtsGenericObject). */
// PK: methods are defined in mtsCommandQueuedWriteBase.cpp
class CISST_EXPORT mtsCommandQueuedWrite: public mtsCommandWrite
{
 protected:
    typedef mtsCommandWrite BaseType;
    typedef mtsCommandQueuedWrite ThisType;

    /*! Mailbox used to queue the commands */
    mtsMailBox * MailBox;

    size_t ArgumentQueueSize; // size used for queue

    /*! Queue to store arguments */
    mtsQueueGeneric ArgumentsQueue;

    /*! Queue of flags to indicate if the command is blocking or
      not */
    mtsQueue<mtsBlockingType> BlockingFlagQueue;

private:
    /*! Private default constructor to prevent use. */
    inline mtsCommandQueuedWrite(void);

    /*! Private copy constructor to prevent copies */
    inline mtsCommandQueuedWrite(const ThisType & CMN_UNUSED(other));

public:
    /*! Constructor, requires a mailbox to queue commands, a pointer
      on actual command and size used to create the argument queue.
      If the actual command doesn't provide an argument prototype, the
      argument queue is not allocated.  Queue allocation will
      potentially occur later, i.e. when SetArgumentPrototype is
      used.  This is useful when the queued command is added to a
      multicast command. */
    template <class __argumentType>
    mtsCommandQueuedWrite(mtsCallableWriteBase * callable,
                          const std::string & name,
                          const __argumentType & argumentPrototype,
                          mtsMailBox * mailBox,
                          size_t size):
        BaseType(callable, name, argumentPrototype),
        MailBox(mailBox),
        ArgumentQueueSize(size),
        ArgumentsQueue(),
        BlockingFlagQueue(size, MTS_NOT_BLOCKING)
    {
        ArgumentsQueue.SetSize(size, argumentPrototype);
    }

    /*! Destructor */
    inline virtual ~mtsCommandQueuedWrite() {}

    inline virtual mtsCommandQueuedWrite * Clone(mtsMailBox * mailBox, size_t size) const {
        return new mtsCommandQueuedWrite(this->Callable, this->Name, *(this->ArgumentPrototype),
                                         mailBox, size);
    }

    // Allocate should be called when a task calls GetMethodXXX().
    virtual void Allocate(size_t size);


    inline virtual void SetArgumentPrototype(const mtsGenericObject * argumentPrototype) {
        BaseType::SetArgumentPrototype(argumentPrototype);
        this->Allocate(this->ArgumentQueueSize);
    }

    /* documented in base class */
    mtsExecutionResult Execute(const mtsGenericObject & argument,
                               mtsBlockingType blocking);

    inline const mtsGenericObject * ArgumentPeek(void) const {
        return ArgumentsQueue.Peek();
    }


    inline mtsGenericObject * ArgumentGet(void) {
        return ArgumentsQueue.Get();
    }


    mtsBlockingType BlockingFlagGet(void);

    const std::string GetMailBoxName(void) const;

    virtual void ToStream(std::ostream & outputStream) const;
};

#endif // _mtsCommandQueuedWrite_h
