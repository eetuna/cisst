/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCommandQueuedWriteProxy.h 75 2009-02-24 16:47:20Z adeguet1 $

  Author(s):  Min Yang Jung
  Created on: 2009-04-29

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights Reserved.

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

#ifndef _mtsCommandQueuedWriteProxy_h
#define _mtsCommandQueuedWriteProxy_h

#include <cisstMultiTask/mtsCommandQueuedWriteBase.h>

#include <cisstMultiTask/mtsExport.h>

class CISST_EXPORT mtsCommandQueuedWriteProxy: public mtsCommandQueuedWriteBase {
public:
    typedef mtsCommandQueuedWriteBase BaseType;

    inline mtsCommandQueuedWriteProxy(void) : BaseType() 
    {}

    inline mtsCommandQueuedWriteProxy(mtsCommandWriteProxy * actualCommand)
        : BaseType(0, actualCommand)
        //
        // TODO:
        //      MailBox, Execute() method implementation!!!
        //
        //MailBox(mailBox),
        //ActualCommand(actualCommand)
    {}

    const cmnGenericObject * GetArgumentPrototype(void) const
    // TODO: FIX THIS
    { return reinterpret_cast<const cmnGenericObject *>(1); }

    mtsCommandQueuedWriteProxy * Clone(mtsMailBox* mailBox, unsigned int size) const
    // TODO: FIX THIS
    { return reinterpret_cast<mtsCommandQueuedWriteProxy * >(1); }

    // Allocate should be called when a task calls GetMethodXXX().
    void Allocate(unsigned int size)
    {}

    mtsCommandBase::ReturnType Execute(const cmnGenericObject & argument) {
        static int cnt = 0;
        std::cout << "mtsCommandQueuedWriteProxy called (" << ++cnt << "): " << Name << std::endl;
        return BaseType::DEV_OK;
    }
    
    const cmnGenericObject * ArgumentPeek(void) const
    // TODO: FIX THIS
    { return reinterpret_cast<const cmnGenericObject *>(1); }

    cmnGenericObject * ArgumentGet(void)
    // TODO: FIX THIS
    { return reinterpret_cast<cmnGenericObject *>(1); }
};

#endif // _mtsCommandQueuedWrite_h

