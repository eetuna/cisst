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
  \brief Defines a base class for a command with no argument
 */

#ifndef _mtsCommandVoidReturnBase_h
#define _mtsCommandVoidReturnBase_h

#include <cisstMultiTask/mtsCommandBase.h>

// Always include last
#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  A base class command object with an execute method that takes no
  arguments.  To be used to contain 0*Methods. */
class mtsCommandVoidReturnBase: public mtsCommandBase
{
public:
    typedef mtsCommandBase BaseType;

    /*! The constructor. Does nothing */
    mtsCommandVoidReturnBase(void): BaseType() {}

    /*! Constructor with a name. */
    mtsCommandVoidReturnBase(const std::string & name): BaseType(name) {}

    /*! The destructor. Does nothing */
    virtual ~mtsCommandVoidReturnBase() {}

    /*! The execute method. Abstract method to be implemented by derived
      classes to run the actual operation on the receiver
      \result Boolean value, true if success, false otherwise */
    virtual mtsExecutionResult Execute(mtsGenericObject & result) = 0;

    /* documented in base class */
    inline size_t NumberOfArguments(void) const {
        return 0;
    }

    /* documented in base class */
    inline bool Returns(void) const {
        return true;
    }

    /*! Return a pointer on the return prototype */
    inline virtual const mtsGenericObject * GetReturnPrototype(void) const {
        return this->ReturnPrototype;
    }

    /* documented in base class */
    virtual void ToStream(std::ostream & outputStream) const = 0;

protected:
    inline virtual void SetReturnPrototype(const mtsGenericObject * returnPrototype) {
        this->ReturnPrototype = returnPrototype;
    }

    const mtsGenericObject * ReturnPrototype;

};

#endif // _mtsCommandVoidReturnBase_h

