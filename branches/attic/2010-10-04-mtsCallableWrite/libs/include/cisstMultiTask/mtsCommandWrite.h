/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Ankur Kapoor, Peter Kazanzides, Anton Deguet
  Created on: 2004-04-30

  (C) Copyright 2004-2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


/*!
  \file
  \brief Defines a command with one argument
*/

#ifndef _mtsCommandWrite_h
#define _mtsCommandWrite_h


#include <cisstMultiTask/mtsCommandBase.h>
#include <cisstMultiTask/mtsGenericObjectProxy.h>


/*!
  \ingroup cisstMultiTask

  A templated version of command object with one argument for
  execute. The template argument is the interface type whose method is
  contained in the command object. */
class mtsCommandWrite: public mtsCommandBase
{
    friend class mtsMulticastCommandWrite;
public:
    typedef mtsCommandBase BaseType;

    /*! This type. */
    typedef mtsCommandWrite ThisType;

private:
    /*! Private copy constructor to prevent copies */
    inline mtsCommandWrite(const ThisType & CMN_UNUSED(other));

    /*! The constructor. Does nothing */
    mtsCommandWrite(void);

protected:
    /*! The pointer to member function of the receiver class that
      is to be invoked for a particular instance of the command*/
    mtsCallableWriteBase * Callable;

public:
    /*! The constructor. */
    template <class __argumentType>
    mtsCommandWrite(mtsCallableWriteBase * callable,
                    const std::string & name,
                    const __argumentType & argumentPrototype):
        BaseType(name),
        Callable(callable)
    {
        this->ArgumentPrototype = mtsGenericTypes<__argumentType>::ConditionalCreate(argumentPrototype, name);
    }


    mtsCommandWrite(mtsCallableWriteBase * callable,
                    const std::string & name):
        BaseType(name),
        Callable(callable),
        ArgumentPrototype(0)
    {
    }


    /*! The destructor. Deletes the internal argument prototype. */
    virtual ~mtsCommandWrite();

    /*! The execute method. Calling the execute method from the invoker
      applies the operation on the receiver.
      \param obj The data passed to the operation method
    */
    virtual mtsExecutionResult Execute(const mtsGenericObject & argument,
                                       mtsBlockingType CMN_UNUSED(blocking));
    /* documented in base class */
    size_t NumberOfArguments(void) const;

    /* documented in base class */
    bool Returns(void) const;

    /*! Return a pointer on the argument prototype */
    virtual const mtsGenericObject * GetArgumentPrototype(void) const;

    /*! Get a direct pointer to the callable object.  This method is
      used for queued commands.  The caller should still use the
      Execute method which will queue the command.  When the command
      is de-queued, one needs access to the callable object to call
      the final method or function. */
    mtsCallableWriteBase * GetCallable(void) const;

    /* commented in base class */
    virtual void ToStream(std::ostream & outputStream) const;

protected:

    virtual void SetArgumentPrototype(const mtsGenericObject * argumentPrototype);

    const mtsGenericObject * ArgumentPrototype;
};


#endif // _mtsCommandWrite_h

