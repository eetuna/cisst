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
  \brief Defines a command with no argument
*/

#ifndef _mtsCommandVoidReturn_h
#define _mtsCommandVoidReturn_h


#include <cisstMultiTask/mtsCommandVoidReturnBase.h>
#include <string>


/*!
  \ingroup cisstMultiTask

  A templated version of command object with zero arguments for
  execute. The template argument is the class type whose method is
  contained in the command object.  This command is based on a void
  method, i.e. it requires the class and method name as well as an
  instantiation of the class to get and actual pointer on the
  method. */
template <class _classType, class _returnType>
class mtsCommandVoidReturn: public mtsCommandVoidReturnBase {

public:
    typedef mtsCommandVoidReturnBase BaseType;
    typedef _returnType ReturnType;

    /*! Typedef for the specific interface. */
    typedef _classType ClassType;

    /*! This type. */
    typedef mtsCommandVoidReturn<ClassType, ReturnType> ThisType;

    /*! Typedef for pointer to member function (method) of a specific
      class (_classType). */
    typedef void(_classType::*ActionType)(ReturnType & result);

private:
    /*! Private copy constructor to prevent copies */
    inline mtsCommandVoidReturn(const ThisType & CMN_UNUSED(other)) {}

protected:
    /*! The pointer to member function of the receiver class that
      is to be invoked for a particular instance of the command. */
    ActionType Action;

    /*! Stores the receiver object of the command. */
    ClassType * ClassInstantiation;

    template <bool, typename _dummy = void>
    class ConditionalCast {
        // Default case: ReturnType not derived from mtsGenericObjectProxy
    public:
        static mtsExecutionResult CallMethod(ClassType * classInstantiation, ActionType action, mtsGenericObject & result) {
            ReturnType * resultCasted = mtsGenericTypes<ReturnType>::CastArg(result);
            if (resultCasted == 0) {
                return mtsExecutionResult::BAD_INPUT;
            }
            (classInstantiation->*action)(*resultCasted);
            return mtsExecutionResult::DEV_OK;
        }
    };

    template <typename _dummy>
    class ConditionalCast<true, _dummy> {
        // Specialization: ReturnType is derived from mtsGenericObjectProxy (and thus also from mtsGenericObject)
        // In this case, we may need to create a temporary Proxy object.
    public:
        static mtsExecutionResult CallMethod(ClassType * classInstantiation, ActionType action, mtsGenericObject & result) {
            // First, check if a Proxy object was passed.
            ReturnType * resultCasted = dynamic_cast<ReturnType *>(&result);
            if (resultCasted) {
                (classInstantiation->*action)(*resultCasted);
                return mtsExecutionResult::DEV_OK;
            }
            // If it isn't a Proxy, maybe it is a ProxyRef
            typedef typename ReturnType::RefType ReturnRefType;
            ReturnRefType * dataRef = dynamic_cast<ReturnRefType *>(&result);
            if (!dataRef) {
                CMN_LOG_INIT_ERROR << "mtsCommandVoidReturn: CallMethod could not cast from " << typeid(result).name()
                                   << " to " << typeid(ReturnRefType).name() << std::endl;
                return mtsExecutionResult::BAD_INPUT;
            }
            // Now, make the call using the temporary
            ReturnType temp;
            (classInstantiation->*action)(temp);
            // Finally, copy the data to the return
            *dataRef = temp;
            return mtsExecutionResult::DEV_OK;
        }
    };

public:
    /*! The constructor. Does nothing. */
    mtsCommandVoidReturn(void): BaseType(), ClassInstantiation(0) {}

    /*! The constructor.
      \param action Pointer to the member function that is to be called
      by the invoker of the command
      \param classInstantiation Pointer to the receiver of the command
      \param name A string to identify the command. */
    mtsCommandVoidReturn(ActionType action, ClassType * classInstantiation, const std::string & name,
                         const ReturnType & returnPrototype):
        BaseType(name),
        Action(action),
        ClassInstantiation(classInstantiation)
    {
        this->ReturnPrototype = mtsGenericTypes<ReturnType>::ConditionalCreate(returnPrototype, name);
    }

    /*! The destructor. Does nothing */
    virtual ~mtsCommandVoidReturn() {
        if (this->ReturnPrototype) {
            delete this->ReturnPrototype;
        }
    }

    /*! The execute method. Calling the execute method from the
      invoker applies the operation on the receiver.
    */
    mtsExecutionResult Execute(mtsGenericObject & result) {
        if (this->IsEnabled()) {
            return ConditionalCast<cmnIsDerivedFromTemplated<ReturnType, mtsGenericObjectProxy>::YES
                                  >::CallMethod(ClassInstantiation, Action, result);
        }
        return mtsExecutionResult::DISABLED;
    }

    /* documented in base class */
    void ToStream(std::ostream & outputStream) const {
        outputStream << "mtsCommandVoidReturn: ";
        if (this->ClassInstantiation) {
            outputStream << this->Name << "(" << this->GetReturnPrototype()->Services()->GetName() << "&) using class/object \""
                         << mtsObjectName(this->ClassInstantiation) << "\" currently "
                         << (this->IsEnabled() ? "enabled" : "disabled");
        } else {
            outputStream << "Not initialized properly";
        }
    }
};

#endif // _mtsCommandVoidReturn_h

