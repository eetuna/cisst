/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Anton Deguet
  Created on: 2010-09-30

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
  \brief Defines a command with one argument
*/

#ifndef _mtsCallableWriteMethod_h
#define _mtsCallableWriteMethod_h


#include <cisstMultiTask/mtsCallableWriteBase.h>
#include <cisstMultiTask/mtsGenericObjectProxy.h>


/*!
  \ingroup cisstMultiTask

  A templated version of command object with one argument for
  execute. The template argument is the interface type whose method is
  contained in the command object. */
template <class _classType, class _argumentType>
class mtsCallableWriteMethod: public mtsCallableWriteBase {

public:
    typedef _argumentType ArgumentType;
    typedef mtsCallableWriteBase BaseType;

    /*! Typedef for the specific interface. */
    typedef _classType ClassType;

    /*! This type. */
    typedef mtsCallableWriteMethod<ClassType, ArgumentType> ThisType;

    /*! Typedef for pointer to member function of the specific interface
      class. */
    typedef void(_classType::*ActionType)(const ArgumentType &);

private:
    /*! Private copy constructor to prevent copies */
    inline mtsCallableWriteMethod(const ThisType & CMN_UNUSED(other));

protected:
    /*! The pointer to member function of the receiver class that
      is to be invoked for a particular instance of the command*/
    ActionType Action;

    /*! Stores the receiver object of the command */
    ClassType * ClassInstantiation;

    template <bool, typename _dummy = void>
    class ConditionalCast {
        // Default case: ArgumentType not derived from mtsGenericObjectProxy
    public:
        static mtsExecutionResult CallMethod(ClassType * classInstantiation,
                                             ActionType action,
                                             const mtsGenericObject & argument)
        {
            const ArgumentType * data = mtsGenericTypes<ArgumentType>::CastArg(argument);
            if (data == 0) {
                return mtsExecutionResult::BAD_INPUT;
            }
            (classInstantiation->*action)(*data);
            return mtsExecutionResult::DEV_OK;
        }
    };

    template <typename _dummy>
    class ConditionalCast <true, _dummy> {
        // Specialization: ArgumentType is derived from mtsGenericObjectProxy (and thus also from mtsGenericObject)
        // In this case, we may need to create a temporary Proxy object.
    public:
        static mtsExecutionResult CallMethod(ClassType * classInstantiation,
                                             ActionType action,
                                             const mtsGenericObject & argument)
        {
            // First, check if a Proxy object was passed.
            const ArgumentType * data = dynamic_cast<const ArgumentType *>(&argument);
            if (data) {
                (classInstantiation->*action)(*data);
                return mtsExecutionResult::DEV_OK;
            }
            // If it isn't a Proxy, maybe it is a ProxyRef
            typedef typename ArgumentType::RefType ArgumentRefType;
            const ArgumentRefType * dataRef = dynamic_cast<const ArgumentRefType *>(&argument);
            if (!dataRef) {
                CMN_LOG_INIT_ERROR << "Write CallMethod could not cast from " << typeid(argument).name()
                                   << " to const " << typeid(ArgumentRefType).name() << std::endl;
                return mtsExecutionResult::BAD_INPUT;
            }
            // Now, make the call using the temporary
            ArgumentType temp;
            temp.Assign(*dataRef);
            (classInstantiation->*action)(temp);
            return mtsExecutionResult::DEV_OK;
        }
    };

private:
    /*! The constructor. Does nothing */
    mtsCallableWriteMethod(void): BaseType() {}

public:
    /*! The constructor.
    //
    // FIXME: this needs to be updated.
    //
      \param action Pointer to the member function that is to be called
      by the invoker of the command
      \param interface Pointer to the receiver of the command
      \param name A string to identify the command. */
    mtsCallableWriteMethod(ActionType action, ClassType * classInstantiation):
        BaseType(),
        Action(action),
        ClassInstantiation(classInstantiation)
    {
    }


    /*! The destructor. Does nothing */
    virtual ~mtsCallableWriteMethod() {}


    /* documented in base class */
    mtsExecutionResult Execute(const mtsGenericObject & argument) {
        return ConditionalCast<cmnIsDerivedFromTemplated<ArgumentType, mtsGenericObjectProxy>::YES
                               >::CallMethod(ClassInstantiation, Action, argument);
    }


    /* documented in base class */
    void ToStream(std::ostream & outputStream) const {
        outputStream << "method based callable write object using class/object \""
                     << mtsObjectName(this->ClassInstantiation) << "\"";
    }
};

#endif // _mtsCallableWriteMethod_h
