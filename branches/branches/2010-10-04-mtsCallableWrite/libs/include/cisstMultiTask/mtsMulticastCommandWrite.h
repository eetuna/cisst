/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Ankur Kapoor, Peter Kazanzides, Anton Deguet
  Created on: 2004-04-30

  (C) Copyright 2004-2008 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


/*!
  \file
  \brief Defines a command with one argument sent to multiple interfaces
*/


#ifndef _mtsMulticastCommandWrite_h
#define _mtsMulticastCommandWrite_h


#include <cisstMultiTask/mtsCommandWrite.h>
#include <cisstMultiTask/mtsGenericObjectProxy.h>
#include <vector>

// Always include last
#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  This class contains a vector of two or more command objects.
  The primary use of this class is to send events to all observers.
 */
class mtsMulticastCommandWrite: public mtsCommandWrite
{
public:
    typedef mtsCommandWrite BaseType;

protected:
    std::vector<BaseType *> Commands;

    //    typedef typename mtsGenericTypes<ArgumentType>::FinalBaseType ArgumentFinalType;  // derived from mtsGenericObject

public:
    /*! Default constructor. Does nothing. */
    template <class __argumentType>
    mtsMulticastCommandWrite(const std::string & name, const __argumentType & argumentPrototype):
        BaseType(0, name, argumentPrototype)
    {}

    /*! Default destructor. Does nothing. */
    ~mtsMulticastCommandWrite() {
        if (this->ArgumentPrototype) {
            delete this->ArgumentPrototype;
        }
    }

    /*! Add a command to the composite. */
    void AddCommand(BaseType * command);

    /*! Execute all the commands in the composite. */
    virtual mtsExecutionResult Execute(const mtsGenericObject & argument,
                                       mtsBlockingType CMN_UNUSED(blocking)) {
#if 0 // needs to be fixed
        // cast argument first
        const ArgumentFinalType * data = dynamic_cast< const ArgumentFinalType * >(&argument);
        if (data == 0) {
            return mtsCommandBase::BAD_INPUT;
        }
        // if cast succeeded call using actual type
        size_t index;
        const size_t commandsSize = Commands.size();
        for (index = 0; index < commandsSize; index++) {
            Commands[index]->Execute(*data, MTS_NOT_BLOCKING);
        }
#else
        size_t index;
        const size_t commandsSize = Commands.size();
        for (index = 0; index < commandsSize; index++) {
            Commands[index]->Execute(argument, MTS_NOT_BLOCKING);
        }
#endif
        return mtsExecutionResult::DEV_OK;
    }

    /* documented in base class */
    void ToStream(std::ostream & outputStream) const;
};


#endif // _mtsMulticastCommandWrite_h
