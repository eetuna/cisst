/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCommandReadProxy.h 75 2009-02-24 16:47:20Z adeguet1 $

  Author(s):  Min Yang Jung
  Created on: 2009-04-29

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
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

#ifndef _mtsCommandReadProxy_h
#define _mtsCommandReadProxy_h

#include <cisstMultiTask/mtsCommandReadOrWriteBase.h>

/*!
  \ingroup cisstMultiTask
*/
class mtsCommandReadProxy: public mtsCommandReadBase {
public:
    typedef cmnGenericObject * ArgumentPointerType;
    typedef mtsCommandReadBase BaseType;

protected:
    /*! Argument prototype */
    ArgumentPointerType ArgumentPointerPrototype;

public:
    mtsCommandReadProxy() : BaseType() {}

    mtsCommandReadProxy(const std::string & name, 
                        ArgumentPointerType argumentProtoType) :
        BaseType(name), ArgumentPointerPrototype(argumentProtoType)
    {}

    /*! The destructor. Does nothing */
    virtual ~mtsCommandReadProxy() {}

    /*! The execute method. */
    virtual BaseType::ReturnType Execute(ArgumentType & argument) {
        return BaseType::DEV_OK;
    }

    /*! For debugging. Generate a human readable output for the
      command object */
    void ToStream(std::ostream & outputStream) const {
        // TODO
    }

    /*! Return a pointer on the argument prototype */
    const cmnGenericObject * GetArgumentPrototype(void) const {
        return ArgumentPointerPrototype;
    }
};

#endif // _mtsCommandReadProxy_h
