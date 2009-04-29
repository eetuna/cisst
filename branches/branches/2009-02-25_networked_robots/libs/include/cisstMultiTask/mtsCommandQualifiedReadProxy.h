/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCommandQualifiedReadProxy.h 75 2009-02-24 16:47:20Z adeguet1 $

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

#ifndef _mtsCommandQualifiedReadProxy_h
#define _mtsCommandQualifiedReadProxy_h

#include <cisstMultiTask/mtsCommandQualifiedReadOrWriteBase.h>

/*!
  \ingroup cisstMultiTask
*/
class mtsCommandQualifiedReadProxy: public mtsCommandQualifiedReadBase {
public:
    typedef const cmnGenericObject * Argument1PointerType;
    typedef cmnGenericObject * Argument2PointerType;
    typedef mtsCommandQualifiedReadBase BaseType;

protected:
    /*! Argument prototype */
    Argument1PointerType Argument1PointerPrototype;
    Argument2PointerType Argument2PointerPrototype;

public:
    mtsCommandQualifiedReadProxy() : BaseType() {}

    mtsCommandQualifiedReadProxy(const std::string & name, 
                                 Argument1PointerType argument1ProtoType,
                                 Argument2PointerType argument2ProtoType) :
        BaseType(name), 
        Argument1PointerPrototype(argument1ProtoType),
        Argument2PointerPrototype(argument2ProtoType)
    {}

    /*! The destructor. Does nothing */
    ~mtsCommandQualifiedReadProxy() {}

    /*! The execute method. */    
    //mtsCommandBase::ReturnType Execute(Argument1PointerType & argument1, 
    //                             Argument2PointerType & argument2) 
    virtual mtsCommandBase::ReturnType Execute(const cmnGenericObject & argument1,
                                               cmnGenericObject & argument2) {
    { 
        return BaseType::DEV_OK;
    }

    /*! For debugging. Generate a human QualifiedReadable output for the
      command object */
    void ToStream(std::ostream & outputStream) const {
        // TODO
    }

    /*! Return a pointer on the argument prototype */
    Argument1PointerType GetArgument1Prototype(void) const {
        return Argument1PointerPrototype;
    }

    Argument2PointerType GetArgument2Prototype(void) const {
        return Argument2PointerPrototype;
    }
};

#endif // _mtsCommandQualifiedReadProxy_h
