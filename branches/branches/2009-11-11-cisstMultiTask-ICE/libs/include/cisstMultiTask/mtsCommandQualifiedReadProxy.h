/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCommandQualifiedReadProxy.h 75 2009-02-24 16:47:20Z adeguet1 $

  Author(s):  Min Yang Jung
  Created on: 2009-04-29

  (C) Copyright 2009-2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


/*!
  \file
  \brief Defines a command with two arguments.
*/

#ifndef _mtsCommandQualifiedReadProxy_h
#define _mtsCommandQualifiedReadProxy_h

#include <cisstMultiTask/mtsCommandQualifiedReadOrWriteBase.h>
#include <cisstMultiTask/mtsCommandProxyBase.h>
#include <cisstMultiTask/mtsProxySerializer.h>

/*!
  \ingroup cisstMultiTask

  mtsCommandQualifiedReadProxy is a proxy for mtsCommandQualifiedRead. 
  This proxy contains CommandId set as a function pointer of which type is 
  mtsFunctionQualifiedRead. When Execute() method is called, the CommandId 
  is sent to the server task over networks with two payloads. 
  The provided interface proxy manages this process.
*/
class mtsCommandQualifiedReadProxy : public mtsCommandQualifiedReadBase, public mtsCommandProxyBase
{
    friend class mtsComponentProxy;

protected:
    /*! Per-command serializer and deserializer */
    mtsProxySerializer Serializer;

    /*! Argument prototypes. Deserialization recovers the original argument
        prototype objects. */
    mtsGenericObject *Argument1Prototype, *Argument2Prototype;

public:
    /*! Typedef for base type */
    typedef mtsCommandQualifiedReadBase BaseType;
    
    mtsCommandQualifiedReadProxy(const std::string & commandName) : BaseType(commandName)
    {
        // Command proxy is disabled by default (enabled when command id and
        // network proxy are set).
        Disable();
    }

    /*! The destructor. Does nothing */
    virtual ~mtsCommandQualifiedReadProxy() 
    {
        //
        // TODO: Don't need to release these???
        //
        //if (this->ArgumentPrototype) {
        //    delete this->ArgumentPrototype;
        //}
    }

    /*! Set command id */
    virtual void SetCommandId(const CommandIDType & commandId) {
        mtsCommandProxyBase::SetCommandId(commandId);
        //
        // TODO: What's this???
        //
        //NetworkProxyServer->AddPerCommandSerializer(CommandId, &Serializer);
    }

    /*! The execute method. */
    mtsCommandBase::ReturnType Execute(const mtsGenericObject & argument1, mtsGenericObject & argument2) {
        if (this->IsDisabled()) mtsCommandBase::DISABLED;

        //NetworkProxyServer->SendExecuteCommandQualifiedReadSerialized(
        //    CommandId, argument1, argument2);

        return mtsCommandBase::DEV_OK;
    }

    /*! Generate human readable description of this object */
    void ToStream(std::ostream & outputStream) const {
        outputStream << "mtsCommandWriteProxy: " << Name << ", " << CommandId << " with ";
        if (NetworkProxyServer) {
            outputStream << NetworkProxyServer->ClassServices()->GetName();
        } else {
            outputStream << NetworkProxyClient->ClassServices()->GetName();
        }
        outputStream << ": currently " << (this->IsEnabled() ? "enabled" : "disabled");
    }

    /*! Set argument prototypes */
    void SetArgumentPrototype(mtsGenericObject * argument1Prototype, 
                              mtsGenericObject * argument2Prototype) 
    {
        Argument1Prototype = argument1Prototype;
        Argument2Prototype = argument2Prototype;
    }

    /*! Return a pointer on the argument prototype */
    const mtsGenericObject * GetArgument1Prototype(void) const {
        return Argument1Prototype;
    }

    const mtsGenericObject * GetArgument2Prototype(void) const {
        return Argument2Prototype;
    }
};

#endif // _mtsCommandQualifiedReadProxy_h
