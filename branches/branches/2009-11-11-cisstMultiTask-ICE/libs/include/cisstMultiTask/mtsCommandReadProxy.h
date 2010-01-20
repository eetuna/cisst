/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCommandReadProxy.h 75 2009-02-24 16:47:20Z adeguet1 $

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
  \brief Defines a command with one argument 
*/

#ifndef _mtsCommandReadProxy_h
#define _mtsCommandReadProxy_h

#include <cisstMultiTask/mtsCommandReadOrWriteBase.h>
#include <cisstMultiTask/mtsCommandProxyBase.h>
#include <cisstMultiTask/mtsProxySerializer.h>

/*!
  \ingroup cisstMultiTask

  mtsCommandReadProxy is a proxy for mtsCommandRead. This proxy contains
  CommandId set as a function pointer of which type is mtsFunctionRead.
  When Execute() method is called, the CommandId is sent to the server task
  over networks with one payload. The provided interface proxy manages 
  this process.
*/
class mtsCommandReadProxy : public mtsCommandReadBase, public mtsCommandProxyBase 
{
    friend class mtsComponentProxy;

protected:
    /*! Per-command serializer and deserializer */
    mtsProxySerializer Serializer;

public:
    /*! Typedef for base type */
    typedef mtsCommandReadBase BaseType;

    /*! Constructor */
    mtsCommandReadProxy(const std::string & commandName) : BaseType(commandName)
    {
        // Command proxy is disabled by default (enabled when command id and
        // network proxy are set).
        Disable();
    }

    /*! Destructor */
    virtual ~mtsCommandReadProxy() {
        if (this->ArgumentPrototype) {
            delete this->ArgumentPrototype;
        }
    }

    /*! Set command id */
    virtual void SetCommandId(const CommandIDType & commandId) {
        mtsCommandProxyBase::SetCommandId(commandId);
        //
        // TODO: What's this???
        //
        //NetworkProxyServer->AddPerCommandSerializer(CommandId, &Serializer);
    }

    /*! Set an argument prototype */
    void SetArgumentPrototype(mtsGenericObject * argumentPrototype) {
        this->ArgumentPrototype = argumentPrototype;
    }
    
    /*! The execute method. */
    virtual mtsCommandBase::ReturnType Execute(mtsGenericObject & argument) {
        if (this->IsDisabled()) mtsCommandBase::DISABLED;

        //NetworkProxyServer->SendExecuteCommandReadSerialized(CommandId, argument);
        return mtsCommandBase::DEV_OK;
    }
    
    /*! Generate human readable description of this object */
    void ToStream(std::ostream & outputStream) const {
        outputStream << "mtsCommandReadProxy: " << Name << ", " << CommandId << " with "
                     << NetworkProxyServer->ClassServices()->GetName()
                     << ": currently " << (this->IsEnabled() ? "enabled" : "disabled");
    }
};

#endif // _mtsCommandReadProxy_h
