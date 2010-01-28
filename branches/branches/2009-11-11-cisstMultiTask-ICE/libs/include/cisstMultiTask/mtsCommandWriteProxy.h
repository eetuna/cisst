/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCommandWriteProxy.h 75 2009-02-24 16:47:20Z adeguet1 $

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
  \brief Defines a command proxy class with one argument 
*/

#ifndef _mtsCommandWriteProxy_h
#define _mtsCommandWriteProxy_h

#include <cisstMultiTask/mtsCommandReadOrWriteBase.h>
#include <cisstMultiTask/mtsCommandProxyBase.h>
#include <cisstMultiTask/mtsProxySerializer.h>

/*!
  \ingroup cisstMultiTask

  mtsCommandWriteProxy is a proxy class for mtsCommandWrite. When Execute() 
  method is called, the command id with payload is sent to the connected peer 
  interface across a network.

  //
  // TODO: Rewrite the following comments
  //
  Note that there are two different usages of this class: as a command or an event.
  If this class used as COMMANDS, an instance of mtsComponentInterfaceProxyClient 
  class should be provided and this is used to execute a write command at a server.
  When this is used for EVENTS, an instance of mtsComponentInterfaceProxyServer class
  takes care of the process of event propagation across a network so that an event
  is sent to a client and an event handler is called at a client side.
  Currently, only one of them can be initialized as a valid value while the other 
  has to be 0.
*/
class mtsCommandWriteProxy : public mtsCommandWriteBase, public mtsCommandProxyBase 
{
    friend class mtsComponentProxy;
    friend class mtsMulticastCommandWriteBase;
    
protected:
    /*! Per-command (de)serializer */
    mtsProxySerializer Serializer;

public:
    /*! Typedef for base type */
    typedef mtsCommandWriteBase BaseType;

    /*! Constructor */
    mtsCommandWriteProxy(const std::string & commandName) : BaseType(commandName) {
        // Command proxy is disabled by default (enabled when command id and
        // network proxy are set).
        Disable();
    }

    /*! Destructor */
    ~mtsCommandWriteProxy() {
        if (ArgumentPrototype) {
            delete ArgumentPrototype;
        }
    }
    
    /*! Set command id and register serializer to network proxy. This method
        should be called after SetNetworkProxy() is called. */
    void SetCommandID(const CommandIDType & commandID) {
        mtsCommandProxyBase::SetCommandID(commandID);

        if (NetworkProxyServer) {
            NetworkProxyServer->RegisterPerCommandSerializer(CommandID, &Serializer);
        } else {
            NetworkProxyClient->RegisterPerEventSerializer(CommandID, &Serializer);
        }
    }

    /*! Set an argument prototype */
    void SetArgumentPrototype(mtsGenericObject * argumentPrototype) {
        this->ArgumentPrototype = argumentPrototype;
    }

    /*! Direct execute can be used for mtsMulticastCommandWrite. */
    inline mtsCommandBase::ReturnType Execute(const ArgumentType & argument) {
        if (IsDisabled()) return mtsCommandBase::DISABLED;

        if (NetworkProxyServer) {
            NetworkProxyServer->SendExecuteCommandWriteSerialized(ClientID, CommandID, argument);
        } else {
            NetworkProxyClient->SendExecuteEventWriteSerialized(CommandID, argument);
        }

        return mtsCommandBase::DEV_OK;
    }

    /*! Getter for per-command (de)serializer */
    inline mtsProxySerializer * GetSerializer() {
        return &Serializer;
    }

    /*! Generate human readable description of this object */
    void ToStream(std::ostream & outputStream) const {
        ToStreamBase("mtsCommandWriteProxy", Name, CommandID, IsEnabled(), outputStream);
    }
};

#endif // _mtsCommandWriteProxy_h
