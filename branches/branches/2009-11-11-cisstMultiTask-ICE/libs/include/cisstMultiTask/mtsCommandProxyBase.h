/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCommandProxyBase.h 75 2009-02-24 16:47:20Z adeguet1 $

  Author(s):  Min Yang Jung
  Created on: 2010-01-20

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#ifndef _mtsCommandProxyBase_h
#define _mtsCommandProxyBase_h

#include <cisstMultiTask/mtsProxyBaseCommon.h>
#include <cisstMultiTask/mtsComponentInterfaceProxyServer.h>
#include <cisstMultiTask/mtsComponentInterfaceProxyClient.h>

class mtsComponentProxy;

class mtsCommandProxyBase {
protected:
    /*! Pointer to mtsFunctionXXX object at the peer's memory space */
    CommandIDType CommandId;

    /*! Network (ICE) proxy which enables communication with the connected
        interface across a network. This is an instance of either
        mtsComponentInterfaceProxyClient or mtsComponentInterfaceProxyServer */
    mtsProxyBaseCommon<mtsComponentProxy> * NetworkProxy;

    /*! Actual pointer to network proxy. Either one of them should be set and 
        the other one should be NULL. */
    mtsComponentInterfaceProxyClient * NetworkProxyClient;
    mtsComponentInterfaceProxyServer * NetworkProxyServer;

public:
    /*! Constructor */
    mtsCommandProxyBase() : CommandId(0) {}

    /*! Set command id */
    virtual void SetCommandId(const CommandIDType & commandId) {
        CommandId = commandId;
    }

    /*! Set network proxy */
    bool SetNetworkProxy(mtsProxyBaseCommon<mtsComponentProxy> * networkProxy) {
        NetworkProxyClient = dynamic_cast<mtsComponentInterfaceProxyClient*>(networkProxy);
        NetworkProxyServer = dynamic_cast<mtsComponentInterfaceProxyServer*>(networkProxy);
        
        return ((!NetworkProxyClient && NetworkProxyServer) || (NetworkProxyClient && !NetworkProxyServer));
    }
};

#endif // _mtsCommandProxyBase_h
