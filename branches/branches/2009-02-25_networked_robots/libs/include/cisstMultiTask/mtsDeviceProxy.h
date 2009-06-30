/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsDeviceProxy.h 291 2009-04-28 01:49:13Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-05-06

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


/*!
  \file
  \brief Defines a periodic task.
*/

// MJUNG:
// A server task proxy is of mtsDevice type, not mtsTask because we assume that
// there is only one required interface at client side, which means there is
// only one user thread. Thus, we don't need to consider thread synchronization
// issues at client side.
// However, if there can be more than one required interface at client side, we
// need to consider them. Moreover, if one required interface can connect to 
// more than one provided interface, things get more complicated. (However, the 
// current design doesn't support such connection.)
#ifndef _mtsDeviceProxy_h
#define _mtsDeviceProxy_h

#include <cisstMultiTask/mtsDevice.h>
#include <cisstMultiTask/mtsDeviceInterfaceProxy.h>

#include <cisstMultiTask/mtsExport.h>

class mtsDeviceInterfaceProxyClient;

class mtsDeviceProxy : public mtsDevice 
{
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

protected:
    //mtsDeviceInterfaceProxyClient * RequiredInterfaceProxy;

public:
    mtsDeviceProxy(const std::string & deviceName) : 
        mtsDevice(deviceName)//, RequiredInterfaceProxy(requiredInterfaceProxy)
    {}
    ~mtsDeviceProxy() 
    {}

    void Configure(const std::string & deviceName) {};

    /*! Create a local provided interface and returns the pointer to it. */
    mtsDeviceInterface * CreateProvidedInterfaceProxy(
        mtsDeviceInterfaceProxyClient * requiredInterfaceProxy,
        const mtsDeviceInterfaceProxy::ProvidedInterfaceInfo & providedInterfaceInfo);

    /*! Return a name for a server device proxy. */
    // Server task proxy naming rule:
    //    
    //   Server-TS:PI-TC:RI
    //
    //   where TS: server task name
    //         PI: provided interface name
    //         TC: client task name
    //         RI: required interface name
    static std::string GetServerTaskProxyName(
        const std::string & resourceTaskName, const std::string & providedInterfaceName,
        const std::string & userTaskName, const std::string & requiredInterfaceName);

    /*! Return a name for a client task proxy. */
    // Client task proxy naming rule:
    //    
    //   Client-TS:PI-TC:RI
    //
    //   where TS: server task name
    //         PI: provided interface name
    //         TC: client task name
    //         RI: required interface name
    static std::string GetClientTaskProxyName(
        const std::string & resourceTaskName, const std::string & providedInterfaceName,
        const std::string & userTaskName, const std::string & requiredInterfaceName);
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsDeviceProxy)

#endif // _mtsDeviceProxy_h

