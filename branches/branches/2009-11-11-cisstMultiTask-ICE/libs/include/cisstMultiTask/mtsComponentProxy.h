/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsComponentProxy.h 291 2009-04-28 01:49:13Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-12-18

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


/*!
  \file
  \brief Definition of Component Proxy
  \ingroup cisstMultiTask

  A component proxy is of mtsDevice type rather than mtsTask type. This 
  helps avoiding potential thread synchronization issues between ICE and cisst.  
*/

#ifndef _mtsComponentProxy_h
#define _mtsComponentProxy_h

#include <cisstMultiTask/mtsDevice.h>
#include <cisstMultiTask/mtsInterfaceCommon.h>

#include <cisstMultiTask/mtsExport.h>

#include <string>

class mtsComponentProxy : public mtsDevice
{
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    mtsComponentProxy(const std::string & componentProxyName) : mtsDevice(componentProxyName) {}
    virtual ~mtsComponentProxy() {}

    inline void Configure(const std::string & CMN_UNUSED(componentProxyName)) {};

    /*! Create or remove a provided interface proxy */
    bool CreateProvidedInterfaceProxy(ProvidedInterfaceDescription & providedInterfaceDescription);
    bool RemoveProvidedInterfaceProxy(const std::string & providedInterfaceProxyName);

    /*! Create or remove a required interface proxy */
    bool CreateRequiredInterfaceProxy(RequiredInterfaceDescription & requiredInterfaceDescription);
    bool RemoveRequiredInterfaceProxy(const std::string & requiredInterfaceProxyName);

    /*! Return a total number of interfaces (used to determine if this componet 
        proxy should be removed; when all interfaces are removed, the component
        proxy has to be cleaned up) */
    unsigned int GetInterfaceCount() const {
        return ProvidedInterfaces.size() + RequiredInterfaces.size();
    }
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsComponentProxy)

#endif

