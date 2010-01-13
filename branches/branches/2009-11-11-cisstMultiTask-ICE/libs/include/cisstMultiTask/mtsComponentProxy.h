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

class mtsComponentProxy : public mtsDevice
{
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

    //-------------------------------------------------------------------------
    //  Data Structures for Server Component
    //-------------------------------------------------------------------------
protected:
    /*! Function proxies. CreateProvidedInterfaceProxy() uses these maps to
        assign commandIDs of the command proxies in a provided interface proxy. */
    //typedef cmnNamedMap<mtsFunctionVoid>  FunctionVoidProxyMapType;
    //typedef cmnNamedMap<mtsFunctionWrite> FunctionWriteProxyMapType;
    //typedef cmnNamedMap<mtsFunctionRead>  FunctionReadProxyMapType;
    //typedef cmnNamedMap<mtsFunctionQualifiedRead> FunctionQualifiedReadProxyMapType;
    //FunctionVoidProxyMapType FunctionVoidProxyMap;
    //FunctionWriteProxyMapType FunctionWriteProxyMap;
    //FunctionReadProxyMapType FunctionReadProxyMap;
    //FunctionQualifiedReadProxyMapType FunctionQualifiedReadProxyMap;

    /*! Event handler proxies. CreateRequiredInterfaceProxy() uses these maps
        to assign ----- TODO ----- */
    //typedef cmnNamedMap<mtsCommandVoidProxy>  EventHandlerVoidProxyMapType;
    //typedef cmnNamedMap<mtsCommandWriteProxy> EventHandlerWriteProxyMapType;
    //EventHandlerVoidProxyMapType  EventHandlerVoidProxyMap;
    //EventHandlerWriteProxyMapType EventHandlerWriteProxyMap;

    //-------------------------------------------------------------------------
    //  Data Structures for Client Component
    //-------------------------------------------------------------------------

    //
    // TODO: add codes here
    //



    /*! Typedef to manage provided interface proxies of which type is 
        mtsComponentInterfaceProxyServer. */
    typedef cmnNamedMap<mtsComponentInterfaceProxyServer> ProvidedInterfaceProxyMapType;
    ProvidedInterfaceProxyMapType ProvidedInterfaceProxies;

    /*! Typedef to manage required interface proxies of which type is 
        mtsComponentInterfaceProxyClient. */
    //typedef cmnNamedMap<mtsDeviceInterfaceProxyClient> RequiredInterfaceProxyMapType;
    //RequiredInterfaceProxyMapType RequiredInterfaceProxies;

public:
    mtsComponentProxy(const std::string & componentProxyName) : mtsDevice(componentProxyName) {}
    virtual ~mtsComponentProxy();

    inline void Configure(const std::string & CMN_UNUSED(componentProxyName)) {};

    //-------------------------------------------------------------------------
    //  Methods to Manage Interface Proxy
    //-------------------------------------------------------------------------
    /*! Create or remove a provided interface proxy */
    bool CreateProvidedInterfaceProxy(ProvidedInterfaceDescription & providedInterfaceDescription);
    bool RemoveProvidedInterfaceProxy(const std::string & providedInterfaceProxyName);

    /*! Create or remove a required interface proxy */
    bool CreateRequiredInterfaceProxy(RequiredInterfaceDescription & requiredInterfaceDescription);
    bool RemoveRequiredInterfaceProxy(const std::string & requiredInterfaceProxyName);

    //-------------------------------------------------------------------------
    //  Methods to Manage Network Proxy
    //-------------------------------------------------------------------------
    /*! Create a network proxy which corresponds to a provided interface proxy. */
    bool CreateInterfaceProxyServer(const std::string & providedInterfaceProxyName,
                                    std::string & adapterName,
                                    std::string & endpointAccessInfo,
                                    std::string & communicatorId);

    /*! Create a network proxy which corresponds to a required interface proxy. */
    bool CreateInterfaceProxyClient(const std::string & requiredInterfaceProxyName);

    //-------------------------------------------------------------------------
    //  Getters
    //-------------------------------------------------------------------------
    /*! Return a total number of interfaces (used to determine if this componet 
        proxy should be removed; when all interfaces are removed, the component
        proxy has to be cleaned up) */
    unsigned int GetInterfaceCount() const {
        return ProvidedInterfaces.size() + RequiredInterfaces.size();
    }

    //-------------------------------------------------------------------------
    //  Utilities
    //-------------------------------------------------------------------------
    const std::string GetNewPortNumberAsString(const unsigned int id) const;
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsComponentProxy)

#endif

