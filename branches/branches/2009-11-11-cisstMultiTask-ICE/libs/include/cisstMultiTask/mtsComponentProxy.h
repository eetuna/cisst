/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsComponentProxy.h 291 2009-04-28 01:49:13Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-12-18

  (C) Copyright 2009-2010 Johns Hopkins University (JHU), All Rights Reserved.

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
  
  The following shows how proxy components exchange data across a network:

       Client Process                             Server Process
  ---------------------------             --------------------------------
   Original function object 
   -> Command proxy object   
   -> Serialization          ->  Network  -> Deserialization
                                          -> Function proxy object
                                          -> Original command object
                                          -> Execution (Void, Write, ...)
                                          -> Argument calculation, if any
                                          -> Serialization
            Deserialization  <-  Network  <- Return data (if any)
            -> Return data
*/

#ifndef _mtsComponentProxy_h
#define _mtsComponentProxy_h

#include <cisstMultiTask/mtsDevice.h>
#include <cisstMultiTask/mtsDeviceInterface.h>
#include <cisstMultiTask/mtsInterfaceCommon.h>
#include <cisstMultiTask/mtsComponentInterfaceProxy.h>
#include <cisstMultiTask/mtsForwardDeclarations.h>

#include <cisstMultiTask/mtsExport.h>

class mtsComponentProxy : public mtsDevice
{
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

protected:
    //-------------------------------------------------------------------------
    //  Data Structures for Server Component Proxy
    //-------------------------------------------------------------------------
    
    //-------------------------------------------------------------------------
    //  Data Structures for Client Component Proxy
    //-------------------------------------------------------------------------
    //class RequiredInterfaceProxyResources {
    //};
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

    /*! Typedef to manage provided interface proxies of which type is 
        mtsComponentInterfaceProxyServer. */
    typedef cmnNamedMap<mtsComponentInterfaceProxyServer> ProvidedInterfaceNetworkProxyMapType;
    ProvidedInterfaceNetworkProxyMapType ProvidedInterfaceNetworkProxies;

    /*! Typedef to manage required interface proxies of which type is 
        mtsComponentInterfaceProxyClient. */
    typedef cmnNamedMap<mtsComponentInterfaceProxyClient> RequiredInterfaceNetworkProxyMapType;
    RequiredInterfaceNetworkProxyMapType RequiredInterfaceNetworkProxies;

    /*! Typedef to manage provided interface proxy resources */
    //typedef std::map<unsigned int, ProvidedInterfaceProxyResources *> ProvidedInterfaceProxyResourceMapType;
    typedef std::map<unsigned int, mtsProvidedInterface *> ProvidedInterfaceProxyInstanceMapType;
    ProvidedInterfaceProxyInstanceMapType ProvidedInterfaceProxyInstanceMap;

    /*! Counter for provided interface proxy instances */
    unsigned int ProvidedInterfaceProxyInstanceID;

public:
    mtsComponentProxy(const std::string & componentProxyName);
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

    /*! Create a provided interface instance by cloning a provided interface
        proxy. Conceptually, this method corresponds to 
        mtsDeviceInterface::AllocateResources(). */
    mtsProvidedInterface * CreateProvidedInterfaceInstance(
        const mtsProvidedInterface * providedInterfaceProxy, unsigned int & instanceID);

    //-------------------------------------------------------------------------
    //  Methods to Manage Network Proxy
    //-------------------------------------------------------------------------
    /*! Create a network proxy which corresponds to a provided interface proxy. */
    bool CreateInterfaceProxyServer(const std::string & providedInterfaceProxyName,
                                    std::string & adapterName,
                                    std::string & endpointAccessInfo,
                                    std::string & communicatorID);

    /*! Create a network proxy which corresponds to a required interface proxy. */
    bool CreateInterfaceProxyClient(const std::string & requiredInterfaceProxyName,
                                    const std::string & serverEndpointInfo,
                                    const std::string & communicatorID,
                                    const unsigned int providedInterfaceProxyInstanceId);

    /*! Check whether a network proxy server for a provided interface proxy has 
        been created or not. */
    inline bool FindInterfaceProxyServer(const std::string & providedInterfaceName) const {
        return ProvidedInterfaceNetworkProxies.FindItem(providedInterfaceName);
    }

    /*! Check whether a network proxy client for a required interface proxy has 
        been created or not. */
    inline bool FindInterfaceProxyClient(const std::string & requiredInterfaceName) const {
        return RequiredInterfaceNetworkProxies.FindItem(requiredInterfaceName);
    }

    /*! Set command proxy IDs in a provided interface proxy at client side as 
        function proxy IDs fetched from a required interface proxy at server 
        side. */
    bool UpdateCommandProxyID(
        const std::string & serverProvidedInterfaceName, const std::string & clientComponentName, 
        const std::string & clientRequiredInterfaceName, const unsigned int providedInterfaceProxyInstanceId);

    /*! Set event handler IDs in a required interface proxy at the server side
        as event generator IDs fetched from a provided interface proxy at the 
        client side. */
    bool UpdateEventHandlerProxyID(const std::string & requiredInterfaceName);

    //-------------------------------------------------------------------------
    //  Getters
    //-------------------------------------------------------------------------
    /*! Return a total number of interfaces (used to determine if this componet 
        proxy should be removed; when all interfaces are removed, the component
        proxy has to be cleaned up) */
    inline unsigned int GetInterfaceCount() const {
        return ProvidedInterfaces.size() + RequiredInterfaces.size();
    }

    /*! Check if a network proxy is active */
    bool IsActiveProxy(const std::string & proxyName, const bool isProxyServer) const;

    /*! Extract function proxy pointers */
    bool GetFunctionProxyPointers(const std::string & requiredInterfaceName, 
        mtsComponentInterfaceProxy::FunctionProxyPointerSet & functionProxyPointers);

    //-------------------------------------------------------------------------
    //  Utilities
    //-------------------------------------------------------------------------
    const std::string GetNewPortNumberAsString(const unsigned int id) const;
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsComponentProxy)

#endif

