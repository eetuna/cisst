/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerGlobalInterface.h 794 2009-09-01 21:43:56Z pkazanz1 $

  Author(s):  Min Yang Jung
  Created on: 2009-11-15

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
  \brief Definition of mtsManagerGlobalInterface
  \ingroup cisstMultiTask

  This class defines an interface used by local component manager to communicate 
  with the global component manager.  The interface is defined as a pure abstract 
  class because there are two different cases that the interface are used for:

  Standalone Mode: Inter-thread communication, no ICE.  A local component manager 
    directly connects to the global component manager that runs in the same process.
    In this case, mtsManagerLocal::ManagerGlobal is of type mtsManagerGlobal.

  Network mode: Inter-process communication, ICE enabled.  A local component 
    manager connects to the global component manager via a proxy for it.
    In this case, mtsManagerLocal::ManagerGlobal is of type mtsManagerGlobalProxyClient.

  \note Please refer to mtsManagerGlobal and mtsManagerGlobalProxyClient for details.
*/

#ifndef _mtsManagerGlobalInterface_h
#define _mtsManagerGlobalInterface_h

#include <cisstMultiTask/mtsInterfaceCommon.h>

class mtsManagerLocalInterface;

class CISST_EXPORT mtsManagerGlobalInterface 
{
public:
    /* Typedef for the state of connection. See comments on Connect() for details. */
    typedef enum {
        CONNECT_ERROR = 0,
        CONNECT_LOCAL,
        CONNECT_REMOTE_BASE
    } CONNECT_STATE;

    //-------------------------------------------------------------------------
    //  Process Management
    //-------------------------------------------------------------------------
    /*! Register a process */
    virtual bool AddProcess(mtsManagerLocalInterface * localManager) = 0;

    /*! Find a process. */
    virtual bool FindProcess(const std::string & processName) const = 0;

    /*! Get a process object (local component manager object) */
    virtual mtsManagerLocalInterface * GetProcessObject(const std::string & processName) = 0;

    /*! Remove a process. */
    virtual bool RemoveProcess(const std::string & processName) = 0;

    //-------------------------------------------------------------------------
    //  Component Management
    //-------------------------------------------------------------------------
    /*! Register a component. */
    virtual bool AddComponent(const std::string & processName, const std::string & componentName) = 0;

    /*! Find a component using process name and component name */
    virtual bool FindComponent(const std::string & processName, const std::string & componentName) const = 0;

    /*! Remove a component. */
    virtual bool RemoveComponent(const std::string & processName, const std::string & componentName) = 0;

    //-------------------------------------------------------------------------
    //  Interface Management
    //-------------------------------------------------------------------------
    /*! Register an interface. An interface can be added/removed dynamically. */
    virtual bool AddProvidedInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName, const bool isProxyInterface) = 0;

    virtual bool AddRequiredInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName, const bool isProxyInterface) = 0;

    /*! Find an interface using process name, component name, and interface name */
    virtual bool FindProvidedInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName) const = 0;

    virtual bool FindRequiredInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName) const = 0;

    /*! Remove an interface. An interface can be added/removed dynamically. */
    virtual bool RemoveProvidedInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName) = 0;

    virtual bool RemoveRequiredInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName) = 0;

    //-------------------------------------------------------------------------
    //  Connection Management
    //-------------------------------------------------------------------------
    /*! Connect two interfaces with timeout.

        Returns CONNECT_ERROR : in case of errors
                CONNECT_LOCAL : in case of local connection.
                CONNECT_REMOTE_BASE + n : if connection can be established.
                                          (n: unsigned integer)

        A CONNECT_ERROR is returned when the two components are managed by the 
        same local component manager (i.e., they are in the same process). 
        In this case, no timeout is set, no proxy component is created, and 
        therefore local component managers don't have to call ConnectConfirm().

        A return value of (CONNECT_REMOTE_BASE + n) is a temporary session 
        id assigned by the global component manager, which is used by 
        ConnectConfirm() to find connection information.

        //
        TODO: Update the following comments
        //
        If Connect() is called, timer is set at the global component manager. If
        timeout elapses before BOTH local component managers confirm successful 
        connection, the global component manager calls Disconnect() to clean up 
        the connection. */
    virtual unsigned int Connect(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName) = 0;

    /*! Local component manager confirms that connection has been successfully 
        established.
        Return true if the global component manager acknowledged the connection. */
    virtual bool ConnectConfirm(unsigned int connectionSessionID) = 0;

    /*! Disconnect two interfaces.
        Return true if disconnecting already disconnected connection. */
    virtual bool Disconnect(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName) = 0;

    //-------------------------------------------------------------------------
    //  Networking
    //-------------------------------------------------------------------------
    /*! Add access information of a server proxy (i.e., provided interface proxy)
        which a client proxy (i.e., required interface proxy) connects to. */
    virtual bool SetProvidedInterfaceProxyAccessInfo(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName,
        const std::string & endpointInfo, const std::string & communicatorID) = 0;

    /*! Fetch access information of a server proxy (i.e., provided interface 
        proxy) with the complete specification of connection  */
    virtual bool GetProvidedInterfaceProxyAccessInfo(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName,
        std::string & endpointInfo, std::string & communicatorID) = 0;

    /*! Fetch access information of a server proxy (i.e., provided interface 
        proxy) without specifying client interface. 
        Because one proxy server at client side can handle multiple connections,
        it is required to have a way to fetch the access information of a proxy 
        server only with a server interface specification. */
    virtual bool GetProvidedInterfaceProxyAccessInfo(
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName,
        std::string & endpointInfo, std::string & communicatorID) = 0;

    /*! Make a client process initiate connection process. 
        When LCM::Connect() is called at the server side, the server process
        internally calls this method to start connection process at the client 
        side. */
    virtual bool InitiateConnect(const unsigned int connectionID,
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName) = 0;

    /*! Make a server process connect components. Internally, a required 
        interface network proxy (of type mtsComponentInterfaceProxyClient)
        is created, run, and connects to a provided interface network proxy
        (of type mtsComponentInterfaceProxyServer). */
    virtual bool ConnectServerSideInterface(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName) = 0;
};

#endif // _mtsManagerGlobalInterface_h

