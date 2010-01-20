/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerGlobal.h 794 2009-09-01 21:43:56Z pkazanz1 $

  Author(s):  Min Yang Jung
  Created on: 2009-11-12

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
  \brief Definition of the global manager
  \ingroup cisstMultiTask

  Please see mtsManagerGlobalInterface.h for detailed comments on methods
  defined as pure virtual in mtsManagerGlobalInterface.
*/

#ifndef _mtsManagerGlobal_h
#define _mtsManagerGlobal_h

#include <cisstCommon/cmnGenericObject.h>
#include <cisstCommon/cmnClassRegister.h>
#include <cisstCommon/cmnNamedMap.h>
#include <cisstOSAbstraction/osaMutex.h>
#include <cisstMultiTask/mtsManagerLocalInterface.h>
#include <cisstMultiTask/mtsManagerGlobalInterface.h>
#include <cisstMultiTask/mtsForwardDeclarations.h>

#include <cisstMultiTask/mtsExport.h>

class CISST_EXPORT mtsManagerGlobal : public mtsManagerGlobalInterface {

    friend class mtsManagerGlobalTest;
    friend class mtsManagerLocalTest;

    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

protected:
    //-------------------------------------------------------------------------
    //  Data Structure of Process Map
    //-------------------------------------------------------------------------
    /*
        P1 - C1 - p1 - r1   Process Map: (P, ComponentMap)
        |    |    |    |    Component Map: (C, InterfaceMap)
        |    |    |    r2   Interface Map: { (PI, ConnectionMap) where PI is Provided Interface
        |    |    |                          (RI, ConnectionMap) where RI is Required Interface }
        |    |    p2 - r3   Connection Map: (name of connected interface, ConnectedInterfaceInfo)
        |    |    |
        |    |    r1 - p1
        |    |
        |    C2 - r1 - p2
        |
        P2 - C1
             |
             C2
    */

    /*! Data structure to keep the information about a connected interface */
    class ConnectedInterfaceInfo {
    protected:
        // Names (IDs)
        const std::string ProcessName;
        const std::string ComponentName;
        const std::string InterfaceName;
        // True if this interface is remote
        const bool RemoteConnection;
#if CISST_MTS_HAS_ICE
        // Server proxy access information (sent to client proxy as requested)
        std::string EndpointInfo;
        std::string CommunicatorID;
#endif

        ConnectedInterfaceInfo() : ProcessName(""), ComponentName(""), InterfaceName(""), RemoteConnection(false)
#if CISST_MTS_HAS_ICE
            , EndpointInfo(""), CommunicatorID("")
#endif
        {}

    public:
        ConnectedInterfaceInfo(const std::string & processName, const std::string & componentName, 
                               const std::string & interfaceName, const bool isRemoteConnection)
            : ProcessName(processName), ComponentName(componentName), InterfaceName(interfaceName), RemoteConnection(isRemoteConnection)
        {}

        // Getters
        const std::string GetProcessName() const   { return ProcessName; }
        const std::string GetComponentName() const { return ComponentName; }
        const std::string GetInterfaceName() const { return InterfaceName; }
        const bool IsRemoteConnection() const      { return RemoteConnection; }
#if CISST_MTS_HAS_ICE
        std::string GetEndpointInfo() const        { return EndpointInfo; }
        std::string GetCommunicatorID() const      { return CommunicatorID; }

        // Setters
        void SetProxyAccessInfo(const std::string & endpointInfo, const std::string & communicatorID)
        {
            EndpointInfo = endpointInfo;
            CommunicatorID = communicatorID;
        }
#endif
    };

    /*! Connection map: (connected interface name, connected interface information)
        Map name: a name of component that has these interfaces. */
    typedef cmnNamedMap<mtsManagerGlobal::ConnectedInterfaceInfo> ConnectionMapType;

    // Interface map consists of two pairs of containers: 
    // containers for connection map (provided/required interface maps) and
    // containers for interface type flag map (provided/required interface maps)

    /*! Interface map: a map of registered interfaces in a component
        key=(interface name), value=(connection map)
        value can be null if an interface does not have any connection. */
    typedef cmnNamedMap<ConnectionMapType> ConnectedInterfaceMapType;

    /*! Interface type flag map: a map of registered interfaces in a component
        key=(interface name), value=(bool)
        value is false if an interface is an original interface
                 true  if an interface is a proxy interface 
        This information is used to determine if an interface should be removed 
        (cleaned up) when a connection is disconnected. See 
        mtsManagerGlobal::Disconnect() for more details. */
    typedef std::map<std::string, bool> InterfaceTypeMapType;

    typedef struct {
        ConnectedInterfaceMapType ProvidedInterfaceMap;
        ConnectedInterfaceMapType RequiredInterfaceMap;
        InterfaceTypeMapType ProvidedInterfaceTypeMap;
        InterfaceTypeMapType RequiredInterfaceTypeMap;
    } InterfaceMapType;

    /*! Component map: a map of registered components in a process
        key=(component name), value=(interface map) 
        value can be null if a component does not have any interface. */
    typedef cmnNamedMap<InterfaceMapType> ComponentMapType;

    /*! Process map: a map of registered processes (i.e., local component managers)
        key=(process name), value=(component map) 
        value can be null if a process does not have any component. */
    typedef cmnNamedMap<ComponentMapType> ProcessMapType;
    ProcessMapType ProcessMap;

    //
    // TODO: NEED TO ADD MUTEX FOR PROCESS_MAP
    //

    /*! Lookup table to access local component manager with process id:
        (process name, local component manager id) */
    typedef cmnNamedMap<mtsManagerLocalInterface> LocalManagerMapType;
    LocalManagerMapType LocalManagerMap;

    /*! Mutex to safely use LocalManagerMap and LocalManagerMapByProcessID */
    osaMutex LocalManagerMapChange;

    /*! Connection id */
    unsigned int ConnectionID;

    /* TODO: Remove me. This is for test purpose */
    typedef std::map<unsigned int, unsigned int> AllocatedPointerType;
    AllocatedPointerType AllocatedPointers;

    //-------------------------------------------------------------------------
    //  Processing Methods
    //-------------------------------------------------------------------------
    /*! Clean up the internal variables */
    bool CleanUp(void);

    /*! Get a map containing connection information for a provided interface */
    ConnectionMapType * GetConnectionsOfProvidedInterface(
        const std::string & serverProcessName, const std::string & serverComponentName, 
        const std::string & providedInterfaceName, InterfaceMapType ** interfaceMap);
    ConnectionMapType * GetConnectionsOfProvidedInterface(
        const std::string & serverProcessName, const std::string & serverComponentName, 
        const std::string & providedInterfaceName) const;

    /*! Get a map containing connection information for a required interface */
    ConnectionMapType * GetConnectionsOfRequiredInterface(
        const std::string & clientProcessName, const std::string & clientComponentName, 
        const std::string & requiredInterfaceName, InterfaceMapType ** interfaceMap);
    
    ConnectionMapType * GetConnectionsOfRequiredInterface(
        const std::string & clientProcessName, const std::string & clientComponentName, 
        const std::string & requiredInterfaceName) const;
    
    /*! Add this interface to connectionMap as connected interface */
    bool AddConnectedInterface(ConnectionMapType * connectionMap, 
        const std::string & processName, const std::string & componentName,
        const std::string & interfaceName, const bool isRemoteConnection = false);

    /*! Check if two interfaces are connected */
    bool IsAlreadyConnected(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName);

public:
    /*! Constructor and destructor */
    mtsManagerGlobal();

    ~mtsManagerGlobal();
    
    //-------------------------------------------------------------------------
    //  Process Management
    //-------------------------------------------------------------------------
    bool AddProcess(mtsManagerLocalInterface * localManager);

    bool FindProcess(const std::string & processName) const;

    mtsManagerLocalInterface * GetProcessObject(const std::string & processName);    

    bool RemoveProcess(const std::string & processName);

    //-------------------------------------------------------------------------
    //  Component Management
    //-------------------------------------------------------------------------
    bool AddComponent(const std::string & processName, const std::string & componentName);

    bool FindComponent(const std::string & processName, const std::string & componentName) const;

    bool RemoveComponent(const std::string & processName, const std::string & componentName);

    //-------------------------------------------------------------------------
    //  Interface Management
    //-------------------------------------------------------------------------
    bool AddProvidedInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName, const bool isProxyInterface = false);

    bool AddRequiredInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName, const bool isProxyInterface = false);

    bool FindProvidedInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName) const;

    bool FindRequiredInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName) const;

    bool RemoveProvidedInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName);

    bool RemoveRequiredInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName);

    //-------------------------------------------------------------------------
    //  Connection Management
    //-------------------------------------------------------------------------
    unsigned int Connect(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName);

    bool ConnectConfirm(unsigned int connectionSessionID);

    bool Disconnect(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName);

    //-------------------------------------------------------------------------
    //  Public Getters
    //-------------------------------------------------------------------------
    /*! Generate unique id of an interface as string */
    static const std::string GetInterfaceUID(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName)
    {
        return processName + ":" + componentName + ":" + interfaceName;
    }

    /*! Generate unique name of a proxy component */
    static const std::string GetComponentProxyName(const std::string & processName, const std::string & componentName)
    {
        return processName + ":" + componentName + "Proxy";
    }

    //-------------------------------------------------------------------------
    //  Networking
    //-------------------------------------------------------------------------
#if CISST_MTS_HAS_ICE
    bool SetProvidedInterfaceProxyAccessInfo(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName,
        const std::string & endpointInfo, const std::string & communicatorID);

    bool GetProvidedInterfaceProxyAccessInfo(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName,
        std::string & endpointInfo, std::string & communicatorID);

    bool GetProvidedInterfaceProxyAccessInfo(
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName,
        std::string & endpointInfo, std::string & communicatorID);

    bool InitiateConnect(const unsigned int connectionID,
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName);

    bool ConnectServerSideInterface(const unsigned int providedInterfaceProxyInstanceId,
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName);
#endif
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsManagerGlobal)

#endif // _mtsManagerGlobal_h
