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
    //  Data Structures
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
    class ConnectedInterfaceInfo {
    protected:
        const std::string ProcessName;
        const std::string ComponentName;
        const std::string InterfaceName;

        ConnectedInterfaceInfo() : ProcessName(""), ComponentName(""), InterfaceName("")
        {}

    public:
        ConnectedInterfaceInfo(const std::string & processName, const std::string & componentName, 
                               const std::string & interfaceName)
            : ProcessName(processName), ComponentName(componentName), InterfaceName(interfaceName)
        {}

        std::string GetProcessName()     { return ProcessName; }
        std::string GetComponentName()   { return ComponentName; }
        std::string GetInterfaceName()   { return InterfaceName; }
    };

    /*! Connection map: (connected interface name, connected interface information)
        Map name: name of component that has these interfaces. */
    typedef cmnNamedMap<mtsManagerGlobal::ConnectedInterfaceInfo> ConnectionMapType;

    /*! Interface map: (interface name, connection map)
        Map name: name of component that has these interfaces. */
    typedef cmnNamedMap<ConnectionMapType> ConnectedInterfaceMapType;
    typedef struct {    
        ConnectedInterfaceMapType ProvidedInterfaceMap;
        ConnectedInterfaceMapType RequiredInterfaceMap;
    } InterfaceMapType;

    /*! Component map: (component name, interface map) 
        Map name: name of process that manages these components. */
    typedef cmnNamedMap<InterfaceMapType> ComponentMapType;

    /*! Process map: (process name, component map) */
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

    /*! Generate unique id of an interface as string */
    inline std::string GetInterfaceUID(const std::string & processName,
        const std::string & componentName, const std::string & interfaceName) const
    {
        return processName + ":" + componentName + ":" + interfaceName;
    }

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
        const std::string & interfaceName);

    /*! Check if two interfaces are connected */
    bool IsAlreadyConnected(
        const std::string & clientProcessName,
        const std::string & clientComponentName,
        const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName,
        const std::string & serverComponentName,
        const std::string & serverProvidedInterfaceName);

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
        const std::string & processName, const std::string & componentName, const std::string & interfaceName);

    bool AddRequiredInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName);

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
        const std::string & thisProcessName,
        const std::string & clientProcessName,
        const std::string & clientComponentName,
        const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName,
        const std::string & serverComponentName,
        const std::string & serverProvidedInterfaceName);

    bool ConnectConfirm(unsigned int connectionSessionID);

    void Disconnect(
        const std::string & clientProcessName,
        const std::string & clientComponentName,
        const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName,
        const std::string & serverComponentName,
        const std::string & serverProvidedInterfaceName);
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsManagerGlobal)

#endif // _mtsManagerGlobal_h

