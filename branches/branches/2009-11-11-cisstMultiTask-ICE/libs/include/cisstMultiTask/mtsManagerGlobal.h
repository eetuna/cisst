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

  TODO: add description
*/

#ifndef _mtsManagerGlobal_h
#define _mtsManagerGlobal_h

#include <cisstCommon/cmnGenericObject.h>
#include <cisstCommon/cmnClassRegister.h>
#include <cisstCommon/cmnNamedMap.h>
#include <cisstMultiTask/mtsManagerGlobalInterface.h>
#include <cisstMultiTask/mtsForwardDeclarations.h>

#include <cisstMultiTask/mtsExport.h>

#include <map>

class CISST_EXPORT mtsManagerGlobal : public mtsManagerGlobalInterface {

    friend class mtsManagerGlobalTest;

    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

protected:
    /*! List of connected local component managers */
    // TODO: currently this considers standalone mode only. Thus. this should be extended later
    // to be able to handle network mode together.
    typedef cmnNamedMap<mtsTaskManager> LocalComponentManagerMapType;
    LocalComponentManagerMapType LocalComponentManagerMap;

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

    //-------------------------------------------------------------------------
    //  Processing Methods
    //-------------------------------------------------------------------------
    /*! Clean up the internal variables */
    void CleanUp(void);

    /*! Helper methods to easily access internal data structure */
    /*
    ComponentMapType * GetComponentMap(const std::string & processName);

    ConnectedInterfaceMapType * GetProvidedInterfaceMap(
        const std::string & processName, const std::string & componentName);

    ConnectedInterfaceMapType * GetRequiredInterfaceMap(
        const std::string & processName, const std::string & componentName);

    ConnectionMapType * GetProvidedInterfaceConnectionMap(
        const std::string & processName, const std::string & componentName, const std::string & providedInterfaceName);

    ConnectionMapType * GetRequiredInterfaceConnectionMap(
        const std::string & processName, const std::string & componentName, const std::string & requiredInterfaceName);
    */

public:
    /*! Constructor and destructor */
    mtsManagerGlobal();

    ~mtsManagerGlobal();

    /*! Add local component manager */
    // TODO: currently this considers standalone mode only. Thus. this should be extended later
    // to be able to handle network mode together.
    //void AddLocalComponentManager(const std::string & localComponentManagerID

    //-------------------------------------------------------------------------
    //  Process Management
    //-------------------------------------------------------------------------
    /*! Register a process. */
    bool AddProcess(const std::string & processName);

    /*! Find a process. */
    bool FindProcess(const std::string & processName) const;

    /*! Remove a process. */
    bool RemoveProcess(const std::string & processName);

    //-------------------------------------------------------------------------
    //  Component Management
    //-------------------------------------------------------------------------
    /*! Register a component. */
    bool AddComponent(const std::string & processName, const std::string & componentName);

    /*! Find a component using process name and component name */
    bool FindComponent(const std::string & processName, const std::string & componentName) const;

    /*! Remove a component. */
    bool RemoveComponent(const std::string & processName, const std::string & componentName);

    //-------------------------------------------------------------------------
    //  Interface Management
    //-------------------------------------------------------------------------
    /*! Register an interface. Note that adding/removing an interface can be run-time. */
    bool AddProvidedInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName);

    bool AddRequiredInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName);

    /*! Find an interface using process name, component name, and interface name */
    bool FindProvidedInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName) const;

    bool FindRequiredInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName) const;

    /*! Remove an interface. Note that adding/removing an interface can be run-time. */
    bool RemoveProvidedInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName);

    bool RemoveRequiredInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName);

    //-------------------------------------------------------------------------
    //  Connection Management
    //-------------------------------------------------------------------------
    /*! Connect two components. */
    bool Connect(
        const std::string & clientProcessName,
        const std::string & clientComponentName,
        const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName,
        const std::string & serverComponentName,
        const std::string & serverProvidedInterfaceName);

    bool Disconnect(
        const std::string & clientProcessName,
        const std::string & clientComponentName,
        const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName,
        const std::string & serverComponentName,
        const std::string & serverProvidedInterfaceName);

    /*! Get a connection information map of the provided/required interface specified.
        Note that this does not validity check. Thus, validity should be checked first
        using FindProvidedInterface()/FindRequiredInterface() before calling this method. */
    ConnectionMapType * GetProvidedInterfaceConnectionMap(
        const std::string & serverProcessName, const std::string & serverComponentName, 
        const std::string & providedInterfaceName, InterfaceMapType ** interfaceMap);

    ConnectionMapType * GetRequiredInterfaceConnectionMap(
        const std::string & clientProcessName, const std::string & clientComponentName, 
        const std::string & requiredInterfaceName, InterfaceMapType ** interfaceMap);

};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsManagerGlobal)

#endif // _mtsManagerGlobal_h

