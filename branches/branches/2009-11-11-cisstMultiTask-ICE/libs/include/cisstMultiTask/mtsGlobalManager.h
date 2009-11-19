/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsGlobalManager.h 794 2009-09-01 21:43:56Z pkazanz1 $

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

#ifndef _mtsGlobalManager_h
#define _mtsGlobalManager_h

#include <cisstCommon/cmnGenericObject.h>
#include <cisstCommon/cmnClassRegister.h>
#include <cisstCommon/cmnNamedMap.h>
#include <cisstMultiTask/mtsGlobalManagerInterface.h>
#include <cisstMultiTask/mtsForwardDeclarations.h>

#include <cisstMultiTask/mtsExport.h>

#include <map>

class CISST_EXPORT mtsGlobalManager : public mtsGlobalManagerInterface {

    friend class mtsGlobalManagerTest;

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
    /*! Connection map: (interface id, connected interface info) */
    class ConnectedInterfaceInfo {
    public:
        std::string ProcessName;
        std::string ComponentName;
        std::string InterfaceName;

        void ConnectWith(const std::string & processName,
                         const std::string & componentName, 
                         const std::string & interfaceName)
        {
            ProcessName = processName;
            ComponentName = componentName;
            InterfaceName = interfaceName;
        }

        ConnectedInterfaceInfo() : ProcessName(""), ComponentName(""), InterfaceName("") {}
    };

    /*! Connection map: (interface name, connected interface information)
        The name of this map is assigned as the name of the component that these
        interfaces are managed by. */
    typedef cmnNamedMap<mtsGlobalManager::ConnectedInterfaceInfo> ConnectionMapType;

    /*! Component map: (component name, connection map) 
        The name of this map is assigned as the name of the process that these
        components are managed by. */
    typedef cmnNamedMap<ConnectionMapType> ComponentMapType;

    /*! Process map: (connected process name, component map) */
    typedef cmnNamedMap<ComponentMapType> ProcessMapType;
    ProcessMapType ProcessMap;

    //-------------------------------------------------------------------------
    //  Processing Methods
    //-------------------------------------------------------------------------
    /*! Clean up the internal variables */
    void CleanUp(void);

    /*! Get the information about connection of a given component. If the component 
        is not yet connected to any interface, this returns ConnectionMapType object
        with a size of 0, not NULL. */
    ConnectionMapType * GetConnectionMap(
        const std::string & processName, const std::string & componentName);

public:
    /*! Constructor and destructor */
    mtsGlobalManager();

    ~mtsGlobalManager();

    /*! Add local component manager */
    // TODO: currently this considers standalone mode only. Thus. this should be extended later
    // to be able to handle network mode together.
    //void AddLocalComponentManager(const std::string & localComponentManagerID

    //-------------------------------------------------------------------------
    //  Component Management
    //-------------------------------------------------------------------------
    /*! Register a component to the global manager. */
    bool AddComponent(const std::string & processName, const std::string & componentName);

    /*! Find a component using process name and component name */
    bool FindComponent(const std::string & processName, const std::string & componentName) const;

    /*! Remove a component from the global manager. */
    bool RemoveComponent(
        const std::string & processName, const std::string & componentName);

    //-------------------------------------------------------------------------
    //  Interface Management
    //-------------------------------------------------------------------------
    ///*! Add an interface. Note that adding/removing an interface can be run-time. */
    //bool AddInterface(
    //    const std::string & processName, const std::string & componentName,
    //    const std::string & interfaceName, const bool isProvidedInterface = true);

    ///*! Find an interface using process name, component name, and interface name */
    //bool FindInterface(
    //    const std::string & processName, const std::string & componentName,
    //    const std::string & interfaceName) const;

    ///*! Remove an interface. Note that adding/removing an interface can be run-time. */
    //bool RemoveInterface(
    //    const std::string & processName, const std::string & componentName,
    //    const std::string & interfaceName, const bool isProvidedInterface = true);

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

    bool Disconnect();
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsGlobalManager)

#endif // _mtsGlobalManager_h

