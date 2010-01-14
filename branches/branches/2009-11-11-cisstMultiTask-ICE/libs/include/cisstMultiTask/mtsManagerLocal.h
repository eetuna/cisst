/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerLocal.h 978 2009-11-22 03:02:48Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-12-07

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
  \brief Definition of Local Component Manager
  \ingroup cisstMultiTask

  This class implements the local component manager which replaces the previous
  task manager (mtsTaskManager) used to manage the connections between tasks and 
  devices. 
  
  Major differences between them are:
  
  1) The local component manager does not differentiate tasks and devices as 
  the task manager did and manages them as components which is of type mtsDevice.
  That is, the local component manager consolidates tasks and devices into a 
  single data structure.

  2) The local component manager does not keep the connection information.
  All the information are now managed by the global component manager
  (mtsManagerGlobal) either in the same process (standalone mode) or through a 
  network (network mode). (Refer to mtsManagerGlobalInterface.h)
  
  This class is also a singleton.

  \note Related classes: mtsManagerLocalInterface, mtsManagerLocalProxyServer
*/


#ifndef _mtsManagerLocal_h
#define _mtsManagerLocal_h

#include <cisstCommon/cmnGenericObject.h>
#include <cisstCommon/cmnClassRegister.h>
//#include <cisstCommon/cmnAssert.h>
#include <cisstCommon/cmnNamedMap.h>

#include <cisstOSAbstraction/osaThreadBuddy.h>
#include <cisstOSAbstraction/osaTimeServer.h>
//#include <cisstOSAbstraction/osaSocket.h>
#include <cisstOSAbstraction/osaMutex.h>

#include <cisstMultiTask/mtsManagerLocalInterface.h>
#include <cisstMultiTask/mtsManagerGlobalInterface.h>
#include <cisstMultiTask/mtsForwardDeclarations.h>

//#include <cisstMultiTask/mtsConfig.h>

//#if CISST_MTS_HAS_ICE
//#include <cisstMultiTask/mtsProxyBaseCommon.h>
//#include <cisstMultiTask/mtsDeviceInterfaceProxy.h>
//#endif // CISST_MTS_HAS_ICE

//#include <set>

#include <cisstMultiTask/mtsExport.h>

class CISST_EXPORT mtsManagerLocal: public mtsManagerLocalInterface, public cmnGenericObject {

    friend class mtsManagerLocalTest;
    friend class mtsManagerGlobalTest;
    friend class mtsManagerGlobal;

    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

private:
    /*! Singleton object */
    static mtsManagerLocal * Instance;

    /*! Flag to skip network-related processing such as proxy creation, proxy
        startup, connection, and so on. This flag is set to false by default 
        and turned on only by unit tests. */
    bool UnitTestOn;

protected:
    /*! Typedef for component map: (component name, component object)
        component object is a pointer to mtsDevice object. */
    typedef cmnNamedMap<mtsDevice> ComponentMapType;
    ComponentMapType ComponentMap;

    // TODO: time synchronization between tasks (conversion local relative time
    // into absolute time or vice versa)

    /*! Time server used by all tasks. */
    osaTimeServer TimeServer;

    //osaSocket JGraphSocket;
    //bool JGraphSocketConnected;

    /*! Process name of this local component manager */
    const std::string ProcessName;

    /*! IP address of this machine. This is internally set by SetIPAddress(). */
    std::string ProcessIP;

    /*! Mutex to use ComponentMap in a thread safe manner */
    osaMutex ComponentMapChange;

    /*! Pointer to the global component manager.
        Depending on configurations, this has two different meanings.
        - In standalone mode, a pointer to the global component manager running 
        in the same process.
        - In network mode, a pointer to a proxy object for the global manager 
        (mtsManagerGlobalProxyClient) that connects to the actual global manager 
        which probably runs in a different process (or different machine). */
    mtsManagerGlobalInterface * ManagerGlobal;

    /*! Constructor.  Protected because this is a singleton.
        Unless all two arguments are valid, this local component manager runs in
        standalone mode, by default.
        In case of real-time OSs, OS-specific initialization should be handled here. 
        
        TODO: Currently I'm just checking if it is empty string or not.
        Maybe we need to add some more strict process naming rule and/or IP validation
        check routine?
    */
    mtsManagerLocal(const std::string & thisProcessName, const std::string & thisProcessIP);

    /*! Constructor for unit tests (used only by unit tests) */
    mtsManagerLocal(const std::string & thisProcessName);

    /*! Destructor. Includes OS-specific cleanup. */
    virtual ~mtsManagerLocal();

    /*! Connect two local components (interfaces). Note that this method assumes 
        that two components are in the same process.
        If connectionSessionID is -1 (default value), this local component manager
        should inform the global task manager of the successful local connection.
        */
    bool ConnectLocally(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName);

    //-------------------------------------------------------------------------
    //  Methods required by mtsManagerLocalInterface
    //-------------------------------------------------------------------------
    /*! Create a component proxy. This should be called before an interface 
        proxy is created. */
    bool CreateComponentProxy(const std::string & componentProxyName);

    /*! Remove a component proxy. Note that all the interface proxies that the
        proxy manages should be automatically removed when removing a component
        proxy. */
    bool RemoveComponentProxy(const std::string & componentProxyName);

    /*! Create a provided interface proxy using ProvidedInterfaceDescription */
    bool CreateProvidedInterfaceProxy(
        const std::string & serverComponentProxyName,
        ProvidedInterfaceDescription & providedInterfaceDescription);

    /*! Create a required interface proxy using RequiredInterfaceDescription */
    bool CreateRequiredInterfaceProxy(
        const std::string & clientComponentProxyName,
        RequiredInterfaceDescription & requiredInterfaceDescription);

    /*! Remove a provided interface proxy */
    bool RemoveProvidedInterfaceProxy(
        const std::string & clientComponentProxyName, const std::string & providedInterfaceProxyName);

    /*! Remove a required interface proxy */
    bool RemoveRequiredInterfaceProxy(
        const std::string & serverComponentProxyName, const std::string & requiredInterfaceProxyName);

    /*! Extract all the information on a provided interface such as command 
        objects and events with serialization */
    bool GetProvidedInterfaceDescription(
        const std::string & componentName,
        const std::string & providedInterfaceName, 
        ProvidedInterfaceDescription & providedInterfaceDescription) const;

    /*! Extract all the information on a required interface such as function
        objects and events with serialization */
    bool GetRequiredInterfaceDescription(
        const std::string & componentName,
        const std::string & requiredInterfaceName, 
        RequiredInterfaceDescription & requiredInterfaceDescription) const;

    /*! Returns the total number of interfaces that are running on a component */
    const int GetCurrentInterfaceCount(const std::string & componentName) const;

    bool ConnectByGlobalComponentManager(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName);

public:
    /*! Create the static instance of local task manager. */
    static mtsManagerLocal * GetInstance(
        const std::string & thisProcessName = "", const std::string & thisProcessIP = "");

    /*! Return a reference to the time server. */
    inline const osaTimeServer & GetTimeServer(void) {
        return TimeServer;
    }

    //-------------------------------------------------------------------------
    //  Component Management
    //-------------------------------------------------------------------------    
    /*! Add a component to this local component manager. */
    bool AddComponent(mtsDevice * component);
    bool CISST_DEPRECATED AddTask(mtsTask * component); // For backward compatibility
    bool CISST_DEPRECATED AddDevice(mtsDevice * component); // For backward compatibility

    /*! Remove a component from this local component manager. */
    bool RemoveComponent(mtsDevice * component);
    bool RemoveComponent(const std::string & componentName);

    /*! Retrieve a component by name. */
    mtsDevice * GetComponent(const std::string & componentName) const;
    mtsTask CISST_DEPRECATED * GetTask(const std::string & taskName); // For backward compatibility
    mtsDevice CISST_DEPRECATED * GetDevice(const std::string & deviceName); // For backward compatibility

    /*! Check the existence of a component by name. */
    const bool FindComponent(const std::string & componentName) const;

    /* Connect two interfaces (limited to connect two local interfaces) */
    bool Connect(
        const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverComponentName, const std::string & serverProvidedInterfaceName);

    /* Connect two interfaces (can connect any two interfaces) */
    bool Connect(
        const std::string & clientProcessName,
        const std::string & clientComponentName,
        const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName,
        const std::string & serverComponentName,
        const std::string & serverProvidedInterfaceName);

    /*! Disconnect two interfaces */
    bool Disconnect(
        const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverComponentName, const std::string & serverProvidedInterfaceName);

    bool Disconnect(
        const std::string & clientProcessName,
        const std::string & clientComponentName,
        const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName,
        const std::string & serverComponentName,
        const std::string & serverProvidedInterfaceName);

    /*! Create all components added if a component is of mtsTask type. 
        Internally, this calls mtsTask::Create() for each task. */
    void CreateAll(void);

    /*! Start all components if a component is of mtsTask type.  
        Internally, this calls mtsTask::Start() for each task.
        If a task will use the current thread, it is called last because its Start method
        will not return until the task is terminated. There should be no more than one task
        using the current thread. */
    void StartAll(void);

    /*! Stop all components.
        Internally, this calls mtsTask::Kill() for each task. */
    void KillAll(void);

    /*! Cleanup.  Since the local component manager is a singleton, the
      destructor will be called when the program exits but the
      user/programmer will not be able to control when exactly.  If
      the cleanup requires some objects to still be instantiated (log
      files, ...), this might lead to crashes.  To avoid this, the
      Cleanup method should be called before the program quits. */
    void Cleanup(void);

    /*! TODO: is this for immediate exit??? */
    inline void Kill(void) {
        __os_exit();
    }

    //-------------------------------------------------------------------------
    //  Utilities
    //-------------------------------------------------------------------------
    /*! Enumerate all the names of components added */
    std::vector<std::string> GetNamesOfComponents(void) const;
    std::vector<std::string> CISST_DEPRECATED GetNamesOfDevices(void) const;  // For backward compatibility
    std::vector<std::string> CISST_DEPRECATED GetNamesOfTasks(void) const;  // For backward compatibility    

    void GetNamesOfComponents(std::vector<std::string>& namesOfComponents) const;
    void CISST_DEPRECATED GetNamesOfDevices(std::vector<std::string>& namesOfDevices) const; // For backward compatibility
    void CISST_DEPRECATED GetNamesOfTasks(std::vector<std::string>& namesOfTasks) const; // For backward compatibility
    
    /*! Returns the name of this local component manager */
    inline const std::string GetProcessName() const {
        return ProcessName;
    }

    /*! Setter to set an IP address of this machine. 
        In standalone mode, IP address is set to localhost by default.
        In network mode, if there are more than one network interface in this 
        machine, a user should choose what to use as an IP address of this 
        machine, which clients connect to. */
    void SetIPAddress();

    /*! For debugging. Dumps to stream the maps maintained by the manager. */
    void CISST_DEPRECATED ToStream(std::ostream & outputStream) const {}

    /*! Create a dot file to be used by graphviz to generate a nice
      graph of connections between tasks/interfaces. */
    void CISST_DEPRECATED ToStreamDot(std::ostream & outputStream) const {}

    //-------------------------------------------------------------------------
    //  Networking
    //-------------------------------------------------------------------------
#if CISST_MTS_HAS_ICE
public:
    /*! Check if a component is a proxy object based on a component name */
    inline const bool IsProxyComponent(const std::string & componentName) {
        const std::string proxyStr = "Proxy";
        size_t found = componentName.find(proxyStr);
        return found != std::string::npos;
    }

    /*! Return IP address of this machine. */
    std::string GetIPAddress() const { return ProcessIP; }

    bool SetProvidedInterfaceProxyAccessInfo(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName,
        const std::string & endpointInfo, const std::string & communicatorID);

    bool ConnectServerProcess(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName);

    bool ConnectClientProcess(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName);

    //
    // TODO: Double check the following comments
    //
    /*! Fetch event generator proxy pointers from the connected provided interface.
        requiredInterfaceProxyUID is a name of the proxy connected to the interface. */
    bool FetchEventGeneratorProxyPointersFrom(const std::string & requiredInterfaceProxyUID);

    /*! Fetch function proxy pointers from the connected required interface.
        providedInterfaceProxyUID is a name of the proxy connected to the interface. */
    bool FetchFunctionProxyPointersFrom(const std::string & providedInterfaceProxyUID);

#endif
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsManagerLocal)

#endif // _mtsManagerLocal_h

