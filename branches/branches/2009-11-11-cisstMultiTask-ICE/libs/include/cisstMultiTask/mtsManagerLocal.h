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

class CISST_EXPORT mtsManagerLocal: public mtsManagerLocalInterface, public cmnGenericObject 
{
    friend class mtsManagerLocalTest;
    friend class mtsManagerGlobalTest;
    friend class mtsManagerGlobal;

    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

private:
    /*! Singleton object */
    static mtsManagerLocal * Instance;

    /*! Flag for unit tests. Enabled only by unit tests. Set false by default */
    bool UnitTestEnabled;

    /*! Flag to skip network-related processing such as network proxy creation,
        network proxy startup, remote connection, and so on. Set false by default */
    bool UnitTestNetworkProxyEnabled;

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

    /*! IP address of the global component manager. Set in network mode */
    const std::string GlobalComponentManagerIP;

    /*! IP address of this machine. This is internally set by SetIPAddress(). */
    std::string ProcessIP;

    /*! True if this local component manager has received ProxyCreationCompleted
        message from the global component manager and all the proxy objects are 
        created successfully. */
    bool isProxyCreationCompleted;

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

    /*! Protected constructor (singleton local component manager)
        If thisProcessName is not given and thus set as "", this local component
        manager will run as standalone mode and globalComponentManagerIP is
        ignored.
        If thisProcessName is given and globalComponentManagerIP is not,
        process name is set and GlobalComponentManagerIP is set as default value
        which is localhost (127.0.0.1).
        If both thisProcessName and globalComponentManagerIP are specified,
        both will be set.
    */
    mtsManagerLocal(const std::string & thisProcessName = "", 
                    const std::string & globalComponentManagerIP = "localhost");

    /*! Constructor for unit tests (used only by unit tests) */
    //mtsManagerLocal(const std::string & thisProcessName);

    /*! Destructor. Includes OS-specific cleanup. */
    virtual ~mtsManagerLocal();

    /*! Initialization */
    void Initialize(void);

    /*! Connect two local components (interfaces). Note that this method assumes 
        that two components are in the same process.
        Returns provided interface proxy instance id, 
                      if server component is a proxy component
                zero, if server component is an original component
                -1,   if error occurs
        */
    int ConnectLocally(
        const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverComponentName, const std::string & serverProvidedInterfaceName);

    //-------------------------------------------------------------------------
    //  Methods required by mtsManagerLocalInterface
    //-------------------------------------------------------------------------
public:
#if CISST_MTS_HAS_ICE
    /*! Create a component proxy. This should be called before an interface 
        proxy is created. */
    bool CreateComponentProxy(const std::string & componentProxyName, const std::string & listenerID = "");

    /*! Remove a component proxy. Note that all the interface proxies that the
        proxy manages should be automatically removed when removing a component
        proxy. */
    bool RemoveComponentProxy(const std::string & componentProxyName, const std::string & listenerID = "");

    /*! Create a provided interface proxy using ProvidedInterfaceDescription */
    bool CreateProvidedInterfaceProxy(
        const std::string & serverComponentProxyName,
        const ProvidedInterfaceDescription & providedInterfaceDescription, const std::string & listenerID = "");

    /*! Create a required interface proxy using RequiredInterfaceDescription */
    bool CreateRequiredInterfaceProxy(
        const std::string & clientComponentProxyName,
        const RequiredInterfaceDescription & requiredInterfaceDescription, const std::string & listenerID = "");

    /*! Remove a provided interface proxy */
    bool RemoveProvidedInterfaceProxy(
        const std::string & clientComponentProxyName, const std::string & providedInterfaceProxyName, const std::string & listenerID = "");

    /*! Remove a required interface proxy */
    bool RemoveRequiredInterfaceProxy(
        const std::string & serverComponentProxyName, const std::string & requiredInterfaceProxyName, const std::string & listenerID = "");

    /*! The GCM informs LCM of the successful completion of proxy creation */
    void ProxyCreationCompleted(const std::string & listenerID = "");

    /*! Extract all the information on a provided interface such as command 
        objects and events with serialization */
    bool GetProvidedInterfaceDescription(
        const std::string & componentName,
        const std::string & providedInterfaceName, 
        ProvidedInterfaceDescription & providedInterfaceDescription, const std::string & listenerID = "");

    /*! Extract all the information on a required interface such as function
        objects and event handlers with arguments serialized */
    bool GetRequiredInterfaceDescription(
        const std::string & componentName,
        const std::string & requiredInterfaceName, 
        RequiredInterfaceDescription & requiredInterfaceDescription, const std::string & listenerID = "");

    /*! Returns the total number of interfaces that are running on a component */
    const int GetCurrentInterfaceCount(const std::string & componentName, const std::string & listenerID = "");
#endif

    bool ConnectServerSideInterface(const unsigned int providedInterfaceProxyInstanceId,
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName, const std::string & listenerID = "");

    bool ConnectClientSideInterface(const unsigned int connectionID,
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName, const std::string & listenerID = "");

    /*! Create an instance of local component manager (singleton) */
    static mtsManagerLocal * GetInstance(
        const std::string & thisProcessName = "", const std::string & globalComponentManagerIP = "");

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

    /* Connect two interfaces */
    bool Connect(
        const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverComponentName, const std::string & serverProvidedInterfaceName);

    bool Connect(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName);

    /*! Disconnect two interfaces */
    bool Disconnect(
        const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverComponentName, const std::string & serverProvidedInterfaceName);

    bool Disconnect(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName);

    /*! Create all components added if a component is of mtsTask type. 
        Internally, this calls mtsTask::Create() for each task. */
    void CreateAll(void);

    /*! Start all components if a component is of mtsTask type.  
        Internally, this calls mtsTask::Start() for each task.
        If a task will use the current thread, it is called last because its Start method
        will not return until the task is terminated. There should be no more than one task
        using the current thread. 

        TODO: After all thread are created, proxy client should be created and run!!!
        
        */
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
    inline const std::string GetProcessName(const std::string & listenerID = "") {
        return ProcessName;
    }

    //
    // TODO: shouldn't this be inline const std::string GetName() const??? (CONST keyword!)
    //

    /*! Returns the name of this local component manager (for mtsProxyBaseCommon.h) */
    inline const std::string GetName() {
        return GetProcessName();
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
    /*! Check if a component is a proxy object based on a component name */
    const bool IsProxyComponent(const std::string & componentName) const {
        const std::string proxyStr = "Proxy";
        size_t found = componentName.find(proxyStr);
        return found != std::string::npos;
    }

    /*! Return IP address of this machine. */
    inline std::string GetIPAddress() const { return ProcessIP; }

    bool SetProvidedInterfaceProxyAccessInfo(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName,
        const std::string & endpointInfo, const std::string & communicatorID);

#endif
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsManagerLocal)

#endif // _mtsManagerLocal_h

