/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerLocal.h 978 2009-11-22 03:02:48Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-12-07

  (C) Copyright 2009-2010 Johns Hopkins University (JHU), All Rights
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

  This class defines the local component manager (LCM) which replaces the 
  previous task manager (mtsTaskManager) and is implemented as a singleton.
  
  Major differences between the two are:
  
  1) The LCM manages tasks and devices as a unified object, a component, which 
  is of type mtsDevice. For this, task map and device map in the task manager 
  has been consolidated into a single data structure, component map.

  2) The LCM does not keep the connection information; All connection information
  are now maintained and managed by the global component manager (GCM).
  
  \note Related classes: mtsManagerLocalInterface, mtsManagerGlobalInterface, mtsManagerGlobal
*/


#ifndef _mtsManagerLocal_h
#define _mtsManagerLocal_h

#include <cisstCommon/cmnGenericObject.h>
#include <cisstCommon/cmnClassRegister.h>
#include <cisstCommon/cmnNamedMap.h>

#include <cisstOSAbstraction/osaThreadBuddy.h>
#include <cisstOSAbstraction/osaTimeServer.h>
#include <cisstOSAbstraction/osaMutex.h>

#include <cisstMultiTask/mtsManagerLocalInterface.h>
#include <cisstMultiTask/mtsManagerGlobalInterface.h>
#include <cisstMultiTask/mtsForwardDeclarations.h>

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

    /*! Flag for unit tests. Enabled only for unit tests and set as false by default */
    bool UnitTestEnabled;

    /*! Flag to skip network-related processing such as network proxy creation,
        network proxy startup, remote connection, and so on. Set as false by default */
    bool UnitTestNetworkProxyEnabled;

protected:
    /*! Typedef for component map: (component name, component object)
        component object is a pointer to mtsDevice object. */
    typedef cmnNamedMap<mtsDevice> ComponentMapType;
    ComponentMapType ComponentMap;

    // TODO: time synchronization between tasks (conversion local relative time
    // into absolute time or vice versa) ???

    /*! Time server used by all tasks. */
    osaTimeServer TimeServer;

    // TODO: Determine JGraph should be implemented at LCM as well
    //osaSocket JGraphSocket;
    //bool JGraphSocketConnected;

    /*! Process name of this local component manager. Should be globally unique 
        across a system. */
    const std::string ProcessName;

    /*! IP address of the global component manager */
    const std::string GlobalComponentManagerIP;

    /*! IP address of this machine. Set internally by SetIPAddress(). */
    std::string ProcessIP;

    /*! Mutex to use ComponentMap safely */
    osaMutex ComponentMapChange;

    /*! A pointer to the global component manager.
        Depending on configurations, this points to two different objects:
        - In standalone mode, this is an instance of the GCM (of type 
          mtsManagerGlobal) running in the same process.
        - In network mode, this is a pointer to a proxy object for the GCM
          (of type mtsManagerGlobalProxyClient) that links this LCM with the 
          GCM. In this case, the GCM normally runs in a different process. */
    mtsManagerGlobalInterface * ManagerGlobal;

    /*! Protected constructor (singleton)
        If a process name is not specified and thus set as "", this local component
        manager will run in standalone mode and the GCM IP is ignored.
        If a process name is given and the GCM IP is not, a process name is set 
        as specified and GlobalComponentManagerIP is set as default value, which 
        is localhost (127.0.0.1).
        If both a process name and the GCM IP are specified, both will be set.
    */
    mtsManagerLocal(const std::string & thisProcessName = "", 
                    const std::string & globalComponentManagerIP = "localhost");

    /*! Destructor. Includes OS-specific cleanup. */
    virtual ~mtsManagerLocal();

    /*! Initialization */
    void Initialize(void);

    /*! Connect two local interfaces. This method assumes two components are in 
        the same process.
        
        Returns: provided interface proxy instance id, if server component is a proxy,
                 zero, if server component is an original component,
                 -1,   if error occurs */
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
        proxy manages is automatically removed when removing a component proxy. */
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

    /*! Extract all the information on a provided interface such as command 
        objects and events with arguments serialized */
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

    /*! Returns a total number of interfaces that are running on a component */
    const int GetCurrentInterfaceCount(const std::string & componentName, const std::string & listenerID = "");
#endif

    /*! Connect interfaces at server side */
    bool ConnectServerSideInterface(const unsigned int providedInterfaceProxyInstanceID,
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName, const std::string & listenerID = "");

    /*! Connect interfaces at client side */
    bool ConnectClientSideInterface(const unsigned int connectionID,
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName, const std::string & listenerID = "");

    /*! Get an instance of local component manager */
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

    /*! Check if a component exists by its name */
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

    /*! Create all components. If a component is of type mtsTask, mtsTask::Create()
        is called internally. */
    void CreateAll(void);

    /*! Start all components. If a component is of type mtsTask, mtsTask::Start() 
        is called internally. */
    void StartAll(void);

    /*! Stop all components. If a component is of type mtsTask, mtsTask::Kill()
        is called internally. */
    void KillAll(void);

    /*! Cleanup.  Since a local component manager is a singleton, the
      destructor will be called when the program exits but a library user
      is not capable of handling the timing. Thus, for safe termination, this 
      method should be called before an application quits. */
    void Cleanup(void);

    /*! Do nothing except LINUX RTAI */
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
    
    /*! Returns name of this local component manager */
    inline const std::string GetProcessName(const std::string & listenerID = "") {
        return ProcessName;
    }

    /*! Returns name of this local component manager (for mtsProxyBaseCommon.h) */
    inline const std::string GetName() {
        return GetProcessName();
    }

    /*! Set an IP address of this machine.
        In standalone mode, IP address is set to "localhost" by default.
        In network mode, if there are more than one network interfaces in this 
        machine, all IPs detected are registered to the GCM so that interface 
        proxy clients in other processes can use them one-by-one until it 
        successfully connects to this machine. */
    void SetIPAddress();

    // TODO: do we need this?
    /*! For debugging. Dumps to stream the maps maintained by the manager. */
    void CISST_DEPRECATED ToStream(std::ostream & outputStream) const {}

    // TODO: do we need this?
    /*! Create a dot file to be used by graphviz to generate a nice
      graph of connections between tasks/interfaces. */
    void CISST_DEPRECATED ToStreamDot(std::ostream & outputStream) const {}

    //-------------------------------------------------------------------------
    //  Networking
    //-------------------------------------------------------------------------
#if CISST_MTS_HAS_ICE
    /*! Check if a component is a proxy object based on its name */
    const bool IsProxyComponent(const std::string & componentName) const {
        const std::string proxyStr = "Proxy";
        size_t found = componentName.find(proxyStr);
        return found != std::string::npos;
    }

    /*! Return IP address of this machine. */
    inline std::string GetIPAddress() const { return ProcessIP; }

    /*! Set endpoint access information */
    bool SetProvidedInterfaceProxyAccessInfo(
        const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName,
        const std::string & endpointInfo, const std::string & communicatorID);

#endif
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsManagerLocal)

#endif // _mtsManagerLocal_h

