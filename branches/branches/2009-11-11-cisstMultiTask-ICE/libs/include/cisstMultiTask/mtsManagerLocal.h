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
#include <cisstMultiTask/mtsForwardDeclarations.h>
//#include <cisstMultiTask/mtsConfig.h>

//#if CISST_MTS_HAS_ICE
//#include <cisstMultiTask/mtsProxyBaseCommon.h>
//#include <cisstMultiTask/mtsDeviceInterfaceProxy.h>
//#endif // CISST_MTS_HAS_ICE

//#include <set>

#include <cisstMultiTask/mtsExport.h>

class CISST_EXPORT mtsManagerLocal: public mtsManagerLocalInterface {

    friend class mtsManagerLocalTest;
    friend class mtsManagerGlobalTest;

    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

private:
    /*! Singleton object */
    static mtsManagerLocal * Instance;

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

    /*! ID for this local component manager. Consists of a process name IP.
        Note that a process name of "localhost" is reserved for the name of local
        component manager in the standalone mode and should not be used by users. */
    const std::string ProcessName;
    const std::string ProcessIP;

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
        Maybe needs to add some more strict process naming rule and/or IP validation
        check routine?
    */
    mtsManagerLocal(const std::string & thisProcessName, const std::string & thisProcessIP);
    
    /*! Destructor. Includes OS-specific cleanup. */
    virtual ~mtsManagerLocal();

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

    /*! Pull out a cmponent from this manager. */
    bool RemoveComponent(mtsDevice * component);
    bool RemoveComponent(const std::string & componentName);

    /*! Retrieve a component by name. */
    mtsDevice * GetComponent(const std::string & componentName);
    mtsTask CISST_DEPRECATED * GetTask(const std::string & taskName); // For backward compatibility

    /* Interfaces for Connect/Disconnect (note that actual connection is managed
       and established by the global component manager). */
    bool Connect(const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
                 const std::string & serverComponentName, const std::string & serverProvidedInterfaceName);

    /*! Disconnect two interfaces */
    void Disconnect(const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
                    const std::string & serverComponentName, const std::string & serverProvidedInterfaceName);

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
    void GetNamesOfComponents(std::vector<std::string>& namesOfComponents) const;

    /*! Getters */
    inline const std::string GetProcessName() const {
        return ProcessName;
    }

    /*! For debugging. Dumps to stream the maps maintained by the manager. */
    void ToStream(std::ostream & outputStream) const;

    /*! Create a dot file to be used by graphviz to generate a nice
      graph of connections between tasks/interfaces. */
    void ToStreamDot(std::ostream & outputStream) const;
    
//#if CISST_MTS_HAS_ICE
//    //-------------------------------------------------------------------------
//    //  Proxy-related
//    //-------------------------------------------------------------------------
//protected:
//    /*! Task manager type. */
//    TaskManagerType TaskManagerTypeMember;
//
//    /*! Task manager communicator ID. Used as one of ICE proxy object properties. */
//    const std::string TaskManagerCommunicatorID;
//
//    /*! Task manager proxy objects. Both are initialized as null at first and 
//      will be assigned later. Either one of the objects should be null and the 
//      other has to be valid.
//      ProxyServer is valid iff this is the global task manager.
//      ProxyClient is valid iff this is a general task manager.
//    */
//    mtsManagerLocalProxyServer * ProxyGlobalTaskManager;
//    mtsManagerLocalProxyClient * ProxyTaskManagerClient;
//
//    /*! IP address information. */
//    std::string GlobalTaskManagerIP;
//    std::string ServerTaskIP;
//
//    /*! Start two kinds of proxies.
//      Task Manager Layer: Start either GlobalTaskManagerProxy of TaskManagerClientProxy
//      according to the type of this task manager.
//      Task Layer: While iterating all tasks, create and start all provided interface 
//      proxies (see mtsTask::RunProvidedInterfaceProxy()).
//    */
//    void StartProxies();
//
//public:
//    /*! Set the type of task manager-global task manager (server) or conventional
//      task manager (client)-and start an appropriate task manager proxy.
//      Also start a task interface proxy. */
//    void SetTaskManagerType(const TaskManagerType taskManagerType) {
//        TaskManagerTypeMember = taskManagerType;
//        StartProxies();
//    }
//
//    /*! Getter */
//    inline TaskManagerType GetTaskManagerType() { return TaskManagerTypeMember; }
//
//    inline mtsManagerLocalProxyServer * GetProxyGlobalTaskManager() const {
//        return ProxyGlobalTaskManager;
//    }
//
//    inline mtsManagerLocalProxyClient * GetProxyTaskManagerClient() const {
//        return ProxyTaskManagerClient;
//    }
//
//    /*! Setter */
//    inline void SetGlobalTaskManagerIP(const std::string & globalTaskManagerIP) {
//        GlobalTaskManagerIP = globalTaskManagerIP;
//    }
//
//    inline void SetServerTaskIP(const std::string & serverTaskIP) {
//        ServerTaskIP = serverTaskIP;
//    }
//#endif // CISST_MTS_HAS_ICE
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsManagerLocal)

#endif // _mtsManagerLocal_h

