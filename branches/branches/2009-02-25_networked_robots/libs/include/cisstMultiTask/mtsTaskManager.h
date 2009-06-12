/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Ankur Kapoor, Peter Kazanzides, Anton Deguet
  Created on: 2004-04-30

  (C) Copyright 2004-2008 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


/*!
  \file
  \brief Define the task manager
*/


#ifndef _mtsTaskManager_h
#define _mtsTaskManager_h

#include <cisstCommon/cmnGenericObject.h>
#include <cisstCommon/cmnClassRegister.h>
#include <cisstCommon/cmnAssert.h>
#include <cisstOSAbstraction/osaThreadBuddy.h>
#include <cisstOSAbstraction/osaTimeServer.h>
#include <cisstMultiTask/mtsForwardDeclarations.h>
#include <cisstMultiTask/mtsMap.h>
#include <cisstMultiTask/mtsProxyBaseCommon.h>
#include <cisstMultiTask/mtsTaskManagerProxyClient.h>
#include <cisstMultiTask/mtsTaskManagerProxyServer.h>
#include <cisstMultiTask/mtsDeviceInterfaceProxy.h>

#include <set>

#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  The Task Manager is used to manage the connections between tasks
  and devices.  It is a Singleton object.
*/
class mtsTaskManager;
class mtsTaskManagerProxyServer;
class mtsTaskManagerProxyClient;

class CISST_EXPORT mtsTaskManager: public cmnGenericObject {
    
    friend class mtsTaskManagerTest;

    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

    /*! Typedef for task name and pointer map. */
    typedef mtsMap<mtsTask> TaskMapType;

    /*! Typedef for device name and pointer map. */
    typedef mtsMap<mtsDevice> DeviceMapType;

    /*! Typedef for user task, composed of task name and "output port"
        name. */
    typedef std::pair<std::string, std::string> UserType;
    /*! Typedef for resource device or task, composed of device or
        task name and interface name. */ 
    typedef std::pair<std::string, std::string> ResourceType;
    /*! Typedef for associations between users and resources. */
    typedef std::pair<UserType, ResourceType> AssociationType;
    typedef std::set<AssociationType> AssociationSetType;

public:

    // Default mailbox size -- perhaps this should be specified elsewhere
    enum { MAILBOX_DEFAULT_SIZE = 16 };

    // Typedef for task manager type
    typedef enum {
        TASK_MANAGER_LOCAL,
        TASK_MANAGER_SERVER, // global task manager
        TASK_MANAGER_CLIENT  // conventional task manager
    } TaskManagerType;

protected:

    /*! Mapping of task name (key) and pointer to mtsTask object. */
    TaskMapType TaskMap;

    /*! Mapping of interface name (key) and pointer to mtsInterface
      object. */
    DeviceMapType DeviceMap;

    /*! Mapping of task name (key) and associated interface name. */
    AssociationSetType AssociationSet;

    /*! Time server used by all tasks. */
    osaTimeServer TimeServer;

    /*! Constructor.  Protected because this is a singleton.
        Does OS-specific initialization to start real-time operations. */
    mtsTaskManager(void);
    
    /*! Destructor.  Does OS-specific cleanup. */
    virtual ~mtsTaskManager();

 public:
    /*! Create the static instance of this class. */
    static mtsTaskManager * GetInstance(void) ;

    /*! Return a reference to the time server. */
    const osaTimeServer &GetTimeServer(void) { return TimeServer; }

    /*! Put a task under the control of the Manager. */
    bool AddTask(mtsTask * task);

    /*! Pull out a task from the Manager. */
    bool RemoveTask(mtsTask * task);

    /*! Put a device under the control of the Manager. */
    bool AddDevice(mtsDevice * device);

    /*! Put a task or device under the control of the Manager
      (calls AddTask or AddDevice based on dynamic type of parameter). */
    bool Add(mtsDevice * device);

    /*! List all devices already added */
    std::vector<std::string> GetNamesOfDevices(void) const;

    /*! List all tasks already added */
    std::vector<std::string> GetNamesOfTasks(void) const;
    
    /*! Fetch all tasks already added. (overloaded) */
    void GetNamesOfTasks(std::vector<std::string>& taskNameContainer) const;

    /*! Retrieve a device by name.  Return 0 if the device is not
        known. */
    mtsDevice * GetDevice(const std::string & deviceName);

    /*! Retrieve a task by name.  Return 0 if the task is not
        known. */
    mtsTask * GetTask(const std::string & taskName);

    /*! Connect the required interface of a user task to the provided
      interface of a resource task (or device).
    */
    bool Connect(const std::string & userTaskName, const std::string & requiredInterfaceName,
                 const std::string & resourceTaskName, const std::string & providedInterfaceName);

    /*! Disconnect the required interface of a user task to the provided
      interface of a resource task (or device).
    */
    bool Disconnect(const std::string & userTaskName, const std::string & requiredInterfaceName,
                    const std::string & resourceTaskName, const std::string & providedInterfaceName);
    
    /*! Create all tasks, i.e. create the threads for each already
        added task.  This method will call the mtsTask::Create method
        for each task. */
    void CreateAll(void);

    /*! Start all tasks.  This method will call the mtsTask::Start method for each task.
        If a task will use the current thread, it is called last because its Start method
        will not return until the task is terminated. There should be no more than one task
        using the current thread. */
    void StartAll(void);

    /*! Stop all tasks.  This method will call the mtsTask::Kill method for each task. */
    void KillAll(void);

    /*! For debugging. Dumps to stream the maps maintained by the manager.
      Here is a typical output: 
      <CODE>
      Task Map: {Name, Address}
      { BSVO, 0x80a59a0 }
      { TRAJ, 0x81e3080 }
      { DISP, 0x81e38d8 }
      Interface Map: {Name, Address}
      { BSVO, 0x80a59a0 }
      { LoPoMoCo, 0x81e40e0 }
      Interface Association Map: {Task, Task/Interface}
      { BSVO, LoPoMoCo }
      { TRAJ, BSVO }
      </CODE>
     */
    void ToStream(std::ostream & outputStream) const;

    /*! Create a dot file to be used by graphviz to generate a nice
      graph of connections between tasks/interfaces. */
    void ToStreamDot(std::ostream & outputStream) const;
    
    inline void Kill(void) {
        __os_exit();
    }

    //-------------------------------------------------------------------------
    //  Proxy-related
    //-------------------------------------------------------------------------
protected:
    /*! Dynamic-casted proxy object. 
        Both objects are initialized as null at first and will be assigned later.
        Either one of pointers should be null and the other has to be valid. 
        ProxyServer is valid only if this is the global task manager.
        ProxyClient is valid only if this is a general task manager.
       */
    mtsTaskManagerProxyServer * ProxyServer;
    mtsTaskManagerProxyClient * ProxyClient;

    /*! Task manager type (server or client) */
    TaskManagerType TaskManagerTypeMember;

    /*! Task manager communicator ID. Used for ICE proxy object property. */
    const std::string TaskManagerCommunicatorID;

    /*! Proxy instance. This will be dynamically created. */
    mtsProxyBaseCommon<mtsTaskManager> * Proxy;

    /*! Start two kinds of proxies
        1) [Task Manager Layer] Start mtsTaskManagerProxyServer or mtsTaskManagerProxyClient.
        2) [Task Layer] Iterating all tasks registered, start mtsDeviceInterfaceProxyServer.
    */
    void StartProxies();

    /*! Connect across networks. This is called internally from Connect(). */
    mtsDeviceInterface * GetResourceInterface(
        const std::string & resourceTaskName, const std::string & providedInterfaceName,
        const std::string & userTaskName, const std::string & interfaceRequiredName,
        mtsTask * userTask);

    /*! Create a provided interface proxy and populate it with the complete specification 
        on the remote provided interface. */
    bool CreateProvidedInterfaceProxy(const mtsDeviceInterfaceProxy::ProvidedInterface & providedInterface,
                                      mtsDevice * serverTaskProxy, mtsTask * clientTask);

    /*! Try to connect at server-side. */
    const bool ConnectAtServerSide(const std::string requiredInterfaceName,
                                   const std::string providedInterfaceName);

public:
    /*! Set the type of task manager-global task manager (server) or conventional
        task manager (client)-and start an appropriate task manager proxy.
        Also start a task interface proxy. */
    void SetTaskManagerType(const TaskManagerType taskManagerType) {
        TaskManagerTypeMember = taskManagerType;
        StartProxies();
    }

    /*! Getter */
    inline TaskManagerType GetTaskManagerType() { return TaskManagerTypeMember; }

    //-------------------------------------------------------------------------
    //  Task Manager Layer Processing
    //-------------------------------------------------------------------------
    /*! Inform the global task manager of the addition of a new provided interface. */
    const bool InvokeAddProvidedInterface(const std::string & newProvidedInterfaceName,
                                    const std::string & adapterName,
                                    const std::string & endpointInfo,
                                    const std::string & communicatorID,
                                    const std::string & taskName);

    const bool InvokeAddRequiredInterface(const std::string & newRequiredInterfaceName,
                                    const std::string & taskName);

    const bool InvokeIsRegisteredProvidedInterface(const std::string & taskName,
                                             const std::string & providedInterfaceName);

    const bool InvokeGetProvidedInterfaceInfo(const ::std::string & taskName,
                                        const std::string & providedInterfaceName,
                                        ::mtsTaskManagerProxy::ProvidedInterfaceInfo & info) const;

    void InvokeNotifyInterfaceConnectionResult(
        const bool isServerTask, const bool isSuccess,
        const std::string & userTaskName,     const std::string & requiredInterfaceName,
        const std::string & resourceTaskName, const std::string & providedInterfaceName);

    //-------------------------------------------------------------------------
    //  Task Layer Processing
    //-------------------------------------------------------------------------

    //
    // TODO: FIX ME
    //
    std::string GlobalTaskManagerIP;
    std::string ServerTaskIP;
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsTaskManager)

#endif // _mtsTaskManager_h

