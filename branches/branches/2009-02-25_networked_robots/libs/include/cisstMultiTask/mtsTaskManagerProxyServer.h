/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsTaskManagerProxyServer.h 142 2009-03-11 23:02:34Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-03-17

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#ifndef _mtsTaskManagerProxyServer_h
#define _mtsTaskManagerProxyServer_h

#include <cisstMultiTask/mtsTaskManager.h>
#include <cisstMultiTask/mtsProxyBaseServer.h>
#include <cisstMultiTask/mtsTaskManagerProxy.h>
#include <cisstMultiTask/mtsMap.h>
#include <cisstMultiTask/mtsTaskGlobal.h>

#include <cisstMultiTask/mtsExport.h>

#include <set>
#include <map>
#include <string>

/*!
  \ingroup cisstMultiTask

  TODO: add class summary here
*/
class CISST_EXPORT mtsTaskManagerProxyServer : public mtsProxyBaseServer<mtsTaskManager> {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

    friend class TaskManagerServerI;

public:
    /*! Container: (in: task name, out: task information) */
    typedef std::map<std::string, mtsTaskGlobal> GlobalTaskMapType;

    /*! Container: (in: task manager ID, out: list of task names) */
    typedef std::vector<std::string> TaskList;
    typedef std::map<std::string, TaskList> GlobalTaskManagerMapType;

protected:
    //--------------------------- Protected member data ---------------------//
    /*! Definitions for send thread */
    class TaskManagerServerI;
    typedef IceUtil::Handle<TaskManagerServerI> TaskManagerServerIPtr;
    TaskManagerServerIPtr Sender;
    
    /*! Map for managing tasks. */
    GlobalTaskMapType GlobalTaskMap;

    /*! Map for managing task managers. */
    GlobalTaskManagerMapType GlobalTaskManagerMap;

    //-------------------------- Protected methods --------------------------//
    /*! Resource clean-up */
    void OnClose();

    //----------------------- Proxy Implementation --------------------------//
    
    //-----------------------------------
    //  Task Manager Processing
    //-----------------------------------
    /*! Update the information of all tasks. 
        If a task manager of which name is identical to the one speicified 
        already exists, the old one is removed first and completely replaced 
        with the new one. */
    void AddTaskManager(const ::mtsTaskManagerProxy::TaskList& localmtsTaskGlobal);

    /*! Check if a specific task manager exists. */
    const bool FindTaskManager(const std::string taskManagerID) const;

    /*! Remove a specific task manager. */
    const bool RemoveTaskManager(const std::string taskManagerID);

    /*! Add a new provided interface. */
    bool AddProvidedInterface(
        const ::mtsTaskManagerProxy::ProvidedInterfaceInfo & providedInterfaceInfo);

    //-----------------------------------
    //  Task Processing
    //-----------------------------------
    /*! Return the information on the specified task. */
    mtsTaskGlobal * GetTask(const std::string & taskName);

    /*! Add a task. Return false if the specified task has been already registered. */
    //const bool AddTask(const std::string taskName);

    /*! Remove a task. Return false if the specified task is not found. */
    //const bool RemoveTask(const std::string taskName);

    //---------------------------- Proxy Support ----------------------------//
    /*! Create a servant which serves TaskManager clients. */
    Ice::ObjectPtr CreateServant() {
        Sender = new TaskManagerServerI(IceCommunicator, Logger, this);
        return Sender;
    }
    
    /*! Entry point to run a proxy. */
    void Start(mtsTaskManager * callingTaskManager);

    /*! Start a send thread and wait for shutdown (blocking call). */
    void StartServer();

    /*! Thread runner */
    static void Runner(ThreadArguments<mtsTaskManager> * arguments);

    /*! Clean up thread-related resources. */
    void OnThreadEnd();
    
public:
    mtsTaskManagerProxyServer(const std::string& adapterName,
                              const std::string& endpointInfo,
                              const std::string& communicatorID)
        : mtsProxyBaseServer(adapterName, endpointInfo, communicatorID)
    {}
    ~mtsTaskManagerProxyServer();

    //-------------------------------------------------------------------------
    //  Definition by mtsTaskManagerProxy.ice
    //-------------------------------------------------------------------------
protected:
    class TaskManagerServerI : public mtsTaskManagerProxy::TaskManagerServer,
                               public IceUtil::Monitor<IceUtil::Mutex> 
    {
    private:
        Ice::CommunicatorPtr Communicator;
        bool Runnable;
        std::set<mtsTaskManagerProxy::TaskManagerClientPrx> _clients;
        IceUtil::ThreadPtr Sender;
        Ice::LoggerPtr Logger;
        mtsTaskManagerProxyServer * TaskManagerServer;

    public:
        TaskManagerServerI(const Ice::CommunicatorPtr& communicator, 
                           const Ice::LoggerPtr& logger,
                           mtsTaskManagerProxyServer * taskManagerServer);

        void Start();
        void Run();
        void Destroy();

        void AddClient(const ::Ice::Identity&, const ::Ice::Current&);
        void AddTaskManager(const ::mtsTaskManagerProxy::TaskList&, const ::Ice::Current&);
        bool AddProvidedInterface(const ::mtsTaskManagerProxy::ProvidedInterfaceInfo&, const ::Ice::Current&);
    };
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsTaskManagerProxyServer)

#endif // _mtsTaskManagerProxyServer_h

