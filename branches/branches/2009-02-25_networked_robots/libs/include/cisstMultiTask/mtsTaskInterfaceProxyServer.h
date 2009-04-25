/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsTaskInterfaceProxyServer.h 142 2009-03-11 23:02:34Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-04-24

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#ifndef _mtsTaskInterfaceProxyServer_h
#define _mtsTaskInterfaceProxyServer_h

#include <cisstMultiTask/mtsTaskInterface.h>
#include <cisstMultiTask/mtsProxyBaseServer.h>
#include <cisstMultiTask/mtsTaskInterfaceProxy.h>

#include <cisstMultiTask/mtsExport.h>

#include <string>

/*!
  \ingroup cisstMultiTask

  TODO: add class summary here
*/
class CISST_EXPORT mtsTaskInterfaceProxyServer : public mtsProxyBaseServer<mtsTask> {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

    friend class TaskInterfaceServerI;

protected:
    //--------------------------- Protected member data ---------------------//
    /*! Definitions for send thread */
    class TaskInterfaceServerI;
    typedef IceUtil::Handle<TaskInterfaceServerI> TaskInterfaceServerIPtr;
    TaskInterfaceServerIPtr Sender;
    
    //-------------------------- Protected methods --------------------------//
    /*! Resource clean-up */
    void OnClose();

    //------------------- Methods for proxy implementation ------------------//
    /*! Create a servant which serves TaskManager clients. */
    Ice::ObjectPtr CreateServant() {
        Sender = new TaskInterfaceServerI(IceCommunicator, Logger, this);
        return Sender;
    }
    
    /*! Entry point to run a proxy. */
    void Start(mtsTask * callingTask);

    /*! Start a send thread and wait for shutdown (blocking call). */
    void StartServer();

    /*! Thread runner */
    static void Runner(ThreadArguments<mtsTask> * arguments);

    /*! Clean up thread-related resources. */
    void OnThreadEnd();
    
public:
    mtsTaskInterfaceProxyServer(const std::string& propertyFileName, 
                                const std::string& propertyName) 
        : mtsProxyBaseServer(propertyFileName, propertyName)
    {}
    ~mtsTaskInterfaceProxyServer();
    
    //----------------------------- Proxy Support ---------------------------//
    /*! Update the information of all tasks. */
    //void AddTaskManager(const ::mtsTaskManagerProxy::TaskList& localTaskInfo);

    //-------------------------------------------------------------------------
    //  Definition by mtsTaskInterfaceProxy.ice
    //-------------------------------------------------------------------------
protected:
    class TaskInterfaceServerI : public mtsTaskInterfaceProxy::TaskInterfaceServer,
                                 public IceUtil::Monitor<IceUtil::Mutex> 
    {
    private:
        Ice::CommunicatorPtr Communicator;
        bool Runnable;
        std::set<mtsTaskInterfaceProxy::TaskInterfaceClientPrx> _clients;
        IceUtil::ThreadPtr Sender;
        Ice::LoggerPtr Logger;
        mtsTaskInterfaceProxyServer * TaskInterfaceServer;

    public:
        TaskInterfaceServerI(const Ice::CommunicatorPtr& communicator, 
                             const Ice::LoggerPtr& logger,
                             mtsTaskInterfaceProxyServer * taskInterfaceServer);

        void Start();
        void Run();
        void Destroy();

        void AddClient(const ::Ice::Identity&, const ::Ice::Current&);
    };

    //------------------ Methods for global task manager --------------------//
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsTaskInterfaceProxyServer)

#endif // _mtsTaskInterfaceProxyServer_h

