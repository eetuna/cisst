/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsTaskManagerProxyClient.h 142 2009-03-11 23:02:34Z mjung5 $

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

#ifndef _mtsTaskManagerProxyClient_h
#define _mtsTaskManagerProxyClient_h

#include <cisstMultiTask/mtsTaskManager.h>
#include <cisstMultiTask/mtsProxyBaseClient.h>
#include <cisstMultiTask/mtsTaskManagerProxy.h>

#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  TODO: add class summary here
*/

class CISST_EXPORT mtsTaskManagerProxyClient : public mtsProxyBaseClient<mtsTaskManager> {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

protected:
    /*! Send thread.
        We need a seperate send thread because the bi-directional communication is
        used between proxies. This is the major limitation of using bi-directional 
        communication. (Another approach is to use Glacier2.) */
    class TaskManagerClientI;
    typedef IceUtil::Handle<TaskManagerClientI> TaskManagerClientIPtr;
    TaskManagerClientIPtr Sender;

    /*! TaskManagerServer proxy */
    mtsTaskManagerProxy::TaskManagerServerPrx TaskManagerServer;

public:
    mtsTaskManagerProxyClient(const std::string& propertyFileName, 
                              const std::string& propertyName)
        : mtsProxyBaseClient(propertyFileName, propertyName)
    {}
    ~mtsTaskManagerProxyClient() {}

    /*! Create a proxy object and a send thread. */
    void CreateProxy() {
        TaskManagerServer = 
            mtsTaskManagerProxy::TaskManagerServerPrx::checkedCast(ProxyObject);
        if (!TaskManagerServer) {
            throw "Invalid proxy";
        }

        Sender = new TaskManagerClientI(IceCommunicator, Logger, TaskManagerServer, this);
    }

    /*! Entry point to run a proxy. */
    void Start(mtsTaskManager * callingTaskManager);

    /*! Start a send thread and wait for shutdown (blocking call). */
    void StartClient();

    /*! Thread runner */
    static void Runner(ThreadArguments<mtsTaskManager> * arguments);

    /*! Clean up thread-related resources. */
    void OnThreadEnd();

    //-------------------------------------------------------------------------
    //  
    //-------------------------------------------------------------------------
    /*! Add a new provided interface. */
    bool AddProvidedInterface(
        const std::string & newProvidedInterfaceName,
        const std::string & adapterName,
        const std::string & endpointInfo,
        const std::string & communicatorID);

    //-------------------------------------------------------------------------
    //  Definition by mtsTaskManagerProxy.ice
    //-------------------------------------------------------------------------
protected:
    class TaskManagerClientI : public mtsTaskManagerProxy::TaskManagerClient,
                               public IceUtil::Monitor<IceUtil::Mutex>
    {
    private:
        Ice::CommunicatorPtr Communicator;
        bool Runnable;
        //std::set<mtsTaskManagerProxy::TaskManagerServerPrx> _servers;
        
        IceUtil::ThreadPtr Sender;
        Ice::LoggerPtr Logger;
        mtsTaskManagerProxy::TaskManagerServerPrx Server;
        mtsTaskManagerProxyClient * TaskManagerClient;

    public:
        TaskManagerClientI(const Ice::CommunicatorPtr& communicator,                           
                           const Ice::LoggerPtr& logger,
                           const mtsTaskManagerProxy::TaskManagerServerPrx& server,
                           mtsTaskManagerProxyClient * taskManagerClient);

        void Start();
        void Run();
        void Destroy();

        virtual void ReceiveData(::Ice::Int num, const ::Ice::Current&);
    };
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsTaskManagerProxyClient)

#endif // _mtsTaskManagerProxyClient_h

