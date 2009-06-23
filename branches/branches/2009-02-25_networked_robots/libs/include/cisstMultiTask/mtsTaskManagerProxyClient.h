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

#include <cisstMultiTask/mtsProxyBaseClient.h>
#include <cisstMultiTask/mtsTaskManagerProxy.h>

#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  TODO: add class summary here
*/

class mtsTaskManager;

class CISST_EXPORT mtsTaskManagerProxyClient : public mtsProxyBaseClient<mtsTaskManager> {

    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);    

public:
    mtsTaskManagerProxyClient(const std::string & propertyFileName, 
                              const std::string & propertyName);
    ~mtsTaskManagerProxyClient();

protected:
    typedef mtsProxyBaseClient<mtsTaskManager> BaseType;

    /*! Send thread.
        We need a seperate send thread because the bi-directional communication is
        used between proxies. This is the major limitation of using bi-directional 
        communication. (Another approach is to use Glacier2.) */
    class TaskManagerClientI;
    typedef IceUtil::Handle<TaskManagerClientI> TaskManagerClientIPtr;
    TaskManagerClientIPtr Sender;

    /*! TaskManagerServer proxy */
    mtsTaskManagerProxy::TaskManagerServerPrx TaskManagerServer;

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
    //  Send Methods
    //-------------------------------------------------------------------------
public:
    bool SendAddProvidedInterface(const std::string & newProvidedInterfaceName,
                                  const std::string & adapterName,
                                  const std::string & endpointInfo,
                                  const std::string & communicatorID,
                                  const std::string & taskName);

    bool SendAddRequiredInterface(const std::string & newRequiredInterfaceName,
                                  const std::string & taskName);

    bool SendIsRegisteredProvidedInterface(const std::string & taskName, 
                                           const std::string & providedInterfaceName) const;

    bool SendGetProvidedInterfaceInfo(const std::string & taskName,
                                      const std::string & providedInterfaceName,
                                      mtsTaskManagerProxy::ProvidedInterfaceInfo & info) const;

    //void SendNotifyInterfaceConnectionResult(
    //    const bool isServerTask, const bool isSuccess,
    //    const std::string & userTaskName,     const std::string & requiredInterfaceName,
    //    const std::string & resourceTaskName, const std::string & providedInterfaceName);

    //-------------------------------------------------------------------------
    //  Methods to Receive and Process Events
    //-------------------------------------------------------------------------

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

        void ReceiveData(::Ice::Int num, const ::Ice::Current&);
    };
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsTaskManagerProxyClient)

#endif // _mtsTaskManagerProxyClient_h

