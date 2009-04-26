/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsTaskInterfaceProxyClient.h 142 2009-03-11 23:02:34Z mjung5 $

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

#ifndef _mtsTaskInterfaceProxyClient_h
#define _mtsTaskInterfaceProxyClient_h

#include <cisstMultiTask/mtsTaskInterface.h>
#include <cisstMultiTask/mtsProxyBaseClient.h>
#include <cisstMultiTask/mtsTaskInterfaceProxy.h>

#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  TODO: add class summary here
*/

class CISST_EXPORT mtsTaskInterfaceProxyClient : public mtsProxyBaseClient<mtsTask> {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

protected:
    /*! Send thread set up. */
    class TaskInterfaceClientI;
    typedef IceUtil::Handle<TaskInterfaceClientI> TaskInterfaceClientIPtr;
    TaskInterfaceClientIPtr Sender;

    /*! TaskInterfaceServer proxy */
    mtsTaskInterfaceProxy::TaskInterfaceServerPrx TaskInterfaceServer;

public:
    mtsTaskInterfaceProxyClient(const std::string& propertyFileName, 
                                const std::string& propertyName) 
        : mtsProxyBaseClient(propertyFileName, propertyName)
    {}
    ~mtsTaskInterfaceProxyClient() {}

    /*! Create a proxy object and a send thread. */
    void CreateProxy() {
        TaskInterfaceServer = 
            mtsTaskInterfaceProxy::TaskInterfaceServerPrx::checkedCast(ProxyObject);
        if (!TaskInterfaceServer) {
            throw "Invalid proxy";
        }

        Sender = new TaskInterfaceClientI(IceCommunicator, Logger, TaskInterfaceServer, this);
    }

    /*! Entry point to run a proxy. */
    void Start(mtsTask * callingTask);

    /*! Start a send thread and wait for shutdown (blocking call). */
    void StartClient();

    /*! Thread runner */
    static void Runner(ThreadArguments<mtsTask> * arguments);

    /*! Clean up thread-related resources. */
    void OnThreadEnd();

    //-------------------------------------------------------------------------
    //  Definition by mtsTaskInterfaceProxy.ice
    //-------------------------------------------------------------------------
protected:
    class TaskInterfaceClientI : public mtsTaskInterfaceProxy::TaskInterfaceClient,
                               public IceUtil::Monitor<IceUtil::Mutex>
    {
    private:
        Ice::CommunicatorPtr Communicator;
        bool Runnable;
        
        IceUtil::ThreadPtr Sender;
        Ice::LoggerPtr Logger;
        mtsTaskInterfaceProxy::TaskInterfaceServerPrx Server;
        mtsTaskInterfaceProxyClient * TaskInterfaceClient;

    public:
        TaskInterfaceClientI(const Ice::CommunicatorPtr& communicator,                           
                           const Ice::LoggerPtr& logger,
                           const mtsTaskInterfaceProxy::TaskInterfaceServerPrx& server,
                           mtsTaskInterfaceProxyClient * TaskInterfaceClient);

        void Start();
        void Run();
        void Destroy();
    };
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsTaskInterfaceProxyClient)

#endif // _mtsTaskInterfaceProxyClient_h

