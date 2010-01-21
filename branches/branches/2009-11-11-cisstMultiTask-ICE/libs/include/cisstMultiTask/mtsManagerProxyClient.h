/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerProxyClient.h 142 2009-03-11 23:02:34Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2010-01-20

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#ifndef _mtsManagerProxyClient_h
#define _mtsManagerProxyClient_h

#include <cisstMultiTask/mtsManagerLocal.h>
#include <cisstMultiTask/mtsManagerProxy.h>
#include <cisstMultiTask/mtsProxyBaseClient.h>

#include <cisstMultiTask/mtsExport.h>

class CISST_EXPORT mtsManagerProxyClient : public mtsProxyBaseClient<mtsManagerLocal> {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    mtsManagerProxyClient(const std::string & serverEndpointInfo, const std::string & communicatorID);
    ~mtsManagerProxyClient();

    /*! Entry point to run a proxy. */
    bool Start(mtsManagerLocal * proxyOwner);

    /*! Stop the proxy (clean up thread-related resources) */
    void Stop(void);

protected:
    /*! Typedef for base type. */
    typedef mtsProxyBaseClient<mtsManagerLocal> BaseClientType;

    /*! Typedef for connected server proxy. */
    typedef mtsManagerProxy::ManagerServerPrx ManagerServerProxyType;
    ManagerServerProxyType ManagerServerProxy;

    /*! Definitions for send thread */
    class ManagerClientI;
    typedef IceUtil::Handle<ManagerClientI> ManagerClientIPtr;
    ManagerClientIPtr Sender;

    //-------------------------------------------------------------------------
    //  Proxy Implementation
    //-------------------------------------------------------------------------
    /*! Create a proxy object and a send thread. */
    void CreateProxy() {
        ManagerServerProxy = 
            mtsManagerProxy::ManagerServerPrx::checkedCast(ProxyObject);
        if (!ManagerServerProxy) {
            throw "mtsManagerProxyClient: Invalid proxy";
        }

        Sender = new ManagerClientI(IceCommunicator, IceLogger, ManagerServerProxy, this);
    }
    
    /*! Start a send thread and wait for shutdown (blocking call). */
    void StartClient();

    /*! Resource clean-up when a client disconnects or is disconnected.
        TODO: add session
        TODO: add resource clean up
        TODO: review/add safe termination  */
    void OnClose();

    /*! Thread runner */
    static void Runner(ThreadArguments<mtsManagerLocal> * arguments);

    //-------------------------------------------------------------------------
    //  Event Handlers : Server -> Client
    //-------------------------------------------------------------------------
    void TestReceiveMessageFromServerToClient(const std::string & str) const;

    //-------------------------------------------------------------------------
    //  Event Generators (Event Sender) : Client -> Server
    //-------------------------------------------------------------------------
public:
    void SendTestMessageFromClientToServer(const std::string & str) const;

    //-------------------------------------------------------------------------
    //  Definition by mtsDeviceInterfaceProxy.ice
    //-------------------------------------------------------------------------
protected:
    class ManagerClientI : 
        public mtsManagerProxy::ManagerClient,
        public IceUtil::Monitor<IceUtil::Mutex>
    {
    private:
        /*! Ice objects */
        Ice::CommunicatorPtr Communicator;
        IceUtil::ThreadPtr SenderThreadPtr;
        Ice::LoggerPtr IceLogger;

        // TODO: Do I really need this flag??? what about mtsProxyBaseCommon::Runnable???
        /*! True if ICE proxy is running */
        bool Runnable;

        /*! Network event processor */
        mtsManagerProxyClient * ManagerProxyClient;

        /*! Connected server proxy */
        mtsManagerProxy::ManagerServerPrx Server;

    public:
        ManagerClientI(
            const Ice::CommunicatorPtr& communicator,                           
            const Ice::LoggerPtr& logger,
            const mtsManagerProxy::ManagerServerPrx& server,
            mtsManagerProxyClient * ManagerClient);

        void Start();
        void Run();
        void Stop();

        //-------------------------------------------------
        //  Network Event handlers (Server -> Client)
        //-------------------------------------------------
        void TestSendMessageFromServerToClient(const std::string & str, const ::Ice::Current & current);

    };
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsManagerProxyClient)

#endif // _mtsManagerProxyClient_h
