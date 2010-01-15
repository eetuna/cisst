/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsComponentInterfaceProxyServer.h 142 2009-03-11 23:02:34Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2010-01-12

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#ifndef _mtsComponentInterfaceProxyServer_h
#define _mtsComponentInterfaceProxyServer_h

#include <cisstMultiTask/mtsComponentProxy.h>
#include <cisstMultiTask/mtsComponentInterfaceProxy.h>
#include <cisstMultiTask/mtsProxyBaseServer.h>

#include <cisstMultiTask/mtsExport.h>

class CISST_EXPORT mtsComponentInterfaceProxyServer : 
    public mtsProxyBaseServer<mtsComponentProxy, mtsComponentInterfaceProxy::ComponentInterfaceClientPrx>
{
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

    /*! Typedef for client proxy type */
    typedef mtsComponentInterfaceProxy::ComponentInterfaceClientPrx ComponentInterfaceClientProxyType;

    /*! Typedef for base type */
    typedef mtsProxyBaseServer<mtsComponentProxy, ComponentInterfaceClientProxyType> BaseServerType;

public:
    mtsComponentInterfaceProxyServer(
        const std::string & adapterName, const std::string & endpointInfo, const std::string & communicatorID)
        : BaseServerType(adapterName, endpointInfo, communicatorID)
    {}

    ~mtsComponentInterfaceProxyServer();

    /*! Entry point to run a proxy. */
    void Start(mtsComponentProxy * owner);

    /*! Stop the proxy (clean up thread-related resources) */
    void Stop();

protected:
    /*! Definitions for send thread */
    class ComponentInterfaceServerI;
    typedef IceUtil::Handle<ComponentInterfaceServerI> ComponentInterfaceServerIPtr;
    ComponentInterfaceServerIPtr Sender;

    //-------------------------------------------------------------------------
    //  Proxy Implementation
    //-------------------------------------------------------------------------
    /*! Create a servant */
    Ice::ObjectPtr CreateServant() {
        Sender = new ComponentInterfaceServerI(IceCommunicator, IceLogger, this);
        return Sender;
    }
    
    /*! Start a send thread and wait for shutdown (this is a blocking method). */
    void StartServer();

    /*! Resource clean-up when a client disconnects or is disconnected.
        TODO: add session
        TODO: add resource clean up
        TODO: review/add safe termination  */
    void OnClose();

    /*! Thread runner */
    static void Runner(ThreadArguments<mtsComponentProxy> * arguments);

    //-------------------------------------------------------------------------
    //  Event Handlers (Client -> Server)
    //-------------------------------------------------------------------------
    void TestReceiveMessageFromClientToServer(const std::string & str) const;

    /*! When a new client connects, add it to the client management list. */
    void ReceiveAddClient(const ConnectionIDType & connectionID, ComponentInterfaceClientProxyType & clientProxy);

    //-------------------------------------------------------------------------
    //  Event Generators (Event Sender) : Client -> Server
    //-------------------------------------------------------------------------
public:
    void SendTestMessageFromServerToClient(const std::string & str);

    ////-------------------------------------------------------------------------
    ////  Methods to Process Events
    ////-------------------------------------------------------------------------
protected:
    ///*! Update the information on the newly connected task manager. */
    //bool ReceiveUpdateTaskManagerClient(const ConnectionIDType & connectionID,
    //                                    const ::mtsComponentInterfaceProxy::TaskList& localTaskInfo);

    ///*! Add a new provided interface. */
    //bool ReceiveAddProvidedInterface(
    //    const ConnectionIDType & connectionID,
    //    const mtsComponentInterfaceProxy::ProvidedInterfaceAccessInfo & providedInterfaceAccessInfo);

    ///*! Add a new required interface. */
    //bool ReceiveAddRequiredInterface(
    //    const ConnectionIDType & connectionID,
    //    const ::mtsComponentInterfaceProxy::RequiredInterfaceAccessInfo & requiredInterfaceAccessInfo);

    ///*! Check if the provided interface has been registered before. */
    //bool ReceiveIsRegisteredProvidedInterface(
    //    const ConnectionIDType & connectionID,
    //    const std::string & taskName, const std::string & providedInterfaceName);

    ///*! Get the information about the provided interface. */
    //bool ReceiveGetProvidedInterfaceAccessInfo(
    //    const ConnectionIDType & connectionID,
    //    const std::string & taskName, const std::string & providedInterfaceName,
    //    mtsComponentInterfaceProxy::ProvidedInterfaceAccessInfo & info);

    ///*! Inform the global task manager of the fact that connect() succeeded. */
    //void ReceiveNotifyInterfaceConnectionResult(
    //    const ConnectionIDType & connectionID,
    //    const bool isServerTask, const bool isSuccess,
    //    const std::string & userTaskName,     const std::string & requiredInterfaceName,
    //    const std::string & resourceTaskName, const std::string & providedInterfaceName);

    //-------------------------------------------------------------------------
    //  Definition by mtsComponentInterfaceProxy.ice
    //-------------------------------------------------------------------------
    class ComponentInterfaceServerI : 
        public mtsComponentInterfaceProxy::ComponentInterfaceServer,
        public IceUtil::Monitor<IceUtil::Mutex>
    {
    private:
        /*! Ice objects */
        Ice::CommunicatorPtr Communicator;
        IceUtil::ThreadPtr SenderThreadPtr;
        Ice::LoggerPtr Logger;

        // TODO: Do I really need this flag??? what about mtsProxyBaseCommon::Runnable???
        /*! True if ICE proxy is running */
        bool Runnable;

        /*! Network event handler */
        mtsComponentInterfaceProxyServer * ComponentInterfaceProxyServer;
        
    public:
        ComponentInterfaceServerI(
            const Ice::CommunicatorPtr& communicator, 
            const Ice::LoggerPtr& logger,
            mtsComponentInterfaceProxyServer * componentInterfaceProxyServer);

        void Start();
        void Run();
        void Stop();

        //---------------------------------------
        //  Event Handlers (Client -> Server)
        //---------------------------------------
        /*! Add a client proxy. Called when a proxy client connects to server proxy. */
        void AddClient(const Ice::Identity&, const Ice::Current&);

        /*! Shutdown this session; prepare shutdown for safe and clean termination. */
        void Shutdown(const ::Ice::Current&);

        //void UpdateTaskManager(const mtsComponentInterfaceProxy::TaskList&, const Ice::Current&);
        //bool AddProvidedInterface(const mtsComponentInterfaceProxy::ProvidedInterfaceAccessInfo&, const Ice::Current&);
        //bool AddRequiredInterface(const mtsComponentInterfaceProxy::RequiredInterfaceAccessInfo&, const Ice::Current&);
        //bool IsRegisteredProvidedInterface(const std::string&, const ::std::string&, const Ice::Current&) const;
        //bool GetProvidedInterfaceAccessInfo(const std::string&, const std::string&, mtsComponentInterfaceProxy::ProvidedInterfaceAccessInfo & info, const Ice::Current&) const;
        //void NotifyInterfaceConnectionResult(bool, bool, const ::std::string&, const ::std::string&, const ::std::string&, const ::std::string&, const Ice::Current&);

        void TestSendMessageFromClientToServer(const std::string & str, const ::Ice::Current & current);
    };
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsComponentInterfaceProxyServer)

#endif // _mtsComponentInterfaceProxyServer_h

