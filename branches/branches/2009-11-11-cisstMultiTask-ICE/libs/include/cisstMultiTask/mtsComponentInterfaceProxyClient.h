/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsComponentInterfaceProxyClient.h 142 2009-03-11 23:02:34Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2010-01-13

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#ifndef _mtsComponentInterfaceProxyClient_h
#define _mtsComponentInterfaceProxyClient_h

#include <cisstMultiTask/mtsComponentProxy.h>
#include <cisstMultiTask/mtsComponentInterfaceProxy.h>
#include <cisstMultiTask/mtsProxyBaseClient.h>

#include <cisstMultiTask/mtsExport.h>

// TODO: ADD the following line in the forward declaration.h (???)
//class mtsProxySerializer;

class CISST_EXPORT mtsComponentInterfaceProxyClient : public mtsProxyBaseClient<mtsComponentProxy> {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    mtsComponentInterfaceProxyClient(
        const std::string & serverEndpointInfo, const std::string & communicatorID);
    ~mtsComponentInterfaceProxyClient();

    /*! Entry point to run a proxy. */
    void Start(mtsComponentProxy * proxyOwner);

    /*! Stop the proxy (clean up thread-related resources) */
    void Stop(void);

protected:
    /*! Typedef for base type. */
    typedef mtsProxyBaseClient<mtsComponentProxy> BaseType;

    /*! Typedef for connected server proxy. */
    typedef mtsComponentInterfaceProxy::ComponentInterfaceServerPrx ComponentInterfaceServerProxyType;
    ComponentInterfaceServerProxyType ComponentInterfaceServerProxy;

    /*! Definitions for send thread */
    class ComponentInterfaceClientI;
    typedef IceUtil::Handle<ComponentInterfaceClientI> ComponentInterfaceClientIPtr;
    ComponentInterfaceClientIPtr Sender;

    /*! Typedef for per-command proxy serializer. */
    //typedef std::map<CommandIDType, mtsProxySerializer *> PerCommandSerializerMapType;

    /*! Per-command proxy serializer container. */
    //PerCommandSerializerMapType PerCommandSerializerMap;
    
    //-------------------------------------------------------------------------
    //  Proxy Implementation
    //-------------------------------------------------------------------------
    /*! Create a proxy object and a send thread. */
    void CreateProxy() {
        ComponentInterfaceServerProxy = 
            mtsComponentInterfaceProxy::ComponentInterfaceServerPrx::checkedCast(ProxyObject);
        if (!ComponentInterfaceServerProxy) {
            throw "Invalid proxy";
        }

        Sender = new ComponentInterfaceClientI(IceCommunicator, IceLogger, ComponentInterfaceServerProxy, this);
    }
    
    /*! Start a send thread and wait for shutdown (blocking call). */
    void StartClient();

    /*! Resource clean-up when a client disconnects or is disconnected.
        TODO: add session
        TODO: add resource clean up
        TODO: review/add safe termination  */
    void OnClose();

    /*! Thread runner */
    static void Runner(ThreadArguments<mtsComponentProxy> * arguments);

    ////-------------------------------------------------------------------------
    ////  Method to register per-command serializer
    ////-------------------------------------------------------------------------
    //bool AddPerCommandSerializer(
    //    const CommandIDType commandId, mtsProxySerializer * argumentSerializer);

    ////-------------------------------------------------------------------------
    ////  Methods to Receive and Process Events (Server -> Client)
    ////-------------------------------------------------------------------------
    //void ReceiveExecuteEventVoid(const CommandIDType commandId);
    //void ReceiveExecuteEventWriteSerialized(const CommandIDType commandId, const std::string argument);

    ////-------------------------------------------------------------------------
    ////  Methods to Send Events (Client -> Server)
    ////-------------------------------------------------------------------------
    //bool SendGetProvidedInterfaceInfo(
    //    const std::string & providedInterfaceName,
    //    mtsDeviceInterfaceProxy::ProvidedInterfaceInfo & providedInterfaceInfo);

    //bool SendCreateClientProxies(
    //    const std::string & userTaskName, const std::string & requiredInterfaceName,
    //    const std::string & resourceTaskName, const std::string & providedInterfaceName);

    //bool SendConnectServerSide(
    //    const std::string & userTaskName, const std::string & requiredInterfaceName,
    //    const std::string & resourceTaskName, const std::string & providedInterfaceName);

    //bool SendUpdateEventHandlerId(
    //    const std::string & clientTaskProxyName,
    //    const mtsDeviceInterfaceProxy::ListsOfEventGeneratorsRegistered & eventGeneratorProxies);

    //void SendGetCommandId(
    //    const std::string & clientTaskProxyName,
    //    mtsDeviceInterfaceProxy::FunctionProxySet & functionProxies);

    //void SendExecuteCommandVoid(const CommandIDType commandId) const;
    //void SendExecuteCommandWriteSerialized(const CommandIDType commandId, const mtsGenericObject & argument);
    //void SendExecuteCommandReadSerialized(const CommandIDType commandId, mtsGenericObject & argument);
    //void SendExecuteCommandQualifiedReadSerialized(
    //    const CommandIDType commandId, const mtsGenericObject & argument1, mtsGenericObject & argument2);

    //-------------------------------------------------------------------------
    //  Definition by mtsDeviceInterfaceProxy.ice
    //-------------------------------------------------------------------------
protected:
    class ComponentInterfaceClientI : 
        public mtsComponentInterfaceProxy::ComponentInterfaceClient,
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

        /*! Network event processor */
        mtsComponentInterfaceProxyClient * ComponentInterfaceProxyClient;

        /*! Connected server proxy */
        mtsComponentInterfaceProxy::ComponentInterfaceServerPrx Server;

    public:
        ComponentInterfaceClientI(
            const Ice::CommunicatorPtr& communicator,                           
            const Ice::LoggerPtr& logger,
            const mtsComponentInterfaceProxy::ComponentInterfaceServerPrx& server,
            mtsComponentInterfaceProxyClient * ComponentInterfaceClient);

        void Start();
        void Run();
        void Stop();

        // Server -> Client
        //void ExecuteEventVoid(IceCommandIDType, const ::Ice::Current&);
        //void ExecuteEventWriteSerialized(IceCommandIDType, const ::std::string&, const ::Ice::Current&);
    };
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsComponentInterfaceProxyClient)

#endif // _mtsComponentInterfaceProxyClient_h
