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

//#include <cisstMultiTask/mtsComponentProxy.h>
#include <cisstMultiTask/mtsComponentInterfaceProxy.h>
#include <cisstMultiTask/mtsProxyBaseClient.h>

#include <cisstMultiTask/mtsExport.h>

// TODO: ADD the following line in the forward declaration.h (???)
//class mtsProxySerializer;

class mtsComponentProxy;

class CISST_EXPORT mtsComponentInterfaceProxyClient : public mtsProxyBaseClient<mtsComponentProxy> {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    mtsComponentInterfaceProxyClient(
        const std::string & serverEndpointInfo, const std::string & communicatorID,
        const unsigned int providedInterfaceProxyInstanceId);
    ~mtsComponentInterfaceProxyClient();

    /*! Entry point to run a proxy. */
    bool Start(mtsComponentProxy * proxyOwner);

    /*! Stop the proxy (clean up thread-related resources) */
    void Stop(void);

protected:
    /*! Typedef for base type. */
    typedef mtsProxyBaseClient<mtsComponentProxy> BaseClientType;

    /*! Typedef for connected server proxy. */
    typedef mtsComponentInterfaceProxy::ComponentInterfaceServerPrx ComponentInterfaceServerProxyType;
    ComponentInterfaceServerProxyType ComponentInterfaceServerProxy;

    /*! Definitions for send thread */
    class ComponentInterfaceClientI;
    typedef IceUtil::Handle<ComponentInterfaceClientI> ComponentInterfaceClientIPtr;
    ComponentInterfaceClientIPtr Sender;

    /*! Provided interface proxy instance id that this network proxy client is
        connected to. When a network proxy server sends a message to a network
        proxy client, this information is used to determine with which the server
        should communicate. (Note that a provided interface proxy should be able
        to handle multiple required interface proxies) */
    const unsigned int ProvidedInterfaceProxyInstanceId;

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
            throw "mtsComponentInterfaceProxyClient:: Invalid proxy";
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

    /*! Typedef for per-event argument serializer */
    typedef std::map<CommandIDType, mtsProxySerializer *> PerEventSerializerMapType;
    PerEventSerializerMapType PerEventSerializerMap;

    //-------------------------------------------------------------------------
    //  Event Handlers : Server -> Client
    //-------------------------------------------------------------------------
    void ReceiveTestMessageFromServerToClient(const std::string & str) const;

    /*! Fetch pointers of function proxies from a required interface proxy at 
        server side */
    bool ReceiveFetchFunctionProxyPointers(const std::string & requiredInterfaceName,
        mtsComponentInterfaceProxy::FunctionProxyPointerSet & functionProxyPointers) const;

    /*! Execute commands by callling function proxy objects. */
    void ReceiveExecuteCommandVoid(const CommandIDType commandID);

    void ReceiveExecuteCommandWriteSerialized(const CommandIDType commandID, const std::string & serializedArgument);

    void ReceiveExecuteCommandReadSerialized(const CommandIDType commandID, std::string & serializedArgument);

    void ReceiveExecuteCommandQualifiedReadSerialized(const CommandIDType commandID, const std::string & serializedArgumentIn, std::string & serializedArgumentOut);

    //-------------------------------------------------------------------------
    //  Event Generators (Event Sender) : Client -> Server
    //-------------------------------------------------------------------------
public:
    /*! Test method */
    void SendTestMessageFromClientToServer(const std::string & str) const;

    /*! Register per-command (de)serializer */
    bool RegisterPerEventSerializer(const CommandIDType commandID, mtsProxySerializer * serializer);

    /*! Fetch pointers of event generator proxies from a provided interface 
        proxy at server side */
    bool SendFetchEventGeneratorProxyPointers(
        const std::string & requiredInterfaceName, const std::string & providedInterfaceName,
        mtsComponentInterfaceProxy::EventGeneratorProxyPointerSet & eventGeneratorProxyPointers);

    bool SendExecuteEventVoid(const CommandIDType commandID);

    bool SendExecuteEventWriteSerialized(const CommandIDType commandID, const mtsGenericObject & argument);


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
        Ice::LoggerPtr IceLogger;

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

        //-------------------------------------------------
        //  Network Event handlers (Server -> Client)
        //-------------------------------------------------
        void TestMessageFromServerToClient(const std::string & str, const ::Ice::Current & current);

        bool FetchFunctionProxyPointers(const std::string &, mtsComponentInterfaceProxy::FunctionProxyPointerSet &, const ::Ice::Current & current) const;

        void ExecuteCommandVoid(::Ice::Long, const ::Ice::Current&);

        void ExecuteCommandWriteSerialized(::Ice::Long, const ::std::string&, const ::Ice::Current&);

        void ExecuteCommandReadSerialized(::Ice::Long, ::std::string&, const ::Ice::Current&);

        void ExecuteCommandQualifiedReadSerialized(::Ice::Long, const ::std::string&, ::std::string&, const ::Ice::Current&);
    };
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsComponentInterfaceProxyClient)

#endif // _mtsComponentInterfaceProxyClient_h
