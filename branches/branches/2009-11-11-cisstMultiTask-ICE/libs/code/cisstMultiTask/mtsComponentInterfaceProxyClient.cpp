/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsComponentInterfaceProxyClient.cpp 145 2009-03-18 23:32:40Z mjung5 $

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

#include <cisstMultiTask/mtsComponentProxy.h>
#include <cisstMultiTask/mtsComponentInterfaceProxyClient.h>
#include <cisstMultiTask/mtsFunctionVoid.h>
#include <cisstMultiTask/mtsFunctionReadOrWriteProxy.h>
#include <cisstMultiTask/mtsFunctionQualifiedReadOrWriteProxy.h>

#include <cisstOSAbstraction/osaSleep.h>

CMN_IMPLEMENT_SERVICES(mtsComponentInterfaceProxyClient);

//#define ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG

//-----------------------------------------------------------------------------
//  Constructor, Destructor, Initializer
//-----------------------------------------------------------------------------
mtsComponentInterfaceProxyClient::mtsComponentInterfaceProxyClient(
    const std::string & serverEndpointInfo, const std::string & communicatorID,
    const unsigned int providedInterfaceProxyInstanceId)
    : BaseClientType(serverEndpointInfo, communicatorID),
      ProvidedInterfaceProxyInstanceId(providedInterfaceProxyInstanceId)
{
}

mtsComponentInterfaceProxyClient::~mtsComponentInterfaceProxyClient()
{
}

//-----------------------------------------------------------------------------
//  Proxy Start-up
//-----------------------------------------------------------------------------
bool mtsComponentInterfaceProxyClient::Start(mtsComponentProxy * proxyOwner)
{
    // Initialize Ice object.
    IceInitialize();

    if (!InitSuccessFlag) {
        LogError(mtsComponentInterfaceProxyClient, "ICE proxy initialization failed");
        return false;
    }

    // Client configuration for bidirectional communication
    Ice::ObjectAdapterPtr adapter = IceCommunicator->createObjectAdapter("");
    Ice::Identity ident;
    ident.name = GetGUID();
    ident.category = "";

    mtsComponentInterfaceProxy::ComponentInterfaceClientPtr client = 
        new ComponentInterfaceClientI(IceCommunicator, IceLogger, ComponentInterfaceServerProxy, this);
    adapter->add(client, ident);
    adapter->activate();
    ComponentInterfaceServerProxy->ice_getConnection()->setAdapter(adapter);

    //
    // TODO: we can use a provided interface proxy instance id instead of an implicit context key.
    //
    // Set an implicit context (per proxy context)
    // (see http://www.zeroc.com/doc/Ice-3.3.1/manual/Adv_server.33.12.html)
    IceCommunicator->getImplicitContext()->put(ConnectionIDKey, IceCommunicator->identityToString(ident));

    // Set the owner and name of this proxy object
    std::string thisProcessName = "On";
    mtsManagerLocal * managerLocal = mtsManagerLocal::GetInstance();
    thisProcessName += managerLocal->GetProcessName();

    SetProxyOwner(proxyOwner, thisProcessName);

    // Connect to server proxy through adding this ICE proxy to server proxy
    if (!ComponentInterfaceServerProxy->AddClient(GetProxyName(), (::Ice::Int) ProvidedInterfaceProxyInstanceId, ident)) {
        LogError(mtsComponentInterfaceProxyClient, "AddClient() failed: duplicate proxy name or identity");
        return false;
    }

    // Create a worker thread here but is not running yet.
    //ThreadArgumentsInfo.ProxyOwner = proxyOwner;
    ThreadArgumentsInfo.Proxy = this;        
    ThreadArgumentsInfo.Runner = mtsComponentInterfaceProxyClient::Runner;

    WorkerThread.Create<ProxyWorker<mtsComponentProxy>, ThreadArguments<mtsComponentProxy>*>(
        &ProxyWorkerInfo, &ProxyWorker<mtsComponentProxy>::Run, &ThreadArgumentsInfo, 
        // Set the name of this thread as CIPC which means Component 
        // Interface Proxy Client. Such a very short naming rule is
        // because sometimes there is a limitation of the total number 
        // of characters as a thread name on some systems (e.g. LINUX RTAI).
        "CIPC");

    return true;
}

void mtsComponentInterfaceProxyClient::StartClient()
{
    Sender->Start();

    // This is a blocking call that should be run in a different thread.
    IceCommunicator->waitForShutdown();
}

void mtsComponentInterfaceProxyClient::Runner(ThreadArguments<mtsComponentProxy> * arguments)
{
    mtsComponentInterfaceProxyClient * ProxyClient = 
        dynamic_cast<mtsComponentInterfaceProxyClient*>(arguments->Proxy);
    if (!ProxyClient) {
        CMN_LOG_RUN_ERROR << "mtsComponentInterfaceProxyClient: Failed to create a proxy client." << std::endl;
        return;
    }

    ProxyClient->GetLogger()->trace("mtsComponentInterfaceProxyClient", "Proxy client starts.....");

    try {
        // TODO: By this call, it is 'assumed' that a client proxy is successfully
        // connected to a server proxy.
        // If I can find better way to detect successful connection establishment
        // between a client and a server, this should be updated.
        ProxyClient->SetAsActiveProxy();

        ProxyClient->StartClient();        
    } catch (const Ice::Exception& e) {
        std::string error("mtsComponentInterfaceProxyClient: ");
        error += e.what();
        ProxyClient->GetLogger()->error(error);
    } catch (const char * msg) {
        std::string error("mtsComponentInterfaceProxyClient: ");
        error += msg;
        ProxyClient->GetLogger()->error(error);
    }

    ProxyClient->GetLogger()->trace("mtsComponentInterfaceProxyClient", "Proxy client terminates.....");

    ProxyClient->Stop();
}

void mtsComponentInterfaceProxyClient::Stop()
{
    LogPrint(mtsComponentInterfaceProxyClient, "ComponentInterfaceProxy client ends.");

    // Let a server disconnect this client safely.
    // TODO: gcc says this doesn't exist???
    ComponentInterfaceServerProxy->Shutdown();

    ShutdownSession();
    
    BaseClientType::Stop();
    
    Sender->Stop();
}

/*
void mtsComponentInterfaceProxyClient::OnEnd()
{
    ComponentInterfaceProxyClientLogger("ComponentInterfaceProxy client ends.");

    // Let a server disconnect this client safely.
    // gcc says this doesn't exist
    ComponentInterfaceServerProxy->Shutdown();

    ShutdownSession();
    
    BaseType::OnEnd();
    
    Sender->Stop();
}

//-------------------------------------------------------------------------
//  Method to register per-command serializer
//-------------------------------------------------------------------------
bool mtsComponentInterfaceProxyClient::AddPerCommandSerializer(
    const CommandIDType commandId, mtsProxySerializer * argumentSerializer)
{
    CMN_ASSERT(argumentSerializer);

    PerCommandSerializerMapType::const_iterator it = PerCommandSerializerMap.find(commandId);
    if (it != PerCommandSerializerMap.end()) {
        CMN_LOG_RUN_ERROR << "mtsComponentInterfaceProxyClient: CommandId already exists." << std::endl;
        return false;
    }

    PerCommandSerializerMap[commandId] = argumentSerializer;

    return true;
}
*/

//-------------------------------------------------------------------------
//  Event Handlers (Server -> Client)
//-------------------------------------------------------------------------
void mtsComponentInterfaceProxyClient::ReceiveTestMessageFromServerToClient(const std::string & str) const
{
    std::cout << "Client received (Server -> Client): " << str << std::endl;
}

bool mtsComponentInterfaceProxyClient::ReceiveFetchFunctionProxyPointers(
    const std::string & requiredInterfaceName, mtsComponentInterfaceProxy::FunctionProxyPointerSet & functionProxyPointers) const
{
    // Get proxy owner object (of type mtsComponentProxy)
    mtsComponentProxy * proxyOwner = this->ProxyOwner;
    if (!proxyOwner) {
        LogError(mtsComponentInterfaceProxyClient, "invalid proxy owner");
        return false;
    }

    return proxyOwner->GetFunctionProxyPointers(requiredInterfaceName, functionProxyPointers);
}

void mtsComponentInterfaceProxyClient::ReceiveExecuteCommandVoid(const CommandIDType commandID)
{
    mtsFunctionVoid * functionVoid = reinterpret_cast<mtsFunctionVoid *>(commandID);
    if (!functionVoid) {
        LogError(mtsComponentInterfaceProxyClient, "invalid function void proxy id: " << commandID);
        return;
    }

    (*functionVoid)();
}

void mtsComponentInterfaceProxyClient::ReceiveExecuteCommandWriteSerialized(const CommandIDType commandID, const std::string & serializedArgument)
{
    mtsFunctionWriteProxy * functionWriteProxy = reinterpret_cast<mtsFunctionWriteProxy*>(commandID);
    if (!functionWriteProxy) {
        LogError(mtsComponentInterfaceProxyClient, "invalid function write proxy id: " << commandID);
        return;
    }

#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyClient, "ExecuteCommandWriteSerialized: " << serializedArgument.size() << " bytes received");
#endif

    // Get per-command deserializer
    mtsProxySerializer * deSerializer = functionWriteProxy->GetSerializer();
    
    mtsGenericObject * argument = deSerializer->DeSerialize(serializedArgument);
    if (!argument) {
        LogError(mtsComponentInterfaceProxyClient, "deserialization failed");
        return;
    }

    (*functionWriteProxy)(*argument);
}

void mtsComponentInterfaceProxyClient::ReceiveExecuteCommandReadSerialized(const CommandIDType commandID, std::string & serializedArgument)
{
}

void mtsComponentInterfaceProxyClient::ReceiveExecuteCommandQualifiedReadSerialized(const CommandIDType commandID, const std::string & serializedArgumentIn, std::string & serializedArgumentOut)
{
}

/*
void mtsComponentInterfaceProxyClient::ReceiveExecuteEventVoid(const CommandIDType commandId)
{
    mtsMulticastCommandVoid * eventVoidGeneratorProxy = 
        reinterpret_cast<mtsMulticastCommandVoid*>(commandId);
    CMN_ASSERT(eventVoidGeneratorProxy);

    eventVoidGeneratorProxy->Execute();
}

void mtsComponentInterfaceProxyClient::ReceiveExecuteEventWriteSerialized(
    const CommandIDType commandId, const std::string argument)
{
    static char buf[1024];
    sprintf(buf, "ReceiveExecuteEventWriteSerialized: %lu bytes received", argument.size());
    IceLogger->trace("TIClient", buf);

    mtsMulticastCommandWriteProxy * eventWriteGeneratorProxy = 
        reinterpret_cast<mtsMulticastCommandWriteProxy*>(commandId);
    CMN_ASSERT(eventWriteGeneratorProxy);

    // Get a per-command serializer.
    mtsProxySerializer * deserializer = eventWriteGeneratorProxy->GetSerializer();
        
    mtsGenericObject * serializedArgument = deserializer->DeSerialize(argument);
    CMN_ASSERT(serializedArgument);

    eventWriteGeneratorProxy->Execute(*serializedArgument);
}
*/

//-------------------------------------------------------------------------
//  Event Generators (Event Sender) : Client -> Server
//-------------------------------------------------------------------------
void mtsComponentInterfaceProxyClient::SendTestMessageFromClientToServer(const std::string & str) const
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyClient, ">>>>> SEND: MessageFromClientToServer");
#endif

    ComponentInterfaceServerProxy->TestMessageFromClientToServer(str);
}

bool mtsComponentInterfaceProxyClient::SendFetchEventGeneratorProxyPointers(
    const std::string & clientComponentName, const std::string & requiredInterfaceName,
    mtsComponentInterfaceProxy::ListsOfEventGeneratorsRegistered & eventGeneratorProxyPointers)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyClient, ">>>>> SEND: FetchEventGeneratorProxyPointers");
#endif

    return ComponentInterfaceServerProxy->FetchEventGeneratorProxyPointers(
        clientComponentName, requiredInterfaceName, eventGeneratorProxyPointers);
}

/*
void mtsComponentInterfaceProxyClient::SendExecuteCommandWriteSerialized(
    const CommandIDType commandId, const mtsGenericObject & argument)
{
    if (!IsValidSession) return;

    //Logger->trace("TIClient", ">>>>> SEND: SendExecuteCommandWriteSerialized");

    // Get a per-command serializer.
    mtsProxySerializer * serializer = PerCommandSerializerMap[commandId];
    if (!serializer) {
        CMN_LOG_RUN_ERROR << "mtsComponentInterfaceProxyClient: cannot find serializer (commandWrite)." << std::endl;
        return;
    }

    // Serialize the argument passed.
    std::string serializedArgument;
    serializer->Serialize(argument, serializedArgument);
    if (serializedArgument.size() == 0) {
        CMN_LOG_RUN_ERROR << "mtsComponentInterfaceProxyClient: serialization failure (commandWrite): " 
            << argument.ToString() << std::endl;
        return;
    }
    
    ComponentInterfaceServerProxy->ExecuteCommandWriteSerialized(commandId, serializedArgument);
}

void mtsComponentInterfaceProxyClient::SendExecuteCommandReadSerialized(
    const CommandIDType commandId, mtsGenericObject & argument)
{
    if (!IsValidSession) return;

    //Logger->trace("TIClient", ">>>>> SEND: SendExecuteCommandReadSerialized");

    // Placeholder for an argument of which value is to be set by the peer.
    std::string serializedArgument;

    ComponentInterfaceServerProxy->ExecuteCommandReadSerialized(commandId, serializedArgument);

    // Deserialize the argument.
    // Get a per-command serializer.
    mtsProxySerializer * deserializer = PerCommandSerializerMap[commandId];
    if (!deserializer) {
        CMN_LOG_RUN_ERROR << "mtsComponentInterfaceProxyClient: cannot find deserializer (commandRead)" << std::endl;
        return;
    }

    deserializer->DeSerialize(serializedArgument, argument);
}

void mtsComponentInterfaceProxyClient::SendExecuteCommandQualifiedReadSerialized(
    const CommandIDType commandId, const mtsGenericObject & argument1, mtsGenericObject & argument2)
{
    if (!IsValidSession) return;

    //Logger->trace("TIClient", ">>>>> SEND: SendExecuteCommandQualifiedRead");
    
    // Get a per-command serializer.
    mtsProxySerializer * serializer = PerCommandSerializerMap[commandId];
    if (!serializer) {
        CMN_LOG_RUN_ERROR << "mtsComponentInterfaceProxyClient: cannot find serializer (commandQRead)" << std::endl;
        return;
    }

    // Serialize the argument1.
    std::string serializedArgument1;
    serializer->Serialize(argument1, serializedArgument1);
    if (serializedArgument1.size() == 0) {
        CMN_LOG_RUN_ERROR << "mtsComponentInterfaceProxyClient: serialization failure (commandQRead): " 
            << argument1.ToString() << std::endl;
        return;
    }

    // Placeholder for an argument of which value is to be set by the peer.
    std::string serializedArgument2;

    // Execute the command across networks
    ComponentInterfaceServerProxy->ExecuteCommandQualifiedReadSerialized(
        commandId, serializedArgument1, serializedArgument2);

    // Deserialize the argument2.
    serializer->DeSerialize(serializedArgument2, argument2);
}

//-------------------------------------------------------------------------
//  Send Methods
//-------------------------------------------------------------------------
*/

//-------------------------------------------------------------------------
//  Definition by mtsComponentInterfaceProxy.ice
//-------------------------------------------------------------------------
mtsComponentInterfaceProxyClient::ComponentInterfaceClientI::ComponentInterfaceClientI(
    const Ice::CommunicatorPtr& communicator, 
    const Ice::LoggerPtr& logger,
    const mtsComponentInterfaceProxy::ComponentInterfaceServerPrx& server,
    mtsComponentInterfaceProxyClient * componentInterfaceClient)
    : Communicator(communicator),
      SenderThreadPtr(new SenderThread<ComponentInterfaceClientIPtr>(this)),
      IceLogger(logger),
      Runnable(true), 
      ComponentInterfaceProxyClient(componentInterfaceClient),
      Server(server)
{
}

void mtsComponentInterfaceProxyClient::ComponentInterfaceClientI::Start()
{
    ComponentInterfaceProxyClient->GetLogger()->trace(
        "mtsComponentInterfaceProxyClient", "Send thread starts");

    SenderThreadPtr->start();
}

// TODO: Remove this
//#define _COMMUNICATION_TEST_

void mtsComponentInterfaceProxyClient::ComponentInterfaceClientI::Run()
{
#ifdef _COMMUNICATION_TEST_
    int count = 0;

    while (Runnable) {
        osaSleep(1 * cmn_s);
        std::cout << "\tClient [" << ComponentInterfaceProxyClient->GetProxyName() << "] running (" << ++count << ")" << std::endl;

        std::stringstream ss;
        ss << "Msg " << count << " from Client " << ComponentInterfaceProxyClient->GetProxyName();

        ComponentInterfaceProxyClient->SendTestMessageFromClientToServer(ss.str());
    }
#else
    while (Runnable)
    {
        osaSleep(10 * cmn_ms);
    }
#endif
}

void mtsComponentInterfaceProxyClient::ComponentInterfaceClientI::Stop()
{
    if (!ComponentInterfaceProxyClient->IsActiveProxy()) return;

    // TODO: review the following codes (for thread safety)
    IceUtil::ThreadPtr callbackSenderThread;
    {
        IceUtil::Monitor<IceUtil::Mutex>::Lock lock(*this);

        Runnable = false;

        notify();

        callbackSenderThread = SenderThreadPtr;
        SenderThreadPtr = 0; // Resolve cyclic dependency.
    }
    callbackSenderThread->getThreadControl().join();
}

//-----------------------------------------------------------------------------
//  Network Event handlers (Server -> Client)
//-----------------------------------------------------------------------------
void mtsComponentInterfaceProxyClient::ComponentInterfaceClientI::TestMessageFromServerToClient(
    const std::string & str, const ::Ice::Current & current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyClient, "<<<<< RECV: TestMessageFromServerToClient");
#endif

    ComponentInterfaceProxyClient->ReceiveTestMessageFromServerToClient(str);
}

bool mtsComponentInterfaceProxyClient::ComponentInterfaceClientI::FetchFunctionProxyPointers(
    const std::string & requiredInterfaceName, mtsComponentInterfaceProxy::FunctionProxyPointerSet & functionProxyPointers, 
    const ::Ice::Current & current) const
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyClient, "<<<<< RECV: FetchFunctionProxyPointers: " << requiredInterfaceName);
#endif

    return ComponentInterfaceProxyClient->ReceiveFetchFunctionProxyPointers(requiredInterfaceName, functionProxyPointers);
}

void mtsComponentInterfaceProxyClient::ComponentInterfaceClientI::ExecuteCommandVoid(
    ::Ice::Long commandID, const ::Ice::Current & current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyClient, "<<<<< RECV: ExecuteCommandVoid: " << commandID);
#endif

    ComponentInterfaceProxyClient->ReceiveExecuteCommandVoid(commandID);
}

void mtsComponentInterfaceProxyClient::ComponentInterfaceClientI::ExecuteCommandWriteSerialized(
    ::Ice::Long commandID, const ::std::string & argument, const ::Ice::Current & current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyClient, "<<<<< RECV: ExecuteCommandWriteSerialized: " << commandID);
#endif

    ComponentInterfaceProxyClient->ReceiveExecuteCommandWriteSerialized(commandID, argument);
}

void mtsComponentInterfaceProxyClient::ComponentInterfaceClientI::ExecuteCommandReadSerialized(
    ::Ice::Long commandID, ::std::string & argument, const ::Ice::Current & current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyClient, "<<<<< RECV: ExecuteCommandReadSerialized: " << commandID);
#endif

    ComponentInterfaceProxyClient->ReceiveExecuteCommandReadSerialized(commandID, argument);
}

void mtsComponentInterfaceProxyClient::ComponentInterfaceClientI::ExecuteCommandQualifiedReadSerialized(
    ::Ice::Long commandID, const ::std::string & argumentIn, ::std::string & argumentOut, const ::Ice::Current & current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyClient, "<<<<< RECV: ExecuteCommandQualifiedReadSerialized: " << commandID);
#endif

    ComponentInterfaceProxyClient->ReceiveExecuteCommandQualifiedReadSerialized(commandID, argumentIn, argumentOut);
}