/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsComponentInterfaceProxyServer.cpp 145 2009-03-18 23:32:40Z mjung5 $

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

#include <cisstMultiTask/mtsComponentProxy.h>
#include <cisstMultiTask/mtsComponentInterfaceProxyServer.h>
#include <cisstMultiTask/mtsComponentInterfaceProxyClient.h>
#include <cisstOSAbstraction/osaSleep.h>

CMN_IMPLEMENT_SERVICES(mtsComponentInterfaceProxyServer);

std::string mtsComponentInterfaceProxyServer::InterfaceCommunicatorID = "InterfaceCommunicator";
std::string mtsComponentInterfaceProxyServer::ConnectionIDKey = "InterfaceConnectionID";
unsigned int mtsComponentInterfaceProxyServer::InstanceCounter = 0;

//-----------------------------------------------------------------------------
//  Proxy Start-up
//-----------------------------------------------------------------------------
bool mtsComponentInterfaceProxyServer::Start(mtsComponentProxy * proxyOwner)
{
    // Initialize Ice object.
    IceInitialize();
    
    if (!InitSuccessFlag) {
        LogError(mtsComponentInterfaceProxyServer, "ICE proxy server Initialization failed");
        return false;
    }

    // Set the owner and name of this proxy object
    std::string thisProcessName = "On";
    mtsManagerLocal * managerLocal = mtsManagerLocal::GetInstance();
    thisProcessName += managerLocal->GetProcessName();
    SetProxyOwner(proxyOwner, thisProcessName);

    // Create a worker thread here and returns immediately.
    ThreadArgumentsInfo.Proxy = this;
    ThreadArgumentsInfo.Runner = mtsComponentInterfaceProxyServer::Runner;

    // Set a short name of this thread as CIPS which means "Component Interface 
    // Proxy Server." Such a condensed naming rule is required because a total
    // number of characters in a thread name is sometimes limited to a small
    // number (e.g. LINUX RTAI).
    std::stringstream ss;
    ss << "CIPS" << mtsComponentInterfaceProxyServer::InstanceCounter++;
    std::string threadName = ss.str();

    // Create worker thread. Note that it is created but is not yet running.
    WorkerThread.Create<ProxyWorker<mtsComponentProxy>, ThreadArguments<mtsComponentProxy>*>(
        &ProxyWorkerInfo, &ProxyWorker<mtsComponentProxy>::Run, &ThreadArgumentsInfo, threadName.c_str());

    return true;
}

void mtsComponentInterfaceProxyServer::StartServer()
{
    Sender->Start();

    // This is a blocking call that should be run in a different thread.
    IceCommunicator->waitForShutdown();
}

void mtsComponentInterfaceProxyServer::Runner(ThreadArguments<mtsComponentProxy> * arguments)
{
    mtsComponentInterfaceProxyServer * ProxyServer = 
        dynamic_cast<mtsComponentInterfaceProxyServer*>(arguments->Proxy);
    if (!ProxyServer) {
        CMN_LOG_RUN_ERROR << "mtsComponentInterfaceProxyServer: failed to create a proxy server." << std::endl;
        return;
    }

    ProxyServer->GetLogger()->trace("mtsComponentInterfaceProxyServer", "proxy server starts");

    try {
        ProxyServer->SetAsActiveProxy();
        ProxyServer->StartServer();
    } catch (const Ice::Exception& e) {
        std::string error("mtsComponentInterfaceProxyServer: ");
        error += e.what();
        ProxyServer->GetLogger()->error(error);
    } catch (const char * msg) {
        std::string error("mtsComponentInterfaceProxyServer: ");
        error += msg;
        ProxyServer->GetLogger()->error(error);
    }

    ProxyServer->GetLogger()->trace("mtsComponentInterfaceProxyServer", "Proxy server terminates");

    ProxyServer->Stop();
}

void mtsComponentInterfaceProxyServer::Stop()
{
    LogPrint(mtsComponentInterfaceProxyClient, "ComponentInterfaceProxy server stops.");

    BaseServerType::Stop();

    Sender->Stop();
}

void mtsComponentInterfaceProxyServer::OnClose()
{
    //
    //  TODO: Add OnClose() event handler.
    //
}

mtsComponentInterfaceProxyServer::ComponentInterfaceClientProxyType * mtsComponentInterfaceProxyServer::GetNetworkProxyClient(const ClientIDType clientID)
{
    ComponentInterfaceClientProxyType * clientProxy = GetClientByClientID(clientID);
    if (!clientProxy) {
        LogError(mtsComponentInterfaceProxyServer, "GetNetworkProxyClient: no client proxy connected with client id: " << clientID);
        return NULL;
    }

    // Check if this network proxy server is active. We don't need to check if
    // a proxy client is still active since any disconnection or inactive proxy
    // has already been detected and taken care of.
    //
    // TODO: add client proxy disconnection/inactive client proxy clean-up
    //
    return (IsActiveProxy() ? clientProxy : NULL);
}

//-------------------------------------------------------------------------
//  Event Handlers (Client -> Server)
//-------------------------------------------------------------------------
void mtsComponentInterfaceProxyServer::ReceiveTestMessageFromClientToServer(const ConnectionIDType & connectionID, const std::string & str)
{
    const ClientIDType clientID = GetClientID(connectionID);

#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyServer,
             "ReceiveTestMessageFromClientToServer: " 
             << "\n..... ConnectionID: " << connectionID
             << "\n..... Message: " << str);
#endif

    std::cout << "Server: received from Client " << clientID << ": " << str << std::endl;
}

bool mtsComponentInterfaceProxyServer::ReceiveAddClient(
    const ConnectionIDType & connectionID, const std::string & connectingProxyName,
    const unsigned int providedInterfaceProxyInstanceId, ComponentInterfaceClientProxyType & clientProxy)
{
    if (!AddProxyClient(connectingProxyName, providedInterfaceProxyInstanceId, connectionID, clientProxy)) {
        LogError(mtsComponentInterfaceProxyServer, "ReceiveAddClient: failed to add proxy client: " << connectingProxyName);
        return false;
    }

#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyServer,
             "ReceiveAddClient: added proxy client: " 
             << "\n..... ConnectionID: " << connectionID 
             << "\n..... Proxy Name: " << connectingProxyName 
             << "\n..... ClientID: " << providedInterfaceProxyInstanceId);
#endif

    return true;
}

bool mtsComponentInterfaceProxyServer::ReceiveFetchEventGeneratorProxyPointers(
    const ConnectionIDType & connectionID, const std::string & clientComponentName,
    const std::string & requiredInterfaceName,
    mtsComponentInterfaceProxy::EventGeneratorProxyPointerSet & eventGeneratorProxyPointers)
{
    const ClientIDType clientID = GetClientID(connectionID);

#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyServer,
             "ReceiveFetchEventGeneratorProxyPointers: " 
             << "\n..... ConnectionID: " << connectionID 
             << "\n..... Client component name: " << clientComponentName
             << "\n..... Required interface name: " << requiredInterfaceName);
#endif

    return ProxyOwner->GetEventGeneratorProxyPointer(clientComponentName, requiredInterfaceName, eventGeneratorProxyPointers);
}

bool mtsComponentInterfaceProxyServer::RegisterPerCommandSerializer(const CommandIDType commandID, mtsProxySerializer * serializer)
{
    PerCommandSerializerMapType::const_iterator it = PerCommandSerializerMap.find(commandID);
    if (!serializer || it != PerCommandSerializerMap.end()) {
        LogError(mtsComponentInterfaceProxyServer, "failed to add per-command serializer" << std::endl);
        return false;
    }

    PerCommandSerializerMap[commandID] = serializer;

    return true;
}

void mtsComponentInterfaceProxyServer::ReceiveExecuteEventVoid(const CommandIDType commandID)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyServer, "ReceiveExecuteEventVoid: " << commandID);
#endif

    mtsMulticastCommandVoid * eventVoidGeneratorProxy = reinterpret_cast<mtsMulticastCommandVoid*>(commandID);
    if (!eventVoidGeneratorProxy) {
        LogError(mtsComponentInterfaceProxyServer, "ReceiveExecuteEventVoid: invalid proxy id of event void: " << commandID);
        return;
    }

    eventVoidGeneratorProxy->Execute();
}

void mtsComponentInterfaceProxyServer::ReceiveExecuteEventWriteSerialized(const CommandIDType commandID, const std::string & serializedArgument)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyServer, "ReceiveExecuteEventWriteSerialized: " << commandID << ", " << serializedArgument.size() << " bytes");
#endif


    mtsMulticastCommandWriteProxy * eventWriteGeneratorProxy = reinterpret_cast<mtsMulticastCommandWriteProxy*>(commandID);
    if (!eventWriteGeneratorProxy) {
        LogError(mtsComponentInterfaceProxyServer, "ReceiveExecuteEventWriteSerialized: invalid proxy id of event write: " << commandID);
        return;
    }

    // Get a per-command serializer.
    mtsProxySerializer * deserializer = eventWriteGeneratorProxy->GetSerializer();
    mtsGenericObject * argument = deserializer->DeSerialize(serializedArgument);
    if (!argument) {
        LogError(mtsComponentInterfaceProxyServer, "ReceiveExecuteEventWriteSerialized: Deserialization failed");
        return;
    }

    eventWriteGeneratorProxy->Execute(*argument);
}

//-------------------------------------------------------------------------
//  Event Generators (Event Sender) : Server -> Client
//-------------------------------------------------------------------------
void mtsComponentInterfaceProxyServer::SendTestMessageFromServerToClient(const std::string & str)
{
    if (!this->IsActiveProxy()) return;

#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyServer, ">>>>> SEND: SendMessageFromServerToClient");
#endif

    // iterate client map -> send message to ALL clients (broadcasts)
    ComponentInterfaceClientProxyType * clientProxy;
    ClientIDMapType::iterator it = ClientIDMap.begin();
    ClientIDMapType::const_iterator itEnd = ClientIDMap.end();
    for (; it != itEnd; ++it) {
        clientProxy = &(it->second.ClientProxy);
        try 
        {
            (*clientProxy)->TestMessageFromServerToClient(str);
        }
        catch (const ::Ice::Exception & ex)
        {
            std::cerr << "Error: " << ex << std::endl;
            continue;
        }
    }
}

bool mtsComponentInterfaceProxyServer::SendFetchFunctionProxyPointers(
    const ClientIDType clientID, const std::string & requiredInterfaceName,
    mtsComponentInterfaceProxy::FunctionProxyPointerSet & functionProxyPointers)
{
    ComponentInterfaceClientProxyType * clientProxy = GetNetworkProxyClient(clientID);
    if (!clientProxy) {
        LogError(mtsComponentInterfaceProxyServer, "SendFetchFunctionProxyPointers: no proxy client found or inactive proxy: " << clientID);
        return false;
    }

#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyServer, ">>>>> SEND: SendFetchFunctionProxyPointers: provided interface proxy instance id: " << clientID);
#endif

    return (*clientProxy)->FetchFunctionProxyPointers(requiredInterfaceName, functionProxyPointers);
}

bool mtsComponentInterfaceProxyServer::SendExecuteCommandVoid(const ClientIDType clientID, const CommandIDType commandID)
{
    ComponentInterfaceClientProxyType * clientProxy = GetNetworkProxyClient(clientID);
    if (!clientProxy) {
        LogError(mtsComponentInterfaceProxyServer, "SendExecuteCommandVoid: no proxy client found or inactive proxy: " << clientID);
        return false;
    }

    (*clientProxy)->ExecuteCommandVoid(commandID);

#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyServer, ">>>>> SEND: SendExecuteCommandVoid: " << commandID);
#endif

    return true;
}

bool mtsComponentInterfaceProxyServer::SendExecuteCommandWriteSerialized(const ClientIDType clientID, const CommandIDType commandID, const mtsGenericObject & argument)
{
    ComponentInterfaceClientProxyType * clientProxy = GetNetworkProxyClient(clientID);
    if (!clientProxy) {
        LogError(mtsComponentInterfaceProxyServer, "SendExecuteCommandWriteSerialized: no proxy client found or inactive proxy: " << clientID);
        return false;
    }

    // Get per-command serializer
    mtsProxySerializer * serializer = PerCommandSerializerMap[commandID];
    if (!serializer) {
        LogError(mtsComponentInterfaceProxyServer, "SendExecuteCommandWriteSerialized: cannot find per-command serializer");
        return false;
    }

    // Serialize the argument
    std::string serializedArgument;
    serializer->Serialize(argument, serializedArgument);
    if (serializedArgument.empty()) {
        LogError(mtsComponentInterfaceProxyServer, "SendExecuteCommandWriteSerialized: serialization failure: " << argument.ToString());
        return false;
    }

#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyServer, ">>>>> SEND: SendExecuteCommandWriteSerialized: " << commandID << ", " << serializedArgument.size() << " bytes");
#endif

    (*clientProxy)->ExecuteCommandWriteSerialized(commandID, serializedArgument);

    return true;
}

bool mtsComponentInterfaceProxyServer::SendExecuteCommandReadSerialized(const ClientIDType clientID, const CommandIDType commandID, mtsGenericObject & argument)
{
    ComponentInterfaceClientProxyType * clientProxy = GetNetworkProxyClient(clientID);
    if (!clientProxy) return false;

    // Argument placeholder of which value is set by the read command
    std::string serializedArgument;

    // Execute read command
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyServer, ">>>>> SEND: SendExecuteCommandReadSerialized: " << commandID);
#endif

    (*clientProxy)->ExecuteCommandReadSerialized(commandID, serializedArgument);

#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyServer, ">>>>> SEND: SendExecuteCommandReadSerialized: received " << serializedArgument.size() << " bytes");
#endif

    // Deserialize the argument returned
    mtsProxySerializer * deserializer = PerCommandSerializerMap[commandID];
    if (!deserializer) {
        LogError(mtsComponentInterfaceProxyServer, "SendExecuteCommandReadSerialized: cannot find per-command serializer");
        return false;
    }
    deserializer->DeSerialize(serializedArgument, argument);

    return true;
}

bool mtsComponentInterfaceProxyServer::SendExecuteCommandQualifiedReadSerialized(
    const ClientIDType clientID, const CommandIDType commandID, const mtsGenericObject & argumentIn, mtsGenericObject & argumentOut)
{
    ComponentInterfaceClientProxyType * clientProxy = GetNetworkProxyClient(clientID);
    if (!clientProxy) {
        LogError(mtsComponentInterfaceProxyServer, "SendExecuteCommandQualifiedReadSerialized: no proxy client found or inactive proxy: " << clientID);
        return false;
    }

    // Get per-command serializer
    mtsProxySerializer * serializer = PerCommandSerializerMap[commandID];
    if (!serializer) {
        LogError(mtsComponentInterfaceProxyServer, "SendExecuteCommandQualifiedReadSerialized: cannot find per-command serializer");
        return false;
    }

    // Serialize the input argument
    std::string serializedArgumentIn;
    serializer->Serialize(argumentIn, serializedArgumentIn);
    if (serializedArgumentIn.empty()) {
        LogError(mtsComponentInterfaceProxyServer, "SendExecuteCommandQualifiedReadSerialized: serialization failure: " << argumentIn.ToString());
        return false;
    }

    // Argument placeholder of which value is set by the qualified read command
    std::string serializedArgumentOut;

#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsComponentInterfaceProxyServer, ">>>>> SEND: SendExecuteCommandQualifiedReadSerialized: " << commandID << ", " << serializedArgumentIn.size() << " bytes");
#endif
    (*clientProxy)->ExecuteCommandQualifiedReadSerialized(commandID, serializedArgumentIn, serializedArgumentOut);

    // Deserialize
    serializer->DeSerialize(serializedArgumentOut, argumentOut);
    
    return true;
}

//-------------------------------------------------------------------------
//  Definition by mtsComponentInterfaceProxy.ice
//-------------------------------------------------------------------------
mtsComponentInterfaceProxyServer::ComponentInterfaceServerI::ComponentInterfaceServerI(
    const Ice::CommunicatorPtr& communicator, const Ice::LoggerPtr& logger,
    mtsComponentInterfaceProxyServer * componentInterfaceProxyServer)
    : Communicator(communicator),
      SenderThreadPtr(new SenderThread<ComponentInterfaceServerIPtr>(this)),
      IceLogger(logger),
      ComponentInterfaceProxyServer(componentInterfaceProxyServer)
{
}

void mtsComponentInterfaceProxyServer::ComponentInterfaceServerI::Start()
{
    ComponentInterfaceProxyServer->GetLogger()->trace("mtsComponentInterfaceProxyServer", "Send thread starts");

    SenderThreadPtr->start();
}

//#define _COMMUNICATION_TEST_
void mtsComponentInterfaceProxyServer::ComponentInterfaceServerI::Run()
{
#ifdef _COMMUNICATION_TEST_
    int count = 0;

    while (IsActiveProxy()) 
    {
        osaSleep(1 * cmn_s);
        std::cout << "\tServer [" << ComponentInterfaceProxyServer->GetProxyName() << "] running (" << ++count << ")" << std::endl;

        std::stringstream ss;
        ss << "Msg " << count << " from Server";

        ComponentInterfaceProxyServer->SendTestMessageFromServerToClient(ss.str());
    }
#else
    while(IsActiveProxy()) 
    {
        osaSleep(10 * cmn_ms);

        /*
        if(!clients.empty())
        {
            ++num;
            for(std::set<mtsTaskManagerProxy::TaskManagerClientPrx>::iterator p 
                = clients.begin(); p != clients.end(); ++p)
            {
                try
                {
                    std::cout << "server sends: " << num << std::endl;
                }
                catch(const IceUtil::Exception& ex)
                {
                    std::cerr << "removing client `" << Communicator->identityToString((*p)->ice_getIdentity()) << "':\n"
                        << ex << std::endl;

                    IceUtil::Monitor<IceUtil::Mutex>::Lock lock(*this);
                    _clients.erase(*p);
                }
            }
        }
        */
    }
#endif
}

void mtsComponentInterfaceProxyServer::ComponentInterfaceServerI::Stop()
{
    if (!ComponentInterfaceProxyServer->IsActiveProxy()) return;

    // TODO: Review the following codes
    IceUtil::ThreadPtr callbackSenderThread;
    {
        IceUtil::Monitor<IceUtil::Mutex>::Lock lock(*this);

        // TODO: Change proxy state from active to 'prepare to stop(?)'

        notify();

        callbackSenderThread = SenderThreadPtr;
        SenderThreadPtr = 0; // Resolve cyclic dependency.
    }
    callbackSenderThread->getThreadControl().join();
}

//-----------------------------------------------------------------------------
//  Network Event Handlers
//-----------------------------------------------------------------------------
void mtsComponentInterfaceProxyServer::ComponentInterfaceServerI::TestMessageFromClientToServer(
    const std::string & str, const ::Ice::Current & current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(ComponentInterfaceServerI, "<<<<< RECV: TestMessageFromClientToServer");
#endif

    const ConnectionIDType connectionID = current.ctx.find(mtsComponentInterfaceProxyServer::ConnectionIDKey)->second;

    ComponentInterfaceProxyServer->ReceiveTestMessageFromClientToServer(connectionID, str);
}

bool mtsComponentInterfaceProxyServer::ComponentInterfaceServerI::AddClient(
    const std::string & connectingProxyName, ::Ice::Int providedInterfaceProxyInstanceId, 
    const ::Ice::Identity & identity, const ::Ice::Current& current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
   LogPrint(ComponentInterfaceServerI, "<<<<< RECV: AddClient: " << connectingProxyName << " (" << Communicator->identityToString(identity) << ")");
#endif

    const ConnectionIDType connectionID = current.ctx.find(mtsComponentInterfaceProxyServer::ConnectionIDKey)->second;

    IceUtil::Monitor<IceUtil::Mutex>::Lock lock(*this);

    ComponentInterfaceClientProxyType clientProxy = 
        ComponentInterfaceClientProxyType::uncheckedCast(current.con->createProxy(identity));
    
    return ComponentInterfaceProxyServer->ReceiveAddClient(connectionID,
        connectingProxyName, (unsigned int) providedInterfaceProxyInstanceId, clientProxy);
}

void mtsComponentInterfaceProxyServer::ComponentInterfaceServerI::Shutdown(const ::Ice::Current& current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(ComponentInterfaceServerI, "<<<<< RECV: Shutdown");
#endif

    const ConnectionIDType connectionID = current.ctx.find(mtsComponentInterfaceProxyServer::ConnectionIDKey)->second;

    // TODO:
    // Set as true to represent that this connection (session) is going to be closed.
    // After this flag is set, no message is allowed to be sent to a server.
    //ComponentInterfaceProxyServer->ShutdownSession(current);
}

bool mtsComponentInterfaceProxyServer::ComponentInterfaceServerI::FetchEventGeneratorProxyPointers(
    const std::string & requiredInterfaceName, const std::string & providedInterfaceName,
    mtsComponentInterfaceProxy::EventGeneratorProxyPointerSet & eventGeneratorProxyPointers,
    const ::Ice::Current & current) const
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(ComponentInterfaceServerI, "<<<<< RECV: FetchEventGeneratorProxyPointers: req.int=" << requiredInterfaceName << ", prv.int=" << providedInterfaceName);
#endif

    const ConnectionIDType connectionID = current.ctx.find(mtsComponentInterfaceProxyServer::ConnectionIDKey)->second;

    return ComponentInterfaceProxyServer->ReceiveFetchEventGeneratorProxyPointers(
        connectionID, requiredInterfaceName, providedInterfaceName, eventGeneratorProxyPointers);
}

void mtsComponentInterfaceProxyServer::ComponentInterfaceServerI::ExecuteEventVoid(
    ::Ice::Long commandID, const ::Ice::Current & current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(ComponentInterfaceServerI, "<<<<< RECV: ExecuteEventVoid: " << commandID);
#endif

    ComponentInterfaceProxyServer->ReceiveExecuteEventVoid(commandID);
}

void mtsComponentInterfaceProxyServer::ComponentInterfaceServerI::ExecuteEventWriteSerialized(
    ::Ice::Long commandID, const ::std::string & argument, const ::Ice::Current & current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(ComponentInterfaceServerI, "<<<<< RECV: ExecuteEventWriteSerialized: " << commandID);
#endif

    ComponentInterfaceProxyServer->ReceiveExecuteEventWriteSerialized(commandID, argument);
}