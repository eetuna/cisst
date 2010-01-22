/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerProxyClient.cpp 145 2009-03-18 23:32:40Z mjung5 $

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

#include <cisstMultiTask/mtsManagerProxyClient.h>
#include <cisstMultiTask/mtsFunctionVoid.h>

#include <cisstOSAbstraction/osaSleep.h>

CMN_IMPLEMENT_SERVICES(mtsManagerProxyClient);

#define ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG

//-----------------------------------------------------------------------------
//  Constructor, Destructor, Initializer
//-----------------------------------------------------------------------------
mtsManagerProxyClient::mtsManagerProxyClient(
    const std::string & serverEndpointInfo, const std::string & communicatorID)
    : BaseClientType(serverEndpointInfo, communicatorID)
{
}

mtsManagerProxyClient::~mtsManagerProxyClient()
{
}

//-----------------------------------------------------------------------------
//  Proxy Start-up
//-----------------------------------------------------------------------------
bool mtsManagerProxyClient::Start(mtsManagerLocal * proxyOwner)
{
    // Initialize Ice object.
    IceInitialize();

    if (!InitSuccessFlag) {
        LogError(mtsManagerProxyClient, "ICE proxy initialization failed");
        return false;
    }

    // Client configuration for bidirectional communication
    Ice::ObjectAdapterPtr adapter = IceCommunicator->createObjectAdapter("");
    Ice::Identity ident;
    ident.name = GetGUID();
    ident.category = "";

    mtsManagerProxy::ManagerClientPtr client = 
        new ManagerClientI(IceCommunicator, IceLogger, ManagerServerProxy, this);
    adapter->add(client, ident);
    adapter->activate();
    ManagerServerProxy->ice_getConnection()->setAdapter(adapter);

    //
    // TODO: we can use a provided interface proxy instance id instead of an implicit context key.
    //
    // Set an implicit context (per proxy context)
    // (see http://www.zeroc.com/doc/Ice-3.3.1/manual/Adv_server.33.12.html)
    IceCommunicator->getImplicitContext()->put(ConnectionIDKey, IceCommunicator->identityToString(ident));

    // Set the owner and name of this proxy object
    SetProxyOwner(proxyOwner);

    // Connect to server proxy through adding this ICE proxy to server proxy
    if (!ManagerServerProxy->AddClient(GetProxyName(), ident)) {
        LogError(mtsManagerProxyClient, "AddClient() failed: duplicate proxy name or identity");
        return false;
    }

    // Create a worker thread here but is not running yet.
    //ThreadArgumentsInfo.ProxyOwner = proxyOwner;
    ThreadArgumentsInfo.Proxy = this;        
    ThreadArgumentsInfo.Runner = mtsManagerProxyClient::Runner;

    WorkerThread.Create<ProxyWorker<mtsManagerLocal>, ThreadArguments<mtsManagerLocal>*>(
        &ProxyWorkerInfo, &ProxyWorker<mtsManagerLocal>::Run, &ThreadArgumentsInfo, 
        // Set the name of this thread as CIPC which means Component 
        // Interface Proxy Client. Such a very short naming rule is
        // because sometimes there is a limitation of the total number 
        // of characters as a thread name on some systems (e.g. LINUX RTAI).
        "CIPC");

    return true;
}

void mtsManagerProxyClient::StartClient()
{
    Sender->Start();

    // This is a blocking call that should be run in a different thread.
    IceCommunicator->waitForShutdown();
}

void mtsManagerProxyClient::Runner(ThreadArguments<mtsManagerLocal> * arguments)
{
    mtsManagerProxyClient * ProxyClient = 
        dynamic_cast<mtsManagerProxyClient*>(arguments->Proxy);
    if (!ProxyClient) {
        CMN_LOG_RUN_ERROR << "mtsManagerProxyClient: Failed to create a proxy client." << std::endl;
        return;
    }

    ProxyClient->GetLogger()->trace("mtsManagerProxyClient", "Proxy client starts.....");

    try {
        // TODO: By this call, it is 'assumed' that a client proxy is successfully
        // connected to a server proxy.
        // If I can find better way to detect successful connection establishment
        // between a client and a server, this should be updated.
        ProxyClient->SetAsActiveProxy();

        ProxyClient->StartClient();        
    } catch (const Ice::Exception& e) {
        std::string error("mtsManagerProxyClient: ");
        error += e.what();
        ProxyClient->GetLogger()->error(error);
    } catch (const char * msg) {
        std::string error("mtsManagerProxyClient: ");
        error += msg;
        ProxyClient->GetLogger()->error(error);
    }

    ProxyClient->GetLogger()->trace("mtsManagerProxyClient", "Proxy client terminates.....");

    ProxyClient->Stop();
}

void mtsManagerProxyClient::Stop()
{
    LogPrint(mtsManagerProxyClient, "ManagerProxy client ends.");

    // Let a server disconnect this client safely.
    // TODO: gcc says this doesn't exist???
    ManagerServerProxy->Shutdown();

    ShutdownSession();
    
    BaseClientType::Stop();
    
    Sender->Stop();
}

//-------------------------------------------------------------------------
//  Event Handlers (Server -> Client)
//-------------------------------------------------------------------------
void mtsManagerProxyClient::ReceiveTestMessageFromServerToClient(const std::string & str) const
{
    std::cout << "Client received (Server -> Client): " << str << std::endl;
}

bool mtsManagerProxyClient::ReceiveCreateComponentProxy(const ::std::string & componentProxyName)
{
    return ProxyOwner->CreateComponentProxy(componentProxyName);
}

bool mtsManagerProxyClient::ReceiveRemoveComponentProxy(const ::std::string & componentProxyName)
{
    return ProxyOwner->RemoveComponentProxy(componentProxyName);
}

bool mtsManagerProxyClient::ReceiveCreateProvidedInterfaceProxy(const std::string & serverComponentProxyName, const ::mtsManagerProxy::ProvidedInterfaceDescription & providedInterfaceDescription)
{
    //
    // TODO: Structure conversion from const mtsManagerProxy::ProvidedInterfaceDescription to const ProvidedInterfaceDescription &
    //
    //return ProxyOwner->CreateProvidedInterfaceProxy(serverComponentProxyName, providedInterfaceDescription);
    return false;
}

bool mtsManagerProxyClient::ReceiveCreateRequiredInterfaceProxy(const std::string & clientComponentProxyName, const ::mtsManagerProxy::RequiredInterfaceDescription & requiredInterfaceDescription)
{
    //
    // TODO: Structure conversion from const mtsManagerProxy::ProvidedInterfaceDescription to const ProvidedInterfaceDescription &
    //
    //return ProxyOwner->CreateRequiredInterfaceProxy(clientComponentProxyName, requiredInterfaceDescription);
    return false;
}

bool mtsManagerProxyClient::ReceiveRemoveProvidedInterfaceProxy(const std::string & clientComponentProxyName, const std::string & providedInterfaceProxyName)
{
    return ProxyOwner->RemoveProvidedInterfaceProxy(clientComponentProxyName, providedInterfaceProxyName);
}

bool mtsManagerProxyClient::ReceiveRemoveRequiredInterfaceProxy(const std::string & serverComponentProxyName, const std::string & requiredInterfaceProxyName)
{
    return ProxyOwner->RemoveRequiredInterfaceProxy(serverComponentProxyName, requiredInterfaceProxyName);
}

bool mtsManagerProxyClient::ReceiveConnectServerSideInterface(::Ice::Int providedInterfaceProxyInstanceId, const ::mtsManagerProxy::ConnectionStringSet & connectionStringSet)
{
    //
    // TODO: Extract connectionStringSet
    //
    //return ProxyOwner->ConnectServerSideInterface(providedInterfaceProxyInstanceId, connectionStringSet);
    return false;
}

bool mtsManagerProxyClient::ReceiveConnectClientSideInterface(::Ice::Int connectionID, const ::mtsManagerProxy::ConnectionStringSet & connectionStringSet)
{
    //
    // TODO: Extract connectionStringSet
    //
    //return ProxyOwner->ConnectClientSideInterface(connectionID, connectionStringSet);
    return false;
}

bool mtsManagerProxyClient::ReceiveGetProvidedInterfaceDescription(const std::string & componentName, const std::string & providedInterfaceName, ::mtsManagerProxy::ProvidedInterfaceDescription & providedInterfaceDescription) const
{
    //
    // TODO: Structure conversion from const mtsManagerProxy::ProvidedInterfaceDescription to const ProvidedInterfaceDescription &
    //
    //return ProxyOwner->GetProvidedInterfaceDescription(componentName, providedInterfaceName, providedInterfaceDescription);
    return false;
}

bool mtsManagerProxyClient::ReceiveGetRequiredInterfaceDescription(const std::string & componentName, const std::string & requiredInterfaceName, ::mtsManagerProxy::RequiredInterfaceDescription & requiredInterfaceDescription) const
{
    //
    // TODO: Structure conversion from const mtsManagerProxy::ProvidedInterfaceDescription to const ProvidedInterfaceDescription &
    //
    //return ProxyOwner->GetRequiredInterfaceDescription(componentName, requiredInterfaceName, requiredInterfaceDescription);
    return false;
}

std::string mtsManagerProxyClient::ReceiveGetProcessName()
{
    return ProxyOwner->GetProcessName();
}

::Ice::Int mtsManagerProxyClient::ReceiveGetCurrentInterfaceCount(const std::string & componentName)
{
    return ProxyOwner->GetCurrentInterfaceCount(componentName);
}

//-------------------------------------------------------------------------
//  Event Generators (Event Sender) : Client -> Server
//-------------------------------------------------------------------------
void mtsManagerProxyClient::SendTestMessageFromClientToServer(const std::string & str) const
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsManagerProxyClient, ">>>>> SEND: MessageFromClientToServer");
#endif

    ManagerServerProxy->TestMessageFromClientToServer(str);
}

//bool mtsManagerProxyClient::SendFetchEventGeneratorProxyPointers(
//    const std::string & clientComponentName, const std::string & requiredInterfaceName,
//    mtsManagerProxy::ListsOfEventGeneratorsRegistered & eventGeneratorProxyPointers)
//{
//#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
//    LogPrint(mtsManagerProxyClient, ">>>>> SEND: FetchEventGeneratorProxyPointers");
//#endif
//
//    return ManagerServerProxy->FetchEventGeneratorProxyPointers(
//        clientComponentName, requiredInterfaceName, eventGeneratorProxyPointers);
//}

bool mtsManagerProxyClient::SendAddProcess(const std::string & processName)
{
    return true;
}

bool mtsManagerProxyClient::SendFindProcess(const std::string & processName) const
{
    return true;
}

bool mtsManagerProxyClient::SendRemoveProcess(const std::string & processName)
{
    return true;
}

bool mtsManagerProxyClient::SendAddComponent(const std::string & processName, const std::string & componentName)
{
    return true;
}

bool mtsManagerProxyClient::SendFindComponent(const std::string & processName, const std::string & componentName) const
{
    return true;
}

bool mtsManagerProxyClient::SendRemoveComponent(const std::string & processName, const std::string & componentName)
{
    return true;
}

bool mtsManagerProxyClient::SendAddProvidedInterface(const std::string & processName, const std::string & componentName, const std::string & interfaceName, bool isProxyInterface)
{
    return true;
}

bool mtsManagerProxyClient::SendFindProvidedInterface(const std::string & processName, const std::string & componentName, const std::string & interfaceName) const
{
    return true;
}

bool mtsManagerProxyClient::SendRemoveProvidedInterface(const std::string & processName, const std::string & componentName, const std::string & interfaceName)
{
    return true;
}

bool mtsManagerProxyClient::SendAddRequiredInterface(const std::string & processName, const std::string & componentName, const std::string & interfaceName, bool isProxyInterface)
{
    return true;
}

bool mtsManagerProxyClient::SendFindRequiredInterface(const std::string & processName, const std::string & componentName, const std::string & interfaceName) const
{
    return true;
}

bool mtsManagerProxyClient::SendRemoveRequiredInterface(const std::string & processName, const std::string & componentName, const std::string & interfaceName)
{
    return true;
}

::Ice::Int SendConnect(const ::mtsManagerProxy::ConnectionStringSet & connectionStringSet)
{
    return 0;
}

bool mtsManagerProxyClient::SendConnectConfirm(::Ice::Int connectionSessionID)
{
    return true;
}

bool mtsManagerProxyClient::SendDisconnect(const ::mtsManagerProxy::ConnectionStringSet & connectionStringSet)
{
    return true;
}

bool mtsManagerProxyClient::SendSetProvidedInterfaceProxyAccessInfo(const ::mtsManagerProxy::ConnectionStringSet & connectionStringSet, const std::string & endpointInfo, const std::string & communicatorID)
{
    return true;
}

bool mtsManagerProxyClient::SendGetProvidedInterfaceProxyAccessInfo(const ::mtsManagerProxy::ConnectionStringSet & connectionStringSet, std::string & endpointInfo, std::string & communicatorID)
{
    return true;
}

bool mtsManagerProxyClient::SendInitiateConnect(::Ice::Int connectionID, const ::mtsManagerProxy::ConnectionStringSet & connectionStringSet)
{
    return true;
}

bool mtsManagerProxyClient::SendConnectServerSideInterface(::Ice::Int providedInterfaceProxyInstanceId, const ::mtsManagerProxy::ConnectionStringSet & connectionStringSet)
{
    return true;
}

//-------------------------------------------------------------------------
//  Definition by mtsManagerProxy.ice
//-------------------------------------------------------------------------
mtsManagerProxyClient::ManagerClientI::ManagerClientI(
    const Ice::CommunicatorPtr& communicator, 
    const Ice::LoggerPtr& logger,
    const mtsManagerProxy::ManagerServerPrx& server,
    mtsManagerProxyClient * ManagerClient)
    : Communicator(communicator),
      SenderThreadPtr(new SenderThread<ManagerClientIPtr>(this)),
      IceLogger(logger),
      Runnable(true), 
      ManagerProxyClient(ManagerClient),
      Server(server)
{
}

void mtsManagerProxyClient::ManagerClientI::Start()
{
    ManagerProxyClient->GetLogger()->trace(
        "mtsManagerProxyClient", "Send thread starts");

    SenderThreadPtr->start();
}

// TODO: Remove this
#define _COMMUNICATION_TEST_

void mtsManagerProxyClient::ManagerClientI::Run()
{
#ifdef _COMMUNICATION_TEST_
    int count = 0;

    while (Runnable) {
        osaSleep(1 * cmn_s);
        std::cout << "\tClient [" << ManagerProxyClient->GetProxyName() << "] running (" << ++count << ")" << std::endl;

        std::stringstream ss;
        ss << "Msg " << count << " from Client " << ManagerProxyClient->GetProxyName();

        ManagerProxyClient->SendTestMessageFromClientToServer(ss.str());
    }
#else
    while (Runnable)
    {
        osaSleep(10 * cmn_ms);
    }
#endif
}

void mtsManagerProxyClient::ManagerClientI::Stop()
{
    if (!ManagerProxyClient->IsActiveProxy()) return;

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
void mtsManagerProxyClient::ManagerClientI::TestMessageFromServerToClient(
    const std::string & str, const ::Ice::Current & current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsManagerProxyClient, "<<<<< RECV: TestMessageFromServerToClient");
#endif

    ManagerProxyClient->ReceiveTestMessageFromServerToClient(str);
}

bool mtsManagerProxyClient::ManagerClientI::CreateComponentProxy(const std::string & componentProxyName, const ::Ice::Current & current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(ManagerClientI, "<<<<< RECV: CreateComponentProxy");
#endif

    return ManagerProxyClient->ReceiveCreateComponentProxy(componentProxyName);
}

bool mtsManagerProxyClient::ManagerClientI::RemoveComponentProxy(const std::string & componentProxyName, const ::Ice::Current & current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(ManagerClientI, "<<<<< RECV: RemoveComponentProxy");
#endif

    return ManagerProxyClient->ReceiveRemoveComponentProxy(componentProxyName);
}

bool mtsManagerProxyClient::ManagerClientI::CreateProvidedInterfaceProxy(const std::string & serverComponentProxyName, const ::mtsManagerProxy::ProvidedInterfaceDescription & providedInterfaceDescription, const ::Ice::Current & current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
	LogPrint(ManagerClientI, "<<<<< RECV: CreateProvidedInterfaceProxy");
#endif

    return ManagerProxyClient->ReceiveCreateProvidedInterfaceProxy(serverComponentProxyName, providedInterfaceDescription);
}

bool mtsManagerProxyClient::ManagerClientI::CreateRequiredInterfaceProxy(const std::string & clientComponentProxyName, const ::mtsManagerProxy::RequiredInterfaceDescription & requiredInterfaceDescription, const ::Ice::Current & current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
	LogPrint(ManagerClientI, "<<<<< RECV: CreateRequiredInterfaceProxy");
#endif

    return ManagerProxyClient->ReceiveCreateRequiredInterfaceProxy(clientComponentProxyName, requiredInterfaceDescription);
}

bool mtsManagerProxyClient::ManagerClientI::RemoveProvidedInterfaceProxy(const std::string & clientComponentProxyName, const std::string & providedInterfaceProxyName, const ::Ice::Current & current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
	LogPrint(ManagerClientI, "<<<<< RECV: RemoveProvidedInterfaceProxy");
#endif

    return ManagerProxyClient->ReceiveRemoveProvidedInterfaceProxy(clientComponentProxyName, providedInterfaceProxyName);
}

bool mtsManagerProxyClient::ManagerClientI::RemoveRequiredInterfaceProxy(const std::string & serverComponentProxyName, const std::string & requiredInterfaceProxyName, const ::Ice::Current & current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
	LogPrint(ManagerClientI, "<<<<< RECV: RemoveRequiredInterfaceProxy");
#endif

    return ManagerProxyClient->ReceiveRemoveRequiredInterfaceProxy(serverComponentProxyName, requiredInterfaceProxyName);
}

bool mtsManagerProxyClient::ManagerClientI::ConnectServerSideInterface(::Ice::Int providedInterfaceProxyInstanceId, const ::mtsManagerProxy::ConnectionStringSet & connectionStringSet, const ::Ice::Current & current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
	LogPrint(ManagerClientI, "<<<<< RECV: ConnectServerSideInterface");
#endif

    return ManagerProxyClient->ReceiveConnectServerSideInterface(providedInterfaceProxyInstanceId, connectionStringSet);
}

bool mtsManagerProxyClient::ManagerClientI::ConnectClientSideInterface(::Ice::Int connectionID, const ::mtsManagerProxy::ConnectionStringSet & connectionStringSet, const ::Ice::Current & current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
	LogPrint(ManagerClientI, "<<<<< RECV: ConnectClientSideInterface");
#endif

    return ManagerProxyClient->ReceiveConnectClientSideInterface(connectionID, connectionStringSet);
}

bool mtsManagerProxyClient::ManagerClientI::GetProvidedInterfaceDescription(const std::string & componentName, const std::string & providedInterfaceName, ::mtsManagerProxy::ProvidedInterfaceDescription & providedInterfaceDescription, const ::Ice::Current &) const
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
	LogPrint(ManagerClientI, "<<<<< RECV: GetProvidedInterfaceDescription");
#endif

    return ManagerProxyClient->ReceiveGetProvidedInterfaceDescription(componentName, providedInterfaceName, providedInterfaceDescription);
}

bool mtsManagerProxyClient::ManagerClientI::GetRequiredInterfaceDescription(const std::string & componentName, const std::string & requiredInterfaceName, ::mtsManagerProxy::RequiredInterfaceDescription & requiredInterfaceDescription, const ::Ice::Current &) const
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
	LogPrint(ManagerClientI, "<<<<< RECV: GetRequiredInterfaceDescription");
#endif

    return ManagerProxyClient->ReceiveGetRequiredInterfaceDescription(componentName, requiredInterfaceName, requiredInterfaceDescription);
}

std::string mtsManagerProxyClient::ManagerClientI::GetProcessName(const ::Ice::Current & current) const
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
	LogPrint(ManagerClientI, "<<<<< RECV: GetProcessName ");
#endif

    return ManagerProxyClient->ReceiveGetProcessName();
}

::Ice::Int mtsManagerProxyClient::ManagerClientI::GetCurrentInterfaceCount(const ::std::string & componentName, const ::Ice::Current & current) const
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
	LogPrint(ManagerClientI, "<<<<< RECV: GetCurrentInterfaceCount");
#endif

    return ManagerProxyClient->ReceiveGetCurrentInterfaceCount(componentName);
}
