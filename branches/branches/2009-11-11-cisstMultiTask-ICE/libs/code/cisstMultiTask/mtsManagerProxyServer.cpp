/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerProxyServer.cpp 145 2009-03-18 23:32:40Z mjung5 $

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

#include <cisstMultiTask/mtsManagerProxyServer.h>

#include <cisstOSAbstraction/osaSleep.h>
//#include <cisstMultiTask/mtsManagerProxyClient.h>

CMN_IMPLEMENT_SERVICES(mtsManagerProxyServer);

#define ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG

//-----------------------------------------------------------------------------
//  Constructor, Destructor, Initializer
//-----------------------------------------------------------------------------
mtsManagerProxyServer::~mtsManagerProxyServer()
{
    //// Add any resource clean-up related methods here, if any.
    //TaskManagerMapType::iterator it = TaskManagerMap.begin();
    //for (; it != TaskManagerMap.end(); ++it) {
    //    delete it->second;
    //}
}

//-----------------------------------------------------------------------------
//  Proxy Start-up
//-----------------------------------------------------------------------------
bool mtsManagerProxyServer::Start(mtsManagerGlobal * proxyOwner)
{
    // Initialize Ice object.
    IceInitialize();
    
    if (!InitSuccessFlag) {
        LogError(mtsManagerProxyServer, "ICE proxy Initialization failed");
        return false;
    }

    // Set the owner of this proxy object
    SetProxyOwner(proxyOwner);

    // Create a worker thread here and returns immediately.
    //ThreadArgumentsInfo.ProxyOwner = proxyOwner;
    ThreadArgumentsInfo.Proxy = this;
    ThreadArgumentsInfo.Runner = mtsManagerProxyServer::Runner;

    // Note that a worker thread is created but is not yet running.
    WorkerThread.Create<ProxyWorker<mtsManagerGlobal>, ThreadArguments<mtsManagerGlobal>*>(
        &ProxyWorkerInfo, &ProxyWorker<mtsManagerGlobal>::Run, &ThreadArgumentsInfo,
        // Set the name of this thread as GMPS which means Global
        // Manager Proxy Server. Such a very short naming rule is
        // because sometimes there is a limitation of the total number 
        // of characters as a thread name on some systems (e.g. LINUX RTAI).
        "GMPS");

    return true;
}

void mtsManagerProxyServer::StartServer()
{
    Sender->Start();

    // This is a blocking call that should be run in a different thread.
    IceCommunicator->waitForShutdown();
}

void mtsManagerProxyServer::Runner(ThreadArguments<mtsManagerGlobal> * arguments)
{
    mtsManagerProxyServer * ProxyServer = 
        dynamic_cast<mtsManagerProxyServer*>(arguments->Proxy);
    if (!ProxyServer) {
        CMN_LOG_RUN_ERROR << "mtsManagerProxyServer: Failed to create a proxy server." << std::endl;
        return;
    }

    ProxyServer->GetLogger()->trace("mtsManagerProxyServer", "Proxy server starts.....");

    try {
        ProxyServer->SetAsActiveProxy();
        ProxyServer->StartServer();
    } catch (const Ice::Exception& e) {
        std::string error("mtsManagerProxyServer: ");
        error += e.what();
        ProxyServer->GetLogger()->error(error);
    } catch (const char * msg) {
        std::string error("mtsManagerProxyServer: ");
        error += msg;
        ProxyServer->GetLogger()->error(error);
    }

    ProxyServer->GetLogger()->trace("mtsManagerProxyServer", "Proxy server terminates.....");

    ProxyServer->Stop();
}

void mtsManagerProxyServer::Stop()
{
    BaseServerType::Stop();

    Sender->Stop();
}

void mtsManagerProxyServer::OnClose()
{
    //
    //  TODO: Add OnClose() event handler.
    //

    // remove from TaskManagerMapByTaskName
    // remove from TaskManagerClient
    //RemoveTaskManagerByConnectionID();
}

mtsManagerProxyServer::ManagerClientProxyType * mtsManagerProxyServer::GetNetworkProxyClient(const ClientIDType clientID)
{
    ManagerClientProxyType * clientProxy = GetClientByClientID(clientID);
    if (!clientProxy) {
        LogError(mtsManagerProxyServer, "GetNetworkProxyClient: no client proxy connected with client id: " << clientID);
        return NULL;
    }

    //
    // TODO: Check if the network proxy client is active
    //

    // Check if this network proxy server is active
    return (IsActiveProxy() ? clientProxy : NULL);
}

//-------------------------------------------------------------------------
//  Implementation of mtsManagerLocalInterface
//  (See mtsManagerLocalInterface.h for comments)
//-------------------------------------------------------------------------
//  Proxy Object Control (Creation, Removal)
bool mtsManagerProxyServer::CreateComponentProxy(const std::string & componentProxyName)
{
    // TOOD: Call SendCreateComponentProxy() function on proxy server
    // to call CreateComponentProxy at LCM side
    return true;
}

bool mtsManagerProxyServer::RemoveComponentProxy(const std::string & componentProxyName)
{
    return true;
}

bool mtsManagerProxyServer::CreateProvidedInterfaceProxy(const std::string & serverComponentProxyName,
    ProvidedInterfaceDescription & providedInterfaceDescription)
{
    return true;
}

bool mtsManagerProxyServer::CreateRequiredInterfaceProxy(const std::string & clientComponentProxyName,
    RequiredInterfaceDescription & requiredInterfaceDescription)
{
    return true;
}

bool mtsManagerProxyServer::RemoveProvidedInterfaceProxy(
    const std::string & clientComponentProxyName, const std::string & providedInterfaceProxyName)
{
    return true;
}

bool mtsManagerProxyServer::RemoveRequiredInterfaceProxy(
    const std::string & serverComponentProxyName, const std::string & requiredInterfaceProxyName)
{
    return true;
}

//  Connection Management
bool mtsManagerProxyServer::ConnectServerSideInterface(const unsigned int providedInterfaceProxyInstanceId,
    const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
    const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName)
{
    return true;
}

bool mtsManagerProxyServer::ConnectClientSideInterface(const unsigned int connectionID,
    const std::string & clientProcessName, const std::string & clientComponentName, const std::string & clientRequiredInterfaceName,
    const std::string & serverProcessName, const std::string & serverComponentName, const std::string & serverProvidedInterfaceName)
{
    return true;
}

//  Getters
bool mtsManagerProxyServer::GetProvidedInterfaceDescription(const std::string & componentName, const std::string & providedInterfaceName, 
    ProvidedInterfaceDescription & providedInterfaceDescription) const
{
    return true;
}

bool mtsManagerProxyServer::GetRequiredInterfaceDescription(const std::string & componentName, const std::string & requiredInterfaceName, 
    RequiredInterfaceDescription & requiredInterfaceDescription) const
{
    return true;
}

const std::string mtsManagerProxyServer::GetProcessName() const
{
    return "";
}

const int mtsManagerProxyServer::GetCurrentInterfaceCount(const std::string & componentName) const
{
    return 0;
}

//-------------------------------------------------------------------------
//  Event Handlers (Client -> Server)
//-------------------------------------------------------------------------
void mtsManagerProxyServer::ReceiveTestMessageFromClientToServer(
    const ConnectionIDType & connectionID, const std::string & str)
{
    const ClientIDType clientID = GetClientID(connectionID);

    LogPrint(mtsManagerProxyServer,
             "ReceiveTestMessageFromClientToServer: " 
             << "\n..... ConnectionID: " << connectionID
             << "\n..... Message: " << str);

    std::cout << "Server: received from Client " << clientID << ": " << str << std::endl;
}

bool mtsManagerProxyServer::ReceiveAddClient(
    const ConnectionIDType & connectionID, const std::string & connectingProxyName,
    ManagerClientProxyType & clientProxy)
{
    return true;
}

//
// TODO: Implement ReceiveShutdown()
//

bool mtsManagerProxyServer::ReceiveAddProcess(const std::string &, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ReceiveFindProcess(const std::string &, const ::Ice::Current &) const
{
    return true;
}

bool mtsManagerProxyServer::ReceiveRemoveProcess(const std::string &, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ReceiveAddComponent(const std::string &, const std::string &, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ReceiveFindComponent(const std::string &, const std::string &, const ::Ice::Current &) const
{
    return true;
}

bool mtsManagerProxyServer::ReceiveRemoveComponent(const std::string &, const std::string &, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ReceiveAddProvidedInterface(const std::string &, const std::string &, const std::string &, bool, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ReceiveFindProvidedInterface(const std::string &, const std::string &, const std::string &, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ReceiveRemoveProvidedInterface(const std::string &, const std::string &, const std::string &, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ReceiveAddRequiredInterface(const std::string &, const std::string &, const std::string &, bool, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ReceiveFindRequiredInterface(const std::string &, const std::string &, const std::string &, const ::Ice::Current &) const
{
    return true;
}

bool mtsManagerProxyServer::ReceiveRemoveRequiredInterface(const std::string &, const std::string &, const std::string &, const ::Ice::Current &)
{
    return true;
}

::Ice::Int mtsManagerProxyServer::ReceiveConnect(const ::mtsManagerProxy::ConnectionStringSet&, const ::Ice::Current &)
{
    return 0;
}

bool mtsManagerProxyServer::ReceiveConnectConfirm(::Ice::Int, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ReceiveDisconnect(const ::mtsManagerProxy::ConnectionStringSet&, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ReceiveSetProvidedInterfaceProxyAccessInfo(const ::mtsManagerProxy::ConnectionStringSet&, const std::string &, const std::string &, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ReceiveGetProvidedInterfaceProxyAccessInfo(const ::mtsManagerProxy::ConnectionStringSet&, const std::string &, const std::string &, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ReceiveInitiateConnect(::Ice::Int, const ::mtsManagerProxy::ConnectionStringSet&, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ReceiveConnectServerSideInterface(::Ice::Int, const ::mtsManagerProxy::ConnectionStringSet&, const ::Ice::Current &)
{
    return true;
}

//-------------------------------------------------------------------------
//  Event Generators (Event Sender) : Server -> Client
//-------------------------------------------------------------------------
void mtsManagerProxyServer::SendTestMessageFromServerToClient(const std::string & str)
{
    if (!this->IsActiveProxy()) return;

#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsManagerProxyServer, ">>>>> SEND: SendMessageFromServerToClient");
#endif

    // iterate client map -> send message to ALL clients (broadcasts)
    ManagerClientProxyType * clientProxy;
    ClientIDMapType::iterator it = ClientIDMap.begin();
    ClientIDMapType::const_iterator itEnd = ClientIDMap.end();
    for (; it != itEnd; ++it) {
        clientProxy = &(it->second.ClientProxy);
        try 
        {
            (*clientProxy)->TestSendMessageFromServerToClient(str);
        }
        catch (const ::Ice::Exception & ex)
        {
            std::cerr << "Error: " << ex << std::endl;
            continue;
        }
    }
}

/*
bool mtsManagerProxyServer::SendFetchFunctionProxyPointers(
    const ClientIDType clientID, const std::string & requiredInterfaceName,
    mtsComponentInterfaceProxy::FunctionProxyPointerSet & functionProxyPointers)
{
    ManagerClientProxyType * clientProxy = GetNetworkProxyClient(clientID);
    if (!clientProxy) {
        LogError(mtsManagerProxyServer, "SendFetchFunctionProxyPointers: no proxy client found or inactive proxy: " << clientID);
        return false;
    }

#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(mtsManagerProxyServer, ">>>>> SEND: SendMessageFromServerToClient: provided interface proxy instance id: " << clientID);
#endif

    return (*clientProxy)->FetchFunctionProxyPointers(requiredInterfaceName, functionProxyPointers);
}
*/

//-------------------------------------------------------------------------
//  Send Methods
//-------------------------------------------------------------------------
/*
bool mtsManagerProxyServer::SendConnectServerSide(
    TaskManagerClient * taskManagerWithServerTask,
    const std::string & userTaskName,     const std::string & requiredInterfaceName,
    const std::string & resourceTaskName, const std::string & providedInterfaceName)
{
    GetLogger()->trace("TMServer", ">>>>> SEND: ConnectServerSide: " 
            + resourceTaskName + " : " + providedInterfaceName + " - "
            + userTaskName + " : " + requiredInterfaceName);

    return taskManagerWithServerTask->GetClientProxy()->ConnectServerSide(
        userTaskName, requiredInterfaceName, resourceTaskName, providedInterfaceName);
}
*/

//-------------------------------------------------------------------------
//  Definition by mtsComponentInterfaceProxy.ice
//-------------------------------------------------------------------------
mtsManagerProxyServer::ManagerServerI::ManagerServerI(
    const Ice::CommunicatorPtr& communicator, const Ice::LoggerPtr& logger,
    mtsManagerProxyServer * managerProxyServer)
    : Communicator(communicator),
      SenderThreadPtr(new SenderThread<ManagerServerIPtr>(this)),
      IceLogger(logger),
      Runnable(true),
      ManagerProxyServer(managerProxyServer)
{
}

void mtsManagerProxyServer::ManagerServerI::Start()
{
    ManagerProxyServer->GetLogger()->trace(
        "mtsManagerProxyServer", "Send thread starts");

    SenderThreadPtr->start();
}

// TODO: Remove this
#define _COMMUNICATION_TEST_

void mtsManagerProxyServer::ManagerServerI::Run()
{
#ifdef _COMMUNICATION_TEST_
    int count = 0;

    while (Runnable) 
    {
        osaSleep(1 * cmn_s);
        std::cout << "\tServer [" << ManagerProxyServer->GetProxyName() << "] running (" << ++count << ")" << std::endl;

        std::stringstream ss;
        ss << "Msg " << count << " from Server";

        ManagerProxyServer->SendTestMessageFromServerToClient(ss.str());
    }
#else
    while(Runnable) 
    {
        osaSleep(10 * cmn_ms);

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
    }
#endif
}

void mtsManagerProxyServer::ManagerServerI::Stop()
{
    if (!ManagerProxyServer->IsActiveProxy()) return;

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
//  Network Event Handlers
//-----------------------------------------------------------------------------
void mtsManagerProxyServer::ManagerServerI::TestSendMessageFromClientToServer(
    const std::string & str, const ::Ice::Current & current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(ManagerServerI, "<<<<< RECV: TestSendMessageFromClientToServer");
#endif

    const ConnectionIDType connectionID = current.ctx.find(ConnectionIDKey)->second;

    ManagerProxyServer->ReceiveTestMessageFromClientToServer(connectionID, str);
}

bool mtsManagerProxyServer::ManagerServerI::AddClient(
    const std::string & connectingProxyName, const ::Ice::Identity & identity, const ::Ice::Current& current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
   LogPrint(ManagerServerI, "<<<<< RECV: AddClient: " << connectingProxyName << " (" << Communicator->identityToString(identity) << ")");
#endif

    const ConnectionIDType connectionID = current.ctx.find(ConnectionIDKey)->second;

    IceUtil::Monitor<IceUtil::Mutex>::Lock lock(*this);

    ManagerClientProxyType clientProxy = 
        ManagerClientProxyType::uncheckedCast(current.con->createProxy(identity));
    
    return ManagerProxyServer->ReceiveAddClient(connectionID, connectingProxyName, clientProxy);
}

void mtsManagerProxyServer::ManagerServerI::Shutdown(const ::Ice::Current& current)
{
#ifdef ENABLE_DETAILED_MESSAGE_EXCHANGE_LOG
    LogPrint(ManagerServerI, "<<<<< RECV: Shutdown");
#endif

    const ConnectionIDType connectionID = current.ctx.find(ConnectionIDKey)->second;

    // TODO:
    // Set as true to represent that this connection (session) is going to be closed.
    // After this flag is set, no message is allowed to be sent to a server.
    //ComponentInterfaceProxyServer->ShutdownSession(current);
}

bool mtsManagerProxyServer::ManagerServerI::AddProcess(const std::string &, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ManagerServerI::FindProcess(const std::string &, const ::Ice::Current &) const
{
    return true;
}

bool mtsManagerProxyServer::ManagerServerI::RemoveProcess(const std::string &, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ManagerServerI::AddComponent(const std::string &, const std::string &, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ManagerServerI::FindComponent(const std::string &, const std::string &, const ::Ice::Current &) const
{
    return true;
}

bool mtsManagerProxyServer::ManagerServerI::RemoveComponent(const std::string &, const std::string &, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ManagerServerI::AddProvidedInterface(const std::string &, const std::string &, const std::string &, bool, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ManagerServerI::FindProvidedInterface(const std::string &, const std::string &, const std::string &, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ManagerServerI::RemoveProvidedInterface(const std::string &, const std::string &, const std::string &, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ManagerServerI::AddRequiredInterface(const std::string &, const std::string &, const std::string &, bool, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ManagerServerI::FindRequiredInterface(const std::string &, const std::string &, const std::string &, const ::Ice::Current &) const
{
    return true;
}

bool mtsManagerProxyServer::ManagerServerI::RemoveRequiredInterface(const std::string &, const std::string &, const std::string &, const ::Ice::Current &)
{
    return true;
}

::Ice::Int mtsManagerProxyServer::ManagerServerI::Connect(const ::mtsManagerProxy::ConnectionStringSet&, const ::Ice::Current &)
{
    return 0;
}

bool mtsManagerProxyServer::ManagerServerI::ConnectConfirm(::Ice::Int, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ManagerServerI::Disconnect(const ::mtsManagerProxy::ConnectionStringSet&, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ManagerServerI::SetProvidedInterfaceProxyAccessInfo(const ::mtsManagerProxy::ConnectionStringSet&, const std::string &, const std::string &, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ManagerServerI::GetProvidedInterfaceProxyAccessInfo(const ::mtsManagerProxy::ConnectionStringSet&, const std::string &, const std::string &, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ManagerServerI::InitiateConnect(::Ice::Int, const ::mtsManagerProxy::ConnectionStringSet&, const ::Ice::Current &)
{
    return true;
}

bool mtsManagerProxyServer::ManagerServerI::ConnectServerSideInterface(::Ice::Int, const ::mtsManagerProxy::ConnectionStringSet&, const ::Ice::Current &)
{
    return true;
}