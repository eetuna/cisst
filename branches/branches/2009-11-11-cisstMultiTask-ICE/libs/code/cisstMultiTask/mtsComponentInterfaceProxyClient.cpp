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

#include <cisstMultiTask/mtsComponentInterfaceProxyClient.h>

#include <cisstOSAbstraction/osaSleep.h>

//#include <cisstMultiTask/mtsTaskManager.h>
//#include <cisstMultiTask/mtsCommandVoidProxy.h>
//#include <cisstMultiTask/mtsCommandWriteProxy.h>
//#include <cisstMultiTask/mtsCommandReadProxy.h>
//#include <cisstMultiTask/mtsCommandQualifiedReadProxy.h>
//#include <cisstMultiTask/mtsRequiredInterface.h>
//#include <cisstMultiTask/mtsDeviceProxy.h>
//#include <cisstMultiTask/mtsTask.h>

CMN_IMPLEMENT_SERVICES(mtsComponentInterfaceProxyClient);

#define ComponentInterfaceProxyClientLogger(_log) IceLogger->trace("mtsComponentInterfaceProxyClient", _log)
#define ComponentInterfaceProxyClientLoggerError(_log1, _log2) {\
        std::string s("mtsComponentInterfaceProxyClient: ");\
        s += _log1; s+= _log2;\
        IceLogger->error(s); }

//-----------------------------------------------------------------------------
//  Constructor, Destructor, Initializer
//-----------------------------------------------------------------------------
mtsComponentInterfaceProxyClient::mtsComponentInterfaceProxyClient(
    const std::string & serverEndpointInfo, const std::string & communicatorID)
    : BaseType(serverEndpointInfo, communicatorID)
{
}

mtsComponentInterfaceProxyClient::~mtsComponentInterfaceProxyClient()
{
}

//-----------------------------------------------------------------------------
//  Proxy Start-up
//-----------------------------------------------------------------------------
void mtsComponentInterfaceProxyClient::Start(mtsComponentProxy * proxyOwner)
{
    // Initialize Ice object.
    IceInitialize();

    if (InitSuccessFlag) {
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
        ComponentInterfaceServerProxy->AddClient(ident);

        // Create a worker thread here but is not running yet.
        ThreadArgumentsInfo.ProxyOwner = proxyOwner;
        ThreadArgumentsInfo.Proxy = this;        
        ThreadArgumentsInfo.Runner = mtsComponentInterfaceProxyClient::Runner;

        WorkerThread.Create<ProxyWorker<mtsComponentProxy>, ThreadArguments<mtsComponentProxy>*>(
            &ProxyWorkerInfo, &ProxyWorker<mtsComponentProxy>::Run, &ThreadArgumentsInfo, 
            // Set the name of this thread as CIPC which means Component 
            // Interface Proxy Client. Such a very short naming rule is
            // because sometimes there is a limitation of the total number 
            // of characters as a thread name on some systems (e.g. LINUX RTAI).
            "CIPC");
    }
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

    // Set owner of this proxy object
    ProxyClient->SetProxyOwner(arguments->ProxyOwner);

    ProxyClient->GetLogger()->trace("mtsComponentInterfaceProxyClient", "Proxy client starts.....");

    try {
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
    ComponentInterfaceProxyClientLogger("ComponentInterfaceProxy client ends.");

    // Let a server disconnect this client safely.
    // TODO: gcc says this doesn't exist???
    ComponentInterfaceServerProxy->Shutdown();

    ShutdownSession();
    
    BaseType::Stop();
    
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

//-------------------------------------------------------------------------
//  Methods to Receive and Process Events (Server -> Client)
//-------------------------------------------------------------------------
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

//-------------------------------------------------------------------------
//  Methods to Send Events
//-------------------------------------------------------------------------
bool mtsComponentInterfaceProxyClient::SendGetProvidedInterfaceInfo(
    const std::string & providedInterfaceName,
    mtsComponentInterfaceProxy::ProvidedInterfaceInfo & providedInterfaceInfo)
{
    if (!IsValidSession) return false;

    IceLogger->trace("TIClient", ">>>>> SEND: SendGetProvidedInterface");

    return ComponentInterfaceServerProxy->GetProvidedInterfaceInfo(
        providedInterfaceName, providedInterfaceInfo);
}

bool mtsComponentInterfaceProxyClient::SendCreateClientProxies(
    const std::string & userTaskName, const std::string & requiredInterfaceName,
    const std::string & resourceTaskName, const std::string & providedInterfaceName)
{
    if (!IsValidSession) return false;

    IceLogger->trace("TIClient", ">>>>> SEND: SendCreateClientProxies");

    return ComponentInterfaceServerProxy->CreateClientProxies(
        userTaskName, requiredInterfaceName, resourceTaskName, providedInterfaceName);
}

bool mtsComponentInterfaceProxyClient::SendConnectServerSide(
    const std::string & userTaskName, const std::string & requiredInterfaceName,
    const std::string & resourceTaskName, const std::string & providedInterfaceName)
{
    if (!IsValidSession) return false;

    IceLogger->trace("TIClient", ">>>>> SEND: SendConnectServerSide");

    return ComponentInterfaceServerProxy->ConnectServerSide(
        userTaskName, requiredInterfaceName, resourceTaskName, providedInterfaceName);
}

bool mtsComponentInterfaceProxyClient::SendUpdateEventHandlerId(
    const std::string & clientTaskProxyName,
    const mtsComponentInterfaceProxy::ListsOfEventGeneratorsRegistered & eventGeneratorProxies)
{
    if (!IsValidSession) return false;

    IceLogger->trace("TIClient", ">>>>> SEND: SendUpdateEventHandlerId");

    return ComponentInterfaceServerProxy->UpdateEventHandlerId(
        clientTaskProxyName, eventGeneratorProxies);
}

void mtsComponentInterfaceProxyClient::SendGetCommandId(
    const std::string & clientTaskProxyName,
    mtsComponentInterfaceProxy::FunctionProxySet & functionProxies)
{
    if (!IsValidSession) return;

    IceLogger->trace("TIClient", ">>>>> SEND: SendGetCommandId");

    ComponentInterfaceServerProxy->GetCommandId(clientTaskProxyName, functionProxies);
}

void mtsComponentInterfaceProxyClient::SendExecuteCommandVoid(const CommandIDType commandId) const
{
    if (!IsValidSession) return;

    //Logger->trace("TIClient", ">>>>> SEND: SendExecuteCommandVoid");

    ComponentInterfaceServerProxy->ExecuteCommandVoid(commandId);
}

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
      Logger(logger),
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
#define _COMMUNICATION_TEST_

void mtsComponentInterfaceProxyClient::ComponentInterfaceClientI::Run()
{
#ifdef _COMMUNICATION_TEST_
    int num = 0;
#endif

    while (Runnable)
    {
#ifndef _COMMUNICATION_TEST_
        osaSleep(10 * cmn_ms);
#else
        osaSleep(1 * cmn_s);
        std::cout << "Component interface proxy client: " << ++num << std::endl;
#endif
    }
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
//  Device Interface Proxy Client Implementation
//-----------------------------------------------------------------------------
//bool mtsComponentInterfaceProxyClient::ComponentInterfaceClientI::GetListsOfEventGeneratorsRegistered(
//    const std::string & serverTaskProxyName,
//    const std::string & clientTaskName,
//    const std::string & requiredInterfaceName,
//    mtsComponentInterfaceProxy::ListsOfEventGeneratorsRegistered & eventGeneratorProxies,
//    const ::Ice::Current&) const
//{
//    IceLogger->trace("TIClient", "<<<<< RECV: GetListsOfEventGeneratorsRegistered");
//
//    return ComponentInterfaceClient->ReceiveGetListsOfEventGeneratorsRegistered(
//        serverTaskProxyName, clientTaskName, 
//        requiredInterfaceName, eventGeneratorProxies);
//}

//void mtsComponentInterfaceProxyClient::ComponentInterfaceClientI::ExecuteEventVoid(
//    IceCommandIDType commandId, const ::Ice::Current&)
//{
//    Logger->trace("TIClient", "<<<<< RECV: ExecuteEventVoid");
//
//    ComponentInterfaceClient->ReceiveExecuteEventVoid(commandId);
//}
//
//void mtsComponentInterfaceProxyClient::ComponentInterfaceClientI::ExecuteEventWriteSerialized(
//    IceCommandIDType commandId, const ::std::string& argument, const ::Ice::Current&)
//{
//    Logger->trace("TIClient", "<<<<< RECV: ExecuteEventWriteSerialized");
//
//    ComponentInterfaceClient->ReceiveExecuteEventWriteSerialized(commandId, argument);
//}
