/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsTaskInterfaceProxyServer.cpp 145 2009-03-18 23:32:40Z mjung5 $

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

#include <cisstCommon/cmnAssert.h>
#include <cisstMultiTask/mtsTaskInterfaceProxyServer.h>
#include <cisstMultiTask/mtsTask.h>
#include <cisstMultiTask/mtsDeviceInterface.h>

CMN_IMPLEMENT_SERVICES(mtsTaskInterfaceProxyServer);

#define mtsTaskInterfaceProxyServerLogger(_log) \
            Logger->trace("mtsTaskInterfaceProxyServer", _log);

mtsTaskInterfaceProxyServer::~mtsTaskInterfaceProxyServer()
{
    OnClose();
}

void mtsTaskInterfaceProxyServer::OnClose()
{
}

void mtsTaskInterfaceProxyServer::Start(mtsTask * callingTask)
{
    // Initialize Ice object.
    // Notice that a worker thread is not created right now.
    Init();
    
    if (InitSuccessFlag) {
        // Create a worker thread here and returns immediately.
        ThreadArgumentsInfo.argument = callingTask;
        ThreadArgumentsInfo.proxy = this;
        ThreadArgumentsInfo.Runner = mtsTaskInterfaceProxyServer::Runner;

        WorkerThread.Create<ProxyWorker<mtsTask>, ThreadArguments<mtsTask>*>(
            &ProxyWorkerInfo, &ProxyWorker<mtsTask>::Run, &ThreadArgumentsInfo, "C-PRX");
    }
}

void mtsTaskInterfaceProxyServer::StartServer()
{
    Sender->Start();

    // This is a blocking call that should run in a different thread.
    IceCommunicator->waitForShutdown();
}

void mtsTaskInterfaceProxyServer::Runner(ThreadArguments<mtsTask> * arguments)
{
    mtsTaskInterfaceProxyServer * ProxyServer = 
        dynamic_cast<mtsTaskInterfaceProxyServer*>(arguments->proxy);
    
    ProxyServer->SetConnectedTask(arguments->argument);
    
    //
    // TEST
    //
    //mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq specs;
    //ProxyServer->GetProvidedInterfaceSpecification(specs);

    ProxyServer->GetLogger()->trace("mtsTaskInterfaceProxyServer", "Proxy server starts.");

    try {
        ProxyServer->StartServer();
    } catch (const Ice::Exception& e) {
        ProxyServer->GetLogger()->trace("mtsTaskInterfaceProxyServer error: ", e.what());
    } catch (const char * msg) {
        ProxyServer->GetLogger()->trace("mtsTaskInterfaceProxyServer error: ", msg);
    }

    ProxyServer->OnThreadEnd();
}

void mtsTaskInterfaceProxyServer::OnThreadEnd()
{
    mtsTaskInterfaceProxyServerLogger("Proxy server ends.");

    mtsProxyBaseServer::OnThreadEnd();

    Sender->Destroy();
}

//-------------------------------------------------------------------------
//  Task Processing
//-------------------------------------------------------------------------
const bool mtsTaskInterfaceProxyServer::GetProvidedInterfaceSpecification(
    ::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq & specs)
{
    CMN_ASSERT(ConnectedTask);

    // 1) Iterate all provided interfaces
    mtsDeviceInterface * providedInterface;    

    std::vector<std::string> namesOfProvidedInterfaces = 
        ConnectedTask->GetNamesOfProvidedInterfaces();
    std::vector<std::string>::const_iterator it = namesOfProvidedInterfaces.begin();
    for (; it != namesOfProvidedInterfaces.end(); ++it) {
        mtsTaskInterfaceProxy::ProvidedInterfaceSpecification providedInterfaceSpec;

        // 1) Get a provided interface object
        providedInterface = ConnectedTask->GetProvidedInterface(*it);
        CMN_ASSERT(providedInterface);

        // 2) Get an provided interface name
        providedInterfaceSpec.interfaceName = providedInterface->GetName();

        // 3) Extract all the information on registered command objects, events, and so on.
#define ITERATE_COMMAND_OBJECT_BEGIN( _commandType ) \
        mtsDeviceInterface::Command##_commandType##MapType::MapType::const_iterator iterator##_commandType = \
            providedInterface->Commands##_commandType##.GetMap().begin();\
        mtsDeviceInterface::Command##_commandType##MapType::MapType::const_iterator iterator##_commandType##End = \
            providedInterface->Commands##_commandType##.GetMap().end();\
        for (; iterator##_commandType != iterator##_commandType##End; ++( iterator##_commandType ) ) {\
            mtsTaskInterfaceProxy::Command##_commandType##Info info;

#define ITERATE_COMMAND_OBJECT_END( _commandType ) \
            providedInterfaceSpec.commands##_commandType##.push_back(info);\
        }

        // 3-1) Command: Read
        ITERATE_COMMAND_OBJECT_BEGIN(Void);
            info.Name = iteratorVoid->second->Name;
        ITERATE_COMMAND_OBJECT_END(Void);

        // 3-2) Command: Write
        ITERATE_COMMAND_OBJECT_BEGIN(Write);
            info.Name = iteratorWrite->second->Name;
            info.ArgumentTypeName = iteratorWrite->second->GetArgumentClassServices()->GetName();
        ITERATE_COMMAND_OBJECT_END(Write);

        // 3-3) Command: Read
        ITERATE_COMMAND_OBJECT_BEGIN(Read);
            info.Name = iteratorRead->second->Name;
            info.ArgumentTypeName = iteratorRead->second->GetArgumentClassServices()->GetName();
        ITERATE_COMMAND_OBJECT_END(Read);

        // 3-4) Command: QualifiedRead
        ITERATE_COMMAND_OBJECT_BEGIN(QualifiedRead);
            info.Name = iteratorQualifiedRead->second->Name;
            info.Argument1TypeName = iteratorQualifiedRead->second->GetArgument1Prototype()->Services()->GetName();
            info.Argument2TypeName = iteratorQualifiedRead->second->GetArgument2Prototype()->Services()->GetName();
        ITERATE_COMMAND_OBJECT_END(QualifiedRead);

        // TODO: 
        // 4) Extract events information (void, write)

        specs.push_back(providedInterfaceSpec);
    }

    return true;
}

//-------------------------------------------------------------------------
//  Definition by mtsTaskManagerProxy.ice
//-------------------------------------------------------------------------
mtsTaskInterfaceProxyServer::TaskInterfaceServerI::TaskInterfaceServerI(
    const Ice::CommunicatorPtr& communicator,
    const Ice::LoggerPtr& logger,
    mtsTaskInterfaceProxyServer * taskInterfaceServer) 
    : Communicator(communicator), Logger(logger),
      TaskInterfaceServer(taskInterfaceServer),
      Runnable(true),
      Sender(new SendThread<TaskInterfaceServerIPtr>(this))
{
}

void mtsTaskInterfaceProxyServer::TaskInterfaceServerI::Start()
{
    mtsTaskInterfaceProxyServerLogger("Send thread starts");

    Sender->start();
}

void mtsTaskInterfaceProxyServer::TaskInterfaceServerI::Run()
{
    int num = 0;
    while(true)
    {
        std::set<mtsTaskInterfaceProxy::TaskInterfaceClientPrx> clients;
        {
            IceUtil::Monitor<IceUtil::Mutex>::Lock lock(*this);
            timedWait(IceUtil::Time::seconds(2));

            if(!Runnable)
            {
                break;
            }

            clients = _clients;
        }

#ifdef _COMMUNICATION_TEST_
        if(!clients.empty())
        {
            ++num;
            for(std::set<mtsTaskInterfaceProxy::TaskInterfaceClientPrx>::iterator p 
                = clients.begin(); p != clients.end(); ++p)
            {
                try
                {
                    std::cout << "server sends: " << num << std::endl;
                    (*p)->ReceiveData(num);
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
#endif
    }
}

void mtsTaskInterfaceProxyServer::TaskInterfaceServerI::Destroy()
{
    mtsTaskInterfaceProxyServerLogger("Send thread is terminating.");

    IceUtil::ThreadPtr callbackSenderThread;

    {
        IceUtil::Monitor<IceUtil::Mutex>::Lock lock(*this);

        mtsTaskInterfaceProxyServerLogger("Destroying sender.");
        Runnable = false;

        notify();

        callbackSenderThread = Sender;
        Sender = 0; // Resolve cyclic dependency.
    }

    callbackSenderThread->getThreadControl().join();
}

//-----------------------------------------------------------------------------
//  Proxy Server Implementation
//-----------------------------------------------------------------------------
void mtsTaskInterfaceProxyServer::TaskInterfaceServerI::AddClient(
    const ::Ice::Identity& ident, const ::Ice::Current& current)
{
    IceUtil::Monitor<IceUtil::Mutex>::Lock lock(*this);

    std::string log = "Adding client: " + Communicator->identityToString(ident);
    mtsTaskInterfaceProxyServerLogger(log.c_str());

    mtsTaskInterfaceProxy::TaskInterfaceClientPrx client = 
        mtsTaskInterfaceProxy::TaskInterfaceClientPrx::uncheckedCast(current.con->createProxy(ident));
    _clients.insert(client);    
}

bool mtsTaskInterfaceProxyServer::TaskInterfaceServerI::GetProvidedInterfaceSpecification(
    ::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq & specs, 
    const ::Ice::Current& current) const
{
    Logger->trace("TIServer", "<<<<< RECV: GetProvidedInterfaceSpecification");

    return TaskInterfaceServer->GetProvidedInterfaceSpecification(specs);
}