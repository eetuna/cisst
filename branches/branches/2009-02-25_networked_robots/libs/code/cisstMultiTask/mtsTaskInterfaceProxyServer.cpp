/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsDeviceInterfaceProxyServer.cpp 145 2009-03-18 23:32:40Z mjung5 $

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
#include <cisstMultiTask/mtsDeviceInterfaceProxyServer.h>
#include <cisstMultiTask/mtsTask.h>
#include <cisstMultiTask/mtsDeviceInterface.h>

CMN_IMPLEMENT_SERVICES(mtsDeviceInterfaceProxyServer);

#define mtsDeviceInterfaceProxyServerLogger(_log) \
            Logger->trace("mtsDeviceInterfaceProxyServer", _log);

mtsDeviceInterfaceProxyServer::~mtsDeviceInterfaceProxyServer()
{
    OnClose();
}

void mtsDeviceInterfaceProxyServer::OnClose()
{
}

void mtsDeviceInterfaceProxyServer::Start(mtsTask * callingTask)
{
    // Initialize Ice object.
    // Notice that a worker thread is not created right now.
    Init();
    
    if (InitSuccessFlag) {
        // Create a worker thread here and returns immediately.
        ThreadArgumentsInfo.argument = callingTask;
        ThreadArgumentsInfo.proxy = this;
        ThreadArgumentsInfo.Runner = mtsDeviceInterfaceProxyServer::Runner;

        WorkerThread.Create<ProxyWorker<mtsTask>, ThreadArguments<mtsTask>*>(
            &ProxyWorkerInfo, &ProxyWorker<mtsTask>::Run, &ThreadArgumentsInfo, "C-PRX");
    }
}

void mtsDeviceInterfaceProxyServer::StartServer()
{
    Sender->Start();

    // This is a blocking call that should run in a different thread.
    IceCommunicator->waitForShutdown();
}

void mtsDeviceInterfaceProxyServer::Runner(ThreadArguments<mtsTask> * arguments)
{
    mtsDeviceInterfaceProxyServer * ProxyServer = 
        dynamic_cast<mtsDeviceInterfaceProxyServer*>(arguments->proxy);
    
    ProxyServer->SetConnectedTask(arguments->argument);
    
    //!!!!!!!!!!!!
    //mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq spec;
    //ProxyServer->GetProvidedInterfaceSpecification(spec);

    ProxyServer->GetLogger()->trace("mtsDeviceInterfaceProxyServer", "Proxy server starts.");

    try {
        ProxyServer->StartServer();
    } catch (const Ice::Exception& e) {
        ProxyServer->GetLogger()->trace("mtsDeviceInterfaceProxyServer error: ", e.what());
    } catch (const char * msg) {
        ProxyServer->GetLogger()->trace("mtsDeviceInterfaceProxyServer error: ", msg);
    }

    ProxyServer->OnThreadEnd();
}

void mtsDeviceInterfaceProxyServer::OnThreadEnd()
{
    mtsDeviceInterfaceProxyServerLogger("Proxy server ends.");

    mtsProxyBaseServer::OnThreadEnd();

    Sender->Destroy();
}

//-------------------------------------------------------------------------
//  Task Processing
//-------------------------------------------------------------------------
const bool mtsDeviceInterfaceProxyServer::GetProvidedInterfaceSpecification(
    ::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq & specs)
{
    CMN_ASSERT(ConnectedTask);

    // 1) Iterate all provided interfaces
    mtsDeviceInterface * providedInterface = NULL;    

    std::vector<std::string> namesOfProvidedInterfaces = 
        ConnectedTask->GetNamesOfProvidedInterfaces();
    std::vector<std::string>::const_iterator it = namesOfProvidedInterfaces.begin();
    for (; it != namesOfProvidedInterfaces.end(); ++it) {
        mtsTaskInterfaceProxy::ProvidedInterfaceSpecification providedInterfaceSpec;

        // 1) Get a provided interface object
        providedInterface = ConnectedTask->GetProvidedInterface(*it);
        CMN_ASSERT(providedInterface);

        // 2) Get a provided interface name.
        providedInterfaceSpec.interfaceName = providedInterface->GetName();

        // TODO: Maybe I can just assume that only mtsDeviceInterface is used.
        // Determine the type of the provided interface: is it mtsDeviceInterface or 
        // mtsTaskInterface?
        //if (dynamic_cast<mtsTaskInterface*>(providedInterface)) {
        //    providedInterfaceSpec.providedInterfaceForTask = true;
        //} else {
            providedInterfaceSpec.providedInterfaceForTask = false;
        //}
            
        // 3) Extract all the information on registered command objects, events, and so on.
#define ITERATE_INTERFACE_BEGIN( _commandType ) \
        mtsDeviceInterface::Command##_commandType##MapType::MapType::const_iterator iterator##_commandType = \
            providedInterface->Commands##_commandType##.GetMap().begin();\
        mtsDeviceInterface::Command##_commandType##MapType::MapType::const_iterator iterator##_commandType##End = \
            providedInterface->Commands##_commandType##.GetMap().end();\
        for (; iterator##_commandType != iterator##_commandType##End; ++( iterator##_commandType ) ) {\
            mtsTaskInterfaceProxy::Command##_commandType##Info info;\
            info.Name = iterator##_commandType##->second->GetName();\
            info.CommandSID = reinterpret_cast<int>(iterator##_commandType##->second);

#define ITERATE_INTERFACE_END( _commandType ) \
            providedInterfaceSpec.commands##_commandType##.push_back(info);\
        }

        // 3-1) Command: Void
        ITERATE_INTERFACE_BEGIN(Void);            
        ITERATE_INTERFACE_END(Void);

        // 3-2) Command: Write
        ITERATE_INTERFACE_BEGIN(Write);
            info.ArgumentTypeName = iteratorWrite->second->GetArgumentClassServices()->GetName();
        ITERATE_INTERFACE_END(Write);

        // 3-3) Command: Read
        ITERATE_INTERFACE_BEGIN(Read);
            info.ArgumentTypeName = iteratorRead->second->GetArgumentClassServices()->GetName();
        ITERATE_INTERFACE_END(Read);

        // 3-4) Command: QualifiedRead
        ITERATE_INTERFACE_BEGIN(QualifiedRead);
            info.Argument1TypeName = iteratorQualifiedRead->second->GetArgument1Prototype()->Services()->GetName();
            info.Argument2TypeName = iteratorQualifiedRead->second->GetArgument2Prototype()->Services()->GetName();
        ITERATE_INTERFACE_END(QualifiedRead);

#undef ITERATE_INTERFACE_BEGIN
#undef ITERATE_INTERFACE_END

        // TODO: 
        // 4) Extract events information (void, write)

        specs.push_back(providedInterfaceSpec);
    }

    return true;
}

/*
void mtsDeviceInterfaceProxyServer::SendCommandProxyInfo(
    const ::mtsTaskInterfaceProxy::CommandProxyInfo & info) const
{
    ConnectedTask->ReceiveCommandProxyInfo(info);
}
*/

void mtsDeviceInterfaceProxyServer::ExecuteCommandVoid(const int commandSID) const
{    
    mtsCommandVoidBase * commandVoid = reinterpret_cast<mtsCommandVoidBase *>(commandSID);
    CMN_ASSERT(commandVoid);

    commandVoid->Execute();
}

void mtsDeviceInterfaceProxyServer::ExecuteCommandWrite(const int commandSID, const double argument) const
{    
    mtsCommandWriteBase * commandWrite = reinterpret_cast<mtsCommandWriteBase *>(commandSID);
    CMN_ASSERT(commandWrite);

    static char buf[100];
    sprintf(buf, "ExecuteCommandWrite: %f", argument);
    Logger->trace("TIServer", buf);

    cmnDouble argumentWrapper(argument);
    commandWrite->Execute(argumentWrapper);
}

void mtsDeviceInterfaceProxyServer::ExecuteCommandRead(const int commandSID, double & argument)
{    
    mtsCommandReadBase * commandRead = reinterpret_cast<mtsCommandReadBase *>(commandSID);
    CMN_ASSERT(commandRead);

    cmnDouble argumentWrapper;
    commandRead->Execute(argumentWrapper);
    argument = argumentWrapper.Data;

    static char buf[100];
    sprintf(buf, "ExecuteCommandRead returns: %f", argument);
    Logger->trace("TIServer", buf);
}

void mtsDeviceInterfaceProxyServer::ExecuteCommandQualifiedRead(const int commandSID, const double argument1, double & argument2)
{    
    mtsCommandQualifiedReadBase * commandQualifiedRead = reinterpret_cast<mtsCommandQualifiedReadBase *>(commandSID);
    CMN_ASSERT(commandQualifiedRead);

    cmnDouble argument1Wrapper(argument1);
    cmnDouble argument2Wrapper;
    commandQualifiedRead->Execute(argument1Wrapper, argument2Wrapper);
    argument2 = argument2Wrapper.Data;
}

//-------------------------------------------------------------------------
//  Definition by mtsTaskManagerProxy.ice
//-------------------------------------------------------------------------
mtsDeviceInterfaceProxyServer::TaskInterfaceServerI::TaskInterfaceServerI(
    const Ice::CommunicatorPtr& communicator,
    const Ice::LoggerPtr& logger,
    mtsDeviceInterfaceProxyServer * taskInterfaceServer) 
    : Communicator(communicator), Logger(logger),
      TaskInterfaceServer(taskInterfaceServer),
      Runnable(true),
      Sender(new SendThread<TaskInterfaceServerIPtr>(this))
{
}

void mtsDeviceInterfaceProxyServer::TaskInterfaceServerI::Start()
{
    mtsDeviceInterfaceProxyServerLogger("Send thread starts");

    Sender->start();
}

void mtsDeviceInterfaceProxyServer::TaskInterfaceServerI::Run()
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

void mtsDeviceInterfaceProxyServer::TaskInterfaceServerI::Destroy()
{
    mtsDeviceInterfaceProxyServerLogger("Send thread is terminating.");

    IceUtil::ThreadPtr callbackSenderThread;

    {
        IceUtil::Monitor<IceUtil::Mutex>::Lock lock(*this);

        mtsDeviceInterfaceProxyServerLogger("Destroying sender.");
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
void mtsDeviceInterfaceProxyServer::TaskInterfaceServerI::AddClient(
    const ::Ice::Identity& ident, const ::Ice::Current& current)
{
    IceUtil::Monitor<IceUtil::Mutex>::Lock lock(*this);

    Logger->trace("TIServer", "<<<<< RECV: AddClient: " + Communicator->identityToString(ident));

    mtsTaskInterfaceProxy::TaskInterfaceClientPrx client = 
        mtsTaskInterfaceProxy::TaskInterfaceClientPrx::uncheckedCast(current.con->createProxy(ident));
    _clients.insert(client);
}

bool mtsDeviceInterfaceProxyServer::TaskInterfaceServerI::GetProvidedInterfaceSpecification(
    ::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq & specs, 
    const ::Ice::Current& current) const
{
    Logger->trace("TIServer", "<<<<< RECV: GetProvidedInterfaceSpecification");

    return TaskInterfaceServer->GetProvidedInterfaceSpecification(specs);
}

/*
void mtsDeviceInterfaceProxyServer::TaskInterfaceServerI::SendCommandProxyInfo(
    const ::mtsTaskInterfaceProxy::CommandProxyInfo & info,
    const ::Ice::Current&)
{
    Logger->trace("TIServer", "<<<<< RECV: SendCommandProxyInfo");

    TaskInterfaceServer->SendCommandProxyInfo(info);
}
*/

void mtsDeviceInterfaceProxyServer::TaskInterfaceServerI::ExecuteCommandVoid(
    ::Ice::Int sid, const ::Ice::Current&)
{
    //Logger->trace("TIServer", "<<<<< RECV: ExecuteCommandVoid");

    TaskInterfaceServer->ExecuteCommandVoid(sid);
}

void mtsDeviceInterfaceProxyServer::TaskInterfaceServerI::ExecuteCommandWrite(
    ::Ice::Int sid, ::Ice::Double argument, const ::Ice::Current&)
{
    //Logger->trace("TIServer", "<<<<< RECV: ExecuteCommandWrite");

    TaskInterfaceServer->ExecuteCommandWrite(sid, argument);
}

void mtsDeviceInterfaceProxyServer::TaskInterfaceServerI::ExecuteCommandRead(
    ::Ice::Int sid, ::Ice::Double& argument, const ::Ice::Current&)
{
    //Logger->trace("TIServer", "<<<<< RECV: ExecuteCommandRead");

    TaskInterfaceServer->ExecuteCommandRead(sid, argument);
}

void mtsDeviceInterfaceProxyServer::TaskInterfaceServerI::ExecuteCommandQualifiedRead(
    ::Ice::Int sid, ::Ice::Double argument1, ::Ice::Double& argument2, const ::Ice::Current&)
{
    //Logger->trace("TIServer", "<<<<< RECV: ExecuteCommandQualifiedRead");

    TaskInterfaceServer->ExecuteCommandQualifiedRead(sid, argument1, argument2);
}