/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsTaskManagerProxyClient.cpp 145 2009-03-18 23:32:40Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-03-17

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstOSAbstraction/osaSleep.h>
#include <cisstMultiTask/mtsTaskManager.h>
#include <cisstMultiTask/mtsTaskManagerProxyClient.h>

CMN_IMPLEMENT_SERVICES(mtsTaskManagerProxyClient);

#define TaskManagerProxyClientLogger(_log) BaseType::Logger->trace("mtsTaskManagerProxyClient", _log)
#define TaskManagerProxyClientLoggerError(_log1, _log2) \
    {   std::stringstream s;\
        s << "mtsTaskManagerProxyClient: " << _log1 << _log2;\
        BaseType::Logger->error(s.str());  }

mtsTaskManagerProxyClient::mtsTaskManagerProxyClient(
    const std::string & propertyFileName, const std::string & propertyName) :
        BaseType(propertyFileName, propertyName)
{
}

mtsTaskManagerProxyClient::~mtsTaskManagerProxyClient()
{
}

void mtsTaskManagerProxyClient::Start(mtsTaskManager * callingTaskManager)
{
    // Initialize Ice object.
    // Notice that a worker thread is not created right now.
    Init();
    
    if (InitSuccessFlag) {
        // Client configuration for bidirectional communication
        // (see http://www.zeroc.com/doc/Ice-3.3.1/manual/Connections.38.7.html)
        Ice::ObjectAdapterPtr adapter = IceCommunicator->createObjectAdapter("");
        Ice::Identity ident;
        ident.name = GetGUID();
        ident.category = "";    // not used currently.

        mtsTaskManagerProxy::TaskManagerClientPtr client = 
            new TaskManagerClientI(IceCommunicator, Logger, GlobalTaskManagerProxy, this);
        adapter->add(client, ident);
        adapter->activate();
        GlobalTaskManagerProxy->ice_getConnection()->setAdapter(adapter);

        // Set an implicit context (per proxy context)
        // (see http://www.zeroc.com/doc/Ice-3.3.1/manual/Adv_server.33.12.html)
        IceCommunicator->getImplicitContext()->put(CONNECTION_ID, 
            IceCommunicator->identityToString(ident));

        // Generate an event so that the global task manager register this task manager.
        GlobalTaskManagerProxy->AddClient(ident);

        // Create a worker thread here and returns immediately.
        ThreadArgumentsInfo.argument = callingTaskManager;
        ThreadArgumentsInfo.proxy = this;        
        ThreadArgumentsInfo.Runner = mtsTaskManagerProxyClient::Runner;

        WorkerThread.Create<ProxyWorker<mtsTaskManager>, ThreadArguments<mtsTaskManager>*>(
            &ProxyWorkerInfo, &ProxyWorker<mtsTaskManager>::Run, &ThreadArgumentsInfo, "S-PRX");
    }
}

void mtsTaskManagerProxyClient::StartClient()
{
    Sender->Start();

    // This is a blocking call that should run in a different thread.
    IceCommunicator->waitForShutdown();
}

void mtsTaskManagerProxyClient::Runner(ThreadArguments<mtsTaskManager> * arguments)
{
    mtsTaskManager * TaskManager = reinterpret_cast<mtsTaskManager*>(arguments->argument);

    mtsTaskManagerProxyClient * ProxyClient = 
        dynamic_cast<mtsTaskManagerProxyClient*>(arguments->proxy);

    ProxyClient->GetLogger()->trace("mtsTaskManagerProxyClient", "Proxy client starts.");

    try {
        ProxyClient->StartClient();        
    } catch (const Ice::Exception& e) {
        ProxyClient->GetLogger()->trace("mtsTaskManagerProxyClient exception: ", e.what());
    } catch (const char * msg) {
        ProxyClient->GetLogger()->trace("mtsTaskManagerProxyClient exception: ", msg);
    }

    ProxyClient->OnThreadEnd();
}

void mtsTaskManagerProxyClient::Stop()
{
    OnThreadEnd();
}

void mtsTaskManagerProxyClient::OnThreadEnd()
{
    TaskManagerProxyClientLogger("Proxy client ends.");

    BaseType::OnThreadEnd();

    Sender->Stop();
}

//-----------------------------------------------------------------------------
//  Processing Methods
//-----------------------------------------------------------------------------
mtsDeviceInterface * mtsTaskManagerProxyClient::GetResourceInterface(
    const std::string & resourceTaskName, const std::string & providedInterfaceName,
    const std::string & userTaskName, const std::string & requiredInterfaceName,
    mtsTask * userTask)
{
    /*
    mtsDeviceInterface * resourceInterface = NULL;

    // For the use of consistent notation
    mtsTask * clientTask = userTask;

    // Ask the global task manager (TMServer) if the specific task specified 
    // the specific provided interface has been registered.
    if (!IsRegisteredProvidedInterface(resourceTaskName, providedInterfaceName)) {
        CMN_LOG_CLASS_RUN_ERROR << "Connect across networks: '" << providedInterfaceName << "' has not been registered." << resourceTaskName << ", " << std::endl;
        return NULL;
    }

    // If (task, provided interface) exists,
    // 1) Retrieve information from the global task manager to connect
    //    the requested provided interface (mtsDeviceInterfaceProxyServer).                
    mtsTaskManagerProxy::ProvidedInterfaceInfo info;
    if (!GetProvidedInterfaceInfo(resourceTaskName, providedInterfaceName, info)) {
        CMN_LOG_CLASS_RUN_ERROR << "Connect across networks: failed to retrieve proxy information: " << resourceTaskName << ", " << providedInterfaceName << std::endl;
        return NULL;
    }

    // 2) Using the information, start a proxy client (=server proxy, mtsDeviceInterfaceProxyClient object).
    clientTask->StartProxyClient(info.endpointInfo, info.communicatorID);

    //
    // TODO: Does ICE allow a user to register a callback function? (e.g. OnConnect())
    //       If it does, we can remove the following line.
    //
    osaSleep(1*cmn_s);

    // 3) From the server task, get the complete information on the provided 
    //    interface as a set of string.
    mtsDeviceInterfaceProxy::ProvidedInterfaceSequence providedInterfaces;
    if (!clientTask->GetProvidedInterfaces(providedInterfaces)) {
        CMN_LOG_CLASS_RUN_ERROR << "Connect across networks: failed to retrieve provided interface specification: " << resourceTaskName << ", " << providedInterfaceName << std::endl;
        return NULL;
    }

    // 4) Create a server task proxy that has a provided interface proxy.
    // 
    // TODO: MJUNG: this loop has to be refactored to remove duplicity.
    // (see mtsDeviceInterfaceProxyClient::ReceiveConnectServerSide())
    //    
    std::vector<mtsDeviceInterfaceProxy::ProvidedInterface>::const_iterator it
        = providedInterfaces.begin();
    for (; it != providedInterfaces.end(); ++it) {
        //
        //!!!!!!!!!!!!!!!!
        //
        //CMN_ASSERT(providedInterfaceName == it->InterfaceName);
        if (providedInterfaceName != it->InterfaceName) continue;

        // Create a server task proxy of which name follows the naming rule above.
        // (see mtsDeviceProxy.h as to why serverTaskProxy is of mtsDevice type, not
        // of mtsTask.)
        std::string serverTaskProxyName = mtsDeviceProxy::GetServerTaskProxyName(
            resourceTaskName, providedInterfaceName, userTaskName, requiredInterfaceName);
        mtsDeviceProxy * serverTaskProxy = new mtsDeviceProxy(serverTaskProxyName);

        // Create a provided interface proxy using the information received from the 
        // server task.
        if (!CreateProvidedInterfaceProxy(*it, serverTaskProxy, clientTask)) {
            CMN_LOG_CLASS_RUN_ERROR << "Connect across networks: failed to create a server task proxy: " << serverTaskProxyName << std::endl;
            return NULL;
        }

        // Add the proxy task to the local task manager
        if (!AddDevice(serverTaskProxy)) {
            CMN_LOG_CLASS_RUN_ERROR << "Connect across networks: failed to add a server task proxy: " << serverTaskProxyName << std::endl;
            return NULL;
        }

        // Return a pointer to the provided interface proxy as if the interface was initially
        // created in client's local memory space.
        resourceInterface = serverTaskProxy->GetProvidedInterface(providedInterfaceName);

        //
        // TODO: Currently, it is assumed that there is only one provided interface.
        //
        return resourceInterface;
    }

    // The following line should not be reached.
    CMN_ASSERT(false);
    */

    return NULL;
}

//
// TODO: Move this method to mtsDeviceInterfaceProxyServer class.
//
bool mtsTaskManagerProxyClient::CreateProvidedInterfaceProxy(
    const mtsDeviceInterfaceProxy::ProvidedInterface & providedInterface,
    mtsDevice * serverTaskProxy, mtsTask * clientTask)
{
    /*
    // 1) Create a local provided interface (a provided interface proxy).
    mtsDeviceInterface * providedInterfaceProxy = serverTaskProxy->AddProvidedInterface(providedInterface.InterfaceName);
    if (!providedInterfaceProxy) {
        CMN_LOG_CLASS_RUN_ERROR << "CreateProvidedInterfaceProxy: Could not add provided interface: " 
                                << providedInterface.InterfaceName << std::endl;
        return false;
    }

    // 2) Create command proxies.
    // CommandId is initially set to zero meaning that it needs to be updated.
    // An actual value will be assigned later when UpdateCommandId() is executed.
    int commandId = NULL;
    std::string commandName, eventName;

#define ADD_COMMANDS_BEGIN(_commandType) \
    {\
        mtsCommand##_commandType##Proxy * newCommand##_commandType = NULL;\
        mtsDeviceInterfaceProxy::Command##_commandType##Sequence::const_iterator it\
            = providedInterface.Commands##_commandType.begin();\
        for (; it != providedInterface.Commands##_commandType.end(); ++it) {\
            commandName = it->Name;
#define ADD_COMMANDS_END \
        }\
    }

    // 2-1) Void
    ADD_COMMANDS_BEGIN(Void)
        newCommandVoid = new mtsCommandVoidProxy(
            commandId, clientTask->GetProxyClient(), commandName);
        CMN_ASSERT(newCommandVoid);
        providedInterfaceProxy->GetCommandVoidMap().AddItem(commandName, newCommandVoid);
    ADD_COMMANDS_END

    // 2-2) Write
    ADD_COMMANDS_BEGIN(Write)
        newCommandWrite = new mtsCommandWriteProxy(
            commandId, clientTask->GetProxyClient(), commandName);
        CMN_ASSERT(newCommandWrite);
        providedInterfaceProxy->GetCommandWriteMap().AddItem(commandName, newCommandWrite);
    ADD_COMMANDS_END

    // 2-3) Read
    ADD_COMMANDS_BEGIN(Read)
        newCommandRead = new mtsCommandReadProxy(
            commandId, clientTask->GetProxyClient(), commandName);
        CMN_ASSERT(newCommandRead);
        providedInterfaceProxy->GetCommandReadMap().AddItem(commandName, newCommandRead);
    ADD_COMMANDS_END

    // 2-4) QualifiedRead
    ADD_COMMANDS_BEGIN(QualifiedRead)
        newCommandQualifiedRead = new mtsCommandQualifiedReadProxy(
            commandId, clientTask->GetProxyClient(), commandName);
        CMN_ASSERT(newCommandQualifiedRead);
        providedInterfaceProxy->GetCommandQualifiedReadMap().AddItem(commandName, newCommandQualifiedRead);
    ADD_COMMANDS_END

    //{
    //    mtsFunctionVoid * newEventVoidGenerator = NULL;
    //    mtsDeviceInterfaceProxy::EventVoidSequence::const_iterator it =
    //        providedInterface.EventsVoid.begin();
    //    for (; it != providedInterface.EventsVoid.end(); ++it) {
    //        eventName = it->Name;            
    //        newEventVoidGenerator = new mtsFunctionVoid();
    //        newEventVoidGenerator->Bind(providedInterfaceProxy->AddEventVoid(eventName));            
    //    }
    //}
#define ADD_EVENTS_BEGIN(_eventType)\
    {\
        mtsFunction##_eventType * newEvent##_eventType##Generator = NULL;\
        mtsDeviceInterfaceProxy::Event##_eventType##Sequence::const_iterator it =\
        providedInterface.Events##_eventType.begin();\
        for (; it != providedInterface.Events##_eventType.end(); ++it) {\
            eventName = it->Name;
#define ADD_EVENTS_END \
        }\
    }

    // 3) Create event generator proxies.
    ADD_EVENTS_BEGIN(Void);
        newEventVoidGenerator = new mtsFunctionVoid();
        newEventVoidGenerator->Bind(providedInterfaceProxy->AddEventVoid(eventName));
    ADD_EVENTS_END;
    
    mtsMulticastCommandWriteProxy * newMulticastCommandWriteProxy = NULL;
    ADD_EVENTS_BEGIN(Write);
        newEventWriteGenerator = new mtsFunctionWrite();
        newMulticastCommandWriteProxy = new mtsMulticastCommandWriteProxy(
            it->Name, it->ArgumentTypeName);
        CMN_ASSERT(providedInterfaceProxy->AddEvent(it->Name, newMulticastCommandWriteProxy));
        CMN_ASSERT(newEventWriteGenerator->Bind(newMulticastCommandWriteProxy));
    ADD_EVENTS_END;

#undef ADD_COMMANDS_BEGIN
#undef ADD_COMMANDS_END
#undef ADD_EVENTS_BEGIN
#undef ADD_EVENTS_END
    */

    return true;
}

void mtsTaskManagerProxyClient::UpdateCommandId(
    mtsDeviceInterfaceProxy::FunctionProxySet functionProxies)
{
    /*
    const std::string serverTaskProxyName = functionProxies.ServerTaskProxyName;
    mtsDevice * serverTaskProxy = GetDevice(serverTaskProxyName);
    CMN_ASSERT(serverTaskProxy);

    mtsProvidedInterface * providedInterfaceProxy = 
        serverTaskProxy->GetProvidedInterface(functionProxies.ProvidedInterfaceProxyName);
    CMN_ASSERT(providedInterfaceProxy);

    //mtsCommandVoidProxy * commandVoid = NULL;
    //mtsDeviceInterfaceProxy::FunctionProxySequence::const_iterator it = 
    //    functionProxies.FunctionVoidProxies.begin();
    //for (; it != functionProxies.FunctionVoidProxies.end(); ++it) {
    //    commandVoid = dynamic_cast<mtsCommandVoidProxy*>(
    //        providedInterfaceProxy->GetCommandVoid(it->Name));
    //    CMN_ASSERT(commandVoid);
    //    commandVoid->SetCommandId(it->FunctionProxyPointer);
    //}

    // Replace a command id value with an actual pointer to the function
    // pointer at server side (this resolves thread synchronization issue).
#define REPLACE_COMMAND_ID(_commandType)\
    mtsCommand##_commandType##Proxy * command##_commandType = NULL;\
    mtsDeviceInterfaceProxy::FunctionProxySequence::const_iterator it##_commandType = \
        functionProxies.Function##_commandType##Proxies.begin();\
    for (; it##_commandType != functionProxies.Function##_commandType##Proxies.end(); ++it##_commandType) {\
        command##_commandType = dynamic_cast<mtsCommand##_commandType##Proxy*>(\
            providedInterfaceProxy->GetCommand##_commandType(it##_commandType->Name));\
        if (command##_commandType)\
            command##_commandType->SetCommandId(it##_commandType->FunctionProxyPointer);\
    }

    REPLACE_COMMAND_ID(Void);
    REPLACE_COMMAND_ID(Write);
    REPLACE_COMMAND_ID(Read);
    REPLACE_COMMAND_ID(QualifiedRead);
    */
}

//-------------------------------------------------------------------------
//  Send Methods
//-------------------------------------------------------------------------
bool mtsTaskManagerProxyClient::SendAddProvidedInterface(
    const std::string & newProvidedInterfaceName,
    const std::string & adapterName,
    const std::string & endpointInfo,
    const std::string & communicatorID,
    const std::string & taskName)
{
    ::mtsTaskManagerProxy::ProvidedInterfaceInfo info;
    info.adapterName = adapterName;
    info.endpointInfo = endpointInfo;
    info.communicatorID = communicatorID;
    info.taskName = taskName;
    info.interfaceName = newProvidedInterfaceName;

    GetLogger()->trace("TMClient", ">>>>> SEND: AddProvidedInterface: " 
        + info.taskName + ", " + info.interfaceName);

    return GlobalTaskManagerProxy->AddProvidedInterface(info);
}

bool mtsTaskManagerProxyClient::SendAddRequiredInterface(
    const std::string & newRequiredInterfaceName, const std::string & taskName)
{
    ::mtsTaskManagerProxy::RequiredInterfaceInfo info;
    info.taskName = taskName;
    info.interfaceName = newRequiredInterfaceName;

    GetLogger()->trace("TMClient", ">>>>> SEND: AddRequiredInterface: " 
        + info.taskName + ", " + info.interfaceName);

    return GlobalTaskManagerProxy->AddRequiredInterface(info);
}

bool mtsTaskManagerProxyClient::SendIsRegisteredProvidedInterface(
    const std::string & taskName, const std::string & providedInterfaceName) const
{
    GetLogger()->trace("TMClient", ">>>>> SEND: IsRegisteredProvidedInterface: " 
        + taskName + ", " + providedInterfaceName);

    return GlobalTaskManagerProxy->IsRegisteredProvidedInterface(
        taskName, providedInterfaceName);
}

bool mtsTaskManagerProxyClient::SendGetProvidedInterfaceInfo(
    const ::std::string & taskName, const std::string & providedInterfaceName,
    ::mtsTaskManagerProxy::ProvidedInterfaceInfo & info) const
{
    GetLogger()->trace("TMClient", ">>>>> SEND: GetProvidedInterfaceInfo: " 
        + taskName + ", " + providedInterfaceName);

    return GlobalTaskManagerProxy->GetProvidedInterfaceInfo(
        taskName, providedInterfaceName, info);
}

//void mtsTaskManagerProxyClient::SendNotifyInterfaceConnectionResult(
//    const bool isServerTask, const bool isSuccess,
//    const std::string & userTaskName,     const std::string & requiredInterfaceName,
//    const std::string & resourceTaskName, const std::string & providedInterfaceName)
//{
//    GetLogger()->trace("TMClient", ">>>>> SEND: NotifyInterfaceConnectionResult: " +
//        resourceTaskName + " : " + providedInterfaceName + " - " +
//        userTaskName + " : " + requiredInterfaceName);
//
//    return GlobalTaskManagerProxy->NotifyInterfaceConnectionResult(
//        isServerTask, isSuccess, 
//        userTaskName, requiredInterfaceName, resourceTaskName, providedInterfaceName);
//}

//-------------------------------------------------------------------------
//  Definition by mtsTaskManagerProxy.ice
//-------------------------------------------------------------------------
mtsTaskManagerProxyClient::TaskManagerClientI::TaskManagerClientI(
    const Ice::CommunicatorPtr& communicator,                           
    const Ice::LoggerPtr& logger,
    const mtsTaskManagerProxy::TaskManagerServerPrx& server,
    mtsTaskManagerProxyClient * taskManagerClient)
    : Runnable(true), 
      Communicator(communicator), Logger(logger),
      Server(server), TaskManagerClient(taskManagerClient),      
      Sender(new SendThread<TaskManagerClientIPtr>(this))      
{
}

void mtsTaskManagerProxyClient::TaskManagerClientI::Start()
{
    CMN_LOG_RUN_VERBOSE << "TaskManagerProxyClient: Send thread starts" << std::endl;

    Sender->start();
}

void mtsTaskManagerProxyClient::TaskManagerClientI::Run()
{
    bool flag = true;

    while(Runnable)
    {
#ifdef _COMMUNICATION_TEST_
        static int num = 0;
        std::cout << "client send: " << ++num << std::endl;
        Server->ReceiveDataFromClient(num);
#endif

        if (flag) {
            // Send a set of task names
            mtsTaskManagerProxy::TaskList localTaskList;
            std::vector<std::string> myTaskNames;
            mtsTaskManager::GetInstance()->GetNamesOfTasks(myTaskNames);

            localTaskList.taskNames.insert(
                localTaskList.taskNames.end(),
                myTaskNames.begin(),
                myTaskNames.end());

            localTaskList.taskManagerID = TaskManagerClient->GetGUID();

            Server->UpdateTaskManager(localTaskList);

            flag = false;
        }

        timedWait(IceUtil::Time::milliSeconds(10));
    }
}

void mtsTaskManagerProxyClient::TaskManagerClientI::Stop()
{
    CMN_LOG_RUN_VERBOSE << "TaskManagerProxyClient: Send thread is terminating." << std::endl;

    IceUtil::ThreadPtr callbackSenderThread;

    {
        //IceUtil::Monitor<IceUtil::Mutex>::Lock lock(*this);

        CMN_LOG_RUN_VERBOSE << "TaskManagerProxyClient: Destroying sender." << std::endl;
        Runnable = false;

        notify();

        callbackSenderThread = Sender;
        Sender = 0; // Resolve cyclic dependency.
    }

    callbackSenderThread->getThreadControl().join();
}

// for test purpose
void mtsTaskManagerProxyClient::TaskManagerClientI::ReceiveData(
    ::Ice::Int num, const ::Ice::Current&)
{
    std::cout << "------------ client recv data " << num << std::endl;
}
