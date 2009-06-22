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
            new TaskManagerClientI(IceCommunicator, Logger, TaskManagerServer, this);
        adapter->add(client, ident);
        adapter->activate();
        TaskManagerServer->ice_getConnection()->setAdapter(adapter);

        // Set an implicit context (per proxy context)
        // (see http://www.zeroc.com/doc/Ice-3.3.1/manual/Adv_server.33.12.html)
        IceCommunicator->getImplicitContext()->put(CONNECTION_ID, 
            IceCommunicator->identityToString(ident));

        // Generate an event so that the global task manager register this task manager.
        TaskManagerServer->AddClient(ident);

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

void mtsTaskManagerProxyClient::OnThreadEnd()
{
    TaskManagerProxyClientLogger("Proxy client ends.");

    BaseType::OnThreadEnd();

    Sender->Destroy();
}

//-----------------------------------------------------------------------------
//  Proxy Client Implementation
//-----------------------------------------------------------------------------
//
// TODO: mtsTaskManager::GetResourceInterface() Refactoring!!! 
// (move to mtsTaskManagerProxyServer, renaming)
//
/*
bool mtsTaskManagerProxyClient::ReceiveConnectServerSide(
    const std::string & userTaskName,     const std::string & requiredInterfaceName,
    const std::string & resourceTaskName, const std::string & providedInterfaceName)
{
    const std::string clientTaskProxyName = mtsDeviceProxy::GetClientTaskProxyName(
        resourceTaskName, providedInterfaceName, userTaskName, requiredInterfaceName);

    // Get an original provided interface.
    mtsProvidedInterface * providedInterface = GetProvidedInterface(
        resourceTaskName, providedInterfaceName);
    if (!providedInterface) {
        TaskManagerProxyClientLoggerError("Connect across networks: cannot find a provided interface: ", providedInterface);
        return false;
    }

    // Create a client task proxy (mtsDevice) and a required Interface proxy (mtsRequiredInterface)
    mtsDeviceProxy * clientTaskProxy = new mtsDeviceProxy(clientTaskProxyName);
    mtsRequiredInterface * requiredInterfaceProxy = 
        clientTaskProxy->AddRequiredInterface(requiredInterfaceName);
    if (!requiredInterfaceProxy) {
        TaskManagerProxyClientLoggerError("ReceiveConnectServerSide: cannot add required interface: ", requiredInterfaceName);
        return false;
    }

    // Populate a required Interface proxy 
    if (!PopulateRequiredInterfaceProxy(requiredInterfaceProxy, providedInterface)) {
        TaskManagerProxyClientLoggerError("Connect across networks: failed to create a client task proxy: ", clientTaskProxyName);
        return false;
    }

    // Connect() locally


    // 4. Return the result back to server

    return true;
}
*/

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

    return TaskManagerServer->AddProvidedInterface(info);
}

bool mtsTaskManagerProxyClient::SendAddRequiredInterface(
    const std::string & newRequiredInterfaceName, const std::string & taskName)
{
    ::mtsTaskManagerProxy::RequiredInterfaceInfo info;
    info.taskName = taskName;
    info.interfaceName = newRequiredInterfaceName;

    GetLogger()->trace("TMClient", ">>>>> SEND: AddRequiredInterface: " 
        + info.taskName + ", " + info.interfaceName);

    return TaskManagerServer->AddRequiredInterface(info);
}

bool mtsTaskManagerProxyClient::SendIsRegisteredProvidedInterface(
    const std::string & taskName, const std::string & providedInterfaceName) const
{
    GetLogger()->trace("TMClient", ">>>>> SEND: IsRegisteredProvidedInterface: " 
        + taskName + ", " + providedInterfaceName);

    return TaskManagerServer->IsRegisteredProvidedInterface(
        taskName, providedInterfaceName);
}

bool mtsTaskManagerProxyClient::SendGetProvidedInterfaceInfo(
    const ::std::string & taskName, const std::string & providedInterfaceName,
    ::mtsTaskManagerProxy::ProvidedInterfaceInfo & info) const
{
    GetLogger()->trace("TMClient", ">>>>> SEND: GetProvidedInterfaceInfo: " 
        + taskName + ", " + providedInterfaceName);

    return TaskManagerServer->GetProvidedInterfaceInfo(
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
//    return TaskManagerServer->NotifyInterfaceConnectionResult(
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

    while(true)
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

        {
            IceUtil::Monitor<IceUtil::Mutex>::Lock lock(*this);
            timedWait(IceUtil::Time::seconds(2));
        }
    }
}

void mtsTaskManagerProxyClient::TaskManagerClientI::Destroy()
{
    CMN_LOG_RUN_VERBOSE << "TaskManagerProxyClient: Send thread is terminating." << std::endl;

    IceUtil::ThreadPtr callbackSenderThread;

    {
        IceUtil::Monitor<IceUtil::Mutex>::Lock lock(*this);

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


//bool mtsTaskManagerProxyClient::TaskManagerClientI::ConnectServerSide(
//    const std::string & userTaskName, const std::string & requiredInterfaceName,
//    const std::string & resourceTaskName, const std::string & providedInterfaceName,
//    const ::Ice::Current & current)
//{
//    Logger->trace("TMClient", "<<<<< RECV: ConnectServerSide: " +
//        userTaskName + ":" + requiredInterfaceName + "-" + resourceTaskName + ":" + providedInterfaceName);
//
//    CMN_ASSERT(TaskManagerClient);
//
//    return TaskManagerClient->ReceiveConnectServerSide(
//        userTaskName, requiredInterfaceName, resourceTaskName, providedInterfaceName);
//}