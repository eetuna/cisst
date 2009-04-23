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

#include <cisstMultiTask/mtsTaskManagerProxyClient.h>
#include <cisstOSAbstraction/osaSleep.h>

CMN_IMPLEMENT_SERVICES(mtsTaskManagerProxyClient);

#define mtsTaskManagerProxyClientLogger(_log) \
    Logger->trace("mtsTaskManagerProxyClient", _log)

//#define _COMMUNICATION_TEST_

void mtsTaskManagerProxyClient::Start(mtsTaskManager * callingTaskManager)
{
    // Initialize Ice object.
    // Notice that a worker thread is not created right now.
    Init();
    
    if (InitSuccessFlag) {
        // Client configuration for bidirectional communication
        // (see 
        // http://www.zeroc.com/doc/Ice-3.3.1/manual/Connections.38.7.html
        // for more information.)
        Ice::ObjectAdapterPtr adapter = IceCommunicator->createObjectAdapter("");
        Ice::Identity ident;
        ident.name = GetGUID();
        ident.category = "";

        mtsTaskManagerProxy::TaskManagerClientPtr client = 
            new TaskManagerClientI(IceCommunicator, Logger, TaskManagerServer, this);
        adapter->add(client, ident);
        adapter->activate();
        TaskManagerServer->ice_getConnection()->setAdapter(adapter);
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

    try {
        ProxyClient->StartClient();        
    } catch (const Ice::Exception& e) {
        ProxyClient->GetLogger()->trace("mtsTaskManagerProxyClient", "exception");
        ProxyClient->GetLogger()->trace("mtsTaskManagerProxyClient", e.what());
    } catch (const char * msg) {
        ProxyClient->GetLogger()->trace("mtsTaskManagerProxyClient", "exception");
        ProxyClient->GetLogger()->trace("mtsTaskManagerProxyClient", msg);
    }

    ProxyClient->OnThreadEnd();
}

void mtsTaskManagerProxyClient::OnThreadEnd()
{
    mtsProxyBaseClient::OnThreadEnd();

    Sender->Destroy();
}

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
    mtsTaskManagerProxyClientLogger("Send thread starts");

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
            mtsTaskManagerProxy::TaskInfo localTaskInfo;
            std::vector<std::string> myTaskNames;
            mtsTaskManager::GetInstance()->GetNamesOfTasks(myTaskNames);

            localTaskInfo.taskNames.insert(
                localTaskInfo.taskNames.end(),
                myTaskNames.begin(),
                myTaskNames.end());

            localTaskInfo.taskManagerID = TaskManagerClient->GetGUID();

            Server->UpdateTaskInfo(localTaskInfo);

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
    mtsTaskManagerProxyClientLogger("Send thread is terminating.");

    IceUtil::ThreadPtr callbackSenderThread;

    {
        IceUtil::Monitor<IceUtil::Mutex>::Lock lock(*this);

        mtsTaskManagerProxyClientLogger("Destroying sender.");
        Runnable = false;

        notify();

        callbackSenderThread = Sender;
        Sender = 0; // Resolve cyclic dependency.
    }

    callbackSenderThread->getThreadControl().join();
}

void mtsTaskManagerProxyClient::TaskManagerClientI::ReceiveData(
    ::Ice::Int num, const ::Ice::Current&)
{
    std::cout << "------------ client recv data " << num << std::endl;
}
