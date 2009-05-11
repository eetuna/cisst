/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsDeviceInterfaceProxyClient.cpp 145 2009-03-18 23:32:40Z mjung5 $

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

#include <cisstMultiTask/mtsDeviceInterfaceProxyClient.h>
#include <cisstOSAbstraction/osaSleep.h>

CMN_IMPLEMENT_SERVICES(mtsDeviceInterfaceProxyClient);

//
// TODO: Make a file logger which is compatible with CISST logger or ICE logger
//
#define mtsDeviceInterfaceProxyClientLogger(_log) \
            Logger->trace("mtsDeviceInterfaceProxyClient", _log);

void mtsDeviceInterfaceProxyClient::Start(mtsTask * callingTask)
{
    // Initialize Ice object.
    // Notice that a worker thread is not created right now.
    Init();
    
    if (InitSuccessFlag) {
        // Client configuration for bidirectional communication
        Ice::ObjectAdapterPtr adapter = IceCommunicator->createObjectAdapter("");
        Ice::Identity ident;
        ident.name = GetGUID();
        ident.category = "";

        mtsDeviceInterfaceProxy::TaskInterfaceClientPtr client = 
            new TaskInterfaceClientI(IceCommunicator, Logger, TaskInterfaceServer, this);
        adapter->add(client, ident);
        adapter->activate();
        TaskInterfaceServer->ice_getConnection()->setAdapter(adapter);
        TaskInterfaceServer->AddClient(ident);

        // Create a worker thread here and returns immediately.
        ThreadArgumentsInfo.argument = callingTask;
        ThreadArgumentsInfo.proxy = this;        
        ThreadArgumentsInfo.Runner = mtsDeviceInterfaceProxyClient::Runner;

        WorkerThread.Create<ProxyWorker<mtsTask>, ThreadArguments<mtsTask>*>(
            &ProxyWorkerInfo, &ProxyWorker<mtsTask>::Run, &ThreadArgumentsInfo, "S-PRX");
    }
}

void mtsDeviceInterfaceProxyClient::StartClient()
{
    Sender->Start();

    // This is a blocking call that should run in a different thread.
    IceCommunicator->waitForShutdown();
}

void mtsDeviceInterfaceProxyClient::Runner(ThreadArguments<mtsTask> * arguments)
{
    mtsTaskManager * TaskManager = reinterpret_cast<mtsTaskManager*>(arguments->argument);

    mtsDeviceInterfaceProxyClient * ProxyClient = 
        dynamic_cast<mtsDeviceInterfaceProxyClient*>(arguments->proxy);

    ProxyClient->GetLogger()->trace("mtsDeviceInterfaceProxyClient", "Proxy client starts.");

    try {
        ProxyClient->StartClient();        
    } catch (const Ice::Exception& e) {
        ProxyClient->GetLogger()->trace("mtsDeviceInterfaceProxyClient exception: ", e.what());
    } catch (const char * msg) {
        ProxyClient->GetLogger()->trace("mtsDeviceInterfaceProxyClient exception: ", msg);
    }

    ProxyClient->OnThreadEnd();
}

void mtsDeviceInterfaceProxyClient::OnThreadEnd()
{
    mtsDeviceInterfaceProxyClientLogger("Proxy client ends.");

    mtsProxyBaseClient::OnThreadEnd();

    Sender->Destroy();
}

//-------------------------------------------------------------------------
//  Send Methods
//-------------------------------------------------------------------------
const bool mtsDeviceInterfaceProxyClient::GetProvidedInterfaceSpecification(
        mtsDeviceInterfaceProxy::ProvidedInterfaceSpecificationSeq & specs) const
{
    GetLogger()->trace("TIClient", ">>>>> SEND: GetProvidedInterfaceSpecification");

    return TaskInterfaceServer->GetProvidedInterfaceSpecification(specs);
}

/*
void mtsDeviceInterfaceProxyClient::SendCommandProxyInfo(
    mtsDeviceInterfaceProxy::CommandProxyInfo & info) const
{
    GetLogger()->trace("TIClient", ">>>>> SEND: SendCommandProxyInfo");

    TaskInterfaceServer->SendCommandProxyInfo(info);
}
*/

void mtsDeviceInterfaceProxyClient::InvokeExecuteCommandVoid(const int commandSID) const
{
    //GetLogger()->trace("TIClient", ">>>>> SEND: InvokeExecuteCommandVoid");

    TaskInterfaceServer->ExecuteCommandVoid(commandSID);
}

void mtsDeviceInterfaceProxyClient::InvokeExecuteCommandWrite(
    const int commandSID, const cmnDouble & argument) const
{
    //GetLogger()->trace("TIClient", ">>>>> SEND: InvokeExecuteCommandWrite");

    double value = argument.Data;

    TaskInterfaceServer->ExecuteCommandWrite(commandSID, value);
}

void mtsDeviceInterfaceProxyClient::InvokeExecuteCommandRead(
    const int commandSID, cmnDouble & argument)
{
    //GetLogger()->trace("TIClient", ">>>>> SEND: InvokeExecuteCommandRead");

    double outValue = 0.0;

    TaskInterfaceServer->ExecuteCommandRead(commandSID, outValue);

    cmnDouble out(outValue);
    argument = out;
}

void mtsDeviceInterfaceProxyClient::InvokeExecuteCommandQualifiedRead(
    const int commandSID, const cmnDouble & argument1, cmnDouble & argument2)
{
    //GetLogger()->trace("TIClient", ">>>>> SEND: InvokeExecuteCommandQualifiedRead");
    
    double value = argument1.Data;
    double outValue = 0.0;

    TaskInterfaceServer->ExecuteCommandQualifiedRead(commandSID, value, outValue);

    cmnDouble out(outValue);
    argument2 = out;
}

//-------------------------------------------------------------------------
//  Definition by mtsDeviceInterfaceProxy.ice
//-------------------------------------------------------------------------
mtsDeviceInterfaceProxyClient::TaskInterfaceClientI::TaskInterfaceClientI(
    const Ice::CommunicatorPtr& communicator,                           
    const Ice::LoggerPtr& logger,
    const mtsDeviceInterfaceProxy::TaskInterfaceServerPrx& server,
    mtsDeviceInterfaceProxyClient * taskInterfaceClient)
    : Runnable(true), 
      Communicator(communicator), Logger(logger),
      Server(server), TaskInterfaceClient(taskInterfaceClient),      
      Sender(new SendThread<TaskInterfaceClientIPtr>(this))      
{
}

void mtsDeviceInterfaceProxyClient::TaskInterfaceClientI::Start()
{
    mtsDeviceInterfaceProxyClientLogger("Send thread starts");

    Sender->start();
}

void mtsDeviceInterfaceProxyClient::TaskInterfaceClientI::Run()
{
    bool flag = true;

    while(true)
    {
        if (flag) {
            // Send a set of task names
            /*
            mtsDeviceInterfaceProxy::TaskList localTaskList;
            std::vector<std::string> myTaskNames;
            mtsTaskManager::GetInstance()->GetNamesOfTasks(myTaskNames);

            localTaskList.taskNames.insert(
                localTaskList.taskNames.end(),
                myTaskNames.begin(),
                myTaskNames.end());

            localTaskList.taskManagerID = TaskManagerClient->GetGUID();

            Server->AddTaskManager(localTaskList);
            */

            flag = false;
        }

        {
            IceUtil::Monitor<IceUtil::Mutex>::Lock lock(*this);
            timedWait(IceUtil::Time::seconds(2));
        }
    }
}

void mtsDeviceInterfaceProxyClient::TaskInterfaceClientI::Destroy()
{
    mtsDeviceInterfaceProxyClientLogger("Send thread is terminating.");

    IceUtil::ThreadPtr callbackSenderThread;

    {
        IceUtil::Monitor<IceUtil::Mutex>::Lock lock(*this);

        mtsDeviceInterfaceProxyClientLogger("Destroying sender.");
        Runnable = false;

        notify();

        callbackSenderThread = Sender;
        Sender = 0; // Resolve cyclic dependency.
    }

    callbackSenderThread->getThreadControl().join();
}
