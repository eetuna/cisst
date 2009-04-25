/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsTaskManagerProxyServer.cpp 145 2009-03-18 23:32:40Z mjung5 $

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

#include <cisstMultiTask/mtsTaskManagerProxyServer.h>
#include <cisstCommon/cmnAssert.h>

CMN_IMPLEMENT_SERVICES(mtsTaskManagerProxyServer);

#define mtsTaskManagerProxyServerLogger(_log) Logger->trace("mtsTaskManagerProxyServer", _log)

//#define _COMMUNICATION_TEST_

mtsTaskManagerProxyServer::~mtsTaskManagerProxyServer()
{
    OnClose();
}

void mtsTaskManagerProxyServer::OnClose()
{
}

void mtsTaskManagerProxyServer::Start(mtsTaskManager * callingTaskManager)
{
    // Initialize Ice object.
    // Notice that a worker thread is not created right now.
    Init();
    
    if (InitSuccessFlag) {
        // Create a worker thread here and returns immediately.
        ThreadArgumentsInfo.argument = callingTaskManager;
        ThreadArgumentsInfo.proxy = this;
        ThreadArgumentsInfo.Runner = mtsTaskManagerProxyServer::Runner;

        WorkerThread.Create<ProxyWorker<mtsTaskManager>, ThreadArguments<mtsTaskManager>*>(
            &ProxyWorkerInfo, &ProxyWorker<mtsTaskManager>::Run, &ThreadArgumentsInfo, "C-PRX");
    }
}

void mtsTaskManagerProxyServer::StartServer()
{
    Sender->Start();

    // This is a blocking call that should run in a different thread.
    IceCommunicator->waitForShutdown();
}

void mtsTaskManagerProxyServer::Runner(ThreadArguments<mtsTaskManager> * arguments)
{
    mtsTaskManagerProxyServer * ProxyServer = 
        dynamic_cast<mtsTaskManagerProxyServer*>(arguments->proxy);
    
    try {
        ProxyServer->StartServer();
    } catch (const Ice::Exception& e) {
        CMN_LOG_CLASS_AUX(ProxyServer, 3) << "mtsTaskManagerProxyServer error: " << e << std::endl;
    } catch (const char * msg) {
        CMN_LOG_CLASS_AUX(ProxyServer, 3) << "mtsTaskManagerProxyServer error: " << msg << std::endl;
    }

    ProxyServer->OnThreadEnd();
}

void mtsTaskManagerProxyServer::OnThreadEnd()
{
    mtsProxyBaseServer::OnThreadEnd();

    Sender->Destroy();
}

mtsTaskManagerProxyServer::GlobalTaskMapType *
    mtsTaskManagerProxyServer::GetTaskMap(const std::string taskManagerID)
{
    GlobalTaskManagerMapType::iterator it = TaskManagerMap.find(taskManagerID);
    if (it == TaskManagerMap.end()) {
        return 0;
    } else {
        return &(it->second);
    }
}

void mtsTaskManagerProxyServer::TestShowMe()
{
    GlobalTaskManagerMapType::const_iterator it = TaskManagerMap.begin();
    for (; it != TaskManagerMap.end(); ++it) {
        std::cout << "--------------------------------- " << it->first << std::endl;
        GlobalTaskMapType::const_iterator itr = it->second.begin();
        for (; itr != it->second.end(); ++itr) {
            std::cout << "\t" << itr->first;
            std::cout << " (" << itr->second.TaskManagerID << ", ";
            std::cout << itr->second.TaskName << ", ";
            std::cout << itr->second.ConnectedTaskList.size() << std::endl;
        }
    }
}

void mtsTaskManagerProxyServer::AddTaskManager(
    const ::mtsTaskManagerProxy::TaskList& localTaskInfo)
{
    const std::string taskManagerID = localTaskInfo.taskManagerID;

    GlobalTaskMapType newTaskMap;
    GlobalTaskMapType * taskMap = GetTaskMap(taskManagerID);
    
    // If this task manager has not been registered,
    if (taskMap == 0) {
        taskMap = &newTaskMap;
    }

    std::string taskName;
    mtsTaskManagerProxy::TaskNameSeq::const_iterator it = localTaskInfo.taskNames.begin();
    for (; it != localTaskInfo.taskNames.end(); ++it) {
        taskName = *it;
        TaskInfo taskInfo(taskName, taskManagerID);
        taskMap->insert(make_pair(taskName, taskInfo));

        //
        // TODO: task name duplicity check!!! and if there is any duplications,
        // an task name duplication exception should be generated!!!
        //

        TaskNameMap.insert(make_pair(taskName, taskManagerID));
    }

    TaskManagerMap.insert(make_pair(taskManagerID, *taskMap));

    // FOR TEST
    TestShowMe();
}

/*
const bool mtsTaskManagerProxyServer::AddTask(const std::string taskManagerName,
                                              const std::string taskName)
{
    // TODO: Check if there is a task that has been already registered as the same name.
    //
    // TODO: task name duplicity check!!! and if there is any duplications,
    // an task name duplication exception should be generated!!!
    //

    return true;
}

const bool mtsTaskManagerProxyServer::RemoveTask(const std::string taskName)
{
    return true;
}
*/

//-------------------------------------------------------------------------
//  Definition by mtsTaskManagerProxy.ice
//-------------------------------------------------------------------------
mtsTaskManagerProxyServer::TaskManagerServerI::TaskManagerServerI(
    const Ice::CommunicatorPtr& communicator,
    const Ice::LoggerPtr& logger,
    mtsTaskManagerProxyServer * taskManagerServer) 
    : Communicator(communicator), Logger(logger),
      TaskManagerServer(taskManagerServer),
      Runnable(true),
      Sender(new SendThread<TaskManagerServerIPtr>(this))
{
}

void mtsTaskManagerProxyServer::TaskManagerServerI::Start()
{
    mtsTaskManagerProxyServerLogger("Send thread starts");

    Sender->start();
}

void mtsTaskManagerProxyServer::TaskManagerServerI::Run()
{
    int num = 0;
    while(true)
    {
        std::set<mtsTaskManagerProxy::TaskManagerClientPrx> clients;
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
            for(std::set<mtsTaskManagerProxy::TaskManagerClientPrx>::iterator p 
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

void mtsTaskManagerProxyServer::TaskManagerServerI::Destroy()
{
    mtsTaskManagerProxyServerLogger("Send thread is terminating.");

    IceUtil::ThreadPtr callbackSenderThread;

    {
        IceUtil::Monitor<IceUtil::Mutex>::Lock lock(*this);

        mtsTaskManagerProxyServerLogger("Destroying sender.");
        Runnable = false;

        notify();

        callbackSenderThread = Sender;
        Sender = 0; // Resolve cyclic dependency.
    }

    callbackSenderThread->getThreadControl().join();
}

void mtsTaskManagerProxyServer::TaskManagerServerI::AddClient(
    const ::Ice::Identity& ident, const ::Ice::Current& current)
{
    IceUtil::Monitor<IceUtil::Mutex>::Lock lock(*this);

    std::string log = "Adding client: " + Communicator->identityToString(ident);
    mtsTaskManagerProxyServerLogger(log.c_str());

    mtsTaskManagerProxy::TaskManagerClientPrx client = 
        mtsTaskManagerProxy::TaskManagerClientPrx::uncheckedCast(current.con->createProxy(ident));
    _clients.insert(client);    
}

//void mtsTaskManagerProxyServer::TaskManagerServerI::ReceiveDataFromClient(
//    ::Ice::Int num, const ::Ice::Current&)
//{
//    std::cout << "------------ server recv data " << num << std::endl;
//}

void mtsTaskManagerProxyServer::TaskManagerServerI::AddTaskManager(
    const ::mtsTaskManagerProxy::TaskList& localTaskInfo, const ::Ice::Current& current) const
{
    /*
    std::vector<std::string>::const_iterator it = localTaskInfo.taskNames.begin();
    for (; it != localTaskInfo.taskNames.end(); ++it) {
        //std::cout << current.requestId;
        //std::cout << Communicator->identityToString(current.id) << " ] : ";
        //std::cerr << current.con->toString() << std::endl;
        std::cout << "[ " << localTaskInfo.taskManagerID << " ] ";
        std::cout << *it << std::endl;        
    }
    */

    TaskManagerServer->AddTaskManager(localTaskInfo);
}