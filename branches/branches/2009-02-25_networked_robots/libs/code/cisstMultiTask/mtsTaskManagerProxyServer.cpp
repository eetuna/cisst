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

//-----------------------------------------------------------------------------
//  Constructor, Destructor, Initializer
//-----------------------------------------------------------------------------
mtsTaskManagerProxyServer::~mtsTaskManagerProxyServer()
{
    OnClose();
}

void mtsTaskManagerProxyServer::OnClose()
{
}

//-----------------------------------------------------------------------------
//  Proxy Start-up
//-----------------------------------------------------------------------------
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
    
    ProxyServer->GetLogger()->trace("mtsTaskManagerProxyServer", "Proxy server starts.");

    try {
        ProxyServer->StartServer();
    } catch (const Ice::Exception& e) {
        ProxyServer->GetLogger()->trace("mtsTaskManagerProxyServer error: ", e.what());
    } catch (const char * msg) {
        ProxyServer->GetLogger()->trace("mtsTaskManagerProxyServer error: ", msg);
    }

    ProxyServer->OnThreadEnd();
}

void mtsTaskManagerProxyServer::OnThreadEnd()
{
    mtsTaskManagerProxyServerLogger("Proxy server ends.");

    mtsProxyBaseServer::OnThreadEnd();

    Sender->Destroy();
}

//-----------------------------------------------------------------------------
//  Task Manager Processing
//-----------------------------------------------------------------------------

//mtsTaskManagerProxyServer::GlobalTaskMapType *
//    mtsTaskManagerProxyServer::GetTaskMap(const std::string taskManagerID)
//{
//    GlobalTaskManagerMapType::iterator it = GlobalTaskManagerMap.find(taskManagerID);
//    if (it == GlobalTaskManagerMap.end()) {
//        return 0;
//    } else {
//        return &(it->second);
//    }
//}
const bool mtsTaskManagerProxyServer::FindTaskManager(const std::string taskManagerID) const
{
    GlobalTaskManagerMapType::const_iterator it = GlobalTaskManagerMap.find(taskManagerID);
    if (it == GlobalTaskManagerMap.end()) {
        return false;
    } else {
        return true;
    }
}

const bool mtsTaskManagerProxyServer::RemoveTaskManager(const std::string taskManagerID)
{
    GlobalTaskManagerMapType::iterator it = GlobalTaskManagerMap.find(taskManagerID);
    if (it == GlobalTaskManagerMap.end()) {
        Logger->error("Can't find a task manager: " + taskManagerID);
        return false;
    } else {
        GlobalTaskManagerMap.erase(it);
        return true;
    }
}

mtsTaskGlobal * mtsTaskManagerProxyServer::GetTask(const std::string & taskName)
{
    GlobalTaskMapType::iterator it = GlobalTaskMap.find(taskName);
    if (it == GlobalTaskMap.end()) {
        return 0;
    } else {
        return &it->second;
    }
}

//-----------------------------------------------------------------------------
//  Proxy Server Implementation
//-----------------------------------------------------------------------------
void mtsTaskManagerProxyServer::AddTaskManager(
    const ::mtsTaskManagerProxy::TaskList& localTaskInfo)
{
    const std::string taskManagerID = localTaskInfo.taskManagerID;

    const bool exist = FindTaskManager(taskManagerID);
    if (exist) {
        CMN_ASSERT(RemoveTaskManager(taskManagerID));
    }

    TaskList newTaskList;
    {
        mtsTaskManagerProxy::TaskNameSeq::const_iterator it = localTaskInfo.taskNames.begin();
        std::string taskName;
        for (; it != localTaskInfo.taskNames.end(); ++it) {
            taskName = *it;
            newTaskList.push_back(taskName);
            Logger->trace("mtsTaskManagerProxyServer", "Adding a new task");
            Logger->trace("New task", "(" + taskManagerID + ", " + taskName + ")");

            mtsTaskGlobal taskInfo(taskName, taskManagerID);
            GlobalTaskMap.insert(make_pair(taskName, taskInfo));
            Logger->print(taskInfo.ShowTaskInfo());
        }
    }
    GlobalTaskManagerMap.insert(make_pair(taskManagerID, newTaskList));
}

bool mtsTaskManagerProxyServer::AddProvidedInterface(
    const ::mtsTaskManagerProxy::ProvidedInterfaceInfo & providedInterfaceInfo)
{
    mtsTaskGlobal * taskInfo = NULL;

    // Check if the task has been registered.
    const std::string taskName = providedInterfaceInfo.taskName;
    GlobalTaskMapType::iterator it = GlobalTaskMap.find(taskName);
    if (it == GlobalTaskMap.end()) {
        Logger->error("No task found: " + taskName);
        return false;
    }

    // Add a new provided interface to the task
    bool ret = it->second.AddProvidedInterface(providedInterfaceInfo);
    Logger->print(GetTask(taskName)->ShowTaskInfo());

    return ret;
}

bool mtsTaskManagerProxyServer::AddRequiredInterface(
    const ::mtsTaskManagerProxy::RequiredInterfaceInfo & requiredInterfaceInfo)
{
    mtsTaskGlobal * taskInfo = NULL;

    // Check if the task has been registered.
    const std::string taskName = requiredInterfaceInfo.taskName;
    GlobalTaskMapType::iterator it = GlobalTaskMap.find(taskName);
    if (it == GlobalTaskMap.end()) {
        Logger->error("No task found: " + taskName);
        return false;
    }

    // Add a new required interface to the task
    bool ret = it->second.AddRequiredInterface(requiredInterfaceInfo);
    Logger->print(GetTask(taskName)->ShowTaskInfo());

    return ret;
}

bool mtsTaskManagerProxyServer::IsRegisteredProvidedInterface(
    const std::string & taskName, const std::string & providedInterfaceName) const
{
    GlobalTaskMapType::const_iterator it = GlobalTaskMap.find(taskName);
    if (it == GlobalTaskMap.end()) {
        return false;
    } else {
        return it->second.IsRegisteredProvidedInterface(providedInterfaceName);
    }
}

bool mtsTaskManagerProxyServer::GetProvidedInterfaceInfo(
    const std::string & taskName, const std::string & providedInterfaceName,
    mtsTaskManagerProxy::ProvidedInterfaceInfo & info)
{
    GlobalTaskMapType::iterator it = GlobalTaskMap.find(taskName);
    if (it == GlobalTaskMap.end()) {
        return false;
    } else {
        return it->second.GetProvidedInterfaceInfo(providedInterfaceName, info);
    }
}

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

    Logger->trace("TMServer", "<<<<< RECV: AddClient: " + Communicator->identityToString(ident));

    mtsTaskManagerProxy::TaskManagerClientPrx client = 
        mtsTaskManagerProxy::TaskManagerClientPrx::uncheckedCast(current.con->createProxy(ident));
    _clients.insert(client);    
}

void mtsTaskManagerProxyServer::TaskManagerServerI::AddTaskManager(
    const ::mtsTaskManagerProxy::TaskList& localTaskInfo, const ::Ice::Current& current)
{
    TaskManagerServer->AddTaskManager(localTaskInfo);
}

bool mtsTaskManagerProxyServer::TaskManagerServerI::AddProvidedInterface(
    const ::mtsTaskManagerProxy::ProvidedInterfaceInfo & providedInterfaceInfo,
    const ::Ice::Current & current)
{
    Logger->trace("TMServer", "<<<<< RECV: AddProvidedInterface: " 
        + providedInterfaceInfo.taskName + ", " + providedInterfaceInfo.interfaceName);

    return TaskManagerServer->AddProvidedInterface(providedInterfaceInfo);
}

bool mtsTaskManagerProxyServer::TaskManagerServerI::AddRequiredInterface(
    const ::mtsTaskManagerProxy::RequiredInterfaceInfo & requiredInterfaceInfo,
    const ::Ice::Current & current)
{
    Logger->trace("TMServer", "<<<<< RECV: AddRequiredInterface: " 
        + requiredInterfaceInfo.taskName + ", " + requiredInterfaceInfo.interfaceName);

    return TaskManagerServer->AddRequiredInterface(requiredInterfaceInfo);
}

bool mtsTaskManagerProxyServer::TaskManagerServerI::IsRegisteredProvidedInterface(
    const ::std::string & taskName, const ::std::string & providedInterfaceName,
    const ::Ice::Current & current) const
{
    Logger->trace("TMServer", "<<<<< RECV: IsRegisteredProvidedInterface: " 
        + taskName + ", " + providedInterfaceName);

    return TaskManagerServer->IsRegisteredProvidedInterface(
        taskName, providedInterfaceName);
}

bool mtsTaskManagerProxyServer::TaskManagerServerI::GetProvidedInterfaceInfo(
    const ::std::string & taskName, const ::std::string & providedInterfaceName,
    ::mtsTaskManagerProxy::ProvidedInterfaceInfo & info, const ::Ice::Current & current) const
{
    Logger->trace("TMServer", "<<<<< RECV: GetProvidedInterfaceInfo: " 
        + taskName + ", " + providedInterfaceName);

    return TaskManagerServer->GetProvidedInterfaceInfo(
        taskName, providedInterfaceName, info);
}