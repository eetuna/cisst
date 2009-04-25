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

#include <cisstMultiTask/mtsTaskInterfaceProxyServer.h>
#include <cisstCommon/cmnAssert.h>

CMN_IMPLEMENT_SERVICES(mtsTaskInterfaceProxyServer);

#define mtsTaskInterfaceProxyServerLogger(_log) Logger->trace("mtsTaskInterfaceProxyServer", _log)

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
    
    try {
        ProxyServer->StartServer();
    } catch (const Ice::Exception& e) {
        CMN_LOG_CLASS_AUX(ProxyServer, 3) << "mtsTaskInterfaceProxyServer error: " << e << std::endl;
    } catch (const char * msg) {
        CMN_LOG_CLASS_AUX(ProxyServer, 3) << "mtsTaskInterfaceProxyServer error: " << msg << std::endl;
    }

    ProxyServer->OnThreadEnd();
}

void mtsTaskInterfaceProxyServer::OnThreadEnd()
{
    mtsProxyBaseServer::OnThreadEnd();

    Sender->Destroy();
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
