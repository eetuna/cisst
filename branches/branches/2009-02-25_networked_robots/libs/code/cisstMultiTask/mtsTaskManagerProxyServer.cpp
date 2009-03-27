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

CMN_IMPLEMENT_SERVICES(mtsTaskManagerProxyServer);

mtsTaskManagerProxyServer::mtsTaskManagerProxyServer() 
{
}

mtsTaskManagerProxyServer::~mtsTaskManagerProxyServer()
{
}

void mtsTaskManagerProxyServer::Init(void)
{
    try {
        IceCommunicator = Ice::initialize();
        IceAdapter = IceCommunicator->createObjectAdapterWithEndpoints(
                "TaskInfoSharingAdapter", // the name of the adapter
                // instructs the adapter to listen for incoming requests 
                // using the default protocol (TCP) at port number 10000
                "default -p 10000");

        // Create a servant for TaskManager interface
        Ice::ObjectPtr object = new mtsTaskManagerProxyServer::TaskManagerI;

        // Inform the object adapter of the presence of a new servant
        IceAdapter->add(object, IceCommunicator->stringToIdentity("TaskInfoSharing"));
        
        InitSuccessFlag = true;
        CMN_LOG_CLASS(3) << "Server proxy initialization success. " << std::endl;
        return;
    } catch (const Ice::Exception& e) {
        CMN_LOG_CLASS(3) << "Server proxy initialization error: " << e << std::endl;
    } catch (const char * msg) {
        CMN_LOG_CLASS(3) << "Server proxy initialization error: " << msg << std::endl;
    }

    if (IceCommunicator) {
        InitSuccessFlag = false;
        try {
            IceCommunicator->destroy();
        } catch (const Ice::Exception& e) {
            CMN_LOG_CLASS(3) << "Server proxy initialization failed. " << std::endl;
        }
    }
}

void mtsTaskManagerProxyServer::StartProxy(mtsTaskManager * callingTaskManager)
{
    // Initialize Ice object.
    // Notice that a worker thread is not created right now.
    Init();

    if (InitSuccessFlag) {
        mtsTaskManagerProxyCommon::communicator = IceCommunicator;

        // Create a worker thread here and returns immediately.
        Arguments.Runner = mtsTaskManagerProxyServer::Runner;
        Arguments.proxy = this;
        Arguments.taskManager = callingTaskManager;

        WorkerThread.Create<ProxyWorker, ThreadArguments *>(
            &ProxyWorkerInfo, &ProxyWorker::Run, &Arguments, "C-PRX");
    }
}

void mtsTaskManagerProxyServer::Runner(ThreadArguments * arguments)
{
    mtsTaskManager * TaskManager = arguments->taskManager;
    mtsTaskManagerProxyServer * ProxyServer = 
        dynamic_cast<mtsTaskManagerProxyServer*>(arguments->proxy);
    Ice::CommunicatorPtr ic = ProxyServer->GetIceCommunicator();

    try {
        // Activate the adapter. The adapter is initially created in a 
        // holding state. The server starts to process incoming requests
        // from clients as soon as the adapter is activated.
        ProxyServer->GetIceAdapter()->activate();

        // Blocking call
        ic->waitForShutdown();
    } catch (const Ice::Exception& e) {
        //CMN_LOG_CLASS(3) << "Proxy initialization error: " << e << std::endl;
        std::cout << "ProxyServerRunner ERROR: " << e << std::endl;
    } catch (const char * msg) {
        //CMN_LOG_CLASS(3) << "Proxy initialization error: " << msg << std::endl;        
        std::cout << "ProxyServerRunner ERROR: " << msg << std::endl;
    }

    ProxyServer->OnThreadEnd();
}

void mtsTaskManagerProxyServer::OnThreadEnd()
{
    if (IceCommunicator) {
        try {
            IceCommunicator->destroy();
            RunningFlag = false;

            CMN_LOG_CLASS(3) << "Proxy cleanup succeeded." << std::endl;
        } catch (const Ice::Exception& e) {
            CMN_LOG_CLASS(3) << "Proxy cleanup failed." << std::endl;
        }
    }    
}