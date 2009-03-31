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

mtsTaskManagerProxyClient::mtsTaskManagerProxyClient() 
    : RunnableFlag(false)
{
}

mtsTaskManagerProxyClient::~mtsTaskManagerProxyClient()
{
}

void mtsTaskManagerProxyClient::Init(void)
{
    try {
        IceCommunicator = Ice::initialize();
        Ice::ObjectPrx base = IceCommunicator->stringToProxy(
            "SimplePrinter:default -p 10000");
        TaskManagerChannelProxy = mtsTaskManagerProxy::TaskManagerChannelPrx::checkedCast(base);
        if (!TaskManagerChannelProxy) {
            throw "Invalid proxy";
        }

        InitSuccessFlag = true;
        RunnableFlag = true;
        CMN_LOG_CLASS(3) << "Client proxy initialization success. " << std::endl;
        return;
    } catch (const Ice::Exception& e) {
        CMN_LOG_CLASS(3) << "Client proxy initialization error: " << e << std::endl;
    } catch (const char * msg) {
        CMN_LOG_CLASS(3) << "Client proxy initialization error: " << msg << std::endl;
    }

    if (IceCommunicator) {
        InitSuccessFlag = false;
        try {
            IceCommunicator->destroy();
        } catch (const Ice::Exception& e) {
            CMN_LOG_CLASS(3) << "Client proxy initialization failed. " << std::endl;
        }
    }
}

void mtsTaskManagerProxyClient::StartProxy(mtsTaskManager * callingTaskManager)
{
    // Initialize Ice object.
    // Notice that a worker thread is not created right now.
    Init();

    if (InitSuccessFlag) {
        // Create a worker thread here and returns immediately.
        Arguments.Runner = mtsTaskManagerProxyClient::Runner;
        Arguments.proxy = this;
        Arguments.taskManager = callingTaskManager;

        WorkerThread.Create<ProxyWorker, ThreadArguments *>(
            &ProxyWorkerInfo, &ProxyWorker::Run, &Arguments, "S-PRX");
    }
}

void mtsTaskManagerProxyClient::Runner(ThreadArguments * arguments)
{
    mtsTaskManager * TaskManager = arguments->taskManager;
    mtsTaskManagerProxyClient * ProxyClient = 
        dynamic_cast<mtsTaskManagerProxyClient*>(arguments->proxy);

    try {
        // for test purpose
        int count = 0;
        char buf[50];        

        while(ProxyClient->IsRunnable()) {            
            sprintf(buf, "Hello World: %d", ++count);
            //ProxyClient->GetTaskManagerProxy()->ShareTaskInfo();

            osaSleep(0.5);
        }
    } catch (const Ice::Exception& e) {
        //CMN_LOG_CLASS(3) << "Proxy initialization error: " << e << std::endl;
        std::cout << "ProxyClientRunner ERROR: " << e << std::endl;
    } catch (const char * msg) {
        //CMN_LOG_CLASS(3) << "Proxy initialization error: " << msg << std::endl;        
        std::cout << "ProxyClientRunner ERROR: " << msg << std::endl;
    }

    ProxyClient->OnThreadEnd();
}

void mtsTaskManagerProxyClient::OnThreadEnd()
{
    if (IceCommunicator) {
        try {
            IceCommunicator->destroy();
            RunningFlag = false;
            RunnableFlag = false;

            CMN_LOG_CLASS(3) << "Proxy cleanup succeeded." << std::endl;
        } catch (const Ice::Exception& e) {
            CMN_LOG_CLASS(3) << "Proxy cleanup failed." << std::endl;
        }
    }    
}
