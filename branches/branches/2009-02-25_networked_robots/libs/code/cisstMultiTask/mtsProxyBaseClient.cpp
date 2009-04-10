/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsProxyBaseClient.cpp 145 2009-03-18 23:32:40Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-04-10

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstMultiTask/mtsProxyBaseClient.h>
#include <cisstOSAbstraction/osaSleep.h>

CMN_IMPLEMENT_SERVICES(mtsProxyBaseClient);

mtsProxyBaseClient::mtsProxyBaseClient() 
    : RunnableFlag(false)
{
}

mtsProxyBaseClient::~mtsProxyBaseClient()
{
}

void mtsProxyBaseClient::Init(void)
{
    try {
        IceCommunicator = Ice::initialize();

        std::string stringifiedProxy = TaskManagerCommunicatorIdentity + ":default -p 10000";
        Ice::ObjectPrx base = IceCommunicator->stringToProxy(stringifiedProxy);

        //TaskManagerCommunicatorProxy = mtsTaskManagerProxy::TaskManagerCommunicatorPrx::checkedCast(base);
        //if (!TaskManagerCommunicatorProxy) {
        //    throw "Invalid proxy";
        //}

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
            CMN_LOG_CLASS(3) << "Client proxy initialization failed: " << e << std::endl;
        }
    }
}

void mtsProxyBaseClient::StartProxy(mtsTaskManager * callingTaskManager)
{
    // Initialize Ice object.
    // Notice that a worker thread is not created right now.
    Init();

    if (InitSuccessFlag) {
        // Create a worker thread here and returns immediately.
        Arguments.Runner = mtsProxyBaseClient::Runner;
        Arguments.proxy = this;
        Arguments.taskManager = callingTaskManager;

        WorkerThread.Create<ProxyWorker, ThreadArguments *>(
            &ProxyWorkerInfo, &ProxyWorker::Run, &Arguments, "S-PRX");
    }
}

void mtsProxyBaseClient::Runner(ThreadArguments * arguments)
{
    mtsTaskManager * TaskManager = arguments->taskManager;
    mtsProxyBaseClient * ProxyClient = 
        dynamic_cast<mtsProxyBaseClient*>(arguments->proxy);

    try {
        mtsTaskManagerProxy::TaskInfo myTaskInfo, peerTaskInfo;
        
        std::vector<std::string> myTaskNames;
        mtsTaskManager::GetInstance()->GetNamesOfTasks(myTaskNames);

        myTaskInfo.taskNames.insert(
            myTaskInfo.taskNames.end(),
            myTaskNames.begin(),
            myTaskNames.end());

        // FOR TEST
        bool flag = true;

        //while(ProxyClient->IsRunnable()) {            
        //    //
        //    //  TODO: If this should be done in a nonblocking way, AMI feature can be applied.
        //    //
        //    if (flag) {
        //        // The following operation is a blocking call.
        //        ProxyClient->GetTaskManagerCommunicatorProxy()->ShareTaskInfo(myTaskInfo, peerTaskInfo);

        //        mtsTaskManagerProxy::TaskNameSeq::const_iterator it = 
        //            peerTaskInfo.taskNames.begin();
        //        for (; it != peerTaskInfo.taskNames.end(); ++it) {
        //            CMN_LOG_CLASS_AUX(ProxyClient, 5) << "SERVER TASK NAME: " << *it << std::endl;
        //        }

        //        flag = false;
        //    }

        //    osaSleep(1 * cmn_ms);
        //}
    } catch (const Ice::Exception& e) {        
        CMN_LOG_CLASS_AUX(ProxyClient, 3) << "Proxy initialization error: " << e << std::endl;        
    } catch (const char * msg) {
        CMN_LOG_CLASS_AUX(ProxyClient, 3) << "Proxy initialization error: " << msg << std::endl;        
    }

    ProxyClient->OnThreadEnd();
}

void mtsProxyBaseClient::OnThreadEnd()
{
    if (IceCommunicator) {
        try {
            IceCommunicator->destroy();
            RunningFlag = false;
            RunnableFlag = false;

            CMN_LOG_CLASS(3) << "Proxy cleanup succeeded." << std::endl;
        } catch (const Ice::Exception& e) {
            CMN_LOG_CLASS(3) << "Proxy cleanup failed: " << e << std::endl;
        }
    }    
}
