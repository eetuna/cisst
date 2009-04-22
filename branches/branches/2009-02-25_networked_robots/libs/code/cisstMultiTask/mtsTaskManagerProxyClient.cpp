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

void mtsTaskManagerProxyClient::StartProxy(mtsTaskManager * callingTaskManager)
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
        ident.name = IceUtil::generateUUID();
        ident.category = "";

        mtsTaskManagerProxy::TaskManagerClientPtr client = new TaskManagerClientI;
        adapter->add(client, ident);
        adapter->activate();
        TaskManagerServer->ice_getConnection()->setAdapter(adapter);
        TaskManagerServer->AddClient(ident);
        //IceCommunicator->watiForShutdown();

        // Create a worker thread here and returns immediately.
        ThreadArgumentsInfo.argument = callingTaskManager;
        ThreadArgumentsInfo.proxy = this;        
        ThreadArgumentsInfo.Runner = mtsTaskManagerProxyClient::Runner;

        WorkerThread.Create<ProxyWorker<mtsTaskManager>, ThreadArguments<mtsTaskManager>*>(
            &ProxyWorkerInfo, &ProxyWorker<mtsTaskManager>::Run, &ThreadArgumentsInfo, "S-PRX");
    }
}

void mtsTaskManagerProxyClient::Runner(ThreadArguments<mtsTaskManager> * arguments)
{
    mtsTaskManager * TaskManager = reinterpret_cast<mtsTaskManager*>(arguments->argument);

    mtsTaskManagerProxyClient * ProxyClient = 
        dynamic_cast<mtsTaskManagerProxyClient*>(arguments->proxy);

    try {
        ProxyClient->GetIceCommunicator()->waitForShutdown();
        /*
        mtsTaskManagerProxy::TaskInfo myTaskInfo, peerTaskInfo;
        
        std::vector<std::string> myTaskNames;
        mtsTaskManager::GetInstance()->GetNamesOfTasks(myTaskNames);

        myTaskInfo.taskNames.insert(
            myTaskInfo.taskNames.end(),
            myTaskNames.begin(),
            myTaskNames.end());

        // FOR TEST
        bool flag = true;

        while(ProxyClient->IsRunnable()) {            
            //
            //  TODO: If this should be done in a nonblocking way, AMI feature can be applied.
            //
            if (flag) {
                // The following operation is a blocking call.
                ProxyClient->GetTaskManagerCommunicatorProxy()->ShareTaskInfo(myTaskInfo, peerTaskInfo);

                std::string s;
                mtsTaskManagerProxy::TaskNameSeq::const_iterator it = 
                    peerTaskInfo.taskNames.begin();
                for (; it != peerTaskInfo.taskNames.end(); ++it) {
                    s = "SERVER TASK NAME: " + (*it);
                    ProxyClient->GetLogger()->trace("mtsTaskManagerProxyClient", s);
                    //CMN_LOG_CLASS_AUX(ProxyClient, 5) << "SERVER TASK NAME: " << *it << std::endl;
                }

                flag = false;
            }

            osaSleep(1 * cmn_ms);
        }
        */
    } catch (const Ice::Exception& e) {
        ProxyClient->GetLogger()->trace("mtsTaskManagerProxyClient", "exception");
        ProxyClient->GetLogger()->trace("mtsTaskManagerProxyClient", e.what());
        //CMN_LOG_CLASS_AUX(ProxyClient, 3) << "Proxy initialization error: " << e << std::endl;        
    } catch (const char * msg) {
        ProxyClient->GetLogger()->trace("mtsTaskManagerProxyClient", "exception");
        ProxyClient->GetLogger()->trace("mtsTaskManagerProxyClient", msg);
        //CMN_LOG_CLASS_AUX(ProxyClient, 3) << "Proxy initialization error: " << msg << std::endl;        
    }

    ProxyClient->OnThreadEnd();
}
