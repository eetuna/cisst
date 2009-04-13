/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsProxyBaseClient.h 142 2009-03-11 23:02:34Z mjung5 $

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

#ifndef _mtsProxyBaseClient_h
#define _mtsProxyBaseClient_h

#include <cisstMultiTask/mtsProxyBaseCommon.h>
#include <cisstOSAbstraction/osaSleep.h>

#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  TODO: add class summary here
*/

template<class _ArgumentType>
class CISST_EXPORT mtsProxyBaseClient : public mtsProxyBaseCommon<_ArgumentType> {
    
protected:
    bool RunnableFlag;
    Ice::ObjectPrx ProxyObject;

    virtual void CreateProxy() = 0;

    void Init(void)
    {
        try {
            IceCommunicator = Ice::initialize();

            std::string stringifiedProxy = 
                GetCommunicatorIdentity(TASK_MANAGER_COMMUNICATOR) + ":default -p 10705";
            ProxyObject = IceCommunicator->stringToProxy(stringifiedProxy);

            // If a proxy fails to be created, an exception is thrown.
            CreateProxy();

            InitSuccessFlag = true;
            RunnableFlag = true;

            Logger = IceCommunicator->getLogger();
            Logger->trace("mtsProxyBaseClient", "Client proxy initialization success");
            //CMN_LOG_CLASS(3) << "Client proxy initialization success. " << std::endl;
            return;
        } catch (const Ice::Exception& e) {
            Logger->trace("mtsProxyBaseClient", "Client proxy initialization error");
            Logger->trace("mtsProxyBaseClient", e.what());
            //CMN_LOG_CLASS(3) << "Client proxy initialization error: " << e << std::endl;
        } catch (const char * msg) {
            Logger->trace("mtsProxyBaseClient", "Client proxy initialization error");
            Logger->trace("mtsProxyBaseClient", msg);
            //CMN_LOG_CLASS(3) << "Client proxy initialization error: " << msg << std::endl;
        }

        if (IceCommunicator) {
            InitSuccessFlag = false;
            try {
                IceCommunicator->destroy();
            } catch (const Ice::Exception& e) {
                Logger->trace("mtsProxyBaseClient", "Client proxy clean-up error");
                Logger->trace("mtsProxyBaseClient", e.what());
                //CMN_LOG_CLASS(3) << "Client proxy initialization failed: " << e << std::endl;
            }
        }
    }

public:
    mtsProxyBaseClient(void) : RunnableFlag(false) {}
    virtual ~mtsProxyBaseClient() {}

    inline const bool IsRunnable() const { return RunnableFlag; }

    //virtual void Runner(ThreadArguments<_ArgumentType> * arguments) = 0;

    virtual void StartProxy(_ArgumentType * callingClass) = 0;

    void OnThreadEnd(void)
    {
        if (IceCommunicator) {
            try {
                IceCommunicator->destroy();
                RunningFlag = false;
                RunnableFlag = false;

                Logger->trace("mtsProxyBaseClient", "Client proxy clean-up success.");
                //CMN_LOG_CLASS(3) << "Proxy cleanup succeeded." << std::endl;
            } catch (const Ice::Exception& e) {
                Logger->trace("mtsProxyBaseClient", "Client proxy clean-up failed.");
                Logger->trace("mtsProxyBaseClient", e.what());
                //CMN_LOG_CLASS(3) << "Proxy cleanup failed: " << e << std::endl;
            }
        }    
    }

    ///////////////////////////////////////////////////////////////////////////
    // From SLICE definition
    //inline mtsTaskManagerProxy::TaskManagerCommunicatorPrx GetTaskManagerCommunicatorProxy() const {
    //    return TaskManagerCommunicatorProxy; 
    //}    
    ///////////////////////////////////////////////////////////////////////////
};

//typedef mtsProxyBaseClient<mtsTaskManager> mtsTaskManagerProxyClient;
//CMN_DECLARE_SERVICES_INSTANTIATION(mtsTaskManagerProxyClient);

#endif // _mtsProxyBaseClient_h

