/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsTaskManagerProxyClient.h 142 2009-03-11 23:02:34Z mjung5 $

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

#ifndef _mtsTaskManagerProxyClient_h
#define _mtsTaskManagerProxyClient_h

#include <cisstMultiTask/mtsTaskManager.h>
#include <cisstMultiTask/mtsProxyBaseClient.h>
#include <cisstMultiTask/mtsTaskManagerProxy.h>

#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  TODO: add class summary here
*/

class CISST_EXPORT mtsTaskManagerProxyClient : public mtsProxyBaseClient<mtsTaskManager> {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

public:
    mtsTaskManagerProxyClient(const std::string& propertyFileName, 
                              const std::string& propertyName) 
        : mtsProxyBaseClient(propertyFileName, propertyName)
    {}
    ~mtsTaskManagerProxyClient() {}

    void StartProxy(mtsTaskManager * callingTaskManager);

    static void Runner(ThreadArguments<mtsTaskManager> * arguments);

    //-------------------------------------------------------------------------
    // From SLICE definition
    //-------------------------------------------------------------------------
protected:
    mtsTaskManagerProxy::TaskManagerServerPrx TaskManagerServer;

    class TaskManagerClientI : public mtsTaskManagerProxy::TaskManagerClient
    {
    public:
        virtual void ReceiveData(::Ice::Int num, const ::Ice::Current&)
        {
            std::cout << "###################### " << num << std::endl;
        }
    
        virtual void SendMyTaskInfo(const ::mtsTaskManagerProxy::TaskInfo&, const ::Ice::Current&)
        {
            //
            // TO BE IMPLEMENTED
            //
        }
    };

public:
    void CreateProxy() {
        TaskManagerServer = 
            mtsTaskManagerProxy::TaskManagerServerPrx::checkedCast(ProxyObject);
        if (!TaskManagerServer) {
            throw "Invalid proxy";
        }
    }

    inline mtsTaskManagerProxy::TaskManagerServerPrx GetTaskManagerServerProxy() const {
        return TaskManagerServer; 
    }    
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsTaskManagerProxyClient)

#endif // _mtsTaskManagerProxyClient_h

