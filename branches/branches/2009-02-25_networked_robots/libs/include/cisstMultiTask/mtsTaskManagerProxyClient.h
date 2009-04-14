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

    ///////////////////////////////////////////////////////////////////////////
    // From SLICE definition
    mtsTaskManagerProxy::TaskManagerCommunicatorPrx TaskManagerCommunicatorProxy;
    ///////////////////////////////////////////////////////////////////////////

public:
    mtsTaskManagerProxyClient(const std::string& propertyFileName, 
                              const std::string& propertyName) 
        : mtsProxyBaseClient(propertyFileName, propertyName)
    {}
    ~mtsTaskManagerProxyClient() {}

    void CreateProxy() {
        TaskManagerCommunicatorProxy = 
            mtsTaskManagerProxy::TaskManagerCommunicatorPrx::checkedCast(ProxyObject);
        if (!TaskManagerCommunicatorProxy) {
            throw "Invalid proxy";
        }
    }

    void StartProxy(mtsTaskManager * callingTaskManager);

    static void Runner(ThreadArguments<mtsTaskManager> * arguments);

    ///////////////////////////////////////////////////////////////////////////
    // From SLICE definition
    inline mtsTaskManagerProxy::TaskManagerCommunicatorPrx GetTaskManagerCommunicatorProxy() const {
        return TaskManagerCommunicatorProxy; 
    }    
    ///////////////////////////////////////////////////////////////////////////    
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsTaskManagerProxyClient)

#endif // _mtsTaskManagerProxyClient_h

