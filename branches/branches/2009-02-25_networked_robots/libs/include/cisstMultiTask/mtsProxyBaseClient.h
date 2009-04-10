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

#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  TODO: add class summary here
*/

class CISST_EXPORT mtsProxyBaseClient : public mtsProxyBaseCommon {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

    ///////////////////////////////////////////////////////////////////////////
    // From SLICE definition
    //mtsTaskManagerProxy::TaskManagerCommunicatorPrx TaskManagerCommunicatorProxy;
    ///////////////////////////////////////////////////////////////////////////
protected:
    bool RunnableFlag;

    void Init(void);

public:
    mtsProxyBaseClient(void);
    virtual ~mtsProxyBaseClient();

    void StartProxy(mtsTaskManager * callingTaskManager);    
    void OnThreadEnd(void);

    static void Runner(ThreadArguments * arguments);

    inline const bool IsRunnable() const { return RunnableFlag; }

    ///////////////////////////////////////////////////////////////////////////
    // From SLICE definition
    //inline mtsTaskManagerProxy::TaskManagerCommunicatorPrx GetTaskManagerCommunicatorProxy() const {
    //    return TaskManagerCommunicatorProxy; 
    //}    
    ///////////////////////////////////////////////////////////////////////////
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsProxyBaseClient)

#endif // _mtsProxyBaseClient_h

