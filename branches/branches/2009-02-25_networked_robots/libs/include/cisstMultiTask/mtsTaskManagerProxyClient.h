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

#include <Ice/Ice.h>
#include <cisstMultiTask/mtsTaskManagerProxy.h>
#include <cisstMultiTask/mtsTaskManagerProxyCommon.h>

#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  TODO: add class summary here
*/

class CISST_EXPORT mtsTaskManagerProxyClient : public mtsTaskManagerProxyCommon {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

    //mtsTaskManagerProxy::PrinterPrx Printer;
    ///////////////////////////////////////////////////////////////////////////
    // From SLICE definition
    mtsTaskManagerProxy::TaskManagerCommunicatorPrx TaskManagerCommunicatorProxy;
    ///////////////////////////////////////////////////////////////////////////

    bool RunnableFlag;

    void Init(void);

public:
    mtsTaskManagerProxyClient(void);
    virtual ~mtsTaskManagerProxyClient();

    void StartProxy(mtsTaskManager * callingTaskManager);    
    void OnThreadEnd(void);

    static void Runner(ThreadArguments * arguments);

    inline const bool IsRunnable() const { return RunnableFlag; }

    //inline mtsTaskManagerProxy::PrinterPrx GetPrinter() const { return Printer; }
    ///////////////////////////////////////////////////////////////////////////
    // From SLICE definition
    inline mtsTaskManagerProxy::TaskManagerCommunicatorPrx GetTaskManagerCommunicatorProxy() const {
        return TaskManagerCommunicatorProxy; 
    }    
    ///////////////////////////////////////////////////////////////////////////
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsTaskManagerProxyClient)

#endif // _mtsTaskManagerProxyClient_h

