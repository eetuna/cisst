/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsProxyBaseServer.h 142 2009-03-11 23:02:34Z mjung5 $

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

#ifndef _mtsProxyBaseServer_h
#define _mtsProxyBaseServer_h

#include <cisstMultiTask/mtsProxyBaseCommon.h>
#include <cisstMultiTask/mtsExport.h>

#include <set>

/*!
  \ingroup cisstMultiTask

  TODO: add class summary here
*/
class CISST_EXPORT mtsProxyBaseServer : public mtsProxyBaseCommon {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

    /////////////////////////////////////////////////////////////////////////////
    //// From SLICE definition
    //class TaskManagerChannelI : public mtsTaskManagerProxy::TaskManagerCommunicator {
    //public:
    //    virtual void ShareTaskInfo(const ::mtsTaskManagerProxy::TaskInfo& clientTaskInfo,
    //                               ::mtsTaskManagerProxy::TaskInfo& serverTaskInfo, 
    //                               const ::Ice::Current&);
    //};
    ////
    /////////////////////////////////////////////////////////////////////////////
protected:

    Ice::ObjectAdapterPtr IceAdapter;

    void Init(void);

public:
    mtsProxyBaseServer(void);
    virtual ~mtsProxyBaseServer();

    void StartProxy(mtsTaskManager * callingTaskManager);    
    void OnThreadEnd(void);

    inline Ice::ObjectAdapterPtr GetIceAdapter() const { return IceAdapter; }

    virtual void Runner(ThreadArguments * arguments) = 0;
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsProxyBaseServer)

#endif // _mtsProxyBaseServer_h

