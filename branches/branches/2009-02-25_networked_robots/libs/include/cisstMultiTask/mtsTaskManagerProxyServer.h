/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsTaskManagerProxyServer.h 142 2009-03-11 23:02:34Z mjung5 $

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

#ifndef _mtsTaskManagerProxyServer_h
#define _mtsTaskManagerProxyServer_h

#include <cisstMultiTask/mtsTaskManager.h>
#include <cisstMultiTask/mtsProxyBaseServer.h>
#include <cisstMultiTask/mtsTaskManagerProxy.h>

#include <cisstMultiTask/mtsExport.h>

#include <set>

/*!
  \ingroup cisstMultiTask

  TODO: add class summary here
*/
class CISST_EXPORT mtsTaskManagerProxyServer : public mtsProxyBaseServer<mtsTaskManager> {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

    //-------------------------------------------------------------------------
    // From SLICE definition
    //-------------------------------------------------------------------------
    class TaskManagerChannelI : public mtsTaskManagerProxy::TaskManagerCommunicator {
    public:
        /*! Called when the connected client requests task information. */
        virtual void ShareTaskInfo(const ::mtsTaskManagerProxy::TaskInfo& clientTaskInfo,
                                   ::mtsTaskManagerProxy::TaskInfo& serverTaskInfo, 
                                   const ::Ice::Current&);

        //
        //  TODO: What if a client/server disconnects abruptly so that data integrity of
        //  two sides are ruined? How to gurantee both peers have synchronized data?
        //  => Periodically let them communicate to each other (periodic synchronization)
        //  => What if a server collapses? (DISASTER... Thus the p2p architecture rather than
        //  the server-client architecture has a merit in this case.)
        //
    };

public:
    mtsTaskManagerProxyServer(const std::string& propertyFileName, 
                              const std::string& propertyName) 
        : mtsProxyBaseServer(propertyFileName, propertyName)
    {}
    ~mtsTaskManagerProxyServer() {}

    Ice::ObjectPtr CreateServant() { 
        return new mtsTaskManagerProxyServer::TaskManagerChannelI;
    }

    void StartProxy(mtsTaskManager * callingTaskManager);

    static void Runner(ThreadArguments<mtsTaskManager> * arguments);
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsTaskManagerProxyServer)

#endif // _mtsTaskManagerProxyServer_h

