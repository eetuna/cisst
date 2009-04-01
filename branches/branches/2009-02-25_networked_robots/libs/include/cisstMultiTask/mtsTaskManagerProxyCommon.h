/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsTaskManagerProxyCommon.h 142 2009-03-11 23:02:34Z mjung5 $

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

#ifndef _mtsTaskManagerProxyCommon_h
#define _mtsTaskManagerProxyCommon_h

#include <cisstCommon/cmnGenericObject.h>
#include <cisstCommon/cmnClassRegister.h>
#include <cisstMultiTask/mtsTaskManager.h>
#include <cisstMultiTask/mtsTaskManagerProxy.h>
#include <cisstOSAbstraction/osaThread.h>
#include <cisstMultiTask/mtsExport.h>

#include <IceUtil/IceUtil.h>
#include <Ice/Ice.h>

#define ICE_TASKMANAGER_COMMUNICATOR_IDENTITY "TaskManagerCommunicator"

/*  Limitations of Ice::Application
    Ice::Application is a singleton class that creates a single communicator.
    If you are using multiple communicators, you cannot use Ice::Application.
*/
class CISST_EXPORT mtsTaskManagerProxyCommon { //: public Ice::Application {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

protected:
    class ThreadArguments {
    public:
        mtsTaskManager * taskManager;
        mtsTaskManagerProxyCommon * proxy;
        void (*Runner)(ThreadArguments *);
    };

    class ProxyWorker {
    public:
        ProxyWorker(void) {}
        virtual ~ProxyWorker(void) {}

        void * Run(ThreadArguments * argument) {
            argument->Runner(argument);
            return 0;
        }
    };

    /*! Was the initiliazation successful? */
    bool InitSuccessFlag;

    /*! Is this thread running? */
    bool RunningFlag;

    /*! Worker thread for network communication */
    osaThread WorkerThread;

    /*! Ice communicator for proxy */
    Ice::CommunicatorPtr IceCommunicator;

    /*! Containers for thread creation */
    ProxyWorker ProxyWorkerInfo;
    ThreadArguments Arguments;

    /*! Ice module initialization */
    virtual void Init(void) = 0;

    /*! A function to be run by a thread */
    //virtual void Run(ThreadArguments * arguments) = 0;

    /*! run() and its overloaded function family are defined by Ice::Application */
    //virtual int run(int argc, char* argv[]) = 0;

    /*! Settings for ICE components */
    std::string TaskManagerCommunicatorIdentity;

public:
    mtsTaskManagerProxyCommon(void);
    virtual ~mtsTaskManagerProxyCommon();    

    /*! Initialize and start a proxy. Returns immediately. */
    virtual void StartProxy(mtsTaskManager * callingTaskManager) = 0;
    
    /*! Called when the worker thread ends. */
    virtual void OnThreadEnd(void) = 0;

    inline const bool IsInitalized() const  { return InitSuccessFlag; }
    inline const bool IsRunning() const     { return RunningFlag; }    

    inline Ice::CommunicatorPtr GetIceCommunicator() const { return IceCommunicator; }

    //
    // TODO: should replace this with map<communicatorPtr, mtsTaskManagerProxy *>
    //
    static Ice::CommunicatorPtr communicator;
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsTaskManagerProxyCommon)

#endif // _mtsTaskManagerProxyCommon_h
