/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsProxyBaseCommon.h 142 2009-03-11 23:02:34Z mjung5 $

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

#ifndef _mtsProxyBaseCommon_h
#define _mtsProxyBaseCommon_h

#include <cisstCommon/cmnGenericObject.h>
#include <cisstCommon/cmnClassRegister.h>
#include <cisstMultiTask/mtsTaskManager.h>
#include <cisstMultiTask/mtsTaskManagerProxy.h>
#include <cisstOSAbstraction/osaThread.h>
#include <cisstMultiTask/mtsExport.h>

#include <IceUtil/IceUtil.h>
#include <Ice/Ice.h>

#include <string>

//#define ICE_TASKMANAGER_COMMUNICATOR_IDENTITY "TaskManagerCommunicator"

class CISST_EXPORT mtsProxyBaseCommon {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

protected:
    //--------------------- Auxiliary Class Definition ----------------------//
    class ThreadArguments {
    public:
        mtsTaskManager * taskManager;
        mtsProxyBaseCommon * proxy;
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

    //-------------------------- Thread Management --------------------------//
    /*! Was the initiliazation successful? */
    bool InitSuccessFlag;

    /*! Is this thread running? */
    bool RunningFlag;

    /*! Worker thread for network communication */
    osaThread WorkerThread;

    /*! Containers for thread creation */
    ProxyWorker ProxyWorkerInfo;
    ThreadArguments Arguments;    

    //---------------------------- ICE Related ------------------------------//
    /*! Settings for ICE components */
    std::string TaskManagerCommunicatorIdentity;

    /*! Ice communicator for proxy */
    Ice::CommunicatorPtr IceCommunicator;

    /*! Ice module initialization */
    virtual void Init(void) = 0;

    /*! Define a string that represents unique ID. */
    virtual std::string GetCommunicatorIdentity() const = 0;

    //
    // TODO: should replace this with map<communicatorPtr, mtsTaskManagerProxy *>
    //
    //static Ice::CommunicatorPtr communicator;
    //Ice::CommunicatorPtr communicator;

public:
    mtsProxyBaseCommon(void);
    virtual ~mtsProxyBaseCommon();

    /*! Initialize and start a proxy. Returns immediately. */
    virtual void StartProxy(mtsTaskManager * callingTaskManager) = 0;
    
    /*! Called when the worker thread ends. */
    virtual void OnThreadEnd(void) = 0;

    //------------------------------- Getters -------------------------------//
    inline const bool IsInitalized() const  { return InitSuccessFlag; }
    
    inline const bool IsRunning() const     { return RunningFlag; }    

    inline Ice::CommunicatorPtr GetIceCommunicator() const { return IceCommunicator; }    
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsProxyBaseCommon)

#endif // _mtsProxyBaseCommon_h
