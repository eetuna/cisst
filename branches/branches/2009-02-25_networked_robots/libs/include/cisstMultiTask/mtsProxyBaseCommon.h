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

//#include <cisstCommon/cmnGenericObject.h>
//#include <cisstCommon/cmnClassRegister.h>
#include <cisstOSAbstraction/osaThread.h>
#include <cisstMultiTask/mtsExport.h>

#include <IceUtil/IceUtil.h>
#include <Ice/Ice.h>

#include <string>

template<class _ArgumentType>
class CISST_EXPORT mtsProxyBaseCommon {
    
protected:
    //--------------------- Auxiliary Class Definition ----------------------//
    template<class _ArgumentType>
    class ThreadArguments {
    public:
        _ArgumentType * argument;
        mtsProxyBaseCommon * proxy;
        void (*Runner)(ThreadArguments<_ArgumentType> *);
    };

    template<class _ArgumentType>
    class ProxyWorker {
    public:
        ProxyWorker(void) {}
        virtual ~ProxyWorker(void) {}

        void * Run(ThreadArguments<_ArgumentType> * arguments) {
            arguments->Runner(arguments);
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
    ProxyWorker<_ArgumentType> ProxyWorkerInfo;
    ThreadArguments<_ArgumentType> ThreadArgumentsInfo;

    //---------------------------- ICE Related ------------------------------//
    /*! Property file name which contains settings for proxy configuration. */
    std::string PropertyFileName;

    /*! Property name (one of setting option in 'PropertyFileName' file). */
    std::string PropertyName;

    /*! Settings for ICE components */
    //std::string TaskManagerCommunicatorIdentity;

    /*! Ice communicator for proxy */
    Ice::CommunicatorPtr IceCommunicator;

    /*! Ice default logger */
    Ice::LoggerPtr Logger;

    /*! Ice run-time */
    Ice::CommunicatorPtr communicator;

    /*! Ice module initialization */
    virtual void Init(void) = 0;

    /*! Define a string that represents unique ID. */
    typedef enum {
        TASK_MANAGER_COMMUNICATOR,
    } CommunicatorIdentity;

    //std::string GetCommunicatorIdentity(CommunicatorIdentity id) const 
    //{
    //    switch (id) {
    //        case TASK_MANAGER_COMMUNICATOR:
    //            return "TaskManagerCommunicator";
    //    }

    //    return "NOT_DEFINED";
    //}

public:
    mtsProxyBaseCommon(const std::string& propertyFileName, const std::string& propertyName) 
        : RunningFlag(false), InitSuccessFlag(false), IceCommunicator(NULL),
          PropertyFileName(propertyFileName), PropertyName(propertyName)
    {
        //IceUtil::CtrlCHandler ctrCHandler(onCtrlC);
    }
    virtual ~mtsProxyBaseCommon() {}

    /*! Initialize and start a proxy. Returns immediately. */
    virtual void StartProxy(_ArgumentType * callingClass) = 0;
    
    /*! Called when the worker thread ends. */
    virtual void OnThreadEnd(void) = 0;

    //------------------------------- Getters -------------------------------//
    inline const bool IsInitalized() const  { return InitSuccessFlag; }
    
    inline const bool IsRunning() const     { return RunningFlag; }

    inline const Ice::LoggerPtr GetLogger() const { return Logger; }
};

#endif // _mtsProxyBaseCommon_h
