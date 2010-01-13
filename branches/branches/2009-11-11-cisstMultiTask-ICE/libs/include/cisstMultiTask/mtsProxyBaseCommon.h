/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsProxyBaseCommon.h 142 2009-03-11 23:02:34Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-04-10

  (C) Copyright 2009-2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#ifndef _mtsProxyBaseCommon_h
#define _mtsProxyBaseCommon_h

#include <cisstOSAbstraction/osaThread.h>
#include <cisstOSAbstraction/osaMutex.h>
#include <cisstCommon/cmnSerializer.h>
#include <cisstCommon/cmnDeSerializer.h>

#include <cisstMultiTask/mtsConfig.h>

#include <cisstMultiTask/mtsExport.h>

#include <IceUtil/IceUtil.h>
#include <Ice/Ice.h>

/*!
  \ingroup cisstMultiTask

  This class implements core features for ICE proxy objects and is used by 
  mtsProxyBaseServer or mtsProxyBaseClient. That is, if a new type of proxy 
  server or proxy client class is to be defined using ICE, it should be derived
  from either mtsProxyBaseServer or mtsProxyBaseClient, respectively, rather 
  than directly from this common class.
*/

/*! Typedef for Command ID */
typedef cmnDeSerializer::TypeId CommandIDType;
typedef ::Ice::Long IceCommandIDType;

// TODO: Replace #define with other (const std::string??)
#define ConnectionIDKey "ConnectionID"

//-----------------------------------------------------------------------------
//  Common Base Class Definitions
//-----------------------------------------------------------------------------
template<class _proxyOwner>
class CISST_EXPORT mtsProxyBaseCommon {

public:
    /*! Implicit per-proxy context to set connection id. */
    //static const char * ConnectionIDKey;

    /*! Typedef for proxy type. */
    enum ProxyType { PROXY_SERVER, PROXY_CLIENT };

    /*! The proxy status definition. This is adopted from mtsTask.h with slight
        modification.

        PROXY_CONSTRUCTED  -- Set by mtsProxyBaseCommon constructor. 
                              Initial state.
        PROXY_INITIALIZING -- Set by either mtsProxyBaseServer::IceInitialize() or
                              mtsProxyBaseClient::IceInitialize().
                              This state means a proxy object is created but not 
                              yet successfully initialized.
        PROXY_READY        -- Set by either mtsProxyBaseServer::IceInitialize() or
                              mtsProxyBaseClient::IceInitialize().
                              This state represents that a proxy object is 
                              successfully initialized and is ready to run.
        PROXY_ACTIVE       -- Set by either mtsProxyBaseServer::SetAsActiveProxy() 
                              or mtsProxyBaseClient::SetAsActiveProxy().
                              If a proxy is in this state, it is running and can 
                              process events.
        PROXY_FINISHING    -- Set by either mtsProxyBaseServer::Stop() or
                              mtsProxyBaseClient::Stop() before trying to stop ICE 
                              proxy processing.
        PROXY_FINISHED     -- Set by either mtsProxyBaseServer::Stop() or
                              mtsProxyBaseClient::Stop() after successful clean-up.
    */
    enum ProxyStateType { 
        PROXY_CONSTRUCTED, 
        PROXY_INITIALIZING, 
        PROXY_READY,
        PROXY_ACTIVE, 
        PROXY_FINISHING, 
        PROXY_FINISHED 
    };

protected:
    ProxyType ProxyTypeMember;
    ProxyStateType ProxyState;

    //-----------------------------------------------------
    // Auxiliary Class Definitions
    //-----------------------------------------------------
    /*! Logger class using the internal logging mechanism of cisst */
    class ProxyLoggerForCISST : public Ice::Logger
    {
    public:
        void print(const ::std::string & message) {
            CMN_LOG_RUN_VERBOSE << "ICE: " << message << std::endl;
        }
        void trace(const ::std::string & category, const ::std::string & message) {
            CMN_LOG_RUN_DEBUG << "ICE: " << category << " :: " << message << std::endl;
        }
        void warning(const ::std::string & message) {
            CMN_LOG_RUN_WARNING << "ICE: " << message << std::endl;
        }
        void error(const ::std::string & message) {
            CMN_LOG_RUN_ERROR << "ICE: " << message << std::endl;
        }
    };

    /*! Logger class using OutputDebugString() on Windows */
    class ProxyLogger : public Ice::Logger
    {
    public:
        void print(const ::std::string & message) { 
            Log(message); 
        }
        void trace(const ::std::string & category, const ::std::string & message) {
            Log(category, message);
        }
        void warning(const ::std::string & message) {
            Log("##### WARNING: " + message);
        }
        void error(const ::std::string & message) {
            Log("##### ERROR: " + message);
        }

        void Log(const std::string & className, const std::string & description) {
            std::string log = className + ": ";
            log += description;
            Log(log);
        }

    protected:
        void Log(const std::string& log)
        {
#if (CISST_OS == CISST_WINDOWS)
            OutputDebugString(log.c_str());
#else
            CMN_LOG_RUN_VERBOSE << log << std::endl;
#endif
        }        
    };

    /* Internal class for thread arguments */
    template<class __proxyOwner>
    class ThreadArguments {
    public:
        _proxyOwner * argument;
        mtsProxyBaseCommon * proxy;
        void (*Runner)(ThreadArguments<__proxyOwner> *);
    };

    /* Internal class for proxy worker */
    template<class __proxyOwner>
    class ProxyWorker {
    public:
        ProxyWorker(void) {}
        virtual ~ProxyWorker(void) {}

        void * Run(ThreadArguments<__proxyOwner> * arguments) {
            arguments->Runner(arguments);
            return 0;
        }
    };

    /* Internal class for send thread */
    template<class _SenderType>
    class SenderThread : public IceUtil::Thread
    {
    private:
        const _SenderType Sender;

    public:
        SenderThread(const _SenderType& sender) : Sender(sender) {}          
        virtual void run() { Sender->Run(); }
    };

    //-----------------------------------------------------
    //  Thread Management
    //-----------------------------------------------------
    /*! Mutex to change the proxy state. */
    osaMutex StateChange;

    /*! The flag which is true only if all initiliazation process succeeded. */
    bool InitSuccessFlag;

    /*! The flag which is true only if this proxy is runnable. */
    bool Runnable;

    // TODO: is the following comment correct????
    /*! Set as true when a session is to be closed.
        For a client, this is set when a client notifies a server of disconnection.
        For a server, this is set when a client calls Shutdown() which allows safe
        and clean termination. */
    bool IsValidSession;

    /*! The worker thread that actually runs a proxy. */
    osaThread WorkerThread;

    /*! The arguments container used for thread creation */
    ProxyWorker<_proxyOwner> ProxyWorkerInfo;
    ThreadArguments<_proxyOwner> ThreadArgumentsInfo;

    /*! Helper function to change the proxy state in a thread-safe manner */
    void ChangeProxyState(const enum ProxyStateType newProxyState) {
        StateChange.Lock();
        ProxyState = newProxyState;
        StateChange.Unlock();
    }

    //-----------------------------------------------------
    //  ICE Related
    //-----------------------------------------------------
    /*! The name of a property file that configures proxy connection settings. */
    const std::string IcePropertyFileName;

    /*! The identity of an Ice object which can also be set through an Ice property file
        (not supported yet). */
    const std::string IceObjectIdentity;

    /*! The proxy communicator. */
    Ice::CommunicatorPtr IceCommunicator;

    /*! The Ice default logger. */
    Ice::LoggerPtr IceLogger;

    /*! The global unique id of this Ice object. */
    std::string IceGUID;

    /*! Initialize Ice module. */
    virtual void IceInitialize(void) = 0;

    /*! Return the global unique id of this object. Currently, IceUtil::generateUUID()
        is used to set the id which is guaranteed to be unique across networks by ICE.
        We can also use a combination of IP address (or MAC address), process id,
        and object id (or a pointer to this object) as the GUID. */
    std::string GetGUID() {
        if (IceGUID.empty()) {
            IceGUID = IceUtil::generateUUID();
        }
        return IceGUID;
    }

public:
    mtsProxyBaseCommon(const std::string& propertyFileName,
                       const std::string& objectIdentity,
                       const ProxyType& proxyType) :
        ProxyTypeMember(proxyType),
        ProxyState(PROXY_CONSTRUCTED),
        InitSuccessFlag(false),
        Runnable(false),
        IsValidSession(true),
        IcePropertyFileName(propertyFileName),
        IceObjectIdentity(objectIdentity),
        IceCommunicator(NULL),
        IceGUID("")
    {
        //IceUtil::CtrlCHandler ctrCHandler(onCtrlC);
    }

    virtual ~mtsProxyBaseCommon() {}

    /*! Initialize and start the proxy (returns immediately). */
    virtual void Start(_proxyOwner * proxyOwner) = 0;
    
    /*! Terminate the proxy. */
    virtual void Stop() = 0;

    /*! Close a session. */
    virtual void ShutdownSession() {
        IsValidSession = false;
    }

    //-----------------------------------------------------
    //  Getters
    //-----------------------------------------------------
    inline bool IsInitalized(void) const  { return InitSuccessFlag; }
    
    inline const Ice::LoggerPtr GetLogger(void) const { return IceLogger; }

    inline Ice::CommunicatorPtr GetIceCommunicator(void) const { return IceCommunicator; }

    /*! Base port numbers. These numbers are not yet registered to IANA (Internet 
        Assigned Numbers Authority) as of January 12th, 2010.
        See http://www.iana.org/assignments/port-numbers for more details. */
    inline static unsigned int GetBasePortNumberForGlobalComponentManager() { return 10705; }
    inline static unsigned int GetBasePortNumberForLocalComponentManager() { return 11705; }
};

#endif // _mtsProxyBaseCommon_h
