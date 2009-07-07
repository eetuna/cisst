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

#include <cisstOSAbstraction/osaThread.h>
#include <cisstCommon/cmnSerializer.h>
#include <cisstCommon/cmnDeSerializer.h>

#include <cisstMultiTask/mtsExport.h>

#include <IceUtil/IceUtil.h>
#include <Ice/Ice.h>

#include <string>

//-----------------------------------------------------------------------------
// Definitions for constant string values
//-----------------------------------------------------------------------------
/*! Implicit per-proxy context to set connection id. */
#define CONNECTION_ID "ConnectionID"

/*! The base port number is assigned as follows, which is not registered yet to 
IANA (Internet Assigned Numbers Authority) as of June 25th, 2009. 
See http://www.iana.org/assignments/port-numbers for more details.

Port number assignment for proxies
----------------------------------
Task manager layer  : 10705 (e.g. the global task manager)
Task layer          : 11705 (e.g. the server task)
*/
#define BASE_PORT_NUMBER_TASK_MANAGER_LAYER 10705
#define BASE_PORT_NUMBER_TASK_LAYER         11705

//-----------------------------------------------------------------------------
// Common Base Class Definitions
//-----------------------------------------------------------------------------
template<class _ArgumentType>
class CISST_EXPORT mtsProxyBaseCommon {
public:
    /*! Typedef for proxy type definition.

        Proxy server: a proxy that WORKS AS a server
        Proxy client: a proxy that WORKS AS a client
        Server proxy: a proxy FOR a server, which works at a client
        Client proxy: a proxy FOR a client, which works at a server
        
        That is, 'proxy server'='client proxy' and 'proxy client'='server proxy'.
    */
    typedef enum {
        PROXY_SERVER,    // Proxy server = Client proxy
        PROXY_CLIENT     // Proxy client = Server proxy
    } ProxyType;

    class ProxyLogger : public Ice::Logger
    {
    public:
        void print(const ::std::string& log) {
            Log(log);
        }
        void trace(const ::std::string& log1, const ::std::string& log2) {
            Log(log1, log2);
        }
        void warning(const ::std::string& log) {
            Log("##### WARNING: " + log);
        }
        void error(const ::std::string& log) {
            Log("##### ERROR: " + log);
        }

        void Log(const std::string& className, const std::string& description)
        {
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
            std::cout << log.c_str() << std::endl;
#endif
        }        
    };

protected:
    ProxyType ProxyTypeMember;

    //-----------------------------------------------------
    // Auxiliary Class Definition
    //-----------------------------------------------------
    template<class __ArgumentType>
    class ThreadArguments {
    public:
        _ArgumentType * argument;
        mtsProxyBaseCommon * proxy;
        void (*Runner)(ThreadArguments<__ArgumentType> *);
    };

    template<class __ArgumentType>
    class ProxyWorker {
    public:
        ProxyWorker(void) {}
        virtual ~ProxyWorker(void) {}

        void * Run(ThreadArguments<__ArgumentType> * arguments) {
            arguments->Runner(arguments);
            return 0;
        }
    };

    template<class _SenderType>
    class SendThread : public IceUtil::Thread
    {
    private:
        const _SenderType Sender;

    public:
        SendThread(const _SenderType& sender) : Sender(sender) {}          
        virtual void run() { Sender->Run(); }
    };

    //-----------------------------------------------------
    // Thread Management
    //-----------------------------------------------------
    /*! Was the initiliazation successful? */
    bool InitSuccessFlag;

    /*! Is this proxy running? */
    bool Runnable;

    /*! Worker thread for network communication */
    osaThread WorkerThread;

    /*! Containers for thread creation */
    ProxyWorker<_ArgumentType> ProxyWorkerInfo;
    ThreadArguments<_ArgumentType> ThreadArgumentsInfo;

    //-----------------------------------------------------
    // ICE Related
    //-----------------------------------------------------
    /*! Property file name which contains settings for proxy configuration. */
    std::string PropertyFileName;

    /*! Property name (one of setting option in 'PropertyFileName' file). */
    std::string PropertyName;

    /*! Ice communicator for proxy */
    Ice::CommunicatorPtr IceCommunicator;

    /*! Ice default logger */
    Ice::LoggerPtr Logger;

    /*! Global UID */
    std::string GUID;

    /*! Ice module initialization */
    virtual void Init(void) = 0;

    /*! Get global UID of this object. This must be unique over networks. */
    //
    // TODO: IP + Process ID + object ID (???)
    //
    const std::string GetGUID() {
        if (GUID.empty()) {
            GUID = IceUtil::generateUUID();
        }
        return GUID;
    }

    //-----------------------------------------------------
    // Serialization and Deserialization
    //-----------------------------------------------------
    /*! Buffers for serialization and deserialization. */
    std::stringstream SerializationBuffer;
    std::stringstream DeSerializationBuffer;

    /*! Per-proxy Serializer and DeSerializer. */
    cmnSerializer * Serializer;
    cmnDeSerializer * DeSerializer;

    void Serialize(const cmnGenericObject & argument, std::string & serializedData)
    {
        SerializationBuffer.str("");    
        Serializer->Serialize(argument);

        serializedData = SerializationBuffer.str();
    }


public:
    mtsProxyBaseCommon(const std::string& propertyFileName, 
                       const std::string& propertyName,
                       const ProxyType proxyType):
        ProxyTypeMember(proxyType),
        InitSuccessFlag(false),
        Runnable(false),
        PropertyFileName(propertyFileName),
        PropertyName(propertyName),
        IceCommunicator(NULL),
        GUID("")
    {
        //IceUtil::CtrlCHandler ctrCHandler(onCtrlC);

        Serializer = new cmnSerializer(SerializationBuffer);
        DeSerializer = new cmnDeSerializer(DeSerializationBuffer);
    }

    virtual ~mtsProxyBaseCommon() 
    {
        delete Serializer;
        delete DeSerializer;
    }

    /*! Initialize and start a proxy. Returns immediately. */
    virtual void Start(_ArgumentType * callingClass) = 0;
    
    /*! End the proxy. */
    virtual void Stop() = 0;

    /*! Called when the worker thread ends. */
    virtual void OnThreadEnd(void) = 0;

    //------------------------------- Getters -------------------------------//
    inline const bool IsInitalized() const  { return InitSuccessFlag; }
    
    inline const Ice::LoggerPtr GetLogger() const { return Logger; }

    inline Ice::CommunicatorPtr GetIceCommunicator() const { return IceCommunicator; }
};

#endif // _mtsProxyBaseCommon_h
