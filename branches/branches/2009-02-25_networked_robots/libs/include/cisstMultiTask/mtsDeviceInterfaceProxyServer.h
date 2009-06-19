/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsDeviceInterfaceProxyServer.h 142 2009-03-11 23:02:34Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-04-24

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#ifndef _mtsDeviceInterfaceProxyServer_h
#define _mtsDeviceInterfaceProxyServer_h

#include <cisstMultiTask/mtsTaskInterface.h>
#include <cisstMultiTask/mtsDeviceInterfaceProxy.h>
#include <cisstMultiTask/mtsProxyBaseServer.h>
#include <cisstCommon/cmnDeSerializer.h>

#include <cisstMultiTask/mtsExport.h>

#include <string>

/*!
  \ingroup cisstMultiTask

  TODO: add class summary here
*/
class mtsTask;

class CISST_EXPORT mtsDeviceInterfaceProxyServer : public mtsProxyBaseServer<mtsTask> {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

    friend class TaskInterfaceServerI;

 public:
    typedef mtsProxyBaseServer<mtsTask> BaseType;

 protected:
    /*! Definitions for send thread */
    class TaskInterfaceServerI;
    typedef IceUtil::Handle<TaskInterfaceServerI> TaskInterfaceServerIPtr;
    TaskInterfaceServerIPtr Sender;

    mtsTask * ConnectedTask;
    
    /*! Resource clean-up */
    void OnClose();

    //-------------------------------------------------------------------------
    //  Serialization and Deserialization
    //-------------------------------------------------------------------------
    /*! Buffers for serialization and deserialization. */
    std::stringstream SerializationBuffer;
    std::stringstream DeSerializationBuffer;

    /*! Serializer and DeSerializer. */
    cmnSerializer * Serializer;
    cmnDeSerializer * DeSerializer;

    //-------------------------------------------------------------------------
    //  Proxy Implementation
    //-------------------------------------------------------------------------
    /*! Create a servant which serves TaskManager clients. */
    Ice::ObjectPtr CreateServant() {
        Sender = new TaskInterfaceServerI(IceCommunicator, Logger, this);
        return Sender;
    }
    
    /*! Entry point to run a proxy. */
    void Start(mtsTask * callingTask);

    /*! Start a send thread and wait for shutdown (blocking call). */
    void StartServer();

    /*! Thread runner */
    static void Runner(ThreadArguments<mtsTask> * arguments);

    /*! Clean up thread-related resources. */
    void OnThreadEnd();
    
public:
    mtsDeviceInterfaceProxyServer(const std::string& adapterName,
                                  const std::string& endpointInfo,
                                  const std::string& communicatorID):
        BaseType(adapterName, endpointInfo, communicatorID),
        ConnectedTask(0)
    {
        Serializer = new cmnSerializer(SerializationBuffer);
        DeSerializer = new cmnDeSerializer(DeSerializationBuffer);
    }

    ~mtsDeviceInterfaceProxyServer();
    
    void SetConnectedTask(mtsTask * task) { ConnectedTask = task; }

    //-------------------------------------------------------------------------
    //  Proxy Support
    //-------------------------------------------------------------------------
    /*! Update the information of all tasks. */
    const bool GetProvidedInterfaces(
        ::mtsDeviceInterfaceProxy::ProvidedInterfaceSequence & providedInterfaces);

    /*! Build a map of (command proxy id, actual command pointer) so that 
        an actual command object can be called by a remote command object proxy. */
    //void SendCommandProxyInfo(const ::mtsDeviceInterfaceProxy::CommandProxyInfo & info) const;

    /*! Execute actual command objects. */
    void ExecuteCommandVoid(const int commandSID) const;
    void ExecuteCommandWriteSerialized(const int commandSID, const std::string argument);
    void ExecuteCommandReadSerialized(const int commandSID, std::string & argument);
    void ExecuteCommandQualifiedReadSerialized(const int commandSID, const std::string argument1, std::string & argument2);

    //-------------------------------------------------------------------------
    //  Definition by mtsDeviceInterfaceProxy.ice
    //-------------------------------------------------------------------------
protected:
    class TaskInterfaceServerI : public mtsDeviceInterfaceProxy::TaskInterfaceServer,
                                 public IceUtil::Monitor<IceUtil::Mutex> 
    {
    private:
        Ice::CommunicatorPtr Communicator;
        bool Runnable;
        std::set<mtsDeviceInterfaceProxy::TaskInterfaceClientPrx> _clients;
        IceUtil::ThreadPtr Sender;
        Ice::LoggerPtr Logger;
        mtsDeviceInterfaceProxyServer * TaskInterfaceServer;

    public:
        TaskInterfaceServerI(const Ice::CommunicatorPtr& communicator, 
                             const Ice::LoggerPtr& logger,
                             mtsDeviceInterfaceProxyServer * taskInterfaceServer);

        void Start();
        void Run();
        void Destroy();

        void AddClient(const ::Ice::Identity&, const ::Ice::Current&);
        
        bool GetProvidedInterfaces(
            ::mtsDeviceInterfaceProxy::ProvidedInterfaceSequence&, 
            const ::Ice::Current&) const;
        
        void ExecuteCommandVoid(::Ice::Int, const ::Ice::Current&);
        void ExecuteCommandWriteSerialized(::Ice::Int, const ::std::string&, const ::Ice::Current&);
        void ExecuteCommandReadSerialized(::Ice::Int, ::std::string&, const ::Ice::Current&);
        void ExecuteCommandQualifiedReadSerialized(::Ice::Int, const ::std::string&, ::std::string&, const ::Ice::Current&);

    };

    //------------------ Methods for global task manager --------------------//
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsDeviceInterfaceProxyServer)

#endif // _mtsDeviceInterfaceProxyServer_h

