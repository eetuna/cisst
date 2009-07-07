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

#include <cisstCommon/cmnDeSerializer.h>
#include <cisstMultiTask/mtsDeviceInterface.h>
#include <cisstMultiTask/mtsDeviceInterfaceProxy.h>
#include <cisstMultiTask/mtsProxyBaseServer.h>

#include <cisstMultiTask/mtsExport.h>

//#include <string>

/*!
  \ingroup cisstMultiTask

  TODO: add class summary here
*/
class mtsTask;
class mtsCommandVoidProxy;
class mtsCommandWriteProxy;

class CISST_EXPORT mtsDeviceInterfaceProxyServer : public mtsProxyBaseServer<mtsTask> {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    mtsDeviceInterfaceProxyServer(const std::string& adapterName,
                                  const std::string& endpointInfo,
                                  const std::string& communicatorID);
    ~mtsDeviceInterfaceProxyServer();
    
    /*! Entry point to run a proxy. */
    void Start(mtsTask * callingTask);

    /*! Stop the proxy. */
    void Stop();

    /*! Set a server task connected to this proxy server. This server task has 
        to provide at least one provided interface. */
    void SetConnectedTask(mtsTask * serverTask) { ConnectedTask = serverTask; }

protected:
    /*! Typedef for client proxy. */
    typedef mtsDeviceInterfaceProxy::DeviceInterfaceClientPrx DeviceInterfaceClientProxyType;

    /*! Typedef for base type. */
    typedef mtsProxyBaseServer<mtsTask> BaseType;

    /*! Pointer to the task connected. */
    mtsTask * ConnectedTask;

    /*! Connected client object. */
    DeviceInterfaceClientProxyType ConnectedClient;

    //-------------------------------------------------------------------------
    //  Proxy Implementation
    //-------------------------------------------------------------------------
    /*! Create a servant which serves TaskManager clients. */
    Ice::ObjectPtr CreateServant() {
        Sender = new DeviceInterfaceServerI(IceCommunicator, Logger, this);
        return Sender;
    }
    
    /*! Start a send thread and wait for shutdown (blocking call). */
    void StartServer();

    /*! Thread runner */
    static void Runner(ThreadArguments<mtsTask> * arguments);

    /*! Clean up thread-related resources. */
    void OnThreadEnd();
    
    /*! Definitions for send thread */
    class DeviceInterfaceServerI;
    typedef IceUtil::Handle<DeviceInterfaceServerI> DeviceInterfaceServerIPtr;
    DeviceInterfaceServerIPtr Sender;

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
    //  Processing Methods
    //-------------------------------------------------------------------------
    /*! Get the local provided interface from the task manager by name. */
    mtsProvidedInterface * GetProvidedInterface(
        const std::string resourceDeviceName, const std::string providedInterfaceName) const;

    //-------------------------------------------------------------------------
    //  Methods to Receive and Process Events (Client -> Server)
    //-------------------------------------------------------------------------
    /*! When a new client connects, add it to the client management list. */
    void ReceiveAddClient(const DeviceInterfaceClientProxyType & clientProxy);

    /*! Update the information of all tasks. */
    const bool ReceiveGetProvidedInterfaceInfo(
        const std::string & providedInterfaceName,
        ::mtsDeviceInterfaceProxy::ProvidedInterfaceInfo & providedInterfaceInfo);

    /*! Connect at server side. 
        This method creates a client task proxy (mtsDeviceProxy) and a required
        interface proxy (mtsRequiredInterface) at server side. */
    bool ReceiveConnectServerSide(
        const std::string & userTaskName, const std::string & requiredInterfaceName,
        const std::string & resourceTaskName, const std::string & providedInterfaceName);

    /*! Update command id. */
    void ReceiveGetCommandId(
        const std::string & clientTaskProxyName,
        mtsDeviceInterfaceProxy::FunctionProxySet & functionProxies);

    /*! Execute actual command objects. */
    void ReceiveExecuteCommandVoid(const int commandId) const;
    void ReceiveExecuteCommandWriteSerialized(const int commandId, const std::string argument);
    void ReceiveExecuteCommandReadSerialized(const int commandId, std::string & argument);
    void ReceiveExecuteCommandQualifiedReadSerialized(const int commandId, const std::string argument1, std::string & argument2);

    //-------------------------------------------------------------------------
    //  Methods to Send Events (Server -> Client)
    //-------------------------------------------------------------------------
public:
    void SendExecuteEventVoid(const int commandId) const;
    void SendExecuteEventWriteSerialized(const int commandId, const cmnGenericObject & argument);

    //-------------------------------------------------------------------------
    //  Definition by mtsDeviceInterfaceProxy.ice
    //-------------------------------------------------------------------------
protected:
    class DeviceInterfaceServerI : public mtsDeviceInterfaceProxy::DeviceInterfaceServer,
                                   public IceUtil::Monitor<IceUtil::Mutex> 
    {
    private:
        Ice::CommunicatorPtr Communicator;
        bool Runnable;
        IceUtil::ThreadPtr Sender;
        Ice::LoggerPtr Logger;
        mtsDeviceInterfaceProxyServer * DeviceInterfaceServer;

    public:
        DeviceInterfaceServerI(const Ice::CommunicatorPtr& communicator, 
                             const Ice::LoggerPtr& logger,
                             mtsDeviceInterfaceProxyServer * DeviceInterfaceServer);

        void Start();
        void Run();
        void Stop();

        void AddClient(const ::Ice::Identity&, const ::Ice::Current&);
        bool GetProvidedInterfaceInfo(const std::string &,
                                      ::mtsDeviceInterfaceProxy::ProvidedInterfaceInfo&,
                                      const ::Ice::Current&) const;
        bool ConnectServerSide(
            const std::string & userTaskName, const std::string & requiredInterfaceName,
            const std::string & resourceTaskName, const std::string & providedInterfaceName,
            const ::Ice::Current&);
        void GetCommandId(
            const std::string & clientTaskProxyName,
            mtsDeviceInterfaceProxy::FunctionProxySet&, const ::Ice::Current&) const;

        void ExecuteCommandVoid(::Ice::Int, const ::Ice::Current&);
        void ExecuteCommandWriteSerialized(::Ice::Int, const ::std::string&, const ::Ice::Current&);
        void ExecuteCommandReadSerialized(::Ice::Int, ::std::string&, const ::Ice::Current&);
        void ExecuteCommandQualifiedReadSerialized(::Ice::Int, const ::std::string&, ::std::string&, const ::Ice::Current&);

    };
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsDeviceInterfaceProxyServer)

#endif // _mtsDeviceInterfaceProxyServer_h

