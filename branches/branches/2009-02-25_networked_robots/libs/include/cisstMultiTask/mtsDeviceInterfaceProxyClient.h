/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsDeviceInterfaceProxyClient.h 142 2009-03-11 23:02:34Z mjung5 $

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

#ifndef _mtsDeviceInterfaceProxyClient_h
#define _mtsDeviceInterfaceProxyClient_h

#include <cisstMultiTask/mtsDeviceInterface.h>
#include <cisstMultiTask/mtsProxyBaseClient.h>
#include <cisstMultiTask/mtsDeviceInterfaceProxy.h>

#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  TODO: add class summary here
*/

class CISST_EXPORT mtsDeviceInterfaceProxyClient : public mtsProxyBaseClient<mtsTask> {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    mtsDeviceInterfaceProxyClient(const std::string & propertyFileName, 
                                  const std::string & propertyName);
    ~mtsDeviceInterfaceProxyClient();

protected:
    typedef mtsProxyBaseClient<mtsTask> BaseType;

    /*! Typedef for server proxy. */
    typedef mtsDeviceInterfaceProxy::DeviceInterfaceServerPrx DeviceInterfaceServerProxyType;

    /*! Send thread set up. */
    class DeviceInterfaceClientI;
    typedef IceUtil::Handle<DeviceInterfaceClientI> DeviceInterfaceClientIPtr;
    DeviceInterfaceClientIPtr Sender;

    /*! DeviceInterfaceServer proxy */
    DeviceInterfaceServerProxyType DeviceInterfaceServerProxy;
    
    //-------------------------------------------------------------------------
    //  Processing Methods
    //-------------------------------------------------------------------------
    /*! Buffers for serialization and deserialization. */
    std::stringstream SerializationBuffer;
    std::stringstream DeSerializationBuffer;

    /*! Per-proxy Serializer and DeSerializer. */
    cmnSerializer * Serializer;
    cmnDeSerializer * DeSerializer;

    void Serialize(const cmnGenericObject & argument, std::string & serializedData);

    /*! Create a proxy object and a send thread. */
    void CreateProxy() {
        DeviceInterfaceServerProxy = 
            mtsDeviceInterfaceProxy::DeviceInterfaceServerPrx::checkedCast(ProxyObject);
        if (!DeviceInterfaceServerProxy) {
            throw "Invalid proxy";
        }

        Sender = new DeviceInterfaceClientI(IceCommunicator, Logger, DeviceInterfaceServerProxy, this);
    }

    /*! Entry point to run a proxy. */
    void Start(mtsTask * callingTask);

    /*! Start a send thread and wait for shutdown (blocking call). */
    void StartClient();

    /*! Thread runner */
    static void Runner(ThreadArguments<mtsTask> * arguments);

    /*! Clean up thread-related resources. */
    void OnThreadEnd();

    //-------------------------------------------------------------------------
    //  Methods to Receive and Process Events
    //-------------------------------------------------------------------------
    void ReceiveUpdateCommandId(const mtsDeviceInterfaceProxy::FunctionProxySet & functionProxies);

    //-------------------------------------------------------------------------
    //  Methods to Send Events
    //-------------------------------------------------------------------------
public:
    const bool SendGetProvidedInterfaces(
        mtsDeviceInterfaceProxy::ProvidedInterfaceSequence & providedInterfaces) const;

    bool SendConnectServerSide(
        const std::string & userTaskName, const std::string & requiredInterfaceName,
        const std::string & resourceTaskName, const std::string & providedInterfaceName);

    void SendExecuteCommandVoid(const int commandId) const;
    void SendExecuteCommandWriteSerialized(const int commandId, const cmnGenericObject & argument);
    void SendExecuteCommandReadSerialized(const int commandId, cmnGenericObject & argument);
    void SendExecuteCommandQualifiedReadSerialized(const int commandId, const std::string & argument1, std::string & argument2);

    //-------------------------------------------------------------------------
    //  Definition by mtsDeviceInterfaceProxy.ice
    //-------------------------------------------------------------------------
protected:
    class DeviceInterfaceClientI : public mtsDeviceInterfaceProxy::DeviceInterfaceClient,
                               public IceUtil::Monitor<IceUtil::Mutex>
    {
    private:
        Ice::CommunicatorPtr Communicator;
        bool Runnable;
        
        IceUtil::ThreadPtr Sender;
        Ice::LoggerPtr Logger;
        mtsDeviceInterfaceProxy::DeviceInterfaceServerPrx Server;
        mtsDeviceInterfaceProxyClient * DeviceInterfaceClient;

    public:
        DeviceInterfaceClientI(const Ice::CommunicatorPtr& communicator,                           
                               const Ice::LoggerPtr& logger,
                               const mtsDeviceInterfaceProxy::DeviceInterfaceServerPrx& server,
                               mtsDeviceInterfaceProxyClient * DeviceInterfaceClient);

        void Start();
        void Run();
        void Destroy();

        void UpdateCommandId(const ::mtsDeviceInterfaceProxy::FunctionProxySet&, const ::Ice::Current&) const;
    };
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsDeviceInterfaceProxyClient)

#endif // _mtsDeviceInterfaceProxyClient_h
