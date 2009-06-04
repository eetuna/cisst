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

#include <cisstMultiTask/mtsTaskInterface.h>
#include <cisstMultiTask/mtsProxyBaseClient.h>
#include <cisstMultiTask/mtsDeviceInterfaceProxy.h>

#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  TODO: add class summary here
*/

class CISST_EXPORT mtsDeviceInterfaceProxyClient : public mtsProxyBaseClient<mtsTask> {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

 public:
    typedef mtsProxyBaseClient<mtsTask> BaseType;

 protected:
    /*! Send thread set up. */
    class TaskInterfaceClientI;
    typedef IceUtil::Handle<TaskInterfaceClientI> TaskInterfaceClientIPtr;
    TaskInterfaceClientIPtr Sender;

    /*! TaskInterfaceServer proxy */
    mtsDeviceInterfaceProxy::TaskInterfaceServerPrx TaskInterfaceServer;

 public:
    mtsDeviceInterfaceProxyClient(const std::string & propertyFileName, 
                                  const std::string & propertyName) :
        BaseType(propertyFileName, propertyName)
    {}
    ~mtsDeviceInterfaceProxyClient() {}

    /*! Create a proxy object and a send thread. */
    void CreateProxy() {
        TaskInterfaceServer = 
            mtsDeviceInterfaceProxy::TaskInterfaceServerPrx::checkedCast(ProxyObject);
        if (!TaskInterfaceServer) {
            throw "Invalid proxy";
        }

        Sender = new TaskInterfaceClientI(IceCommunicator, Logger, TaskInterfaceServer, this);
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
    //  Send Methods
    //-------------------------------------------------------------------------
    const bool GetProvidedInterfaceSpecification(
        mtsDeviceInterfaceProxy::ProvidedInterfaceSpecificationSeq & specs) const;

    //void SendCommandProxyInfo(mtsDeviceInterfaceProxy::CommandProxyInfo & info) const;

    void InvokeExecuteCommandVoid(const int commandSID) const;
    void InvokeExecuteCommandWrite(const int commandSID, const cmnDouble & argument) const;
    void InvokeExecuteCommandRead(const int commandSID, cmnDouble & argument);
    void InvokeExecuteCommandQualifiedRead(const int commandSID, const cmnDouble & argument1, cmnDouble & argument2);

    void InvokeExecuteCommandWriteSerialized(const int commandSID, const std::string & argument) const;
    void InvokeExecuteCommandReadSerialized(const int commandSID, std::string & argument);
    void InvokeExecuteCommandQualifiedReadSerialized(const int commandSID, const std::string & argument1, std::string & argument2);

    //-------------------------------------------------------------------------
    //  Definition by mtsDeviceInterfaceProxy.ice
    //-------------------------------------------------------------------------
protected:
    class TaskInterfaceClientI : public mtsDeviceInterfaceProxy::TaskInterfaceClient,
                               public IceUtil::Monitor<IceUtil::Mutex>
    {
    private:
        Ice::CommunicatorPtr Communicator;
        bool Runnable;
        
        IceUtil::ThreadPtr Sender;
        Ice::LoggerPtr Logger;
        mtsDeviceInterfaceProxy::TaskInterfaceServerPrx Server;
        mtsDeviceInterfaceProxyClient * TaskInterfaceClient;

    public:
        TaskInterfaceClientI(const Ice::CommunicatorPtr& communicator,                           
                           const Ice::LoggerPtr& logger,
                           const mtsDeviceInterfaceProxy::TaskInterfaceServerPrx& server,
                           mtsDeviceInterfaceProxyClient * TaskInterfaceClient);

        void Start();
        void Run();
        void Destroy();
    };
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsDeviceInterfaceProxyClient)

#endif // _mtsDeviceInterfaceProxyClient_h

