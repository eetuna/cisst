/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsTaskInterfaceProxyServer.h 142 2009-03-11 23:02:34Z mjung5 $

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

#ifndef _mtsTaskInterfaceProxyServer_h
#define _mtsTaskInterfaceProxyServer_h

#include <cisstMultiTask/mtsTaskInterface.h>
#include <cisstMultiTask/mtsTaskInterfaceProxy.h>
#include <cisstMultiTask/mtsProxyBaseServer.h>

#include <cisstMultiTask/mtsExport.h>

#include <string>

/*!
  \ingroup cisstMultiTask

  TODO: add class summary here
*/
class mtsTask;

class CISST_EXPORT mtsTaskInterfaceProxyServer : public mtsProxyBaseServer<mtsTask> {
    
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

    friend class TaskInterfaceServerI;

protected:
    //--------------------------- Protected member data ---------------------//
    /*! Definitions for send thread */
    class TaskInterfaceServerI;
    typedef IceUtil::Handle<TaskInterfaceServerI> TaskInterfaceServerIPtr;
    TaskInterfaceServerIPtr Sender;

    mtsTask * ConnectedTask;
    
    //-------------------------- Protected methods --------------------------//
    /*! Resource clean-up */
    void OnClose();

    //----------------------- Proxy Implementation --------------------------//

    //---------------------------- Proxy Support ----------------------------//
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
    mtsTaskInterfaceProxyServer(const std::string& adapterName,
                                const std::string& endpointInfo,
                                const std::string& communicatorID)
        : mtsProxyBaseServer(adapterName, endpointInfo, communicatorID),
          ConnectedTask(0)
    {}
    ~mtsTaskInterfaceProxyServer();
    
    void SetConnectedTask(mtsTask * task) { ConnectedTask = task; }

    //----------------------------- Proxy Support ---------------------------//
    /*! Update the information of all tasks. */
    const bool GetProvidedInterfaceSpecification(
        ::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq & specs);

    /*! Build a map of (command proxy id, actual command pointer) so that 
        an actual command object can be called by a remote command object proxy. */
    //void SendCommandProxyInfo(const ::mtsTaskInterfaceProxy::CommandProxyInfo & info) const;

    /*! Execute actual command objects. */
    void ExecuteCommandVoid(const int commandSID) const;
    void ExecuteCommandWrite(const int commandSID, const double argument) const;
    void ExecuteCommandRead(const int commandSID, double & argument);
    void ExecuteCommandQualifiedRead(
        const int commandSID, const double argument1, double & argument2);

    //-------------------------------------------------------------------------
    //  Definition by mtsTaskInterfaceProxy.ice
    //-------------------------------------------------------------------------
protected:
    class TaskInterfaceServerI : public mtsTaskInterfaceProxy::TaskInterfaceServer,
                                 public IceUtil::Monitor<IceUtil::Mutex> 
    {
    private:
        Ice::CommunicatorPtr Communicator;
        bool Runnable;
        std::set<mtsTaskInterfaceProxy::TaskInterfaceClientPrx> _clients;
        IceUtil::ThreadPtr Sender;
        Ice::LoggerPtr Logger;
        mtsTaskInterfaceProxyServer * TaskInterfaceServer;

    public:
        TaskInterfaceServerI(const Ice::CommunicatorPtr& communicator, 
                             const Ice::LoggerPtr& logger,
                             mtsTaskInterfaceProxyServer * taskInterfaceServer);

        void Start();
        void Run();
        void Destroy();

        void AddClient(const ::Ice::Identity&, const ::Ice::Current&);
        
        bool GetProvidedInterfaceSpecification(
            ::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq&, 
            const ::Ice::Current&) const;
        
        //void SendCommandProxyInfo(
        //    const ::mtsTaskInterfaceProxy::CommandProxyInfo&,
        //    const ::Ice::Current&);
        
        void ExecuteCommandVoid(::Ice::Int, const ::Ice::Current&);
        void ExecuteCommandWrite(::Ice::Int, ::Ice::Double, const ::Ice::Current&);
        void ExecuteCommandRead(::Ice::Int, ::Ice::Double&, const ::Ice::Current&);
        void ExecuteCommandQualifiedRead(::Ice::Int, ::Ice::Double, ::Ice::Double&, const ::Ice::Current&);
    };

    //------------------ Methods for global task manager --------------------//
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsTaskInterfaceProxyServer)

#endif // _mtsTaskInterfaceProxyServer_h

