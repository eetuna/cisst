/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsProxyBaseServer.h 142 2009-03-11 23:02:34Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-03-17

  (C) Copyright 2009-2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#ifndef _mtsProxyBaseServer_h
#define _mtsProxyBaseServer_h

#include <cisstMultiTask/mtsProxyBaseCommon.h>
#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  This class inherits mtsProxyBaseCommon and implements the basic structure of
  ICE proxy object acting as a server. The actual processing routines should be
  implemented by a derived class.
*/
template<class _proxyOwner, class _clientProxyType>
class CISST_EXPORT mtsProxyBaseServer : public mtsProxyBaseCommon<_proxyOwner> {

public:
    typedef mtsProxyBaseCommon<_proxyOwner> BaseType;
    typedef _clientProxyType ClientProxyType;

    /*! Typedef for proxy connection id (defined by ICE). Set as Ice::Identity
        which can be transformed to std::string by Communicator->identityToString()
        See http://www.zeroc.com/doc/Ice-3.3.1/reference/Ice/Identity.html */
    typedef std::string ConnectionIDType;

    /*! Typedef for client id (defined by user) which is an interface name and
        of which value is human readable. */
    typedef std::string ClientIDType;

    /*! Start server proxy */
    virtual void Start(_proxyOwner * proxyOwner) = 0;

    /*! Terminate proxy. */
    virtual void Stop()
    {
        if (this->ProxyState != BaseType::PROXY_ACTIVE) {
            return;
        }

        if (this->ProxyState == BaseType::PROXY_ACTIVE) {
            ChangeProxyState(BaseType::PROXY_FINISHING);

            if (this->IceCommunicator) {                
                try {
                    this->IceCommunicator->destroy();
                    this->ChangeProxyState(BaseType::PROXY_FINISHED);
                    this->IceLogger->trace("mtsProxyBaseServer", "Server proxy clean-up success.");
                } catch (const Ice::Exception & e) {
                    this->IceLogger->error("mtsProxyBaseServer: Server proxy clean-up failure.");
                    this->IceLogger->trace("mtsProxyBaseServer", e.what());
                }
            }
        }
    }

    //-------------------------------------------------------------------------
    //  Networking: ICE
    //-------------------------------------------------------------------------
protected:
    /*! ICE Objects */
    Ice::ObjectAdapterPtr IceAdapter;
    Ice::ObjectPtr Servant;

    /*! Endpoint information for a client to connect to server proxy */
    const std::string AdapterName;
    const std::string EndpointInfo;
    const std::string CommunicatorID;

    /*! Create servant object */
    virtual Ice::ObjectPtr CreateServant() = 0;

    /*! Initialize server proxy */
    void IceInitialize(void)
    {
        try {
            ChangeProxyState(BaseType::PROXY_INITIALIZING);

            Ice::InitializationData initData;
            //initData.logger = new typename BaseType::ProxyLoggerForCISST();
            initData.logger = new typename BaseType::ProxyLogger();
            initData.properties = Ice::createProperties();
            // There are two different modes of using implicit context: 
            // shared vs. PerThread.
            // (see http://www.zeroc.com/doc/Ice-3.3.1/manual/Adv_server.33.12.html)
            initData.properties->setProperty("Ice.ImplicitContext", "Shared");
            //initData.properties->load(IcePropertyFileName);           
            this->IceCommunicator = Ice::initialize(initData);
            
            // Create Logger
            this->IceLogger = this->IceCommunicator->getLogger();

            // Create an adapter (server-side only)
            IceAdapter = this->IceCommunicator->
                createObjectAdapterWithEndpoints(AdapterName, EndpointInfo);

            // Create a servant
            Servant = CreateServant();

            // Inform the object adapter of the presence of a new servant
            IceAdapter->add(Servant, this->IceCommunicator->stringToIdentity(CommunicatorID));

            // Activate the adapter. The adapter is initially created in a 
            // holding state. The server starts to process incoming requests
            // from clients as soon as the adapter is activated.
            IceAdapter->activate();

            this->InitSuccessFlag = true;
            this->Runnable = true;
            
            ChangeProxyState(BaseType::PROXY_READY);

            this->IceLogger->trace("mtsProxyBaseServer", "Server proxy initialization success.");
        } catch (const Ice::Exception& e) {
            if (this->IceLogger) {
                this->IceLogger->error("mtsProxyBaseServer: Server proxy initialization error");
                this->IceLogger->trace("mtsProxyBaseServer", e.what());
            } else {
                CMN_LOG_RUN_ERROR << "mtsProxyBaseServer: Server proxy initialization error." << std::endl;
                CMN_LOG_RUN_ERROR << "mtsProxyBaseServer: " << e.what() << std::endl;
            }
        } catch (const char * msg) {
            if (this->IceLogger) {
                this->IceLogger->error("mtsProxyBaseServer: Server proxy initialization error");
                this->IceLogger->trace("mtsProxyBaseServer", msg);
            } else {
                CMN_LOG_RUN_ERROR << "mtsProxyBaseServer: Server proxy initialization error." << std::endl;
                CMN_LOG_RUN_ERROR << "mtsProxyBaseServer: " << msg << std::endl;
            }
        }

        if (!this->InitSuccessFlag) {
            try {
                this->IceCommunicator->destroy();
            } catch (const Ice::Exception & e) {
                if (this->IceLogger) {
                    this->IceLogger->error("mtsProxyBaseServer: Server proxy clean-up error");
                    this->IceLogger->trace("mtsProxyBaseServer", e.what());
                } else {
                    CMN_LOG_RUN_ERROR << "mtsProxyBaseServer: Server proxy clean-up error." << std::endl;
                    CMN_LOG_RUN_ERROR << e.what() << std::endl;
                }
            }
        }
    }

    /*! Change the proxy state as active. */
    void SetAsActiveProxy(void) {
        ChangeProxyState(BaseType::PROXY_ACTIVE);
    }

    /*! Check if this proxy is active */
    bool IsActiveProxy(void) const {
        return (ProxyState == BaseType::PROXY_ACTIVE);
    }

    /*! Shutdown the current session for graceful termination */
    void ShutdownSession(const Ice::Current & current) {
        current.adapter->getCommunicator()->shutdown();
        BaseType::ShutdownSession();
    }

    //-------------------------------------------------------------------------
    //  Connection and Client Proxy Management
    //-------------------------------------------------------------------------
protected:
    /*! Lookup table to fetch connection id with client id */
    typedef std::map<ClientIDType, ConnectionIDType> ConnectionIDMapType;
    ConnectionIDMapType ConnectionIDMap;

    /*! Container to manage connected client proxies based on its connection id */
    typedef std::map<ConnectionIDType, ClientProxyType*> ClientProxyMapType;
    ClientProxyMapType ClientProxyMap;

    /*! When a client proxy is connected to this server proxy, add it to client 
        proxy map with a key of connection id */
    bool ReceiveAddClient(const ConnectionIDType & connectionID, const ClientProxyType & clientProxy)
    {
        if (GetClientProxyByConnectionID(connectionID)) {
            this->IceLogger->trace("WARNING: duplicate connection id", connectionID);
            return false;
        }

        ClientProxyMap.insert(make_pair(connectionID, &clientProxy));

        return (GetClientProxyByConnectionID(connectionID) != NULL);
    }

    ClientProxyType * GetClientProxyByConnectionID(const ConnectionIDType & connectionID) const
    {
        ClientProxyMapType::const_iterator it = ClientProxyMap.find(connectionID);
        if (it == ClientProxyMap.end()) {
            this->IceLogger->trace("WARNING: can't find client proxy with connection id", connectionID);
            return NULL;
        }

        return it->second;
    }

    ClientProxyType * GetClientProxyByClientID(const ClientIDType & clientID) const
    {
        // Fetch a connection id from connection id map
        ConnectionIDMapType::const_iterator it = ConnectionIDMap.find(clientID);
        if (it == ConnectionIDMap.end()) {
            this->IceLogger->trace("WARNING: can't find client proxy with client id", clientID);
            return NULL;
        }

        return GetClientProxyByConnectionID(it->second);
    }

public:
    mtsProxyBaseServer(const std::string & adapterName,
                       const std::string & endpointInfo,
                       const std::string & communicatorID):
        BaseType("", "", BaseType::PROXY_SERVER),
        AdapterName(adapterName),
        EndpointInfo(endpointInfo),
        CommunicatorID(communicatorID)        
    {}
    virtual ~mtsProxyBaseServer() {}
};

#endif // _mtsProxyBaseServer_h

