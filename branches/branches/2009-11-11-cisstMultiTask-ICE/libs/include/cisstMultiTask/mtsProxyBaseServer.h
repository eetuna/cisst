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

  This base class supports multiple clients regardless the type of clients
  (type is templated) because a proxy server usually handles multiple clients.
  For example, one provided interface proxy (proxy server) should be able to
  handle multiple required interface proxy (proxy client).
*/
template<class _proxyOwner, class _clientProxyType, class _clientIDType>
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
    //typedef std::string ClientIDType;

    /*! Typedef for client id (provided interface proxy instance id) */
    typedef _clientIDType ClientIDType;

    /*! Start server proxy */
    virtual bool Start(_proxyOwner * proxyOwner) = 0;

    /*! Terminate proxy. */
    virtual void Stop(void)
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

            // Use the following line if you want to use CISST logger.
            //initData.logger = new typename BaseType::CisstLogger();
            initData.logger = new typename BaseType::ProxyLogger();

            // Create a set ICE proxy properties
            // TODO: It would be better if we could control these properties 
            // not within codes but using an external property file.
            initData.properties = Ice::createProperties();
            // There are two different modes of using implicit context: 
            // shared vs. PerThread.
            // (see http://www.zeroc.com/doc/Ice-3.3.1/manual/Adv_server.33.12.html)
            initData.properties->setProperty("Ice.ImplicitContext", "Shared");
            // For nested invocation
            initData.properties->setProperty("Ice.ThreadPool.Server.Size", "2");
            initData.properties->setProperty("Ice.ThreadPool.Server.SizeMax", "4");
            //initData.properties->load(IcePropertyFileName);
            this->IceCommunicator = Ice::initialize(initData);
            
            // Create a logger
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

    /*! Shutdown the current session for graceful termination */
    void ShutdownSession(const Ice::Current & current) {
        current.adapter->getCommunicator()->shutdown();
        BaseType::ShutdownSession();
    }

    //-------------------------------------------------------------------------
    //  Connection Management and Client Proxy Management
    //-------------------------------------------------------------------------
protected:
    /*! Client information */
    typedef struct {
        std::string ClientName;
        ClientIDType ClientID;
        ConnectionIDType ConnectionID;
        ClientProxyType ClientProxy;
    } ClientInformation;

    /*! Lookup table to fetch client information with ClientID */
    typedef std::map<ClientIDType, ClientInformation> ClientIDMapType;
    ClientIDMapType ClientIDMap;

    /*! Lookup table to fetch client information with ConnectionID */
    typedef std::map<ConnectionIDType, ClientInformation> ConnectionIDMapType;
    ConnectionIDMapType ConnectionIDMap;

    /*! When a client proxy is connected to this server proxy, add it to client 
        proxy map with a key of connection id */
    bool AddProxyClient(const std::string & clientName, const ClientIDType & clientID, 
        const ConnectionIDType & connectionID, ClientProxyType & clientProxy) 
    {
        // Check the uniqueness of clientID
        if (FindClientByClientID(clientID)) {
            std::stringstream ss;
            ss << "WARNING: duplicate client id: " << clientID;
            std::string s = ss.str();
            this->IceLogger->warning(s);
            return false;
        }

        // Check the uniqueness of connectionID
        if (FindClientByConnectionID(connectionID)) {
            std::stringstream ss;
            ss << "WARNING: duplicate connection id: " << connectionID;
            std::string s = ss.str();
            this->IceLogger->warning(s);
            return false;
        }

        ClientInformation client;
        client.ClientName = clientName;
        client.ClientID = clientID;
        client.ConnectionID = connectionID;
        client.ClientProxy = clientProxy;

        ClientIDMap.insert(std::make_pair(clientID, client));
        ConnectionIDMap.insert(std::make_pair(connectionID, client));

        return (FindClientByClientID(clientID) && FindClientByConnectionID(connectionID));
    }

    /*! Return ClientIDType */
    ClientIDType GetClientID(const ConnectionIDType & connectionID) {
        ConnectionIDMapType::iterator it = ConnectionIDMap.find(connectionID);
        if (it == ConnectionIDMap.end()) {
            return 0;
        }
        return it->second.ClientID;
    }

    /*! Get an ICE proxy object using connection id to send a message to a client */
    ClientProxyType * GetClientByConnectionID(const ConnectionIDType & connectionID) {
        ConnectionIDMapType::iterator it = ConnectionIDMap.find(connectionID);
        if (it == ConnectionIDMap.end()) {
            return NULL;
        }
        return &(it->second.ClientProxy);
    }

    /*! Get an ICE proxy object using client id to send a message to a client */
    ClientProxyType * GetClientByClientID(const ClientIDType & clientID) {
        ClientIDMapType::iterator it = ClientIDMap.find(clientID);
        if (it == ClientIDMap.end()) {
            return NULL;
        }
        return &(it->second.ClientProxy);
    }

    /*! Check if there is an ICE proxy object using connection id */
    bool FindClientByConnectionID(const ConnectionIDType & connectionID) const {
        return (ConnectionIDMap.find(connectionID) != ConnectionIDMap.end());
    }

    /*! Check if there is an ICE proxy object using client id */
    bool FindClientByClientID(const ClientIDType & clientID) const {
        return (ClientIDMap.find(clientID) != ClientIDMap.end());
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

