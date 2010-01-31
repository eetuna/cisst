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

  This class inherits mtsProxyBaseCommon and implements the basic structure and 
  common functionalities of ICE proxy server. They include proxy server setup, 
  proxy initialization, multiple client manage, and connection management.
  Note that this proxy server manages multiple clients regardless of its type
  since the type is templated.
*/
template<class _proxyOwner, class _clientProxyType, class _clientIDType>
class CISST_EXPORT mtsProxyBaseServer : public mtsProxyBaseCommon<_proxyOwner>
{
public:
    typedef mtsProxyBaseCommon<_proxyOwner> BaseType;
    typedef _clientProxyType ClientProxyType;

    /*! Typedef for proxy connection id (defined by ICE). Set as Ice::Identity
        which can be transformed to std::string by identityToString().
        See http://www.zeroc.com/doc/Ice-3.3.1/reference/Ice/Identity.html */
    typedef std::string ConnectionIDType;

    /*! Typedef for client id */
    typedef _clientIDType ClientIDType;

    /*! Start proxy server */
    virtual bool Start(_proxyOwner * proxyOwner) = 0;

    /*! Connection monitor */
    virtual void Monitor(void) {
        //try {
        //    ProxyObject->ice_ping();
        //} catch (const Ice::Exception & e) {
        //    this->IceLogger->warning("mtsProxyBaseClient", e.what());
        //}
    }

    /*! Terminate proxy */
    virtual void Stop(void)
    {
        if (this->ProxyState != BaseType::PROXY_ACTIVE) {
            return;
        }

        ChangeProxyState(BaseType::PROXY_FINISHING);

        if (this->IceCommunicator) {                
            try {
                this->IceCommunicator->destroy();
                this->ChangeProxyState(BaseType::PROXY_FINISHED);
                this->IceLogger->trace("mtsProxyBaseServer", "Proxy server clean-up success.");
            } catch (const Ice::Exception & e) {
                this->IceLogger->error("mtsProxyBaseServer: Proxy server clean-up failure.");
                this->IceLogger->trace("mtsProxyBaseServer", e.what());
            }
        }
    }

    //-------------------------------------------------------------------------
    //  Networking: ICE
    //-------------------------------------------------------------------------
protected:
    /*! ICE objects */
    Ice::ObjectAdapterPtr IceAdapter;
    Ice::ObjectPtr Servant;

    /*! Endpoint information that clients uses to connect to this server */
    const std::string AdapterName;
    const std::string EndpointInfo;
    const std::string CommunicatorID;

    /*! Create ICE servant object */
    virtual Ice::ObjectPtr CreateServant() = 0;

    /*! Initialize server proxy */
    void IceInitialize(void)
    {
        try {
            BaseType::IceInitialize();

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
            ss << "AddProxyClient: duplicate client id: " << clientID;
            std::string s = ss.str();
            this->IceLogger->error(s);
            return false;
        }

        // Check the uniqueness of connectionID
        if (FindClientByConnectionID(connectionID)) {
            std::stringstream ss;
            ss << "AddProxyClient: duplicate connection id: " << connectionID;
            std::string s = ss.str();
            this->IceLogger->error(s);
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

    /*! Close a connection with a specific client */
    void CloseClient(const ConnectionIDType & connectionID) {
        ClientProxyType * clientProxy = GetClientByConnectionID(connectionID);
        if (!clientProxy) {
            std::stringstream ss;
            ss << "CloseClient: cannot find client with connection id: " << connectionID;
            std::string s = ss.str();
            this->IceLogger->warning(s);
            return;
        }

        // Close a connection explicitly (graceful closure)
        Ice::ConnectionPtr conn = ClientIDMap.begin()->second.ClientProxy->ice_getConnection();
        conn->close(false);
    }
    

    /*! Close all the connections with all the clients */

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

