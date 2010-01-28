/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsProxyBaseClient.h 142 2009-03-11 23:02:34Z mjung5 $

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

#ifndef _mtsProxyBaseClient_h
#define _mtsProxyBaseClient_h

#include <cisstMultiTask/mtsProxyBaseCommon.h>
#include <cisstOSAbstraction/osaSleep.h>

#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  This class inherits mtsProxyBaseCommon and implements the basic structure of
  ICE proxy object acting as a client. The actual processing routines should be
  implemented by a derived class.

  Compared to mtsProxyBaseServer, this base class allows only one connection,
  i.e., one server proxy because one required interface can connect to only one
  provided interface in the current cisstMultiTask design.
*/

template<class _proxyOwner>
class CISST_EXPORT mtsProxyBaseClient: public mtsProxyBaseCommon<_proxyOwner> {

public:
    typedef mtsProxyBaseCommon<_proxyOwner> BaseType;

protected:
    /*! Start client proxy */
    virtual bool Start(_proxyOwner * proxyOwner) = 0;

    /*! Terminate proxy */
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
                    
                    ChangeProxyState(BaseType::PROXY_FINISHED);
                    this->IceLogger->trace("mtsProxyBaseClient", "Client proxy clean-up success.");
                } catch (const Ice::Exception& e) {
                    this->IceLogger->trace("mtsProxyBaseClient", "Client proxy clean-up failure.");
                    this->IceLogger->trace("mtsProxyBaseClient", e.what());
                }
            }
        }
    }

    //-------------------------------------------------------------------------
    //  Networking: ICE
    //-------------------------------------------------------------------------
protected:
    /*! ICE Object */
    Ice::ObjectPrx ProxyObject;

    /*! Endpoint information to connect to server proxy. This information is
        feteched from  the global component manager. */
    const std::string EndpointInfo;
    const std::string CommunicatorID;

    /*! Create client proxy object */
    virtual void CreateProxy() = 0;

    /*! Initialize client proxy */
    void IceInitialize(void)
    {
        try {
            ChangeProxyState(BaseType::PROXY_INITIALIZING);

            Ice::InitializationData initData;
            
            // Use the following line if you want to use CISST logger.
            initData.logger = new typename BaseType::CisstLogger();
            //initData.logger = new typename BaseType::ProxyLogger();

            initData.properties = Ice::createProperties();
            // There are two different modes of using implicit context: 
            // shared vs. PerThread.
            // (see http://www.zeroc.com/doc/Ice-3.3.1/manual/Adv_server.33.12.html)
            initData.properties->setProperty("Ice.ImplicitContext", "Shared");
            // For nested invocation
            initData.properties->setProperty("Ice.ThreadPool.Client.Size", "2");
            initData.properties->setProperty("Ice.ThreadPool.Client.SizeMax", "4");
            //initData.properties->load(IcePropertyFileName);
            this->IceCommunicator = Ice::initialize(initData);
            
            // Create a logger
            this->IceLogger = this->IceCommunicator->getLogger();

            // Create a proxy object from stringfied proxy information
            std::string stringfiedProxy = CommunicatorID + EndpointInfo;
            ProxyObject = this->IceCommunicator->stringToProxy(stringfiedProxy);

            // If a proxy fails to be created, an exception is thrown.
            CreateProxy();

            this->InitSuccessFlag = true;
            this->Runnable = true;
            
            ChangeProxyState(BaseType::PROXY_READY);

            this->IceLogger->trace("mtsProxyBaseClient", "Client proxy initialization success.");
        } catch (const Ice::Exception& e) {
            if (this->IceLogger) {
                this->IceLogger->error("mtsProxyBaseClient: Client proxy initialization error");
                this->IceLogger->trace("mtsProxyBaseClient", e.what());
            } else {
                std::cout << "mtsProxyBaseClient: Client proxy initialization error." << std::endl;
                std::cout << "mtsProxyBaseClient: " << e.what() << std::endl;
            }
        } catch (const char * msg) {
            if (this->IceLogger) {
                this->IceLogger->error("mtsProxyBaseClient: Client proxy initialization error");
                this->IceLogger->trace("mtsProxyBaseClient", msg);
            } else {
                std::cout << "mtsProxyBaseClient: Client proxy initialization error." << std::endl;
                std::cout << "mtsProxyBaseClient: " << msg << std::endl;
            }
        }

        if (!this->InitSuccessFlag) {
            try {
                this->IceCommunicator->destroy();
            } catch (const Ice::Exception& e) {
                if (this->IceLogger) {
                    this->IceLogger->error("mtsProxyBaseClient: Client proxy clean-up error");
                    this->IceLogger->trace("mtsProxyBaseClient", e.what());
                } else {
                    std::cerr << "mtsProxyBaseClient: Client proxy clean-up error." << std::endl;
                    std::cerr << e.what() << std::endl;
                }
            }
        }
    }

    // TODO: for safe/clean termination, should a client call shutdown() first?
    // TODO: do I need ShutdownSession() in a client proxy as well??

    ///*! Shutdown the current session for graceful termination */
    //void ShutdownSession(const Ice::Current & current) {
    //    current.adapter->getCommunicator()->shutdown();
    //    BaseType::ShutdownSession();
    //}

public:
    mtsProxyBaseClient(const std::string & endpointInfo, 
                       const std::string & communicatorID):
        BaseType("", "", BaseType::PROXY_CLIENT),
        EndpointInfo(endpointInfo),
        CommunicatorID(communicatorID)
    {}
    virtual ~mtsProxyBaseClient() {}
};

#endif // _mtsProxyBaseClient_h

