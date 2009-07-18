/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsProxyBaseServer.h 142 2009-03-11 23:02:34Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-03-17

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
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

  TODO: add class summary here
*/
template<class _argumentType>
class CISST_EXPORT mtsProxyBaseServer : public mtsProxyBaseCommon<_argumentType> {
public:
    typedef mtsProxyBaseCommon<_argumentType> BaseType;
    
protected:
    Ice::ObjectAdapterPtr IceAdapter;
    Ice::ObjectPtr Servant;
    const std::string AdapterName;
    const std::string EndpointInfo;
    const std::string CommunicatorID;

    virtual Ice::ObjectPtr CreateServant() = 0;

    void Init(void)
    {
        try {
            Ice::InitializationData initData;
            initData.logger = new typename BaseType::ProxyLogger();
            initData.properties = Ice::createProperties();
            // There are two different modes of using implicit context: 
            // shared vs. PerThread.
            // (see http://www.zeroc.com/doc/Ice-3.3.1/manual/Adv_server.33.12.html)
            initData.properties->setProperty("Ice.ImplicitContext", "Shared");
            //initData.properties->load(PropertyFileName);           
            //IceCommunicator = Ice::initialize(initData);
            this->IceCommunicator = Ice::initialize(initData);
            
            // Create Logger
            this->Logger = this->IceCommunicator->getLogger();

            // Create an adapter (server-side only)
            //IceAdapter = IceCommunicator->createObjectAdapter(PropertyName);
            IceAdapter = this->IceCommunicator
                ->createObjectAdapterWithEndpoints(AdapterName, EndpointInfo);

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
            
            this->Logger->trace("mtsProxyBaseServer", "mtsProxyBaseServer initialization: success");
        } catch (const Ice::Exception& e) {
            this->Logger->trace("mtsProxyBaseServer", "Server proxy initialization error");
            this->Logger->trace("mtsProxyBaseServer", e.what());
        } catch (const char * msg) {
            this->Logger->trace("mtsProxyBaseServer", "Server proxy initialization error");
            this->Logger->trace("mtsProxyBaseServer", msg);
        }

        if (!this->InitSuccessFlag) {
            try {
                this->IceCommunicator->destroy();
            } catch (const Ice::Exception & e) {
                this->Logger->trace("mtsProxyBaseServer", "Server proxy clean-up error");
                this->Logger->trace("mtsProxyBaseServer", e.what());
            }
        }
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
    
    virtual void Start(_argumentType * callingClass) = 0;

    virtual void OnThreadEnd(void)
    {
        //
        // TODO: EndServant() should be placed somewhere in this method.
        //
        if (this->IceCommunicator) {
            try {                
                this->IceCommunicator->destroy();

                this->Logger->trace("mtsProxyBaseServer", "Server proxy clean-up success.");
            } catch (const Ice::Exception & e) {
                this->Logger->trace("mtsProxyBaseServer", "Server proxy clean-up failed.");
                this->Logger->trace("mtsProxyBaseServer", e.what());
            }
        } 
    }

    ///*! Returns the base port number for this object. This method only applies to the
    //    server type object such as the global task manager or the task server. */
    //virtual const unsigned int GetBasePortNumber() = 0;

    ///*! Returns the base port number for this object as string. */
    //virtual std::string GetBasePortNumberAsString() = 0;
};

#endif // _mtsProxyBaseServer_h

