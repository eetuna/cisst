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
template<class _ArgumentType>
class CISST_EXPORT mtsProxyBaseServer : public mtsProxyBaseCommon<_ArgumentType> {

protected:
    Ice::ObjectAdapterPtr IceAdapter;
    Ice::ObjectPtr Servant;    

    virtual Ice::ObjectPtr CreateServant() = 0;

    void Init(void)
    {
        try {
            //Ice::InitializationData initData;
            //initData.properties = Ice::createProperties();
            //initData.properties->load(PropertyFileName);           
            //IceCommunicator = Ice::initialize(initData);
            IceCommunicator = Ice::initialize();
            
            // Create Logger
            Logger = IceCommunicator->getLogger();

            // Create an adapter (server-side only)
            //IceAdapter = IceCommunicator->createObjectAdapter(PropertyName);
            IceAdapter = IceCommunicator->createObjectAdapterWithEndpoints(
                "TaskManagerServerAdapter", "tcp -p 10705");

            // Create a servant
            Servant = CreateServant();

            // Inform the object adapter of the presence of a new servant
            //IceAdapter->add(Servant, IceCommunicator->stringToIdentity(PropertyName));
            IceAdapter->add(Servant, IceCommunicator->stringToIdentity("TaskManagerServerSender"));

            // Activate the adapter. The adapter is initially created in a 
            // holding state. The server starts to process incoming requests
            // from clients as soon as the adapter is activated.
            IceAdapter->activate();

            InitSuccessFlag = true;
            
            Logger->trace("mtsProxyBaseServer", "mtsProxyBaseServer initialization: success");
        } catch (const Ice::Exception& e) {
            Logger->trace("mtsProxyBaseServer", "Server proxy initialization error");
            Logger->trace("mtsProxyBaseServer", e.what());
        } catch (const char * msg) {
            Logger->trace("mtsProxyBaseServer", "Server proxy initialization error");
            Logger->trace("mtsProxyBaseServer", msg);
        }

        if (!InitSuccessFlag) {
            try {
                IceCommunicator->destroy();
            } catch (const Ice::Exception& e) {
                Logger->trace("mtsProxyBaseServer", "Server proxy clean-up error");
                Logger->trace("mtsProxyBaseServer", e.what());
            }
        }
    }

public:
    mtsProxyBaseServer(const std::string& propertyFileName, const std::string& propertyName) 
        : mtsProxyBaseCommon(propertyFileName, propertyName) 
    {}
    virtual ~mtsProxyBaseServer() {}
    
    virtual void StartProxy(_ArgumentType * callingClass) = 0;

    virtual void OnThreadEnd(void)
    {
        //
        // TODO: EndServant() should be placed somewhere in this method.
        //
        if (IceCommunicator) {
            try {                
                IceCommunicator->destroy();
                RunningFlag = false;

                Logger->trace("mtsProxyBaseServer", "Server proxy clean-up success.");
                //CMN_LOG_CLASS(3) << "Proxy cleanup succeeded." << std::endl;
            } catch (const Ice::Exception& e) {
                Logger->trace("mtsProxyBaseServer", "Server proxy clean-up failed.");
                Logger->trace("mtsProxyBaseServer", e.what());
                //CMN_LOG_CLASS(3) << "Proxy cleanup failed: " << e << std::endl;
            }
        } 
    }
};

#endif // _mtsProxyBaseServer_h

