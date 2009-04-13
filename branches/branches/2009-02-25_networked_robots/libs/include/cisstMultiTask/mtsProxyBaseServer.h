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

#include <set>

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
            IceCommunicator = Ice::initialize();

            std::string ObjectIdentityName = GetCommunicatorIdentity(TASK_MANAGER_COMMUNICATOR);
            std::string ObjectAdapterName = ObjectIdentityName + "Adapter";

            IceAdapter = IceCommunicator->createObjectAdapterWithEndpoints(
                    ObjectAdapterName.c_str(), // the name of the adapter
                    // instructs the adapter to listen for incoming requests 
                    // using the default protocol (TCP) at port number 10000
                    "default -p 10705");

            // Create a servant
            Servant = CreateServant();

            // Inform the object adapter of the presence of a new servant
            IceAdapter->add(Servant, IceCommunicator->stringToIdentity(ObjectIdentityName));

            InitSuccessFlag = true;
            Logger = IceCommunicator->getLogger();
            Logger->trace("mtsProxyBaseServer", "Server proxy initialization success");

            //CMN_LOG_CLASS(3) << "Server proxy initialization success. " << std::endl;
            return;
        } catch (const Ice::Exception& e) {
            Logger->trace("mtsProxyBaseServer", "Server proxy initialization error");
            Logger->trace("mtsProxyBaseServer", e.what());
            //CMN_LOG_CLASS(3) << "Server proxy initialization error: " << e << std::endl;
        } catch (const char * msg) {
            Logger->trace("mtsProxyBaseServer", "Server proxy initialization error");
            Logger->trace("mtsProxyBaseServer", msg);
            //CMN_LOG_CLASS(3) << "Server proxy initialization error: " << msg << std::endl;
        }

        if (IceCommunicator) {
            InitSuccessFlag = false;
            try {
                IceCommunicator->destroy();
            } catch (const Ice::Exception& e) {
                Logger->trace("mtsProxyBaseServer", "Server proxy clean-up error");
                Logger->trace("mtsProxyBaseServer", e.what());
                //CMN_LOG_CLASS(3) << "Server proxy initialization failed: " << e << std::endl;
            }
        }
    }

public:
    mtsProxyBaseServer(void) {}
    virtual ~mtsProxyBaseServer() {}
    
    //virtual void Runner(ThreadArguments<_ArgumentType> * arguments) = 0;

    virtual void StartProxy(_ArgumentType * callingClass) = 0;

    void ActivateServer()
    {
        // Activate the adapter. The adapter is initially created in a 
        // holding state. The server starts to process incoming requests
        // from clients as soon as the adapter is activated.
        IceAdapter->activate();

        // Blocking call
        IceCommunicator->waitForShutdown();
    }

    void OnThreadEnd(void)
    {
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

