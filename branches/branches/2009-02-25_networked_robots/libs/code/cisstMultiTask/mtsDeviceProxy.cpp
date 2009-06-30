/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsDeviceProxy.cpp 291 2009-04-28 01:49:13Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-06-30

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstMultiTask/mtsDeviceProxy.h>
#include <cisstMultiTask/mtsDeviceInterface.h>
#include <cisstMultiTask/mtsDeviceInterfaceProxy.h>
#include <cisstMultiTask/mtsDeviceInterfaceProxyClient.h>
#include <cisstMultiTask/mtsCommandVoidProxy.h>
#include <cisstMultiTask/mtsCommandWriteProxy.h>
#include <cisstMultiTask/mtsCommandReadProxy.h>
#include <cisstMultiTask/mtsCommandQualifiedReadProxy.h>
#include <cisstMultiTask/mtsMulticastCommandWriteProxy.h>

CMN_IMPLEMENT_SERVICES(mtsDeviceProxy)

std::string mtsDeviceProxy::GetServerTaskProxyName(
    const std::string & resourceTaskName, const std::string & providedInterfaceName,
    const std::string & userTaskName, const std::string & requiredInterfaceName)
{
    return "Server-" +
           resourceTaskName + ":" +      // TS
           providedInterfaceName + "-" + // PI
           userTaskName + ":" +          // TC
           requiredInterfaceName;        // RI
}

std::string mtsDeviceProxy::GetClientTaskProxyName(
    const std::string & resourceTaskName, const std::string & providedInterfaceName,
    const std::string & userTaskName, const std::string & requiredInterfaceName)
{
    return "Client-" +
           resourceTaskName + ":" +      // TS
           providedInterfaceName + "-" + // PI
           userTaskName + ":" +          // TC
           requiredInterfaceName;        // RI
}

mtsDeviceInterface * mtsDeviceProxy::CreateProvidedInterfaceProxy(
    mtsDeviceInterfaceProxyClient * requiredInterfaceProxy,
    const mtsDeviceInterfaceProxy::ProvidedInterfaceInfo & providedInterfaceInfo)
{
    if (!requiredInterfaceProxy) {
        CMN_LOG_RUN_ERROR << "CreateProvidedInterfaceProxy: NULL required interface proxy." << std::endl;
        return NULL;
    }

    // Create a local provided interface (a provided interface proxy).
    mtsDeviceInterface * providedInterfaceProxy = AddProvidedInterface(providedInterfaceInfo.InterfaceName);
    if (!providedInterfaceProxy) {
        CMN_LOG_RUN_ERROR << "CreateProvidedInterfaceProxy: AddProvidedInterface failed." << std::endl;
        return NULL;
    }

    // Create command proxies.
    // CommandId is initially set to zero meaning that it needs to be updated later.
    // An actual value will be assigned later when UpdateCommandId() is executed.
    int commandId = NULL;
    std::string commandName, eventName;

#define ADD_COMMANDS_BEGIN(_commandType) \
    {\
        mtsCommand##_commandType##Proxy * newCommand##_commandType = NULL;\
        mtsDeviceInterfaceProxy::Command##_commandType##Sequence::const_iterator it\
            = providedInterfaceInfo.Commands##_commandType.begin();\
        for (; it != providedInterfaceInfo.Commands##_commandType.end(); ++it) {\
            commandName = it->Name;
#define ADD_COMMANDS_END \
        }\
    }

    // 2-1) Void
    ADD_COMMANDS_BEGIN(Void)
        newCommandVoid = new mtsCommandVoidProxy(
            commandId, requiredInterfaceProxy, commandName);
        CMN_ASSERT(newCommandVoid);
        providedInterfaceProxy->GetCommandVoidMap().AddItem(commandName, newCommandVoid);
    ADD_COMMANDS_END

    // 2-2) Write
    ADD_COMMANDS_BEGIN(Write)
        newCommandWrite = new mtsCommandWriteProxy(
            commandId, requiredInterfaceProxy, commandName);
        CMN_ASSERT(newCommandWrite);
        providedInterfaceProxy->GetCommandWriteMap().AddItem(commandName, newCommandWrite);
    ADD_COMMANDS_END

    // 2-3) Read
    ADD_COMMANDS_BEGIN(Read)
        newCommandRead = new mtsCommandReadProxy(
            commandId, requiredInterfaceProxy, commandName);
        CMN_ASSERT(newCommandRead);
        providedInterfaceProxy->GetCommandReadMap().AddItem(commandName, newCommandRead);
    ADD_COMMANDS_END

    // 2-4) QualifiedRead
    ADD_COMMANDS_BEGIN(QualifiedRead)
        newCommandQualifiedRead = new mtsCommandQualifiedReadProxy(
            commandId, requiredInterfaceProxy, commandName);
        CMN_ASSERT(newCommandQualifiedRead);
        providedInterfaceProxy->GetCommandQualifiedReadMap().AddItem(commandName, newCommandQualifiedRead);
    ADD_COMMANDS_END

    //{
    //    mtsFunctionVoid * newEventVoidGenerator = NULL;
    //    mtsDeviceInterfaceProxy::EventVoidSequence::const_iterator it =
    //        providedInterface.EventsVoid.begin();
    //    for (; it != providedInterface.EventsVoid.end(); ++it) {
    //        eventName = it->Name;            
    //        newEventVoidGenerator = new mtsFunctionVoid();
    //        newEventVoidGenerator->Bind(providedInterfaceProxy->AddEventVoid(eventName));            
    //    }
    //}
#define ADD_EVENTS_BEGIN(_eventType)\
    {\
        mtsFunction##_eventType * newEvent##_eventType##Generator = NULL;\
        mtsDeviceInterfaceProxy::Event##_eventType##Sequence::const_iterator it =\
        providedInterfaceInfo.Events##_eventType.begin();\
        for (; it != providedInterfaceInfo.Events##_eventType.end(); ++it) {\
            eventName = it->Name;
#define ADD_EVENTS_END \
        }\
    }

    // 3) Create event generator proxies.
    ADD_EVENTS_BEGIN(Void);
        newEventVoidGenerator = new mtsFunctionVoid();
        newEventVoidGenerator->Bind(providedInterfaceProxy->AddEventVoid(eventName));
    ADD_EVENTS_END;
    
    mtsMulticastCommandWriteProxy * newMulticastCommandWriteProxy = NULL;
    ADD_EVENTS_BEGIN(Write);
        newEventWriteGenerator = new mtsFunctionWrite();
        newMulticastCommandWriteProxy = new mtsMulticastCommandWriteProxy(
            it->Name, it->ArgumentTypeName);
        CMN_ASSERT(providedInterfaceProxy->AddEvent(it->Name, newMulticastCommandWriteProxy));
        CMN_ASSERT(newEventWriteGenerator->Bind(newMulticastCommandWriteProxy));
    ADD_EVENTS_END;

#undef ADD_COMMANDS_BEGIN
#undef ADD_COMMANDS_END
#undef ADD_EVENTS_BEGIN
#undef ADD_EVENTS_END

    return NULL;
}