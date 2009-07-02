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

CMN_IMPLEMENT_SERVICES(mtsDeviceProxy)

mtsDeviceProxy::~mtsDeviceProxy()
{
    FunctionVoidProxyMap.DeleteAll();
    FunctionWriteProxyMap.DeleteAll();
    FunctionReadProxyMap.DeleteAll();
    FunctionQualifiedReadProxyMap.DeleteAll();
}

/* Server task proxy naming rule:
    
   Server-TS:PI-TC:RI

   where TS: server task name
         PI: provided interface name
         TC: client task name
         RI: required interface name
*/
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

/* Client task proxy naming rule:
    
   Client-TS:PI-TC:RI

   where TS: server task name
         PI: provided interface name
         TC: client task name
         RI: required interface name
*/
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

mtsProvidedInterface * mtsDeviceProxy::CreateProvidedInterfaceProxy(
    mtsDeviceInterfaceProxyClient & requiredInterfaceProxy,
    const mtsDeviceInterfaceProxy::ProvidedInterfaceInfo & providedInterfaceInfo)
{
    //if (!requiredInterfaceProxy) {
    //    CMN_LOG_RUN_ERROR << "CreateProvidedInterfaceProxy: NULL required interface proxy." << std::endl;
    //    return NULL;
    //}

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
            commandId, &requiredInterfaceProxy, commandName);
        CMN_ASSERT(newCommandVoid);
        providedInterfaceProxy->GetCommandVoidMap().AddItem(commandName, newCommandVoid);
    ADD_COMMANDS_END

    // 2-2) Write
    ADD_COMMANDS_BEGIN(Write)
        newCommandWrite = new mtsCommandWriteProxy(
            commandId, &requiredInterfaceProxy, commandName);
        CMN_ASSERT(newCommandWrite);
        providedInterfaceProxy->GetCommandWriteMap().AddItem(commandName, newCommandWrite);
    ADD_COMMANDS_END

    // 2-3) Read
    ADD_COMMANDS_BEGIN(Read)
        newCommandRead = new mtsCommandReadProxy(
            commandId, &requiredInterfaceProxy, commandName);
        CMN_ASSERT(newCommandRead);
        providedInterfaceProxy->GetCommandReadMap().AddItem(commandName, newCommandRead);
    ADD_COMMANDS_END

    // 2-4) QualifiedRead
    ADD_COMMANDS_BEGIN(QualifiedRead)
        newCommandQualifiedRead = new mtsCommandQualifiedReadProxy(
            commandId, &requiredInterfaceProxy, commandName);
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

    return providedInterfaceProxy;
}

mtsRequiredInterface * mtsDeviceProxy::CreateRequiredInterfaceProxy(
    mtsProvidedInterface & providedInterface, const std::string & requiredInterfaceName)
{
    // Create a required Interface proxy (mtsRequiredInterface).
    mtsRequiredInterface * requiredInterfaceProxy = AddRequiredInterface(requiredInterfaceName);
    if (!requiredInterfaceProxy) {
        CMN_LOG_RUN_ERROR << "CreateRequiredInterfaceProxy: Cannot add required interface: "
            << requiredInterfaceName << std::endl;
        return NULL;
    }

    // Now, populate a required Interface proxy.
    // Get the lists of commands
    mtsFunctionVoid  * functionVoidProxy = NULL;
    mtsFunctionWrite * functionWriteProxy = NULL;
    mtsFunctionRead  * functionReadProxy = NULL;
    mtsFunctionQualifiedRead * functionQualifiedReadProxy = NULL;

    //std::vector<std::string> namesOfCommandsVoid = providedInterface.GetNamesOfCommandsVoid();
    //for (unsigned int i = 0; i < namesOfCommandsVoid.size(); ++i) {
    //    functionVoidProxy = new mtsFunctionVoid(providedInterface, namesOfCommandsVoid[i]);
    //    CMN_ASSERT(FunctionVoidProxyMap.AddItem(namesOfCommandsVoid[i], functionVoidProxy));
    //    CMN_ASSERT(requiredInterfaceProxy->AddFunction(namesOfCommandsVoid[i], *functionVoidProxy));
    //}
#define ADD_FUNCTION_PROXY_BEGIN(_commandType)\
    std::vector<std::string> namesOfCommands##_commandType = providedInterface.GetNamesOfCommands##_commandType##();\
    for (unsigned int i = 0; i < namesOfCommands##_commandType.size(); ++i) {\
        function##_commandType##Proxy = new mtsFunction##_commandType##(&providedInterface, namesOfCommands##_commandType##[i]);\
        CMN_ASSERT(Function##_commandType##ProxyMap.AddItem(namesOfCommands##_commandType[i], function##_commandType##Proxy));\
        CMN_ASSERT(requiredInterfaceProxy->AddFunction(namesOfCommands##_commandType##[i], *function##_commandType##Proxy));
#define ADD_FUNCTION_PROXY_END\
    }

    ADD_FUNCTION_PROXY_BEGIN(Void);
    ADD_FUNCTION_PROXY_END;

    ADD_FUNCTION_PROXY_BEGIN(Write);
    ADD_FUNCTION_PROXY_END;

    ADD_FUNCTION_PROXY_BEGIN(Read);
    ADD_FUNCTION_PROXY_END;

    ADD_FUNCTION_PROXY_BEGIN(QualifiedRead);
    ADD_FUNCTION_PROXY_END;

    // Get the lists of events
    mtsCommandVoidProxy  * actualCommandVoidProxy = NULL;
    mtsCommandWriteProxy * actualCommandWriteProxy = NULL;

    //std::vector<std::string> namesOfEventsVoid = providedInterface.GetNamesOfEventsVoid();
    //for (unsigned int i = 0; i < namesOfEventsVoid.size(); ++i) {
    //    // The fourth argument 'queued' should have to be false in order not to queue events.
    //    requiredInterfaceProxy->AddEventHandlerVoid(
    //        &mtsDeviceInterfaceProxyServer::EventHandlerVoid, this, namesOfEventsVoid[i], false);
    //}

    // CommandId is initially set to zero meaning that it needs to be updated.
    // An actual value will be assigned later when UpdateEventCommandId() is executed.
#define ADD_EVENT_PROXY_BEGIN(_eventType) \
    std::vector<std::string> namesOfEvents##_eventType = providedInterface.GetNamesOfEvents##_eventType();\
    for (unsigned int i = 0; i < namesOfEvents##_eventType.size(); ++i) {\
        actualCommand##_eventType##Proxy = new mtsCommand##_eventType##Proxy(NULL, this);\
        CMN_ASSERT(EventHandler##_eventType##Map.AddItem(namesOfEvents##_eventType[i], actualCommand##_eventType##Proxy));\
        CMN_ASSERT(requiredInterfaceProxy->EventHandlers##_eventType.AddItem(namesOfEvents##_eventType[i], actualCommand##_eventType##Proxy));
#define ADD_EVENT_PROXY_END \
    }
        
    //ADD_EVENT_PROXY_BEGIN(Void);
    //ADD_EVENT_PROXY_END;
    
    //ADD_EVENT_PROXY_BEGIN(Write);
    //ADD_EVENT_PROXY_END;

    // Using AllocateResources(), get pointers which has been allocated for this 
    // required interface and is thread-safe to use.
    unsigned int userId;
    userId = providedInterface.AllocateResources(requiredInterfaceProxy->GetName() + "Proxy");

    // Connect to the original device or task that provides allocated resources.
    requiredInterfaceProxy->ConnectTo(&providedInterface);
    if (!requiredInterfaceProxy->BindCommandsAndEvents(userId)) {
        CMN_LOG_RUN_ERROR << "CreateRequiredInterfaceProxy: BindCommandsAndEvents failed: "
            << userId << std::endl;
        return NULL;
    }

    return requiredInterfaceProxy;
}

void mtsDeviceProxy::GetFunctionPointers(
    mtsDeviceInterfaceProxy::FunctionProxySet & functionProxySet)
{
    mtsDeviceInterfaceProxy::FunctionProxyInfo element;

    //FunctionVoidProxyMapType::MapType::const_iterator it;
    //it = FunctionVoidProxyMap.GetMap().begin();
    //for (; it != FunctionVoidProxyMap.GetMap().end(); ++it) {
    //    element.Name = it->first;
    //    element.FunctionProxyId = reinterpret_cast<int>(it->second);
    //    functionProxy.FunctionVoidProxies.push_back(element);
    //}
#define GET_FUNCTION_PROXY_BEGIN(_commandType)\
    Function##_commandType##ProxyMapType::MapType::const_iterator it##_commandType;\
    it##_commandType = Function##_commandType##ProxyMap.GetMap().begin();\
    for (; it##_commandType != Function##_commandType##ProxyMap.GetMap().end(); ++it##_commandType) {\
        element.Name = it##_commandType->first;\
        element.FunctionProxyId = reinterpret_cast<int>(it##_commandType->second);\
        functionProxySet.Function##_commandType##Proxies.push_back(element)
#define GET_FUNCTION_PROXY_END\
    }

    GET_FUNCTION_PROXY_BEGIN(Void);
    GET_FUNCTION_PROXY_END;

    GET_FUNCTION_PROXY_BEGIN(Write);
    GET_FUNCTION_PROXY_END;

    GET_FUNCTION_PROXY_BEGIN(Read);
    GET_FUNCTION_PROXY_END;

    GET_FUNCTION_PROXY_BEGIN(QualifiedRead);
    GET_FUNCTION_PROXY_END;
}
