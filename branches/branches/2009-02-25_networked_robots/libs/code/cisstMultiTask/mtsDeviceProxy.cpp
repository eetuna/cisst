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
    EventVoidGeneratorProxyMap.DeleteAll();
    EventWriteGeneratorProxyMap.DeleteAll();
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
    mtsDeviceInterfaceProxyClient * proxyClient,
    const mtsDeviceInterfaceProxy::ProvidedInterfaceInfo & providedInterfaceInfo)
{
    if (!proxyClient) {
        CMN_LOG_RUN_ERROR << "CreateProvidedInterfaceProxy: NULL required interface proxy." << std::endl;
        return NULL;
    }

    // Create a local provided interface (a provided interface proxy).
    mtsDeviceInterface * providedInterfaceProxy = 
        AddProvidedInterface(providedInterfaceInfo.InterfaceName);
    if (!providedInterfaceProxy) {
        CMN_LOG_RUN_ERROR << "CreateProvidedInterfaceProxy: AddProvidedInterface failed." << std::endl;
        return NULL;
    }

    // Create command proxies.
    // CommandId is initially set to zero meaning that it needs to be updated later.
    // An actual value will be assigned later when UpdateCommandId() is executed.
    int commandId = 0;
    std::string commandName;

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
        newCommandVoid = new mtsCommandVoidProxy(commandId, proxyClient, commandName);
        providedInterfaceProxy->GetCommandVoidMap().AddItem(commandName, newCommandVoid);
    ADD_COMMANDS_END

    // 2-2) Write
    ADD_COMMANDS_BEGIN(Write)
        newCommandWrite = new mtsCommandWriteProxy(commandId, proxyClient, commandName);
        providedInterfaceProxy->GetCommandWriteMap().AddItem(commandName, newCommandWrite);
    ADD_COMMANDS_END

    // 2-3) Read
    ADD_COMMANDS_BEGIN(Read)
        newCommandRead = new mtsCommandReadProxy(commandId, proxyClient, commandName);
        providedInterfaceProxy->GetCommandReadMap().AddItem(commandName, newCommandRead);
    ADD_COMMANDS_END

    // 2-4) QualifiedRead
    ADD_COMMANDS_BEGIN(QualifiedRead)
        newCommandQualifiedRead = new mtsCommandQualifiedReadProxy(commandId, proxyClient, commandName);
        providedInterfaceProxy->GetCommandQualifiedReadMap().AddItem(commandName, newCommandQualifiedRead);
    ADD_COMMANDS_END

#undef ADD_COMMANDS_BEGIN
#undef ADD_COMMANDS_END

    // 3) Create event generator proxies.
    std::string eventName;

    mtsFunctionVoid * eventVoidGeneratorProxy = NULL;
    mtsDeviceInterfaceProxy::EventVoidSequence::const_iterator itEventVoid =
        providedInterfaceInfo.EventsVoid.begin();
    for (; itEventVoid != providedInterfaceInfo.EventsVoid.end(); ++itEventVoid) {
        eventName = itEventVoid->Name;
        eventVoidGeneratorProxy = new mtsFunctionVoid();        
        CMN_ASSERT(EventVoidGeneratorProxyMap.AddItem(eventName, eventVoidGeneratorProxy));
        
        CMN_ASSERT(eventVoidGeneratorProxy->Bind(providedInterfaceProxy->AddEventVoid(eventName)));
    }

    mtsFunctionWrite * eventWriteGeneratorProxy = NULL;
    mtsMulticastCommandWriteProxy * eventMulticastCommandProxy = NULL;

    mtsDeviceInterfaceProxy::EventWriteSequence::const_iterator itEventWrite =
        providedInterfaceInfo.EventsWrite.begin();
    for (; itEventWrite != providedInterfaceInfo.EventsWrite.end(); ++itEventWrite) {
        eventName = itEventWrite->Name;
        eventWriteGeneratorProxy = new mtsFunctionWrite();
        CMN_ASSERT(EventWriteGeneratorProxyMap.AddItem(eventName, eventWriteGeneratorProxy));

        // GOHOME: I should take care of the command id which is initially set as '0'.
        eventMulticastCommandProxy = new mtsMulticastCommandWriteProxy(0, proxyClient, eventName);
        CMN_ASSERT(providedInterfaceProxy->AddEvent(eventName, eventMulticastCommandProxy));
        CMN_ASSERT(eventWriteGeneratorProxy->Bind(eventMulticastCommandProxy));
    }
    
    return providedInterfaceProxy;
}

mtsRequiredInterface * mtsDeviceProxy::CreateRequiredInterfaceProxy(
    mtsProvidedInterface * providedInterface, const std::string & requiredInterfaceName,
    mtsDeviceInterfaceProxyServer * proxyServer)
{
    // Create a required Interface proxy (mtsRequiredInterface).
    mtsRequiredInterface * requiredInterfaceProxy = AddRequiredInterface(requiredInterfaceName);
    if (!requiredInterfaceProxy) {
        CMN_LOG_RUN_ERROR << "CreateRequiredInterfaceProxy: Cannot add required interface: "
            << requiredInterfaceName << std::endl;
        return NULL;
    }

    // Now, populate a required Interface proxy.
    
    // 1. Function proxies
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
    std::vector<std::string> namesOfCommands##_commandType = providedInterface->GetNamesOfCommands##_commandType##();\
    for (unsigned int i = 0; i < namesOfCommands##_commandType.size(); ++i) {\
        function##_commandType##Proxy = new mtsFunction##_commandType##(providedInterface, namesOfCommands##_commandType##[i]);\
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

    // 2. Event handler proxies
    std::string eventName;

    mtsCommandVoidProxy * actualEventVoidCommandProxy = NULL;
    std::vector<std::string> namesOfEventsVoid = providedInterface->GetNamesOfEventsVoid();
    for (unsigned int i = 0; i < namesOfEventsVoid.size(); ++i) {
        // CommandId is initially set to zero meaning that it needs to be updated.
        // An actual value will be assigned later when UpdateEventCommandId() is executed.
        eventName = namesOfEventsVoid[i];
        actualEventVoidCommandProxy = new mtsCommandVoidProxy(NULL, proxyServer, eventName);
        CMN_ASSERT(requiredInterfaceProxy->EventHandlersVoid.AddItem(
            eventName, actualEventVoidCommandProxy));
        CMN_ASSERT(EventHandlerVoidProxyMap.AddItem(eventName, actualEventVoidCommandProxy));
    }

    mtsCommandWriteProxy * actualEventWriteCommandProxy = NULL;    
    std::vector<std::string> namesOfEventsWrite = providedInterface->GetNamesOfEventsWrite();
    for (unsigned int i = 0; i < namesOfEventsWrite.size(); ++i) {
        eventName = namesOfEventsWrite[i];
        // CommandId is initially set to zero meaning that it needs to be updated.
        // An actual value will be assigned later when UpdateEventCommandId() is executed.
        actualEventWriteCommandProxy = new mtsCommandWriteProxy(NULL, proxyServer, eventName);
        CMN_ASSERT(EventHandlerWriteProxyMap.AddItem(eventName, actualEventWriteCommandProxy));
        CMN_ASSERT(requiredInterfaceProxy->EventHandlersWrite.AddItem(
            eventName, actualEventWriteCommandProxy));        
    }    

    // Using AllocateResources(), get pointers which has been allocated for this 
    // required interface and is thread-safe to use.
    unsigned int userId;
    std::string userName = requiredInterfaceProxy->GetName() + "Proxy";
    userId = providedInterface->AllocateResources(userName);

    // Connect to the original device or task that provides allocated resources.
    requiredInterfaceProxy->ConnectTo(providedInterface);
    if (!requiredInterfaceProxy->BindCommandsAndEvents(userId)) {
        CMN_LOG_RUN_ERROR << "CreateRequiredInterfaceProxy: BindCommandsAndEvents failed: "
            << userName << " with userId = " << userId << std::endl;
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

void mtsDeviceProxy::EventVoidHandlerProxyFunction()
{
}

void mtsDeviceProxy::EventWriteHandlerProxyFunction(const mtsGenericObject & argument)
{
}
