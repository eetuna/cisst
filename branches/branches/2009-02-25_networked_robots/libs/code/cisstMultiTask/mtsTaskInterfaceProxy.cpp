// **********************************************************************
//
// Copyright (c) 2003-2008 ZeroC, Inc. All rights reserved.
//
// This copy of Ice is licensed to you under the terms described in the
// ICE_LICENSE file included in this distribution.
//
// **********************************************************************

// Ice version 3.3.0
// Generated from file `mtsTaskInterfaceProxy.ice'

#include <mtsTaskInterfaceProxy.h>
#include <Ice/LocalException.h>
#include <Ice/ObjectFactory.h>
#include <Ice/BasicStream.h>
#include <IceUtil/Iterator.h>
#include <IceUtil/ScopedArray.h>

#ifndef ICE_IGNORE_VERSION
#   if ICE_INT_VERSION / 100 != 303
#       error Ice version mismatch!
#   endif
#   if ICE_INT_VERSION % 100 > 50
#       error Beta header file detected
#   endif
#   if ICE_INT_VERSION % 100 < 0
#       error Ice patch level mismatch!
#   endif
#endif

static const ::std::string __mtsTaskInterfaceProxy__TaskInterfaceServer__AddClient_name = "AddClient";

static const ::std::string __mtsTaskInterfaceProxy__TaskInterfaceServer__GetProvidedInterfaceSpecification_name = "GetProvidedInterfaceSpecification";

static const ::std::string __mtsTaskInterfaceProxy__TaskInterfaceServer__ExecuteCommandVoid_name = "ExecuteCommandVoid";

static const ::std::string __mtsTaskInterfaceProxy__TaskInterfaceServer__ExecuteCommandWrite_name = "ExecuteCommandWrite";

static const ::std::string __mtsTaskInterfaceProxy__TaskInterfaceServer__ExecuteCommandRead_name = "ExecuteCommandRead";

static const ::std::string __mtsTaskInterfaceProxy__TaskInterfaceServer__ExecuteCommandQualifiedRead_name = "ExecuteCommandQualifiedRead";

::Ice::Object* IceInternal::upCast(::mtsTaskInterfaceProxy::TaskInterfaceClient* p) { return p; }
::IceProxy::Ice::Object* IceInternal::upCast(::IceProxy::mtsTaskInterfaceProxy::TaskInterfaceClient* p) { return p; }

::Ice::Object* IceInternal::upCast(::mtsTaskInterfaceProxy::TaskInterfaceServer* p) { return p; }
::IceProxy::Ice::Object* IceInternal::upCast(::IceProxy::mtsTaskInterfaceProxy::TaskInterfaceServer* p) { return p; }

void
mtsTaskInterfaceProxy::__read(::IceInternal::BasicStream* __is, ::mtsTaskInterfaceProxy::TaskInterfaceClientPrx& v)
{
    ::Ice::ObjectPrx proxy;
    __is->read(proxy);
    if(!proxy)
    {
        v = 0;
    }
    else
    {
        v = new ::IceProxy::mtsTaskInterfaceProxy::TaskInterfaceClient;
        v->__copyFrom(proxy);
    }
}

void
mtsTaskInterfaceProxy::__read(::IceInternal::BasicStream* __is, ::mtsTaskInterfaceProxy::TaskInterfaceServerPrx& v)
{
    ::Ice::ObjectPrx proxy;
    __is->read(proxy);
    if(!proxy)
    {
        v = 0;
    }
    else
    {
        v = new ::IceProxy::mtsTaskInterfaceProxy::TaskInterfaceServer;
        v->__copyFrom(proxy);
    }
}

bool
mtsTaskInterfaceProxy::CommandVoidInfo::operator==(const CommandVoidInfo& __rhs) const
{
    if(this == &__rhs)
    {
        return true;
    }
    if(Name != __rhs.Name)
    {
        return false;
    }
    if(CommandSID != __rhs.CommandSID)
    {
        return false;
    }
    return true;
}

bool
mtsTaskInterfaceProxy::CommandVoidInfo::operator<(const CommandVoidInfo& __rhs) const
{
    if(this == &__rhs)
    {
        return false;
    }
    if(Name < __rhs.Name)
    {
        return true;
    }
    else if(__rhs.Name < Name)
    {
        return false;
    }
    if(CommandSID < __rhs.CommandSID)
    {
        return true;
    }
    else if(__rhs.CommandSID < CommandSID)
    {
        return false;
    }
    return false;
}

void
mtsTaskInterfaceProxy::CommandVoidInfo::__write(::IceInternal::BasicStream* __os) const
{
    __os->write(Name);
    __os->write(CommandSID);
}

void
mtsTaskInterfaceProxy::CommandVoidInfo::__read(::IceInternal::BasicStream* __is)
{
    __is->read(Name);
    __is->read(CommandSID);
}

bool
mtsTaskInterfaceProxy::CommandWriteInfo::operator==(const CommandWriteInfo& __rhs) const
{
    if(this == &__rhs)
    {
        return true;
    }
    if(Name != __rhs.Name)
    {
        return false;
    }
    if(ArgumentTypeName != __rhs.ArgumentTypeName)
    {
        return false;
    }
    if(CommandSID != __rhs.CommandSID)
    {
        return false;
    }
    return true;
}

bool
mtsTaskInterfaceProxy::CommandWriteInfo::operator<(const CommandWriteInfo& __rhs) const
{
    if(this == &__rhs)
    {
        return false;
    }
    if(Name < __rhs.Name)
    {
        return true;
    }
    else if(__rhs.Name < Name)
    {
        return false;
    }
    if(ArgumentTypeName < __rhs.ArgumentTypeName)
    {
        return true;
    }
    else if(__rhs.ArgumentTypeName < ArgumentTypeName)
    {
        return false;
    }
    if(CommandSID < __rhs.CommandSID)
    {
        return true;
    }
    else if(__rhs.CommandSID < CommandSID)
    {
        return false;
    }
    return false;
}

void
mtsTaskInterfaceProxy::CommandWriteInfo::__write(::IceInternal::BasicStream* __os) const
{
    __os->write(Name);
    __os->write(ArgumentTypeName);
    __os->write(CommandSID);
}

void
mtsTaskInterfaceProxy::CommandWriteInfo::__read(::IceInternal::BasicStream* __is)
{
    __is->read(Name);
    __is->read(ArgumentTypeName);
    __is->read(CommandSID);
}

bool
mtsTaskInterfaceProxy::CommandReadInfo::operator==(const CommandReadInfo& __rhs) const
{
    if(this == &__rhs)
    {
        return true;
    }
    if(Name != __rhs.Name)
    {
        return false;
    }
    if(ArgumentTypeName != __rhs.ArgumentTypeName)
    {
        return false;
    }
    if(CommandSID != __rhs.CommandSID)
    {
        return false;
    }
    return true;
}

bool
mtsTaskInterfaceProxy::CommandReadInfo::operator<(const CommandReadInfo& __rhs) const
{
    if(this == &__rhs)
    {
        return false;
    }
    if(Name < __rhs.Name)
    {
        return true;
    }
    else if(__rhs.Name < Name)
    {
        return false;
    }
    if(ArgumentTypeName < __rhs.ArgumentTypeName)
    {
        return true;
    }
    else if(__rhs.ArgumentTypeName < ArgumentTypeName)
    {
        return false;
    }
    if(CommandSID < __rhs.CommandSID)
    {
        return true;
    }
    else if(__rhs.CommandSID < CommandSID)
    {
        return false;
    }
    return false;
}

void
mtsTaskInterfaceProxy::CommandReadInfo::__write(::IceInternal::BasicStream* __os) const
{
    __os->write(Name);
    __os->write(ArgumentTypeName);
    __os->write(CommandSID);
}

void
mtsTaskInterfaceProxy::CommandReadInfo::__read(::IceInternal::BasicStream* __is)
{
    __is->read(Name);
    __is->read(ArgumentTypeName);
    __is->read(CommandSID);
}

bool
mtsTaskInterfaceProxy::CommandQualifiedReadInfo::operator==(const CommandQualifiedReadInfo& __rhs) const
{
    if(this == &__rhs)
    {
        return true;
    }
    if(Name != __rhs.Name)
    {
        return false;
    }
    if(Argument1TypeName != __rhs.Argument1TypeName)
    {
        return false;
    }
    if(Argument2TypeName != __rhs.Argument2TypeName)
    {
        return false;
    }
    if(CommandSID != __rhs.CommandSID)
    {
        return false;
    }
    return true;
}

bool
mtsTaskInterfaceProxy::CommandQualifiedReadInfo::operator<(const CommandQualifiedReadInfo& __rhs) const
{
    if(this == &__rhs)
    {
        return false;
    }
    if(Name < __rhs.Name)
    {
        return true;
    }
    else if(__rhs.Name < Name)
    {
        return false;
    }
    if(Argument1TypeName < __rhs.Argument1TypeName)
    {
        return true;
    }
    else if(__rhs.Argument1TypeName < Argument1TypeName)
    {
        return false;
    }
    if(Argument2TypeName < __rhs.Argument2TypeName)
    {
        return true;
    }
    else if(__rhs.Argument2TypeName < Argument2TypeName)
    {
        return false;
    }
    if(CommandSID < __rhs.CommandSID)
    {
        return true;
    }
    else if(__rhs.CommandSID < CommandSID)
    {
        return false;
    }
    return false;
}

void
mtsTaskInterfaceProxy::CommandQualifiedReadInfo::__write(::IceInternal::BasicStream* __os) const
{
    __os->write(Name);
    __os->write(Argument1TypeName);
    __os->write(Argument2TypeName);
    __os->write(CommandSID);
}

void
mtsTaskInterfaceProxy::CommandQualifiedReadInfo::__read(::IceInternal::BasicStream* __is)
{
    __is->read(Name);
    __is->read(Argument1TypeName);
    __is->read(Argument2TypeName);
    __is->read(CommandSID);
}

void
mtsTaskInterfaceProxy::__writeCommandVoidSeq(::IceInternal::BasicStream* __os, const ::mtsTaskInterfaceProxy::CommandVoidInfo* begin, const ::mtsTaskInterfaceProxy::CommandVoidInfo* end)
{
    ::Ice::Int size = static_cast< ::Ice::Int>(end - begin);
    __os->writeSize(size);
    for(int i = 0; i < size; ++i)
    {
        begin[i].__write(__os);
    }
}

void
mtsTaskInterfaceProxy::__readCommandVoidSeq(::IceInternal::BasicStream* __is, ::mtsTaskInterfaceProxy::CommandVoidSeq& v)
{
    ::Ice::Int sz;
    __is->readSize(sz);
    __is->startSeq(sz, 5);
    v.resize(sz);
    for(int i = 0; i < sz; ++i)
    {
        v[i].__read(__is);
        __is->checkSeq();
        __is->endElement();
    }
    __is->endSeq(sz);
}

void
mtsTaskInterfaceProxy::__writeCommandWriteSeq(::IceInternal::BasicStream* __os, const ::mtsTaskInterfaceProxy::CommandWriteInfo* begin, const ::mtsTaskInterfaceProxy::CommandWriteInfo* end)
{
    ::Ice::Int size = static_cast< ::Ice::Int>(end - begin);
    __os->writeSize(size);
    for(int i = 0; i < size; ++i)
    {
        begin[i].__write(__os);
    }
}

void
mtsTaskInterfaceProxy::__readCommandWriteSeq(::IceInternal::BasicStream* __is, ::mtsTaskInterfaceProxy::CommandWriteSeq& v)
{
    ::Ice::Int sz;
    __is->readSize(sz);
    __is->startSeq(sz, 6);
    v.resize(sz);
    for(int i = 0; i < sz; ++i)
    {
        v[i].__read(__is);
        __is->checkSeq();
        __is->endElement();
    }
    __is->endSeq(sz);
}

void
mtsTaskInterfaceProxy::__writeCommandReadSeq(::IceInternal::BasicStream* __os, const ::mtsTaskInterfaceProxy::CommandReadInfo* begin, const ::mtsTaskInterfaceProxy::CommandReadInfo* end)
{
    ::Ice::Int size = static_cast< ::Ice::Int>(end - begin);
    __os->writeSize(size);
    for(int i = 0; i < size; ++i)
    {
        begin[i].__write(__os);
    }
}

void
mtsTaskInterfaceProxy::__readCommandReadSeq(::IceInternal::BasicStream* __is, ::mtsTaskInterfaceProxy::CommandReadSeq& v)
{
    ::Ice::Int sz;
    __is->readSize(sz);
    __is->startSeq(sz, 6);
    v.resize(sz);
    for(int i = 0; i < sz; ++i)
    {
        v[i].__read(__is);
        __is->checkSeq();
        __is->endElement();
    }
    __is->endSeq(sz);
}

void
mtsTaskInterfaceProxy::__writeCommandQualifiedReadSeq(::IceInternal::BasicStream* __os, const ::mtsTaskInterfaceProxy::CommandQualifiedReadInfo* begin, const ::mtsTaskInterfaceProxy::CommandQualifiedReadInfo* end)
{
    ::Ice::Int size = static_cast< ::Ice::Int>(end - begin);
    __os->writeSize(size);
    for(int i = 0; i < size; ++i)
    {
        begin[i].__write(__os);
    }
}

void
mtsTaskInterfaceProxy::__readCommandQualifiedReadSeq(::IceInternal::BasicStream* __is, ::mtsTaskInterfaceProxy::CommandQualifiedReadSeq& v)
{
    ::Ice::Int sz;
    __is->readSize(sz);
    __is->startSeq(sz, 7);
    v.resize(sz);
    for(int i = 0; i < sz; ++i)
    {
        v[i].__read(__is);
        __is->checkSeq();
        __is->endElement();
    }
    __is->endSeq(sz);
}

bool
mtsTaskInterfaceProxy::ProvidedInterfaceSpecification::operator==(const ProvidedInterfaceSpecification& __rhs) const
{
    if(this == &__rhs)
    {
        return true;
    }
    if(interfaceName != __rhs.interfaceName)
    {
        return false;
    }
    if(providedInterfaceForTask != __rhs.providedInterfaceForTask)
    {
        return false;
    }
    if(commandsVoid != __rhs.commandsVoid)
    {
        return false;
    }
    if(commandsWrite != __rhs.commandsWrite)
    {
        return false;
    }
    if(commandsRead != __rhs.commandsRead)
    {
        return false;
    }
    if(commandsQualifiedRead != __rhs.commandsQualifiedRead)
    {
        return false;
    }
    return true;
}

bool
mtsTaskInterfaceProxy::ProvidedInterfaceSpecification::operator<(const ProvidedInterfaceSpecification& __rhs) const
{
    if(this == &__rhs)
    {
        return false;
    }
    if(interfaceName < __rhs.interfaceName)
    {
        return true;
    }
    else if(__rhs.interfaceName < interfaceName)
    {
        return false;
    }
    if(providedInterfaceForTask < __rhs.providedInterfaceForTask)
    {
        return true;
    }
    else if(__rhs.providedInterfaceForTask < providedInterfaceForTask)
    {
        return false;
    }
    if(commandsVoid < __rhs.commandsVoid)
    {
        return true;
    }
    else if(__rhs.commandsVoid < commandsVoid)
    {
        return false;
    }
    if(commandsWrite < __rhs.commandsWrite)
    {
        return true;
    }
    else if(__rhs.commandsWrite < commandsWrite)
    {
        return false;
    }
    if(commandsRead < __rhs.commandsRead)
    {
        return true;
    }
    else if(__rhs.commandsRead < commandsRead)
    {
        return false;
    }
    if(commandsQualifiedRead < __rhs.commandsQualifiedRead)
    {
        return true;
    }
    else if(__rhs.commandsQualifiedRead < commandsQualifiedRead)
    {
        return false;
    }
    return false;
}

void
mtsTaskInterfaceProxy::ProvidedInterfaceSpecification::__write(::IceInternal::BasicStream* __os) const
{
    __os->write(interfaceName);
    __os->write(providedInterfaceForTask);
    if(commandsVoid.size() == 0)
    {
        __os->writeSize(0);
    }
    else
    {
        ::mtsTaskInterfaceProxy::__writeCommandVoidSeq(__os, &commandsVoid[0], &commandsVoid[0] + commandsVoid.size());
    }
    if(commandsWrite.size() == 0)
    {
        __os->writeSize(0);
    }
    else
    {
        ::mtsTaskInterfaceProxy::__writeCommandWriteSeq(__os, &commandsWrite[0], &commandsWrite[0] + commandsWrite.size());
    }
    if(commandsRead.size() == 0)
    {
        __os->writeSize(0);
    }
    else
    {
        ::mtsTaskInterfaceProxy::__writeCommandReadSeq(__os, &commandsRead[0], &commandsRead[0] + commandsRead.size());
    }
    if(commandsQualifiedRead.size() == 0)
    {
        __os->writeSize(0);
    }
    else
    {
        ::mtsTaskInterfaceProxy::__writeCommandQualifiedReadSeq(__os, &commandsQualifiedRead[0], &commandsQualifiedRead[0] + commandsQualifiedRead.size());
    }
}

void
mtsTaskInterfaceProxy::ProvidedInterfaceSpecification::__read(::IceInternal::BasicStream* __is)
{
    __is->read(interfaceName);
    __is->read(providedInterfaceForTask);
    ::mtsTaskInterfaceProxy::__readCommandVoidSeq(__is, commandsVoid);
    ::mtsTaskInterfaceProxy::__readCommandWriteSeq(__is, commandsWrite);
    ::mtsTaskInterfaceProxy::__readCommandReadSeq(__is, commandsRead);
    ::mtsTaskInterfaceProxy::__readCommandQualifiedReadSeq(__is, commandsQualifiedRead);
}

void
mtsTaskInterfaceProxy::__writeProvidedInterfaceSpecificationSeq(::IceInternal::BasicStream* __os, const ::mtsTaskInterfaceProxy::ProvidedInterfaceSpecification* begin, const ::mtsTaskInterfaceProxy::ProvidedInterfaceSpecification* end)
{
    ::Ice::Int size = static_cast< ::Ice::Int>(end - begin);
    __os->writeSize(size);
    for(int i = 0; i < size; ++i)
    {
        begin[i].__write(__os);
    }
}

void
mtsTaskInterfaceProxy::__readProvidedInterfaceSpecificationSeq(::IceInternal::BasicStream* __is, ::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq& v)
{
    ::Ice::Int sz;
    __is->readSize(sz);
    __is->startSeq(sz, 6);
    v.resize(sz);
    for(int i = 0; i < sz; ++i)
    {
        v[i].__read(__is);
        __is->checkSeq();
        __is->endElement();
    }
    __is->endSeq(sz);
}

bool
mtsTaskInterfaceProxy::CommandProxyElement::operator==(const CommandProxyElement& __rhs) const
{
    if(this == &__rhs)
    {
        return true;
    }
    if(Name != __rhs.Name)
    {
        return false;
    }
    if(ID != __rhs.ID)
    {
        return false;
    }
    return true;
}

bool
mtsTaskInterfaceProxy::CommandProxyElement::operator<(const CommandProxyElement& __rhs) const
{
    if(this == &__rhs)
    {
        return false;
    }
    if(Name < __rhs.Name)
    {
        return true;
    }
    else if(__rhs.Name < Name)
    {
        return false;
    }
    if(ID < __rhs.ID)
    {
        return true;
    }
    else if(__rhs.ID < ID)
    {
        return false;
    }
    return false;
}

void
mtsTaskInterfaceProxy::CommandProxyElement::__write(::IceInternal::BasicStream* __os) const
{
    __os->write(Name);
    __os->write(ID);
}

void
mtsTaskInterfaceProxy::CommandProxyElement::__read(::IceInternal::BasicStream* __is)
{
    __is->read(Name);
    __is->read(ID);
}

void
mtsTaskInterfaceProxy::__writeCommandProxyElementSeq(::IceInternal::BasicStream* __os, const ::mtsTaskInterfaceProxy::CommandProxyElement* begin, const ::mtsTaskInterfaceProxy::CommandProxyElement* end)
{
    ::Ice::Int size = static_cast< ::Ice::Int>(end - begin);
    __os->writeSize(size);
    for(int i = 0; i < size; ++i)
    {
        begin[i].__write(__os);
    }
}

void
mtsTaskInterfaceProxy::__readCommandProxyElementSeq(::IceInternal::BasicStream* __is, ::mtsTaskInterfaceProxy::CommandProxyElementSeq& v)
{
    ::Ice::Int sz;
    __is->readSize(sz);
    __is->startSeq(sz, 5);
    v.resize(sz);
    for(int i = 0; i < sz; ++i)
    {
        v[i].__read(__is);
        __is->checkSeq();
        __is->endElement();
    }
    __is->endSeq(sz);
}

bool
mtsTaskInterfaceProxy::CommandProxyInfo::operator==(const CommandProxyInfo& __rhs) const
{
    if(this == &__rhs)
    {
        return true;
    }
    if(ConnectedProvidedInterfaceName != __rhs.ConnectedProvidedInterfaceName)
    {
        return false;
    }
    if(CommandProxyVoidSeq != __rhs.CommandProxyVoidSeq)
    {
        return false;
    }
    if(CommandProxyWriteSeq != __rhs.CommandProxyWriteSeq)
    {
        return false;
    }
    if(CommandProxyReadSeq != __rhs.CommandProxyReadSeq)
    {
        return false;
    }
    if(CommandProxyQualifiedReadSeq != __rhs.CommandProxyQualifiedReadSeq)
    {
        return false;
    }
    return true;
}

bool
mtsTaskInterfaceProxy::CommandProxyInfo::operator<(const CommandProxyInfo& __rhs) const
{
    if(this == &__rhs)
    {
        return false;
    }
    if(ConnectedProvidedInterfaceName < __rhs.ConnectedProvidedInterfaceName)
    {
        return true;
    }
    else if(__rhs.ConnectedProvidedInterfaceName < ConnectedProvidedInterfaceName)
    {
        return false;
    }
    if(CommandProxyVoidSeq < __rhs.CommandProxyVoidSeq)
    {
        return true;
    }
    else if(__rhs.CommandProxyVoidSeq < CommandProxyVoidSeq)
    {
        return false;
    }
    if(CommandProxyWriteSeq < __rhs.CommandProxyWriteSeq)
    {
        return true;
    }
    else if(__rhs.CommandProxyWriteSeq < CommandProxyWriteSeq)
    {
        return false;
    }
    if(CommandProxyReadSeq < __rhs.CommandProxyReadSeq)
    {
        return true;
    }
    else if(__rhs.CommandProxyReadSeq < CommandProxyReadSeq)
    {
        return false;
    }
    if(CommandProxyQualifiedReadSeq < __rhs.CommandProxyQualifiedReadSeq)
    {
        return true;
    }
    else if(__rhs.CommandProxyQualifiedReadSeq < CommandProxyQualifiedReadSeq)
    {
        return false;
    }
    return false;
}

void
mtsTaskInterfaceProxy::CommandProxyInfo::__write(::IceInternal::BasicStream* __os) const
{
    __os->write(ConnectedProvidedInterfaceName);
    if(CommandProxyVoidSeq.size() == 0)
    {
        __os->writeSize(0);
    }
    else
    {
        ::mtsTaskInterfaceProxy::__writeCommandProxyElementSeq(__os, &CommandProxyVoidSeq[0], &CommandProxyVoidSeq[0] + CommandProxyVoidSeq.size());
    }
    if(CommandProxyWriteSeq.size() == 0)
    {
        __os->writeSize(0);
    }
    else
    {
        ::mtsTaskInterfaceProxy::__writeCommandProxyElementSeq(__os, &CommandProxyWriteSeq[0], &CommandProxyWriteSeq[0] + CommandProxyWriteSeq.size());
    }
    if(CommandProxyReadSeq.size() == 0)
    {
        __os->writeSize(0);
    }
    else
    {
        ::mtsTaskInterfaceProxy::__writeCommandProxyElementSeq(__os, &CommandProxyReadSeq[0], &CommandProxyReadSeq[0] + CommandProxyReadSeq.size());
    }
    if(CommandProxyQualifiedReadSeq.size() == 0)
    {
        __os->writeSize(0);
    }
    else
    {
        ::mtsTaskInterfaceProxy::__writeCommandProxyElementSeq(__os, &CommandProxyQualifiedReadSeq[0], &CommandProxyQualifiedReadSeq[0] + CommandProxyQualifiedReadSeq.size());
    }
}

void
mtsTaskInterfaceProxy::CommandProxyInfo::__read(::IceInternal::BasicStream* __is)
{
    __is->read(ConnectedProvidedInterfaceName);
    ::mtsTaskInterfaceProxy::__readCommandProxyElementSeq(__is, CommandProxyVoidSeq);
    ::mtsTaskInterfaceProxy::__readCommandProxyElementSeq(__is, CommandProxyWriteSeq);
    ::mtsTaskInterfaceProxy::__readCommandProxyElementSeq(__is, CommandProxyReadSeq);
    ::mtsTaskInterfaceProxy::__readCommandProxyElementSeq(__is, CommandProxyQualifiedReadSeq);
}

const ::std::string&
IceProxy::mtsTaskInterfaceProxy::TaskInterfaceClient::ice_staticId()
{
    return ::mtsTaskInterfaceProxy::TaskInterfaceClient::ice_staticId();
}

::IceInternal::Handle< ::IceDelegateM::Ice::Object>
IceProxy::mtsTaskInterfaceProxy::TaskInterfaceClient::__createDelegateM()
{
    return ::IceInternal::Handle< ::IceDelegateM::Ice::Object>(new ::IceDelegateM::mtsTaskInterfaceProxy::TaskInterfaceClient);
}

::IceInternal::Handle< ::IceDelegateD::Ice::Object>
IceProxy::mtsTaskInterfaceProxy::TaskInterfaceClient::__createDelegateD()
{
    return ::IceInternal::Handle< ::IceDelegateD::Ice::Object>(new ::IceDelegateD::mtsTaskInterfaceProxy::TaskInterfaceClient);
}

::IceProxy::Ice::Object*
IceProxy::mtsTaskInterfaceProxy::TaskInterfaceClient::__newInstance() const
{
    return new TaskInterfaceClient;
}

void
IceProxy::mtsTaskInterfaceProxy::TaskInterfaceServer::AddClient(const ::Ice::Identity& ident, const ::Ice::Context* __ctx)
{
    int __cnt = 0;
    while(true)
    {
        ::IceInternal::Handle< ::IceDelegate::Ice::Object> __delBase;
        try
        {
            __delBase = __getDelegate(false);
            ::IceDelegate::mtsTaskInterfaceProxy::TaskInterfaceServer* __del = dynamic_cast< ::IceDelegate::mtsTaskInterfaceProxy::TaskInterfaceServer*>(__delBase.get());
            __del->AddClient(ident, __ctx);
            return;
        }
        catch(const ::IceInternal::LocalExceptionWrapper& __ex)
        {
            __handleExceptionWrapper(__delBase, __ex, 0);
        }
        catch(const ::Ice::LocalException& __ex)
        {
            __handleException(__delBase, __ex, 0, __cnt);
        }
    }
}

bool
IceProxy::mtsTaskInterfaceProxy::TaskInterfaceServer::GetProvidedInterfaceSpecification(::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq& providedInterfaceSpecifications, const ::Ice::Context* __ctx)
{
    int __cnt = 0;
    while(true)
    {
        ::IceInternal::Handle< ::IceDelegate::Ice::Object> __delBase;
        try
        {
            __checkTwowayOnly(__mtsTaskInterfaceProxy__TaskInterfaceServer__GetProvidedInterfaceSpecification_name);
            __delBase = __getDelegate(false);
            ::IceDelegate::mtsTaskInterfaceProxy::TaskInterfaceServer* __del = dynamic_cast< ::IceDelegate::mtsTaskInterfaceProxy::TaskInterfaceServer*>(__delBase.get());
            return __del->GetProvidedInterfaceSpecification(providedInterfaceSpecifications, __ctx);
        }
        catch(const ::IceInternal::LocalExceptionWrapper& __ex)
        {
            __handleExceptionWrapperRelaxed(__delBase, __ex, 0, __cnt);
        }
        catch(const ::Ice::LocalException& __ex)
        {
            __handleException(__delBase, __ex, 0, __cnt);
        }
    }
}

void
IceProxy::mtsTaskInterfaceProxy::TaskInterfaceServer::ExecuteCommandVoid(::Ice::Int CommandSID, const ::Ice::Context* __ctx)
{
    int __cnt = 0;
    while(true)
    {
        ::IceInternal::Handle< ::IceDelegate::Ice::Object> __delBase;
        try
        {
            __delBase = __getDelegate(false);
            ::IceDelegate::mtsTaskInterfaceProxy::TaskInterfaceServer* __del = dynamic_cast< ::IceDelegate::mtsTaskInterfaceProxy::TaskInterfaceServer*>(__delBase.get());
            __del->ExecuteCommandVoid(CommandSID, __ctx);
            return;
        }
        catch(const ::IceInternal::LocalExceptionWrapper& __ex)
        {
            __handleExceptionWrapper(__delBase, __ex, 0);
        }
        catch(const ::Ice::LocalException& __ex)
        {
            __handleException(__delBase, __ex, 0, __cnt);
        }
    }
}

void
IceProxy::mtsTaskInterfaceProxy::TaskInterfaceServer::ExecuteCommandWrite(::Ice::Int CommandSID, ::Ice::Double argument, const ::Ice::Context* __ctx)
{
    int __cnt = 0;
    while(true)
    {
        ::IceInternal::Handle< ::IceDelegate::Ice::Object> __delBase;
        try
        {
            __delBase = __getDelegate(false);
            ::IceDelegate::mtsTaskInterfaceProxy::TaskInterfaceServer* __del = dynamic_cast< ::IceDelegate::mtsTaskInterfaceProxy::TaskInterfaceServer*>(__delBase.get());
            __del->ExecuteCommandWrite(CommandSID, argument, __ctx);
            return;
        }
        catch(const ::IceInternal::LocalExceptionWrapper& __ex)
        {
            __handleExceptionWrapper(__delBase, __ex, 0);
        }
        catch(const ::Ice::LocalException& __ex)
        {
            __handleException(__delBase, __ex, 0, __cnt);
        }
    }
}

void
IceProxy::mtsTaskInterfaceProxy::TaskInterfaceServer::ExecuteCommandRead(::Ice::Int CommandSID, ::Ice::Double& argument, const ::Ice::Context* __ctx)
{
    int __cnt = 0;
    while(true)
    {
        ::IceInternal::Handle< ::IceDelegate::Ice::Object> __delBase;
        try
        {
            __checkTwowayOnly(__mtsTaskInterfaceProxy__TaskInterfaceServer__ExecuteCommandRead_name);
            __delBase = __getDelegate(false);
            ::IceDelegate::mtsTaskInterfaceProxy::TaskInterfaceServer* __del = dynamic_cast< ::IceDelegate::mtsTaskInterfaceProxy::TaskInterfaceServer*>(__delBase.get());
            __del->ExecuteCommandRead(CommandSID, argument, __ctx);
            return;
        }
        catch(const ::IceInternal::LocalExceptionWrapper& __ex)
        {
            __handleExceptionWrapper(__delBase, __ex, 0);
        }
        catch(const ::Ice::LocalException& __ex)
        {
            __handleException(__delBase, __ex, 0, __cnt);
        }
    }
}

void
IceProxy::mtsTaskInterfaceProxy::TaskInterfaceServer::ExecuteCommandQualifiedRead(::Ice::Int CommandSID, ::Ice::Double argument1, ::Ice::Double& argument2, const ::Ice::Context* __ctx)
{
    int __cnt = 0;
    while(true)
    {
        ::IceInternal::Handle< ::IceDelegate::Ice::Object> __delBase;
        try
        {
            __checkTwowayOnly(__mtsTaskInterfaceProxy__TaskInterfaceServer__ExecuteCommandQualifiedRead_name);
            __delBase = __getDelegate(false);
            ::IceDelegate::mtsTaskInterfaceProxy::TaskInterfaceServer* __del = dynamic_cast< ::IceDelegate::mtsTaskInterfaceProxy::TaskInterfaceServer*>(__delBase.get());
            __del->ExecuteCommandQualifiedRead(CommandSID, argument1, argument2, __ctx);
            return;
        }
        catch(const ::IceInternal::LocalExceptionWrapper& __ex)
        {
            __handleExceptionWrapper(__delBase, __ex, 0);
        }
        catch(const ::Ice::LocalException& __ex)
        {
            __handleException(__delBase, __ex, 0, __cnt);
        }
    }
}

const ::std::string&
IceProxy::mtsTaskInterfaceProxy::TaskInterfaceServer::ice_staticId()
{
    return ::mtsTaskInterfaceProxy::TaskInterfaceServer::ice_staticId();
}

::IceInternal::Handle< ::IceDelegateM::Ice::Object>
IceProxy::mtsTaskInterfaceProxy::TaskInterfaceServer::__createDelegateM()
{
    return ::IceInternal::Handle< ::IceDelegateM::Ice::Object>(new ::IceDelegateM::mtsTaskInterfaceProxy::TaskInterfaceServer);
}

::IceInternal::Handle< ::IceDelegateD::Ice::Object>
IceProxy::mtsTaskInterfaceProxy::TaskInterfaceServer::__createDelegateD()
{
    return ::IceInternal::Handle< ::IceDelegateD::Ice::Object>(new ::IceDelegateD::mtsTaskInterfaceProxy::TaskInterfaceServer);
}

::IceProxy::Ice::Object*
IceProxy::mtsTaskInterfaceProxy::TaskInterfaceServer::__newInstance() const
{
    return new TaskInterfaceServer;
}

void
IceDelegateM::mtsTaskInterfaceProxy::TaskInterfaceServer::AddClient(const ::Ice::Identity& ident, const ::Ice::Context* __context)
{
    ::IceInternal::Outgoing __og(__handler.get(), __mtsTaskInterfaceProxy__TaskInterfaceServer__AddClient_name, ::Ice::Normal, __context);
    try
    {
        ::IceInternal::BasicStream* __os = __og.os();
        ident.__write(__os);
    }
    catch(const ::Ice::LocalException& __ex)
    {
        __og.abort(__ex);
    }
    bool __ok = __og.invoke();
    if(!__og.is()->b.empty())
    {
        try
        {
            if(!__ok)
            {
                try
                {
                    __og.throwUserException();
                }
                catch(const ::Ice::UserException& __ex)
                {
                    ::Ice::UnknownUserException __uue(__FILE__, __LINE__, __ex.ice_name());
                    throw __uue;
                }
            }
            __og.is()->skipEmptyEncaps();
        }
        catch(const ::Ice::LocalException& __ex)
        {
            throw ::IceInternal::LocalExceptionWrapper(__ex, false);
        }
    }
}

bool
IceDelegateM::mtsTaskInterfaceProxy::TaskInterfaceServer::GetProvidedInterfaceSpecification(::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq& providedInterfaceSpecifications, const ::Ice::Context* __context)
{
    ::IceInternal::Outgoing __og(__handler.get(), __mtsTaskInterfaceProxy__TaskInterfaceServer__GetProvidedInterfaceSpecification_name, ::Ice::Idempotent, __context);
    bool __ok = __og.invoke();
    try
    {
        if(!__ok)
        {
            try
            {
                __og.throwUserException();
            }
            catch(const ::Ice::UserException& __ex)
            {
                ::Ice::UnknownUserException __uue(__FILE__, __LINE__, __ex.ice_name());
                throw __uue;
            }
        }
        bool __ret;
        ::IceInternal::BasicStream* __is = __og.is();
        __is->startReadEncaps();
        ::mtsTaskInterfaceProxy::__readProvidedInterfaceSpecificationSeq(__is, providedInterfaceSpecifications);
        __is->read(__ret);
        __is->endReadEncaps();
        return __ret;
    }
    catch(const ::Ice::LocalException& __ex)
    {
        throw ::IceInternal::LocalExceptionWrapper(__ex, false);
    }
}

void
IceDelegateM::mtsTaskInterfaceProxy::TaskInterfaceServer::ExecuteCommandVoid(::Ice::Int CommandSID, const ::Ice::Context* __context)
{
    ::IceInternal::Outgoing __og(__handler.get(), __mtsTaskInterfaceProxy__TaskInterfaceServer__ExecuteCommandVoid_name, ::Ice::Normal, __context);
    try
    {
        ::IceInternal::BasicStream* __os = __og.os();
        __os->write(CommandSID);
    }
    catch(const ::Ice::LocalException& __ex)
    {
        __og.abort(__ex);
    }
    bool __ok = __og.invoke();
    if(!__og.is()->b.empty())
    {
        try
        {
            if(!__ok)
            {
                try
                {
                    __og.throwUserException();
                }
                catch(const ::Ice::UserException& __ex)
                {
                    ::Ice::UnknownUserException __uue(__FILE__, __LINE__, __ex.ice_name());
                    throw __uue;
                }
            }
            __og.is()->skipEmptyEncaps();
        }
        catch(const ::Ice::LocalException& __ex)
        {
            throw ::IceInternal::LocalExceptionWrapper(__ex, false);
        }
    }
}

void
IceDelegateM::mtsTaskInterfaceProxy::TaskInterfaceServer::ExecuteCommandWrite(::Ice::Int CommandSID, ::Ice::Double argument, const ::Ice::Context* __context)
{
    ::IceInternal::Outgoing __og(__handler.get(), __mtsTaskInterfaceProxy__TaskInterfaceServer__ExecuteCommandWrite_name, ::Ice::Normal, __context);
    try
    {
        ::IceInternal::BasicStream* __os = __og.os();
        __os->write(CommandSID);
        __os->write(argument);
    }
    catch(const ::Ice::LocalException& __ex)
    {
        __og.abort(__ex);
    }
    bool __ok = __og.invoke();
    if(!__og.is()->b.empty())
    {
        try
        {
            if(!__ok)
            {
                try
                {
                    __og.throwUserException();
                }
                catch(const ::Ice::UserException& __ex)
                {
                    ::Ice::UnknownUserException __uue(__FILE__, __LINE__, __ex.ice_name());
                    throw __uue;
                }
            }
            __og.is()->skipEmptyEncaps();
        }
        catch(const ::Ice::LocalException& __ex)
        {
            throw ::IceInternal::LocalExceptionWrapper(__ex, false);
        }
    }
}

void
IceDelegateM::mtsTaskInterfaceProxy::TaskInterfaceServer::ExecuteCommandRead(::Ice::Int CommandSID, ::Ice::Double& argument, const ::Ice::Context* __context)
{
    ::IceInternal::Outgoing __og(__handler.get(), __mtsTaskInterfaceProxy__TaskInterfaceServer__ExecuteCommandRead_name, ::Ice::Normal, __context);
    try
    {
        ::IceInternal::BasicStream* __os = __og.os();
        __os->write(CommandSID);
    }
    catch(const ::Ice::LocalException& __ex)
    {
        __og.abort(__ex);
    }
    bool __ok = __og.invoke();
    try
    {
        if(!__ok)
        {
            try
            {
                __og.throwUserException();
            }
            catch(const ::Ice::UserException& __ex)
            {
                ::Ice::UnknownUserException __uue(__FILE__, __LINE__, __ex.ice_name());
                throw __uue;
            }
        }
        ::IceInternal::BasicStream* __is = __og.is();
        __is->startReadEncaps();
        __is->read(argument);
        __is->endReadEncaps();
    }
    catch(const ::Ice::LocalException& __ex)
    {
        throw ::IceInternal::LocalExceptionWrapper(__ex, false);
    }
}

void
IceDelegateM::mtsTaskInterfaceProxy::TaskInterfaceServer::ExecuteCommandQualifiedRead(::Ice::Int CommandSID, ::Ice::Double argument1, ::Ice::Double& argument2, const ::Ice::Context* __context)
{
    ::IceInternal::Outgoing __og(__handler.get(), __mtsTaskInterfaceProxy__TaskInterfaceServer__ExecuteCommandQualifiedRead_name, ::Ice::Normal, __context);
    try
    {
        ::IceInternal::BasicStream* __os = __og.os();
        __os->write(CommandSID);
        __os->write(argument1);
    }
    catch(const ::Ice::LocalException& __ex)
    {
        __og.abort(__ex);
    }
    bool __ok = __og.invoke();
    try
    {
        if(!__ok)
        {
            try
            {
                __og.throwUserException();
            }
            catch(const ::Ice::UserException& __ex)
            {
                ::Ice::UnknownUserException __uue(__FILE__, __LINE__, __ex.ice_name());
                throw __uue;
            }
        }
        ::IceInternal::BasicStream* __is = __og.is();
        __is->startReadEncaps();
        __is->read(argument2);
        __is->endReadEncaps();
    }
    catch(const ::Ice::LocalException& __ex)
    {
        throw ::IceInternal::LocalExceptionWrapper(__ex, false);
    }
}

void
IceDelegateD::mtsTaskInterfaceProxy::TaskInterfaceServer::AddClient(const ::Ice::Identity& ident, const ::Ice::Context* __context)
{
    class _DirectI : public ::IceInternal::Direct
    {
    public:

        _DirectI(const ::Ice::Identity& ident, const ::Ice::Current& __current) : 
            ::IceInternal::Direct(__current),
            _m_ident(ident)
        {
        }
        
        virtual ::Ice::DispatchStatus
        run(::Ice::Object* object)
        {
            ::mtsTaskInterfaceProxy::TaskInterfaceServer* servant = dynamic_cast< ::mtsTaskInterfaceProxy::TaskInterfaceServer*>(object);
            if(!servant)
            {
                throw ::Ice::OperationNotExistException(__FILE__, __LINE__, _current.id, _current.facet, _current.operation);
            }
            servant->AddClient(_m_ident, _current);
            return ::Ice::DispatchOK;
        }
        
    private:
        
        const ::Ice::Identity& _m_ident;
    };
    
    ::Ice::Current __current;
    __initCurrent(__current, __mtsTaskInterfaceProxy__TaskInterfaceServer__AddClient_name, ::Ice::Normal, __context);
    try
    {
        _DirectI __direct(ident, __current);
        try
        {
            __direct.servant()->__collocDispatch(__direct);
        }
        catch(...)
        {
            __direct.destroy();
            throw;
        }
        __direct.destroy();
    }
    catch(const ::Ice::SystemException&)
    {
        throw;
    }
    catch(const ::IceInternal::LocalExceptionWrapper&)
    {
        throw;
    }
    catch(const ::std::exception& __ex)
    {
        ::IceInternal::LocalExceptionWrapper::throwWrapper(__ex);
    }
    catch(...)
    {
        throw ::IceInternal::LocalExceptionWrapper(::Ice::UnknownException(__FILE__, __LINE__, "unknown c++ exception"), false);
    }
}

bool
IceDelegateD::mtsTaskInterfaceProxy::TaskInterfaceServer::GetProvidedInterfaceSpecification(::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq& providedInterfaceSpecifications, const ::Ice::Context* __context)
{
    class _DirectI : public ::IceInternal::Direct
    {
    public:

        _DirectI(bool& __result, ::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq& providedInterfaceSpecifications, const ::Ice::Current& __current) : 
            ::IceInternal::Direct(__current),
            _result(__result),
            _m_providedInterfaceSpecifications(providedInterfaceSpecifications)
        {
        }
        
        virtual ::Ice::DispatchStatus
        run(::Ice::Object* object)
        {
            ::mtsTaskInterfaceProxy::TaskInterfaceServer* servant = dynamic_cast< ::mtsTaskInterfaceProxy::TaskInterfaceServer*>(object);
            if(!servant)
            {
                throw ::Ice::OperationNotExistException(__FILE__, __LINE__, _current.id, _current.facet, _current.operation);
            }
            _result = servant->GetProvidedInterfaceSpecification(_m_providedInterfaceSpecifications, _current);
            return ::Ice::DispatchOK;
        }
        
    private:
        
        bool& _result;
        ::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq& _m_providedInterfaceSpecifications;
    };
    
    ::Ice::Current __current;
    __initCurrent(__current, __mtsTaskInterfaceProxy__TaskInterfaceServer__GetProvidedInterfaceSpecification_name, ::Ice::Idempotent, __context);
    bool __result;
    try
    {
        _DirectI __direct(__result, providedInterfaceSpecifications, __current);
        try
        {
            __direct.servant()->__collocDispatch(__direct);
        }
        catch(...)
        {
            __direct.destroy();
            throw;
        }
        __direct.destroy();
    }
    catch(const ::Ice::SystemException&)
    {
        throw;
    }
    catch(const ::IceInternal::LocalExceptionWrapper&)
    {
        throw;
    }
    catch(const ::std::exception& __ex)
    {
        ::IceInternal::LocalExceptionWrapper::throwWrapper(__ex);
    }
    catch(...)
    {
        throw ::IceInternal::LocalExceptionWrapper(::Ice::UnknownException(__FILE__, __LINE__, "unknown c++ exception"), false);
    }
    return __result;
}

void
IceDelegateD::mtsTaskInterfaceProxy::TaskInterfaceServer::ExecuteCommandVoid(::Ice::Int CommandSID, const ::Ice::Context* __context)
{
    class _DirectI : public ::IceInternal::Direct
    {
    public:

        _DirectI(::Ice::Int CommandSID, const ::Ice::Current& __current) : 
            ::IceInternal::Direct(__current),
            _m_CommandSID(CommandSID)
        {
        }
        
        virtual ::Ice::DispatchStatus
        run(::Ice::Object* object)
        {
            ::mtsTaskInterfaceProxy::TaskInterfaceServer* servant = dynamic_cast< ::mtsTaskInterfaceProxy::TaskInterfaceServer*>(object);
            if(!servant)
            {
                throw ::Ice::OperationNotExistException(__FILE__, __LINE__, _current.id, _current.facet, _current.operation);
            }
            servant->ExecuteCommandVoid(_m_CommandSID, _current);
            return ::Ice::DispatchOK;
        }
        
    private:
        
        ::Ice::Int _m_CommandSID;
    };
    
    ::Ice::Current __current;
    __initCurrent(__current, __mtsTaskInterfaceProxy__TaskInterfaceServer__ExecuteCommandVoid_name, ::Ice::Normal, __context);
    try
    {
        _DirectI __direct(CommandSID, __current);
        try
        {
            __direct.servant()->__collocDispatch(__direct);
        }
        catch(...)
        {
            __direct.destroy();
            throw;
        }
        __direct.destroy();
    }
    catch(const ::Ice::SystemException&)
    {
        throw;
    }
    catch(const ::IceInternal::LocalExceptionWrapper&)
    {
        throw;
    }
    catch(const ::std::exception& __ex)
    {
        ::IceInternal::LocalExceptionWrapper::throwWrapper(__ex);
    }
    catch(...)
    {
        throw ::IceInternal::LocalExceptionWrapper(::Ice::UnknownException(__FILE__, __LINE__, "unknown c++ exception"), false);
    }
}

void
IceDelegateD::mtsTaskInterfaceProxy::TaskInterfaceServer::ExecuteCommandWrite(::Ice::Int CommandSID, ::Ice::Double argument, const ::Ice::Context* __context)
{
    class _DirectI : public ::IceInternal::Direct
    {
    public:

        _DirectI(::Ice::Int CommandSID, ::Ice::Double argument, const ::Ice::Current& __current) : 
            ::IceInternal::Direct(__current),
            _m_CommandSID(CommandSID),
            _m_argument(argument)
        {
        }
        
        virtual ::Ice::DispatchStatus
        run(::Ice::Object* object)
        {
            ::mtsTaskInterfaceProxy::TaskInterfaceServer* servant = dynamic_cast< ::mtsTaskInterfaceProxy::TaskInterfaceServer*>(object);
            if(!servant)
            {
                throw ::Ice::OperationNotExistException(__FILE__, __LINE__, _current.id, _current.facet, _current.operation);
            }
            servant->ExecuteCommandWrite(_m_CommandSID, _m_argument, _current);
            return ::Ice::DispatchOK;
        }
        
    private:
        
        ::Ice::Int _m_CommandSID;
        ::Ice::Double _m_argument;
    };
    
    ::Ice::Current __current;
    __initCurrent(__current, __mtsTaskInterfaceProxy__TaskInterfaceServer__ExecuteCommandWrite_name, ::Ice::Normal, __context);
    try
    {
        _DirectI __direct(CommandSID, argument, __current);
        try
        {
            __direct.servant()->__collocDispatch(__direct);
        }
        catch(...)
        {
            __direct.destroy();
            throw;
        }
        __direct.destroy();
    }
    catch(const ::Ice::SystemException&)
    {
        throw;
    }
    catch(const ::IceInternal::LocalExceptionWrapper&)
    {
        throw;
    }
    catch(const ::std::exception& __ex)
    {
        ::IceInternal::LocalExceptionWrapper::throwWrapper(__ex);
    }
    catch(...)
    {
        throw ::IceInternal::LocalExceptionWrapper(::Ice::UnknownException(__FILE__, __LINE__, "unknown c++ exception"), false);
    }
}

void
IceDelegateD::mtsTaskInterfaceProxy::TaskInterfaceServer::ExecuteCommandRead(::Ice::Int CommandSID, ::Ice::Double& argument, const ::Ice::Context* __context)
{
    class _DirectI : public ::IceInternal::Direct
    {
    public:

        _DirectI(::Ice::Int CommandSID, ::Ice::Double& argument, const ::Ice::Current& __current) : 
            ::IceInternal::Direct(__current),
            _m_CommandSID(CommandSID),
            _m_argument(argument)
        {
        }
        
        virtual ::Ice::DispatchStatus
        run(::Ice::Object* object)
        {
            ::mtsTaskInterfaceProxy::TaskInterfaceServer* servant = dynamic_cast< ::mtsTaskInterfaceProxy::TaskInterfaceServer*>(object);
            if(!servant)
            {
                throw ::Ice::OperationNotExistException(__FILE__, __LINE__, _current.id, _current.facet, _current.operation);
            }
            servant->ExecuteCommandRead(_m_CommandSID, _m_argument, _current);
            return ::Ice::DispatchOK;
        }
        
    private:
        
        ::Ice::Int _m_CommandSID;
        ::Ice::Double& _m_argument;
    };
    
    ::Ice::Current __current;
    __initCurrent(__current, __mtsTaskInterfaceProxy__TaskInterfaceServer__ExecuteCommandRead_name, ::Ice::Normal, __context);
    try
    {
        _DirectI __direct(CommandSID, argument, __current);
        try
        {
            __direct.servant()->__collocDispatch(__direct);
        }
        catch(...)
        {
            __direct.destroy();
            throw;
        }
        __direct.destroy();
    }
    catch(const ::Ice::SystemException&)
    {
        throw;
    }
    catch(const ::IceInternal::LocalExceptionWrapper&)
    {
        throw;
    }
    catch(const ::std::exception& __ex)
    {
        ::IceInternal::LocalExceptionWrapper::throwWrapper(__ex);
    }
    catch(...)
    {
        throw ::IceInternal::LocalExceptionWrapper(::Ice::UnknownException(__FILE__, __LINE__, "unknown c++ exception"), false);
    }
}

void
IceDelegateD::mtsTaskInterfaceProxy::TaskInterfaceServer::ExecuteCommandQualifiedRead(::Ice::Int CommandSID, ::Ice::Double argument1, ::Ice::Double& argument2, const ::Ice::Context* __context)
{
    class _DirectI : public ::IceInternal::Direct
    {
    public:

        _DirectI(::Ice::Int CommandSID, ::Ice::Double argument1, ::Ice::Double& argument2, const ::Ice::Current& __current) : 
            ::IceInternal::Direct(__current),
            _m_CommandSID(CommandSID),
            _m_argument1(argument1),
            _m_argument2(argument2)
        {
        }
        
        virtual ::Ice::DispatchStatus
        run(::Ice::Object* object)
        {
            ::mtsTaskInterfaceProxy::TaskInterfaceServer* servant = dynamic_cast< ::mtsTaskInterfaceProxy::TaskInterfaceServer*>(object);
            if(!servant)
            {
                throw ::Ice::OperationNotExistException(__FILE__, __LINE__, _current.id, _current.facet, _current.operation);
            }
            servant->ExecuteCommandQualifiedRead(_m_CommandSID, _m_argument1, _m_argument2, _current);
            return ::Ice::DispatchOK;
        }
        
    private:
        
        ::Ice::Int _m_CommandSID;
        ::Ice::Double _m_argument1;
        ::Ice::Double& _m_argument2;
    };
    
    ::Ice::Current __current;
    __initCurrent(__current, __mtsTaskInterfaceProxy__TaskInterfaceServer__ExecuteCommandQualifiedRead_name, ::Ice::Normal, __context);
    try
    {
        _DirectI __direct(CommandSID, argument1, argument2, __current);
        try
        {
            __direct.servant()->__collocDispatch(__direct);
        }
        catch(...)
        {
            __direct.destroy();
            throw;
        }
        __direct.destroy();
    }
    catch(const ::Ice::SystemException&)
    {
        throw;
    }
    catch(const ::IceInternal::LocalExceptionWrapper&)
    {
        throw;
    }
    catch(const ::std::exception& __ex)
    {
        ::IceInternal::LocalExceptionWrapper::throwWrapper(__ex);
    }
    catch(...)
    {
        throw ::IceInternal::LocalExceptionWrapper(::Ice::UnknownException(__FILE__, __LINE__, "unknown c++ exception"), false);
    }
}

::Ice::ObjectPtr
mtsTaskInterfaceProxy::TaskInterfaceClient::ice_clone() const
{
    throw ::Ice::CloneNotImplementedException(__FILE__, __LINE__);
    return 0; // to avoid a warning with some compilers
}

static const ::std::string __mtsTaskInterfaceProxy__TaskInterfaceClient_ids[2] =
{
    "::Ice::Object",
    "::mtsTaskInterfaceProxy::TaskInterfaceClient"
};

bool
mtsTaskInterfaceProxy::TaskInterfaceClient::ice_isA(const ::std::string& _s, const ::Ice::Current&) const
{
    return ::std::binary_search(__mtsTaskInterfaceProxy__TaskInterfaceClient_ids, __mtsTaskInterfaceProxy__TaskInterfaceClient_ids + 2, _s);
}

::std::vector< ::std::string>
mtsTaskInterfaceProxy::TaskInterfaceClient::ice_ids(const ::Ice::Current&) const
{
    return ::std::vector< ::std::string>(&__mtsTaskInterfaceProxy__TaskInterfaceClient_ids[0], &__mtsTaskInterfaceProxy__TaskInterfaceClient_ids[2]);
}

const ::std::string&
mtsTaskInterfaceProxy::TaskInterfaceClient::ice_id(const ::Ice::Current&) const
{
    return __mtsTaskInterfaceProxy__TaskInterfaceClient_ids[1];
}

const ::std::string&
mtsTaskInterfaceProxy::TaskInterfaceClient::ice_staticId()
{
    return __mtsTaskInterfaceProxy__TaskInterfaceClient_ids[1];
}

void
mtsTaskInterfaceProxy::TaskInterfaceClient::__write(::IceInternal::BasicStream* __os) const
{
    __os->writeTypeId(ice_staticId());
    __os->startWriteSlice();
    __os->endWriteSlice();
#if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
    Object::__write(__os);
#else
    ::Ice::Object::__write(__os);
#endif
}

void
mtsTaskInterfaceProxy::TaskInterfaceClient::__read(::IceInternal::BasicStream* __is, bool __rid)
{
    if(__rid)
    {
        ::std::string myId;
        __is->readTypeId(myId);
    }
    __is->startReadSlice();
    __is->endReadSlice();
#if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
    Object::__read(__is, true);
#else
    ::Ice::Object::__read(__is, true);
#endif
}

void
mtsTaskInterfaceProxy::TaskInterfaceClient::__write(const ::Ice::OutputStreamPtr&) const
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type mtsTaskInterfaceProxy::TaskInterfaceClient was not generated with stream support";
    throw ex;
}

void
mtsTaskInterfaceProxy::TaskInterfaceClient::__read(const ::Ice::InputStreamPtr&, bool)
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type mtsTaskInterfaceProxy::TaskInterfaceClient was not generated with stream support";
    throw ex;
}

void 
mtsTaskInterfaceProxy::__patch__TaskInterfaceClientPtr(void* __addr, ::Ice::ObjectPtr& v)
{
    ::mtsTaskInterfaceProxy::TaskInterfaceClientPtr* p = static_cast< ::mtsTaskInterfaceProxy::TaskInterfaceClientPtr*>(__addr);
    assert(p);
    *p = ::mtsTaskInterfaceProxy::TaskInterfaceClientPtr::dynamicCast(v);
    if(v && !*p)
    {
        IceInternal::Ex::throwUOE(::mtsTaskInterfaceProxy::TaskInterfaceClient::ice_staticId(), v->ice_id());
    }
}

bool
mtsTaskInterfaceProxy::operator==(const ::mtsTaskInterfaceProxy::TaskInterfaceClient& l, const ::mtsTaskInterfaceProxy::TaskInterfaceClient& r)
{
    return static_cast<const ::Ice::Object&>(l) == static_cast<const ::Ice::Object&>(r);
}

bool
mtsTaskInterfaceProxy::operator<(const ::mtsTaskInterfaceProxy::TaskInterfaceClient& l, const ::mtsTaskInterfaceProxy::TaskInterfaceClient& r)
{
    return static_cast<const ::Ice::Object&>(l) < static_cast<const ::Ice::Object&>(r);
}

::Ice::ObjectPtr
mtsTaskInterfaceProxy::TaskInterfaceServer::ice_clone() const
{
    throw ::Ice::CloneNotImplementedException(__FILE__, __LINE__);
    return 0; // to avoid a warning with some compilers
}

static const ::std::string __mtsTaskInterfaceProxy__TaskInterfaceServer_ids[2] =
{
    "::Ice::Object",
    "::mtsTaskInterfaceProxy::TaskInterfaceServer"
};

bool
mtsTaskInterfaceProxy::TaskInterfaceServer::ice_isA(const ::std::string& _s, const ::Ice::Current&) const
{
    return ::std::binary_search(__mtsTaskInterfaceProxy__TaskInterfaceServer_ids, __mtsTaskInterfaceProxy__TaskInterfaceServer_ids + 2, _s);
}

::std::vector< ::std::string>
mtsTaskInterfaceProxy::TaskInterfaceServer::ice_ids(const ::Ice::Current&) const
{
    return ::std::vector< ::std::string>(&__mtsTaskInterfaceProxy__TaskInterfaceServer_ids[0], &__mtsTaskInterfaceProxy__TaskInterfaceServer_ids[2]);
}

const ::std::string&
mtsTaskInterfaceProxy::TaskInterfaceServer::ice_id(const ::Ice::Current&) const
{
    return __mtsTaskInterfaceProxy__TaskInterfaceServer_ids[1];
}

const ::std::string&
mtsTaskInterfaceProxy::TaskInterfaceServer::ice_staticId()
{
    return __mtsTaskInterfaceProxy__TaskInterfaceServer_ids[1];
}

::Ice::DispatchStatus
mtsTaskInterfaceProxy::TaskInterfaceServer::___AddClient(::IceInternal::Incoming& __inS, const ::Ice::Current& __current)
{
    __checkMode(::Ice::Normal, __current.mode);
    ::IceInternal::BasicStream* __is = __inS.is();
    __is->startReadEncaps();
    ::Ice::Identity ident;
    ident.__read(__is);
    __is->endReadEncaps();
    AddClient(ident, __current);
    return ::Ice::DispatchOK;
}

::Ice::DispatchStatus
mtsTaskInterfaceProxy::TaskInterfaceServer::___GetProvidedInterfaceSpecification(::IceInternal::Incoming& __inS, const ::Ice::Current& __current) const
{
    __checkMode(::Ice::Idempotent, __current.mode);
    __inS.is()->skipEmptyEncaps();
    ::IceInternal::BasicStream* __os = __inS.os();
    ::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq providedInterfaceSpecifications;
    bool __ret = GetProvidedInterfaceSpecification(providedInterfaceSpecifications, __current);
    if(providedInterfaceSpecifications.size() == 0)
    {
        __os->writeSize(0);
    }
    else
    {
        ::mtsTaskInterfaceProxy::__writeProvidedInterfaceSpecificationSeq(__os, &providedInterfaceSpecifications[0], &providedInterfaceSpecifications[0] + providedInterfaceSpecifications.size());
    }
    __os->write(__ret);
    return ::Ice::DispatchOK;
}

::Ice::DispatchStatus
mtsTaskInterfaceProxy::TaskInterfaceServer::___ExecuteCommandVoid(::IceInternal::Incoming& __inS, const ::Ice::Current& __current)
{
    __checkMode(::Ice::Normal, __current.mode);
    ::IceInternal::BasicStream* __is = __inS.is();
    __is->startReadEncaps();
    ::Ice::Int CommandSID;
    __is->read(CommandSID);
    __is->endReadEncaps();
    ExecuteCommandVoid(CommandSID, __current);
    return ::Ice::DispatchOK;
}

::Ice::DispatchStatus
mtsTaskInterfaceProxy::TaskInterfaceServer::___ExecuteCommandWrite(::IceInternal::Incoming& __inS, const ::Ice::Current& __current)
{
    __checkMode(::Ice::Normal, __current.mode);
    ::IceInternal::BasicStream* __is = __inS.is();
    __is->startReadEncaps();
    ::Ice::Int CommandSID;
    ::Ice::Double argument;
    __is->read(CommandSID);
    __is->read(argument);
    __is->endReadEncaps();
    ExecuteCommandWrite(CommandSID, argument, __current);
    return ::Ice::DispatchOK;
}

::Ice::DispatchStatus
mtsTaskInterfaceProxy::TaskInterfaceServer::___ExecuteCommandRead(::IceInternal::Incoming& __inS, const ::Ice::Current& __current)
{
    __checkMode(::Ice::Normal, __current.mode);
    ::IceInternal::BasicStream* __is = __inS.is();
    __is->startReadEncaps();
    ::Ice::Int CommandSID;
    __is->read(CommandSID);
    __is->endReadEncaps();
    ::IceInternal::BasicStream* __os = __inS.os();
    ::Ice::Double argument;
    ExecuteCommandRead(CommandSID, argument, __current);
    __os->write(argument);
    return ::Ice::DispatchOK;
}

::Ice::DispatchStatus
mtsTaskInterfaceProxy::TaskInterfaceServer::___ExecuteCommandQualifiedRead(::IceInternal::Incoming& __inS, const ::Ice::Current& __current)
{
    __checkMode(::Ice::Normal, __current.mode);
    ::IceInternal::BasicStream* __is = __inS.is();
    __is->startReadEncaps();
    ::Ice::Int CommandSID;
    ::Ice::Double argument1;
    __is->read(CommandSID);
    __is->read(argument1);
    __is->endReadEncaps();
    ::IceInternal::BasicStream* __os = __inS.os();
    ::Ice::Double argument2;
    ExecuteCommandQualifiedRead(CommandSID, argument1, argument2, __current);
    __os->write(argument2);
    return ::Ice::DispatchOK;
}

static ::std::string __mtsTaskInterfaceProxy__TaskInterfaceServer_all[] =
{
    "AddClient",
    "ExecuteCommandQualifiedRead",
    "ExecuteCommandRead",
    "ExecuteCommandVoid",
    "ExecuteCommandWrite",
    "GetProvidedInterfaceSpecification",
    "ice_id",
    "ice_ids",
    "ice_isA",
    "ice_ping"
};

::Ice::DispatchStatus
mtsTaskInterfaceProxy::TaskInterfaceServer::__dispatch(::IceInternal::Incoming& in, const ::Ice::Current& current)
{
    ::std::pair< ::std::string*, ::std::string*> r = ::std::equal_range(__mtsTaskInterfaceProxy__TaskInterfaceServer_all, __mtsTaskInterfaceProxy__TaskInterfaceServer_all + 10, current.operation);
    if(r.first == r.second)
    {
        throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
    }

    switch(r.first - __mtsTaskInterfaceProxy__TaskInterfaceServer_all)
    {
        case 0:
        {
            return ___AddClient(in, current);
        }
        case 1:
        {
            return ___ExecuteCommandQualifiedRead(in, current);
        }
        case 2:
        {
            return ___ExecuteCommandRead(in, current);
        }
        case 3:
        {
            return ___ExecuteCommandVoid(in, current);
        }
        case 4:
        {
            return ___ExecuteCommandWrite(in, current);
        }
        case 5:
        {
            return ___GetProvidedInterfaceSpecification(in, current);
        }
        case 6:
        {
            return ___ice_id(in, current);
        }
        case 7:
        {
            return ___ice_ids(in, current);
        }
        case 8:
        {
            return ___ice_isA(in, current);
        }
        case 9:
        {
            return ___ice_ping(in, current);
        }
    }

    assert(false);
    throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
}

void
mtsTaskInterfaceProxy::TaskInterfaceServer::__write(::IceInternal::BasicStream* __os) const
{
    __os->writeTypeId(ice_staticId());
    __os->startWriteSlice();
    __os->endWriteSlice();
#if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
    Object::__write(__os);
#else
    ::Ice::Object::__write(__os);
#endif
}

void
mtsTaskInterfaceProxy::TaskInterfaceServer::__read(::IceInternal::BasicStream* __is, bool __rid)
{
    if(__rid)
    {
        ::std::string myId;
        __is->readTypeId(myId);
    }
    __is->startReadSlice();
    __is->endReadSlice();
#if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
    Object::__read(__is, true);
#else
    ::Ice::Object::__read(__is, true);
#endif
}

void
mtsTaskInterfaceProxy::TaskInterfaceServer::__write(const ::Ice::OutputStreamPtr&) const
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type mtsTaskInterfaceProxy::TaskInterfaceServer was not generated with stream support";
    throw ex;
}

void
mtsTaskInterfaceProxy::TaskInterfaceServer::__read(const ::Ice::InputStreamPtr&, bool)
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type mtsTaskInterfaceProxy::TaskInterfaceServer was not generated with stream support";
    throw ex;
}

void 
mtsTaskInterfaceProxy::__patch__TaskInterfaceServerPtr(void* __addr, ::Ice::ObjectPtr& v)
{
    ::mtsTaskInterfaceProxy::TaskInterfaceServerPtr* p = static_cast< ::mtsTaskInterfaceProxy::TaskInterfaceServerPtr*>(__addr);
    assert(p);
    *p = ::mtsTaskInterfaceProxy::TaskInterfaceServerPtr::dynamicCast(v);
    if(v && !*p)
    {
        IceInternal::Ex::throwUOE(::mtsTaskInterfaceProxy::TaskInterfaceServer::ice_staticId(), v->ice_id());
    }
}

bool
mtsTaskInterfaceProxy::operator==(const ::mtsTaskInterfaceProxy::TaskInterfaceServer& l, const ::mtsTaskInterfaceProxy::TaskInterfaceServer& r)
{
    return static_cast<const ::Ice::Object&>(l) == static_cast<const ::Ice::Object&>(r);
}

bool
mtsTaskInterfaceProxy::operator<(const ::mtsTaskInterfaceProxy::TaskInterfaceServer& l, const ::mtsTaskInterfaceProxy::TaskInterfaceServer& r)
{
    return static_cast<const ::Ice::Object&>(l) < static_cast<const ::Ice::Object&>(r);
}
