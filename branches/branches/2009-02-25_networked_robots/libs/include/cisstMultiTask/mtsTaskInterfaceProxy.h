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

#ifndef __mtsTaskInterfaceProxy_h__
#define __mtsTaskInterfaceProxy_h__

#include <Ice/LocalObjectF.h>
#include <Ice/ProxyF.h>
#include <Ice/ObjectF.h>
#include <Ice/Exception.h>
#include <Ice/LocalObject.h>
#include <Ice/Proxy.h>
#include <Ice/Object.h>
#include <Ice/Outgoing.h>
#include <Ice/Incoming.h>
#include <Ice/Direct.h>
#include <Ice/StreamF.h>
#include <Ice/Identity.h>
#include <Ice/UndefSysMacros.h>

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

namespace IceProxy
{

namespace mtsTaskInterfaceProxy
{

class TaskInterfaceClient;

class TaskInterfaceServer;

}

}

namespace mtsTaskInterfaceProxy
{

class TaskInterfaceClient;
bool operator==(const TaskInterfaceClient&, const TaskInterfaceClient&);
bool operator<(const TaskInterfaceClient&, const TaskInterfaceClient&);

class TaskInterfaceServer;
bool operator==(const TaskInterfaceServer&, const TaskInterfaceServer&);
bool operator<(const TaskInterfaceServer&, const TaskInterfaceServer&);

}

namespace IceInternal
{

::Ice::Object* upCast(::mtsTaskInterfaceProxy::TaskInterfaceClient*);
::IceProxy::Ice::Object* upCast(::IceProxy::mtsTaskInterfaceProxy::TaskInterfaceClient*);

::Ice::Object* upCast(::mtsTaskInterfaceProxy::TaskInterfaceServer*);
::IceProxy::Ice::Object* upCast(::IceProxy::mtsTaskInterfaceProxy::TaskInterfaceServer*);

}

namespace mtsTaskInterfaceProxy
{

typedef ::IceInternal::Handle< ::mtsTaskInterfaceProxy::TaskInterfaceClient> TaskInterfaceClientPtr;
typedef ::IceInternal::ProxyHandle< ::IceProxy::mtsTaskInterfaceProxy::TaskInterfaceClient> TaskInterfaceClientPrx;

void __read(::IceInternal::BasicStream*, TaskInterfaceClientPrx&);
void __patch__TaskInterfaceClientPtr(void*, ::Ice::ObjectPtr&);

typedef ::IceInternal::Handle< ::mtsTaskInterfaceProxy::TaskInterfaceServer> TaskInterfaceServerPtr;
typedef ::IceInternal::ProxyHandle< ::IceProxy::mtsTaskInterfaceProxy::TaskInterfaceServer> TaskInterfaceServerPrx;

void __read(::IceInternal::BasicStream*, TaskInterfaceServerPrx&);
void __patch__TaskInterfaceServerPtr(void*, ::Ice::ObjectPtr&);

}

namespace mtsTaskInterfaceProxy
{

struct CommandVoidInfo
{
    ::std::string Name;

    bool operator==(const CommandVoidInfo&) const;
    bool operator<(const CommandVoidInfo&) const;
    bool operator!=(const CommandVoidInfo& __rhs) const
    {
        return !operator==(__rhs);
    }
    bool operator<=(const CommandVoidInfo& __rhs) const
    {
        return operator<(__rhs) || operator==(__rhs);
    }
    bool operator>(const CommandVoidInfo& __rhs) const
    {
        return !operator<(__rhs) && !operator==(__rhs);
    }
    bool operator>=(const CommandVoidInfo& __rhs) const
    {
        return !operator<(__rhs);
    }

    void __write(::IceInternal::BasicStream*) const;
    void __read(::IceInternal::BasicStream*);
};

struct CommandWriteInfo
{
    ::std::string Name;
    ::std::string ArgumentTypeName;

    bool operator==(const CommandWriteInfo&) const;
    bool operator<(const CommandWriteInfo&) const;
    bool operator!=(const CommandWriteInfo& __rhs) const
    {
        return !operator==(__rhs);
    }
    bool operator<=(const CommandWriteInfo& __rhs) const
    {
        return operator<(__rhs) || operator==(__rhs);
    }
    bool operator>(const CommandWriteInfo& __rhs) const
    {
        return !operator<(__rhs) && !operator==(__rhs);
    }
    bool operator>=(const CommandWriteInfo& __rhs) const
    {
        return !operator<(__rhs);
    }

    void __write(::IceInternal::BasicStream*) const;
    void __read(::IceInternal::BasicStream*);
};

struct CommandReadInfo
{
    ::std::string Name;
    ::std::string ArgumentTypeName;

    bool operator==(const CommandReadInfo&) const;
    bool operator<(const CommandReadInfo&) const;
    bool operator!=(const CommandReadInfo& __rhs) const
    {
        return !operator==(__rhs);
    }
    bool operator<=(const CommandReadInfo& __rhs) const
    {
        return operator<(__rhs) || operator==(__rhs);
    }
    bool operator>(const CommandReadInfo& __rhs) const
    {
        return !operator<(__rhs) && !operator==(__rhs);
    }
    bool operator>=(const CommandReadInfo& __rhs) const
    {
        return !operator<(__rhs);
    }

    void __write(::IceInternal::BasicStream*) const;
    void __read(::IceInternal::BasicStream*);
};

struct CommandQualifiedReadInfo
{
    ::std::string Name;
    ::std::string Argument1TypeName;
    ::std::string Argument2TypeName;

    bool operator==(const CommandQualifiedReadInfo&) const;
    bool operator<(const CommandQualifiedReadInfo&) const;
    bool operator!=(const CommandQualifiedReadInfo& __rhs) const
    {
        return !operator==(__rhs);
    }
    bool operator<=(const CommandQualifiedReadInfo& __rhs) const
    {
        return operator<(__rhs) || operator==(__rhs);
    }
    bool operator>(const CommandQualifiedReadInfo& __rhs) const
    {
        return !operator<(__rhs) && !operator==(__rhs);
    }
    bool operator>=(const CommandQualifiedReadInfo& __rhs) const
    {
        return !operator<(__rhs);
    }

    void __write(::IceInternal::BasicStream*) const;
    void __read(::IceInternal::BasicStream*);
};

typedef ::std::vector< ::mtsTaskInterfaceProxy::CommandVoidInfo> CommandVoidSeq;
void __writeCommandVoidSeq(::IceInternal::BasicStream*, const ::mtsTaskInterfaceProxy::CommandVoidInfo*, const ::mtsTaskInterfaceProxy::CommandVoidInfo*);
void __readCommandVoidSeq(::IceInternal::BasicStream*, CommandVoidSeq&);

typedef ::std::vector< ::mtsTaskInterfaceProxy::CommandWriteInfo> CommandWriteSeq;
void __writeCommandWriteSeq(::IceInternal::BasicStream*, const ::mtsTaskInterfaceProxy::CommandWriteInfo*, const ::mtsTaskInterfaceProxy::CommandWriteInfo*);
void __readCommandWriteSeq(::IceInternal::BasicStream*, CommandWriteSeq&);

typedef ::std::vector< ::mtsTaskInterfaceProxy::CommandReadInfo> CommandReadSeq;
void __writeCommandReadSeq(::IceInternal::BasicStream*, const ::mtsTaskInterfaceProxy::CommandReadInfo*, const ::mtsTaskInterfaceProxy::CommandReadInfo*);
void __readCommandReadSeq(::IceInternal::BasicStream*, CommandReadSeq&);

typedef ::std::vector< ::mtsTaskInterfaceProxy::CommandQualifiedReadInfo> CommandQualifiedReadSeq;
void __writeCommandQualifiedReadSeq(::IceInternal::BasicStream*, const ::mtsTaskInterfaceProxy::CommandQualifiedReadInfo*, const ::mtsTaskInterfaceProxy::CommandQualifiedReadInfo*);
void __readCommandQualifiedReadSeq(::IceInternal::BasicStream*, CommandQualifiedReadSeq&);

struct ProvidedInterfaceSpecification
{
    ::std::string interfaceName;
    ::mtsTaskInterfaceProxy::CommandVoidSeq commandsVoid;
    ::mtsTaskInterfaceProxy::CommandWriteSeq commandsWrite;
    ::mtsTaskInterfaceProxy::CommandReadSeq commandsRead;
    ::mtsTaskInterfaceProxy::CommandQualifiedReadSeq commandsQualifiedRead;

    bool operator==(const ProvidedInterfaceSpecification&) const;
    bool operator<(const ProvidedInterfaceSpecification&) const;
    bool operator!=(const ProvidedInterfaceSpecification& __rhs) const
    {
        return !operator==(__rhs);
    }
    bool operator<=(const ProvidedInterfaceSpecification& __rhs) const
    {
        return operator<(__rhs) || operator==(__rhs);
    }
    bool operator>(const ProvidedInterfaceSpecification& __rhs) const
    {
        return !operator<(__rhs) && !operator==(__rhs);
    }
    bool operator>=(const ProvidedInterfaceSpecification& __rhs) const
    {
        return !operator<(__rhs);
    }

    void __write(::IceInternal::BasicStream*) const;
    void __read(::IceInternal::BasicStream*);
};

typedef ::std::vector< ::mtsTaskInterfaceProxy::ProvidedInterfaceSpecification> ProvidedInterfaceSpecificationSeq;
void __writeProvidedInterfaceSpecificationSeq(::IceInternal::BasicStream*, const ::mtsTaskInterfaceProxy::ProvidedInterfaceSpecification*, const ::mtsTaskInterfaceProxy::ProvidedInterfaceSpecification*);
void __readProvidedInterfaceSpecificationSeq(::IceInternal::BasicStream*, ProvidedInterfaceSpecificationSeq&);

}

namespace IceProxy
{

namespace mtsTaskInterfaceProxy
{

class TaskInterfaceClient : virtual public ::IceProxy::Ice::Object
{
public:
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_context(const ::Ice::Context& __context) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_context(__context).get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_context(__context).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_adapterId(const std::string& __id) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_adapterId(__id).get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_adapterId(__id).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_endpoints(const ::Ice::EndpointSeq& __endpoints) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_endpoints(__endpoints).get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_endpoints(__endpoints).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_locatorCacheTimeout(int __timeout) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_locatorCacheTimeout(__timeout).get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_locatorCacheTimeout(__timeout).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_connectionCached(bool __cached) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_connectionCached(__cached).get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_connectionCached(__cached).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_endpointSelection(::Ice::EndpointSelectionType __est) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_endpointSelection(__est).get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_endpointSelection(__est).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_secure(bool __secure) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_secure(__secure).get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_secure(__secure).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_preferSecure(bool __preferSecure) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_preferSecure(__preferSecure).get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_preferSecure(__preferSecure).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_router(const ::Ice::RouterPrx& __router) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_router(__router).get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_router(__router).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_locator(const ::Ice::LocatorPrx& __locator) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_locator(__locator).get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_locator(__locator).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_collocationOptimized(bool __co) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_collocationOptimized(__co).get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_collocationOptimized(__co).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_twoway() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_twoway().get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_twoway().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_oneway() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_oneway().get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_oneway().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_batchOneway() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_batchOneway().get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_batchOneway().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_datagram() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_datagram().get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_datagram().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_batchDatagram() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_batchDatagram().get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_batchDatagram().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_compress(bool __compress) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_compress(__compress).get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_compress(__compress).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_timeout(int __timeout) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_timeout(__timeout).get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_timeout(__timeout).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceClient> ice_connectionId(const std::string& __id) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceClient*>(_Base::ice_connectionId(__id).get());
    #else
        return dynamic_cast<TaskInterfaceClient*>(::IceProxy::Ice::Object::ice_connectionId(__id).get());
    #endif
    }
    
    static const ::std::string& ice_staticId();

private: 

    virtual ::IceInternal::Handle< ::IceDelegateM::Ice::Object> __createDelegateM();
    virtual ::IceInternal::Handle< ::IceDelegateD::Ice::Object> __createDelegateD();
    virtual ::IceProxy::Ice::Object* __newInstance() const;
};

class TaskInterfaceServer : virtual public ::IceProxy::Ice::Object
{
public:

    void AddClient(const ::Ice::Identity& ident)
    {
        AddClient(ident, 0);
    }
    void AddClient(const ::Ice::Identity& ident, const ::Ice::Context& __ctx)
    {
        AddClient(ident, &__ctx);
    }
    
private:

    void AddClient(const ::Ice::Identity&, const ::Ice::Context*);
    
public:

    bool GetProvidedInterfaceSpecification(::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq& providedInterfaceSpecifications)
    {
        return GetProvidedInterfaceSpecification(providedInterfaceSpecifications, 0);
    }
    bool GetProvidedInterfaceSpecification(::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq& providedInterfaceSpecifications, const ::Ice::Context& __ctx)
    {
        return GetProvidedInterfaceSpecification(providedInterfaceSpecifications, &__ctx);
    }
    
private:

    bool GetProvidedInterfaceSpecification(::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq&, const ::Ice::Context*);
    
public:
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_context(const ::Ice::Context& __context) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_context(__context).get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_context(__context).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_adapterId(const std::string& __id) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_adapterId(__id).get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_adapterId(__id).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_endpoints(const ::Ice::EndpointSeq& __endpoints) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_endpoints(__endpoints).get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_endpoints(__endpoints).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_locatorCacheTimeout(int __timeout) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_locatorCacheTimeout(__timeout).get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_locatorCacheTimeout(__timeout).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_connectionCached(bool __cached) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_connectionCached(__cached).get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_connectionCached(__cached).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_endpointSelection(::Ice::EndpointSelectionType __est) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_endpointSelection(__est).get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_endpointSelection(__est).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_secure(bool __secure) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_secure(__secure).get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_secure(__secure).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_preferSecure(bool __preferSecure) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_preferSecure(__preferSecure).get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_preferSecure(__preferSecure).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_router(const ::Ice::RouterPrx& __router) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_router(__router).get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_router(__router).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_locator(const ::Ice::LocatorPrx& __locator) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_locator(__locator).get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_locator(__locator).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_collocationOptimized(bool __co) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_collocationOptimized(__co).get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_collocationOptimized(__co).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_twoway() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_twoway().get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_twoway().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_oneway() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_oneway().get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_oneway().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_batchOneway() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_batchOneway().get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_batchOneway().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_datagram() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_datagram().get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_datagram().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_batchDatagram() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_batchDatagram().get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_batchDatagram().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_compress(bool __compress) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_compress(__compress).get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_compress(__compress).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_timeout(int __timeout) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_timeout(__timeout).get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_timeout(__timeout).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskInterfaceServer> ice_connectionId(const std::string& __id) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskInterfaceServer*>(_Base::ice_connectionId(__id).get());
    #else
        return dynamic_cast<TaskInterfaceServer*>(::IceProxy::Ice::Object::ice_connectionId(__id).get());
    #endif
    }
    
    static const ::std::string& ice_staticId();

private: 

    virtual ::IceInternal::Handle< ::IceDelegateM::Ice::Object> __createDelegateM();
    virtual ::IceInternal::Handle< ::IceDelegateD::Ice::Object> __createDelegateD();
    virtual ::IceProxy::Ice::Object* __newInstance() const;
};

}

}

namespace IceDelegate
{

namespace mtsTaskInterfaceProxy
{

class TaskInterfaceClient : virtual public ::IceDelegate::Ice::Object
{
public:
};

class TaskInterfaceServer : virtual public ::IceDelegate::Ice::Object
{
public:

    virtual void AddClient(const ::Ice::Identity&, const ::Ice::Context*) = 0;

    virtual bool GetProvidedInterfaceSpecification(::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq&, const ::Ice::Context*) = 0;
};

}

}

namespace IceDelegateM
{

namespace mtsTaskInterfaceProxy
{

class TaskInterfaceClient : virtual public ::IceDelegate::mtsTaskInterfaceProxy::TaskInterfaceClient,
                            virtual public ::IceDelegateM::Ice::Object
{
public:
};

class TaskInterfaceServer : virtual public ::IceDelegate::mtsTaskInterfaceProxy::TaskInterfaceServer,
                            virtual public ::IceDelegateM::Ice::Object
{
public:

    virtual void AddClient(const ::Ice::Identity&, const ::Ice::Context*);

    virtual bool GetProvidedInterfaceSpecification(::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq&, const ::Ice::Context*);
};

}

}

namespace IceDelegateD
{

namespace mtsTaskInterfaceProxy
{

class TaskInterfaceClient : virtual public ::IceDelegate::mtsTaskInterfaceProxy::TaskInterfaceClient,
                            virtual public ::IceDelegateD::Ice::Object
{
public:
};

class TaskInterfaceServer : virtual public ::IceDelegate::mtsTaskInterfaceProxy::TaskInterfaceServer,
                            virtual public ::IceDelegateD::Ice::Object
{
public:

    virtual void AddClient(const ::Ice::Identity&, const ::Ice::Context*);

    virtual bool GetProvidedInterfaceSpecification(::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq&, const ::Ice::Context*);
};

}

}

namespace mtsTaskInterfaceProxy
{

class TaskInterfaceClient : virtual public ::Ice::Object
{
public:

    typedef TaskInterfaceClientPrx ProxyType;
    typedef TaskInterfaceClientPtr PointerType;
    
    virtual ::Ice::ObjectPtr ice_clone() const;

    virtual bool ice_isA(const ::std::string&, const ::Ice::Current& = ::Ice::Current()) const;
    virtual ::std::vector< ::std::string> ice_ids(const ::Ice::Current& = ::Ice::Current()) const;
    virtual const ::std::string& ice_id(const ::Ice::Current& = ::Ice::Current()) const;
    static const ::std::string& ice_staticId();

    virtual void __write(::IceInternal::BasicStream*) const;
    virtual void __read(::IceInternal::BasicStream*, bool);
    virtual void __write(const ::Ice::OutputStreamPtr&) const;
    virtual void __read(const ::Ice::InputStreamPtr&, bool);
};

class TaskInterfaceServer : virtual public ::Ice::Object
{
public:

    typedef TaskInterfaceServerPrx ProxyType;
    typedef TaskInterfaceServerPtr PointerType;
    
    virtual ::Ice::ObjectPtr ice_clone() const;

    virtual bool ice_isA(const ::std::string&, const ::Ice::Current& = ::Ice::Current()) const;
    virtual ::std::vector< ::std::string> ice_ids(const ::Ice::Current& = ::Ice::Current()) const;
    virtual const ::std::string& ice_id(const ::Ice::Current& = ::Ice::Current()) const;
    static const ::std::string& ice_staticId();

    virtual void AddClient(const ::Ice::Identity&, const ::Ice::Current& = ::Ice::Current()) = 0;
    ::Ice::DispatchStatus ___AddClient(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual bool GetProvidedInterfaceSpecification(::mtsTaskInterfaceProxy::ProvidedInterfaceSpecificationSeq&, const ::Ice::Current& = ::Ice::Current()) const = 0;
    ::Ice::DispatchStatus ___GetProvidedInterfaceSpecification(::IceInternal::Incoming&, const ::Ice::Current&) const;

    virtual ::Ice::DispatchStatus __dispatch(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual void __write(::IceInternal::BasicStream*) const;
    virtual void __read(::IceInternal::BasicStream*, bool);
    virtual void __write(const ::Ice::OutputStreamPtr&) const;
    virtual void __read(const ::Ice::InputStreamPtr&, bool);
};

}

#endif
