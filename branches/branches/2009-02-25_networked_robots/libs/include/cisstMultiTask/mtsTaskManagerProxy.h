// **********************************************************************
//
// Copyright (c) 2003-2008 ZeroC, Inc. All rights reserved.
//
// This copy of Ice is licensed to you under the terms described in the
// ICE_LICENSE file included in this distribution.
//
// **********************************************************************

// Ice version 3.3.0
// Generated from file `mtsTaskManagerProxy.ice'

#ifndef __mtsTaskManagerProxy_h__
#define __mtsTaskManagerProxy_h__

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

namespace mtsTaskManagerProxy
{

class TaskManagerClient;

class TaskManagerServer;

}

}

namespace mtsTaskManagerProxy
{

class TaskManagerClient;
bool operator==(const TaskManagerClient&, const TaskManagerClient&);
bool operator<(const TaskManagerClient&, const TaskManagerClient&);

class TaskManagerServer;
bool operator==(const TaskManagerServer&, const TaskManagerServer&);
bool operator<(const TaskManagerServer&, const TaskManagerServer&);

}

namespace IceInternal
{

::Ice::Object* upCast(::mtsTaskManagerProxy::TaskManagerClient*);
::IceProxy::Ice::Object* upCast(::IceProxy::mtsTaskManagerProxy::TaskManagerClient*);

::Ice::Object* upCast(::mtsTaskManagerProxy::TaskManagerServer*);
::IceProxy::Ice::Object* upCast(::IceProxy::mtsTaskManagerProxy::TaskManagerServer*);

}

namespace mtsTaskManagerProxy
{

typedef ::IceInternal::Handle< ::mtsTaskManagerProxy::TaskManagerClient> TaskManagerClientPtr;
typedef ::IceInternal::ProxyHandle< ::IceProxy::mtsTaskManagerProxy::TaskManagerClient> TaskManagerClientPrx;

void __read(::IceInternal::BasicStream*, TaskManagerClientPrx&);
void __patch__TaskManagerClientPtr(void*, ::Ice::ObjectPtr&);

typedef ::IceInternal::Handle< ::mtsTaskManagerProxy::TaskManagerServer> TaskManagerServerPtr;
typedef ::IceInternal::ProxyHandle< ::IceProxy::mtsTaskManagerProxy::TaskManagerServer> TaskManagerServerPrx;

void __read(::IceInternal::BasicStream*, TaskManagerServerPrx&);
void __patch__TaskManagerServerPtr(void*, ::Ice::ObjectPtr&);

}

namespace mtsTaskManagerProxy
{

typedef ::std::vector< ::std::string> TaskNameSeq;

struct TaskInfo
{
    ::std::string taskManagerID;
    ::mtsTaskManagerProxy::TaskNameSeq taskNames;

    bool operator==(const TaskInfo&) const;
    bool operator<(const TaskInfo&) const;
    bool operator!=(const TaskInfo& __rhs) const
    {
        return !operator==(__rhs);
    }
    bool operator<=(const TaskInfo& __rhs) const
    {
        return operator<(__rhs) || operator==(__rhs);
    }
    bool operator>(const TaskInfo& __rhs) const
    {
        return !operator<(__rhs) && !operator==(__rhs);
    }
    bool operator>=(const TaskInfo& __rhs) const
    {
        return !operator<(__rhs);
    }

    void __write(::IceInternal::BasicStream*) const;
    void __read(::IceInternal::BasicStream*);
};

}

namespace IceProxy
{

namespace mtsTaskManagerProxy
{

class TaskManagerClient : virtual public ::IceProxy::Ice::Object
{
public:

    void ReceiveData(::Ice::Int num)
    {
        ReceiveData(num, 0);
    }
    void ReceiveData(::Ice::Int num, const ::Ice::Context& __ctx)
    {
        ReceiveData(num, &__ctx);
    }
    
private:

    void ReceiveData(::Ice::Int, const ::Ice::Context*);
    
public:
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_context(const ::Ice::Context& __context) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_context(__context).get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_context(__context).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_adapterId(const std::string& __id) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_adapterId(__id).get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_adapterId(__id).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_endpoints(const ::Ice::EndpointSeq& __endpoints) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_endpoints(__endpoints).get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_endpoints(__endpoints).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_locatorCacheTimeout(int __timeout) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_locatorCacheTimeout(__timeout).get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_locatorCacheTimeout(__timeout).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_connectionCached(bool __cached) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_connectionCached(__cached).get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_connectionCached(__cached).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_endpointSelection(::Ice::EndpointSelectionType __est) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_endpointSelection(__est).get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_endpointSelection(__est).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_secure(bool __secure) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_secure(__secure).get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_secure(__secure).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_preferSecure(bool __preferSecure) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_preferSecure(__preferSecure).get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_preferSecure(__preferSecure).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_router(const ::Ice::RouterPrx& __router) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_router(__router).get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_router(__router).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_locator(const ::Ice::LocatorPrx& __locator) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_locator(__locator).get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_locator(__locator).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_collocationOptimized(bool __co) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_collocationOptimized(__co).get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_collocationOptimized(__co).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_twoway() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_twoway().get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_twoway().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_oneway() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_oneway().get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_oneway().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_batchOneway() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_batchOneway().get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_batchOneway().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_datagram() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_datagram().get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_datagram().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_batchDatagram() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_batchDatagram().get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_batchDatagram().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_compress(bool __compress) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_compress(__compress).get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_compress(__compress).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_timeout(int __timeout) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_timeout(__timeout).get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_timeout(__timeout).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerClient> ice_connectionId(const std::string& __id) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerClient*>(_Base::ice_connectionId(__id).get());
    #else
        return dynamic_cast<TaskManagerClient*>(::IceProxy::Ice::Object::ice_connectionId(__id).get());
    #endif
    }
    
    static const ::std::string& ice_staticId();

private: 

    virtual ::IceInternal::Handle< ::IceDelegateM::Ice::Object> __createDelegateM();
    virtual ::IceInternal::Handle< ::IceDelegateD::Ice::Object> __createDelegateD();
    virtual ::IceProxy::Ice::Object* __newInstance() const;
};

class TaskManagerServer : virtual public ::IceProxy::Ice::Object
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

    void UpdateTaskInfo(const ::mtsTaskManagerProxy::TaskInfo& localTaskInfo)
    {
        UpdateTaskInfo(localTaskInfo, 0);
    }
    void UpdateTaskInfo(const ::mtsTaskManagerProxy::TaskInfo& localTaskInfo, const ::Ice::Context& __ctx)
    {
        UpdateTaskInfo(localTaskInfo, &__ctx);
    }
    
private:

    void UpdateTaskInfo(const ::mtsTaskManagerProxy::TaskInfo&, const ::Ice::Context*);
    
public:

    void ReceiveDataFromClient(::Ice::Int num)
    {
        ReceiveDataFromClient(num, 0);
    }
    void ReceiveDataFromClient(::Ice::Int num, const ::Ice::Context& __ctx)
    {
        ReceiveDataFromClient(num, &__ctx);
    }
    
private:

    void ReceiveDataFromClient(::Ice::Int, const ::Ice::Context*);
    
public:
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_context(const ::Ice::Context& __context) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_context(__context).get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_context(__context).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_adapterId(const std::string& __id) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_adapterId(__id).get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_adapterId(__id).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_endpoints(const ::Ice::EndpointSeq& __endpoints) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_endpoints(__endpoints).get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_endpoints(__endpoints).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_locatorCacheTimeout(int __timeout) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_locatorCacheTimeout(__timeout).get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_locatorCacheTimeout(__timeout).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_connectionCached(bool __cached) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_connectionCached(__cached).get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_connectionCached(__cached).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_endpointSelection(::Ice::EndpointSelectionType __est) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_endpointSelection(__est).get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_endpointSelection(__est).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_secure(bool __secure) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_secure(__secure).get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_secure(__secure).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_preferSecure(bool __preferSecure) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_preferSecure(__preferSecure).get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_preferSecure(__preferSecure).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_router(const ::Ice::RouterPrx& __router) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_router(__router).get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_router(__router).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_locator(const ::Ice::LocatorPrx& __locator) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_locator(__locator).get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_locator(__locator).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_collocationOptimized(bool __co) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_collocationOptimized(__co).get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_collocationOptimized(__co).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_twoway() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_twoway().get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_twoway().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_oneway() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_oneway().get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_oneway().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_batchOneway() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_batchOneway().get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_batchOneway().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_datagram() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_datagram().get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_datagram().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_batchDatagram() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_batchDatagram().get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_batchDatagram().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_compress(bool __compress) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_compress(__compress).get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_compress(__compress).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_timeout(int __timeout) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_timeout(__timeout).get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_timeout(__timeout).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<TaskManagerServer> ice_connectionId(const std::string& __id) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<TaskManagerServer*>(_Base::ice_connectionId(__id).get());
    #else
        return dynamic_cast<TaskManagerServer*>(::IceProxy::Ice::Object::ice_connectionId(__id).get());
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

namespace mtsTaskManagerProxy
{

class TaskManagerClient : virtual public ::IceDelegate::Ice::Object
{
public:

    virtual void ReceiveData(::Ice::Int, const ::Ice::Context*) = 0;
};

class TaskManagerServer : virtual public ::IceDelegate::Ice::Object
{
public:

    virtual void AddClient(const ::Ice::Identity&, const ::Ice::Context*) = 0;

    virtual void UpdateTaskInfo(const ::mtsTaskManagerProxy::TaskInfo&, const ::Ice::Context*) = 0;

    virtual void ReceiveDataFromClient(::Ice::Int, const ::Ice::Context*) = 0;
};

}

}

namespace IceDelegateM
{

namespace mtsTaskManagerProxy
{

class TaskManagerClient : virtual public ::IceDelegate::mtsTaskManagerProxy::TaskManagerClient,
                          virtual public ::IceDelegateM::Ice::Object
{
public:

    virtual void ReceiveData(::Ice::Int, const ::Ice::Context*);
};

class TaskManagerServer : virtual public ::IceDelegate::mtsTaskManagerProxy::TaskManagerServer,
                          virtual public ::IceDelegateM::Ice::Object
{
public:

    virtual void AddClient(const ::Ice::Identity&, const ::Ice::Context*);

    virtual void UpdateTaskInfo(const ::mtsTaskManagerProxy::TaskInfo&, const ::Ice::Context*);

    virtual void ReceiveDataFromClient(::Ice::Int, const ::Ice::Context*);
};

}

}

namespace IceDelegateD
{

namespace mtsTaskManagerProxy
{

class TaskManagerClient : virtual public ::IceDelegate::mtsTaskManagerProxy::TaskManagerClient,
                          virtual public ::IceDelegateD::Ice::Object
{
public:

    virtual void ReceiveData(::Ice::Int, const ::Ice::Context*);
};

class TaskManagerServer : virtual public ::IceDelegate::mtsTaskManagerProxy::TaskManagerServer,
                          virtual public ::IceDelegateD::Ice::Object
{
public:

    virtual void AddClient(const ::Ice::Identity&, const ::Ice::Context*);

    virtual void UpdateTaskInfo(const ::mtsTaskManagerProxy::TaskInfo&, const ::Ice::Context*);

    virtual void ReceiveDataFromClient(::Ice::Int, const ::Ice::Context*);
};

}

}

namespace mtsTaskManagerProxy
{

class TaskManagerClient : virtual public ::Ice::Object
{
public:

    typedef TaskManagerClientPrx ProxyType;
    typedef TaskManagerClientPtr PointerType;
    
    virtual ::Ice::ObjectPtr ice_clone() const;

    virtual bool ice_isA(const ::std::string&, const ::Ice::Current& = ::Ice::Current()) const;
    virtual ::std::vector< ::std::string> ice_ids(const ::Ice::Current& = ::Ice::Current()) const;
    virtual const ::std::string& ice_id(const ::Ice::Current& = ::Ice::Current()) const;
    static const ::std::string& ice_staticId();

    virtual void ReceiveData(::Ice::Int, const ::Ice::Current& = ::Ice::Current()) = 0;
    ::Ice::DispatchStatus ___ReceiveData(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual ::Ice::DispatchStatus __dispatch(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual void __write(::IceInternal::BasicStream*) const;
    virtual void __read(::IceInternal::BasicStream*, bool);
    virtual void __write(const ::Ice::OutputStreamPtr&) const;
    virtual void __read(const ::Ice::InputStreamPtr&, bool);
};

class TaskManagerServer : virtual public ::Ice::Object
{
public:

    typedef TaskManagerServerPrx ProxyType;
    typedef TaskManagerServerPtr PointerType;
    
    virtual ::Ice::ObjectPtr ice_clone() const;

    virtual bool ice_isA(const ::std::string&, const ::Ice::Current& = ::Ice::Current()) const;
    virtual ::std::vector< ::std::string> ice_ids(const ::Ice::Current& = ::Ice::Current()) const;
    virtual const ::std::string& ice_id(const ::Ice::Current& = ::Ice::Current()) const;
    static const ::std::string& ice_staticId();

    virtual void AddClient(const ::Ice::Identity&, const ::Ice::Current& = ::Ice::Current()) = 0;
    ::Ice::DispatchStatus ___AddClient(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual void UpdateTaskInfo(const ::mtsTaskManagerProxy::TaskInfo&, const ::Ice::Current& = ::Ice::Current()) const = 0;
    ::Ice::DispatchStatus ___UpdateTaskInfo(::IceInternal::Incoming&, const ::Ice::Current&) const;

    virtual void ReceiveDataFromClient(::Ice::Int, const ::Ice::Current& = ::Ice::Current()) = 0;
    ::Ice::DispatchStatus ___ReceiveDataFromClient(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual ::Ice::DispatchStatus __dispatch(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual void __write(::IceInternal::BasicStream*) const;
    virtual void __read(::IceInternal::BasicStream*, bool);
    virtual void __write(const ::Ice::OutputStreamPtr&) const;
    virtual void __read(const ::Ice::InputStreamPtr&, bool);
};

}

#endif
