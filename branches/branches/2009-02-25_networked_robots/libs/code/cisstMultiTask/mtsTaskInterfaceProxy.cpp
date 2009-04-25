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

static ::std::string __mtsTaskInterfaceProxy__TaskInterfaceServer_all[] =
{
    "AddClient",
    "ice_id",
    "ice_ids",
    "ice_isA",
    "ice_ping"
};

::Ice::DispatchStatus
mtsTaskInterfaceProxy::TaskInterfaceServer::__dispatch(::IceInternal::Incoming& in, const ::Ice::Current& current)
{
    ::std::pair< ::std::string*, ::std::string*> r = ::std::equal_range(__mtsTaskInterfaceProxy__TaskInterfaceServer_all, __mtsTaskInterfaceProxy__TaskInterfaceServer_all + 5, current.operation);
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
            return ___ice_id(in, current);
        }
        case 2:
        {
            return ___ice_ids(in, current);
        }
        case 3:
        {
            return ___ice_isA(in, current);
        }
        case 4:
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
