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

#ifndef ICE_ENABLE_DLL_EXPORTS
#   define ICE_ENABLE_DLL_EXPORTS
#endif
#include <mtsTaskManagerProxy.h>
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

static const ::std::string __mtsTaskManagerProxy__TaskManagerChannel__ShareTaskInfo_name = "ShareTaskInfo";

ICE_DECLSPEC_EXPORT ::Ice::Object* IceInternal::upCast(::mtsTaskManagerProxy::TaskManagerChannel* p) { return p; }
ICE_DECLSPEC_EXPORT ::IceProxy::Ice::Object* IceInternal::upCast(::IceProxy::mtsTaskManagerProxy::TaskManagerChannel* p) { return p; }

void
mtsTaskManagerProxy::__read(::IceInternal::BasicStream* __is, ::mtsTaskManagerProxy::TaskManagerChannelPrx& v)
{
    ::Ice::ObjectPrx proxy;
    __is->read(proxy);
    if(!proxy)
    {
        v = 0;
    }
    else
    {
        v = new ::IceProxy::mtsTaskManagerProxy::TaskManagerChannel;
        v->__copyFrom(proxy);
    }
}

bool
mtsTaskManagerProxy::TaskInfo::operator==(const TaskInfo& __rhs) const
{
    if(this == &__rhs)
    {
        return true;
    }
    if(taskNames != __rhs.taskNames)
    {
        return false;
    }
    return true;
}

bool
mtsTaskManagerProxy::TaskInfo::operator<(const TaskInfo& __rhs) const
{
    if(this == &__rhs)
    {
        return false;
    }
    if(taskNames < __rhs.taskNames)
    {
        return true;
    }
    else if(__rhs.taskNames < taskNames)
    {
        return false;
    }
    return false;
}

void
mtsTaskManagerProxy::TaskInfo::__write(::IceInternal::BasicStream* __os) const
{
    if(taskNames.size() == 0)
    {
        __os->writeSize(0);
    }
    else
    {
        __os->write(&taskNames[0], &taskNames[0] + taskNames.size());
    }
}

void
mtsTaskManagerProxy::TaskInfo::__read(::IceInternal::BasicStream* __is)
{
    __is->read(taskNames);
}
#ifdef __SUNPRO_CC
class ICE_DECLSPEC_EXPORT IceProxy::mtsTaskManagerProxy::TaskManagerChannel;
#endif

void
IceProxy::mtsTaskManagerProxy::TaskManagerChannel::ShareTaskInfo(const ::mtsTaskManagerProxy::TaskInfo& myTaskInfo, ::mtsTaskManagerProxy::TaskInfo& peerTaskInfo, const ::Ice::Context* __ctx)
{
    int __cnt = 0;
    while(true)
    {
        ::IceInternal::Handle< ::IceDelegate::Ice::Object> __delBase;
        try
        {
            __checkTwowayOnly(__mtsTaskManagerProxy__TaskManagerChannel__ShareTaskInfo_name);
            __delBase = __getDelegate(false);
            ::IceDelegate::mtsTaskManagerProxy::TaskManagerChannel* __del = dynamic_cast< ::IceDelegate::mtsTaskManagerProxy::TaskManagerChannel*>(__delBase.get());
            __del->ShareTaskInfo(myTaskInfo, peerTaskInfo, __ctx);
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
IceProxy::mtsTaskManagerProxy::TaskManagerChannel::ice_staticId()
{
    return ::mtsTaskManagerProxy::TaskManagerChannel::ice_staticId();
}

::IceInternal::Handle< ::IceDelegateM::Ice::Object>
IceProxy::mtsTaskManagerProxy::TaskManagerChannel::__createDelegateM()
{
    return ::IceInternal::Handle< ::IceDelegateM::Ice::Object>(new ::IceDelegateM::mtsTaskManagerProxy::TaskManagerChannel);
}

::IceInternal::Handle< ::IceDelegateD::Ice::Object>
IceProxy::mtsTaskManagerProxy::TaskManagerChannel::__createDelegateD()
{
    return ::IceInternal::Handle< ::IceDelegateD::Ice::Object>(new ::IceDelegateD::mtsTaskManagerProxy::TaskManagerChannel);
}

::IceProxy::Ice::Object*
IceProxy::mtsTaskManagerProxy::TaskManagerChannel::__newInstance() const
{
    return new TaskManagerChannel;
}

void
IceDelegateM::mtsTaskManagerProxy::TaskManagerChannel::ShareTaskInfo(const ::mtsTaskManagerProxy::TaskInfo& myTaskInfo, ::mtsTaskManagerProxy::TaskInfo& peerTaskInfo, const ::Ice::Context* __context)
{
    ::IceInternal::Outgoing __og(__handler.get(), __mtsTaskManagerProxy__TaskManagerChannel__ShareTaskInfo_name, ::Ice::Normal, __context);
    try
    {
        ::IceInternal::BasicStream* __os = __og.os();
        myTaskInfo.__write(__os);
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
        peerTaskInfo.__read(__is);
        __is->endReadEncaps();
    }
    catch(const ::Ice::LocalException& __ex)
    {
        throw ::IceInternal::LocalExceptionWrapper(__ex, false);
    }
}

void
IceDelegateD::mtsTaskManagerProxy::TaskManagerChannel::ShareTaskInfo(const ::mtsTaskManagerProxy::TaskInfo& myTaskInfo, ::mtsTaskManagerProxy::TaskInfo& peerTaskInfo, const ::Ice::Context* __context)
{
    class _DirectI : public ::IceInternal::Direct
    {
    public:

        _DirectI(const ::mtsTaskManagerProxy::TaskInfo& myTaskInfo, ::mtsTaskManagerProxy::TaskInfo& peerTaskInfo, const ::Ice::Current& __current) : 
            ::IceInternal::Direct(__current),
            _m_myTaskInfo(myTaskInfo),
            _m_peerTaskInfo(peerTaskInfo)
        {
        }
        
        virtual ::Ice::DispatchStatus
        run(::Ice::Object* object)
        {
            ::mtsTaskManagerProxy::TaskManagerChannel* servant = dynamic_cast< ::mtsTaskManagerProxy::TaskManagerChannel*>(object);
            if(!servant)
            {
                throw ::Ice::OperationNotExistException(__FILE__, __LINE__, _current.id, _current.facet, _current.operation);
            }
            servant->ShareTaskInfo(_m_myTaskInfo, _m_peerTaskInfo, _current);
            return ::Ice::DispatchOK;
        }
        
    private:
        
        const ::mtsTaskManagerProxy::TaskInfo& _m_myTaskInfo;
        ::mtsTaskManagerProxy::TaskInfo& _m_peerTaskInfo;
    };
    
    ::Ice::Current __current;
    __initCurrent(__current, __mtsTaskManagerProxy__TaskManagerChannel__ShareTaskInfo_name, ::Ice::Normal, __context);
    try
    {
        _DirectI __direct(myTaskInfo, peerTaskInfo, __current);
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
mtsTaskManagerProxy::TaskManagerChannel::ice_clone() const
{
    throw ::Ice::CloneNotImplementedException(__FILE__, __LINE__);
    return 0; // to avoid a warning with some compilers
}

static const ::std::string __mtsTaskManagerProxy__TaskManagerChannel_ids[2] =
{
    "::Ice::Object",
    "::mtsTaskManagerProxy::TaskManagerChannel"
};

bool
mtsTaskManagerProxy::TaskManagerChannel::ice_isA(const ::std::string& _s, const ::Ice::Current&) const
{
    return ::std::binary_search(__mtsTaskManagerProxy__TaskManagerChannel_ids, __mtsTaskManagerProxy__TaskManagerChannel_ids + 2, _s);
}

::std::vector< ::std::string>
mtsTaskManagerProxy::TaskManagerChannel::ice_ids(const ::Ice::Current&) const
{
    return ::std::vector< ::std::string>(&__mtsTaskManagerProxy__TaskManagerChannel_ids[0], &__mtsTaskManagerProxy__TaskManagerChannel_ids[2]);
}

const ::std::string&
mtsTaskManagerProxy::TaskManagerChannel::ice_id(const ::Ice::Current&) const
{
    return __mtsTaskManagerProxy__TaskManagerChannel_ids[1];
}

const ::std::string&
mtsTaskManagerProxy::TaskManagerChannel::ice_staticId()
{
    return __mtsTaskManagerProxy__TaskManagerChannel_ids[1];
}

::Ice::DispatchStatus
mtsTaskManagerProxy::TaskManagerChannel::___ShareTaskInfo(::IceInternal::Incoming& __inS, const ::Ice::Current& __current)
{
    __checkMode(::Ice::Normal, __current.mode);
    ::IceInternal::BasicStream* __is = __inS.is();
    __is->startReadEncaps();
    ::mtsTaskManagerProxy::TaskInfo myTaskInfo;
    myTaskInfo.__read(__is);
    __is->endReadEncaps();
    ::IceInternal::BasicStream* __os = __inS.os();
    ::mtsTaskManagerProxy::TaskInfo peerTaskInfo;
    ShareTaskInfo(myTaskInfo, peerTaskInfo, __current);
    peerTaskInfo.__write(__os);
    return ::Ice::DispatchOK;
}

static ::std::string __mtsTaskManagerProxy__TaskManagerChannel_all[] =
{
    "ShareTaskInfo",
    "ice_id",
    "ice_ids",
    "ice_isA",
    "ice_ping"
};

::Ice::DispatchStatus
mtsTaskManagerProxy::TaskManagerChannel::__dispatch(::IceInternal::Incoming& in, const ::Ice::Current& current)
{
    ::std::pair< ::std::string*, ::std::string*> r = ::std::equal_range(__mtsTaskManagerProxy__TaskManagerChannel_all, __mtsTaskManagerProxy__TaskManagerChannel_all + 5, current.operation);
    if(r.first == r.second)
    {
        throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
    }

    switch(r.first - __mtsTaskManagerProxy__TaskManagerChannel_all)
    {
        case 0:
        {
            return ___ShareTaskInfo(in, current);
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
mtsTaskManagerProxy::TaskManagerChannel::__write(::IceInternal::BasicStream* __os) const
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
mtsTaskManagerProxy::TaskManagerChannel::__read(::IceInternal::BasicStream* __is, bool __rid)
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
mtsTaskManagerProxy::TaskManagerChannel::__write(const ::Ice::OutputStreamPtr&) const
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type mtsTaskManagerProxy::TaskManagerChannel was not generated with stream support";
    throw ex;
}

void
mtsTaskManagerProxy::TaskManagerChannel::__read(const ::Ice::InputStreamPtr&, bool)
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type mtsTaskManagerProxy::TaskManagerChannel was not generated with stream support";
    throw ex;
}

void ICE_DECLSPEC_EXPORT 
mtsTaskManagerProxy::__patch__TaskManagerChannelPtr(void* __addr, ::Ice::ObjectPtr& v)
{
    ::mtsTaskManagerProxy::TaskManagerChannelPtr* p = static_cast< ::mtsTaskManagerProxy::TaskManagerChannelPtr*>(__addr);
    assert(p);
    *p = ::mtsTaskManagerProxy::TaskManagerChannelPtr::dynamicCast(v);
    if(v && !*p)
    {
        IceInternal::Ex::throwUOE(::mtsTaskManagerProxy::TaskManagerChannel::ice_staticId(), v->ice_id());
    }
}

bool
mtsTaskManagerProxy::operator==(const ::mtsTaskManagerProxy::TaskManagerChannel& l, const ::mtsTaskManagerProxy::TaskManagerChannel& r)
{
    return static_cast<const ::Ice::Object&>(l) == static_cast<const ::Ice::Object&>(r);
}

bool
mtsTaskManagerProxy::operator<(const ::mtsTaskManagerProxy::TaskManagerChannel& l, const ::mtsTaskManagerProxy::TaskManagerChannel& r)
{
    return static_cast<const ::Ice::Object&>(l) < static_cast<const ::Ice::Object&>(r);
}
