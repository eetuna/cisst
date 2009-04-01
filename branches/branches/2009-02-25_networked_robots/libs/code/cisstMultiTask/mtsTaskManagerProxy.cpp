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

static const ::std::string __mtsTaskManagerProxy__TaskManagerCommunicator__ShareTaskInfo_name = "ShareTaskInfo";

ICE_DECLSPEC_EXPORT ::Ice::Object* IceInternal::upCast(::mtsTaskManagerProxy::TaskManagerCommunicator* p) { return p; }
ICE_DECLSPEC_EXPORT ::IceProxy::Ice::Object* IceInternal::upCast(::IceProxy::mtsTaskManagerProxy::TaskManagerCommunicator* p) { return p; }

void
mtsTaskManagerProxy::__read(::IceInternal::BasicStream* __is, ::mtsTaskManagerProxy::TaskManagerCommunicatorPrx& v)
{
    ::Ice::ObjectPrx proxy;
    __is->read(proxy);
    if(!proxy)
    {
        v = 0;
    }
    else
    {
        v = new ::IceProxy::mtsTaskManagerProxy::TaskManagerCommunicator;
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
class ICE_DECLSPEC_EXPORT IceProxy::mtsTaskManagerProxy::TaskManagerCommunicator;
#endif

void
IceProxy::mtsTaskManagerProxy::TaskManagerCommunicator::ShareTaskInfo(const ::mtsTaskManagerProxy::TaskInfo& clientTaskInfo, ::mtsTaskManagerProxy::TaskInfo& serverTaskInfo, const ::Ice::Context* __ctx)
{
    int __cnt = 0;
    while(true)
    {
        ::IceInternal::Handle< ::IceDelegate::Ice::Object> __delBase;
        try
        {
            __checkTwowayOnly(__mtsTaskManagerProxy__TaskManagerCommunicator__ShareTaskInfo_name);
            __delBase = __getDelegate(false);
            ::IceDelegate::mtsTaskManagerProxy::TaskManagerCommunicator* __del = dynamic_cast< ::IceDelegate::mtsTaskManagerProxy::TaskManagerCommunicator*>(__delBase.get());
            __del->ShareTaskInfo(clientTaskInfo, serverTaskInfo, __ctx);
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
IceProxy::mtsTaskManagerProxy::TaskManagerCommunicator::ice_staticId()
{
    return ::mtsTaskManagerProxy::TaskManagerCommunicator::ice_staticId();
}

::IceInternal::Handle< ::IceDelegateM::Ice::Object>
IceProxy::mtsTaskManagerProxy::TaskManagerCommunicator::__createDelegateM()
{
    return ::IceInternal::Handle< ::IceDelegateM::Ice::Object>(new ::IceDelegateM::mtsTaskManagerProxy::TaskManagerCommunicator);
}

::IceInternal::Handle< ::IceDelegateD::Ice::Object>
IceProxy::mtsTaskManagerProxy::TaskManagerCommunicator::__createDelegateD()
{
    return ::IceInternal::Handle< ::IceDelegateD::Ice::Object>(new ::IceDelegateD::mtsTaskManagerProxy::TaskManagerCommunicator);
}

::IceProxy::Ice::Object*
IceProxy::mtsTaskManagerProxy::TaskManagerCommunicator::__newInstance() const
{
    return new TaskManagerCommunicator;
}

void
IceDelegateM::mtsTaskManagerProxy::TaskManagerCommunicator::ShareTaskInfo(const ::mtsTaskManagerProxy::TaskInfo& clientTaskInfo, ::mtsTaskManagerProxy::TaskInfo& serverTaskInfo, const ::Ice::Context* __context)
{
    ::IceInternal::Outgoing __og(__handler.get(), __mtsTaskManagerProxy__TaskManagerCommunicator__ShareTaskInfo_name, ::Ice::Normal, __context);
    try
    {
        ::IceInternal::BasicStream* __os = __og.os();
        clientTaskInfo.__write(__os);
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
        serverTaskInfo.__read(__is);
        __is->endReadEncaps();
    }
    catch(const ::Ice::LocalException& __ex)
    {
        throw ::IceInternal::LocalExceptionWrapper(__ex, false);
    }
}

void
IceDelegateD::mtsTaskManagerProxy::TaskManagerCommunicator::ShareTaskInfo(const ::mtsTaskManagerProxy::TaskInfo& clientTaskInfo, ::mtsTaskManagerProxy::TaskInfo& serverTaskInfo, const ::Ice::Context* __context)
{
    class _DirectI : public ::IceInternal::Direct
    {
    public:

        _DirectI(const ::mtsTaskManagerProxy::TaskInfo& clientTaskInfo, ::mtsTaskManagerProxy::TaskInfo& serverTaskInfo, const ::Ice::Current& __current) : 
            ::IceInternal::Direct(__current),
            _m_clientTaskInfo(clientTaskInfo),
            _m_serverTaskInfo(serverTaskInfo)
        {
        }
        
        virtual ::Ice::DispatchStatus
        run(::Ice::Object* object)
        {
            ::mtsTaskManagerProxy::TaskManagerCommunicator* servant = dynamic_cast< ::mtsTaskManagerProxy::TaskManagerCommunicator*>(object);
            if(!servant)
            {
                throw ::Ice::OperationNotExistException(__FILE__, __LINE__, _current.id, _current.facet, _current.operation);
            }
            servant->ShareTaskInfo(_m_clientTaskInfo, _m_serverTaskInfo, _current);
            return ::Ice::DispatchOK;
        }
        
    private:
        
        const ::mtsTaskManagerProxy::TaskInfo& _m_clientTaskInfo;
        ::mtsTaskManagerProxy::TaskInfo& _m_serverTaskInfo;
    };
    
    ::Ice::Current __current;
    __initCurrent(__current, __mtsTaskManagerProxy__TaskManagerCommunicator__ShareTaskInfo_name, ::Ice::Normal, __context);
    try
    {
        _DirectI __direct(clientTaskInfo, serverTaskInfo, __current);
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
mtsTaskManagerProxy::TaskManagerCommunicator::ice_clone() const
{
    throw ::Ice::CloneNotImplementedException(__FILE__, __LINE__);
    return 0; // to avoid a warning with some compilers
}

static const ::std::string __mtsTaskManagerProxy__TaskManagerCommunicator_ids[2] =
{
    "::Ice::Object",
    "::mtsTaskManagerProxy::TaskManagerCommunicator"
};

bool
mtsTaskManagerProxy::TaskManagerCommunicator::ice_isA(const ::std::string& _s, const ::Ice::Current&) const
{
    return ::std::binary_search(__mtsTaskManagerProxy__TaskManagerCommunicator_ids, __mtsTaskManagerProxy__TaskManagerCommunicator_ids + 2, _s);
}

::std::vector< ::std::string>
mtsTaskManagerProxy::TaskManagerCommunicator::ice_ids(const ::Ice::Current&) const
{
    return ::std::vector< ::std::string>(&__mtsTaskManagerProxy__TaskManagerCommunicator_ids[0], &__mtsTaskManagerProxy__TaskManagerCommunicator_ids[2]);
}

const ::std::string&
mtsTaskManagerProxy::TaskManagerCommunicator::ice_id(const ::Ice::Current&) const
{
    return __mtsTaskManagerProxy__TaskManagerCommunicator_ids[1];
}

const ::std::string&
mtsTaskManagerProxy::TaskManagerCommunicator::ice_staticId()
{
    return __mtsTaskManagerProxy__TaskManagerCommunicator_ids[1];
}

::Ice::DispatchStatus
mtsTaskManagerProxy::TaskManagerCommunicator::___ShareTaskInfo(::IceInternal::Incoming& __inS, const ::Ice::Current& __current)
{
    __checkMode(::Ice::Normal, __current.mode);
    ::IceInternal::BasicStream* __is = __inS.is();
    __is->startReadEncaps();
    ::mtsTaskManagerProxy::TaskInfo clientTaskInfo;
    clientTaskInfo.__read(__is);
    __is->endReadEncaps();
    ::IceInternal::BasicStream* __os = __inS.os();
    ::mtsTaskManagerProxy::TaskInfo serverTaskInfo;
    ShareTaskInfo(clientTaskInfo, serverTaskInfo, __current);
    serverTaskInfo.__write(__os);
    return ::Ice::DispatchOK;
}

static ::std::string __mtsTaskManagerProxy__TaskManagerCommunicator_all[] =
{
    "ShareTaskInfo",
    "ice_id",
    "ice_ids",
    "ice_isA",
    "ice_ping"
};

::Ice::DispatchStatus
mtsTaskManagerProxy::TaskManagerCommunicator::__dispatch(::IceInternal::Incoming& in, const ::Ice::Current& current)
{
    ::std::pair< ::std::string*, ::std::string*> r = ::std::equal_range(__mtsTaskManagerProxy__TaskManagerCommunicator_all, __mtsTaskManagerProxy__TaskManagerCommunicator_all + 5, current.operation);
    if(r.first == r.second)
    {
        throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
    }

    switch(r.first - __mtsTaskManagerProxy__TaskManagerCommunicator_all)
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
mtsTaskManagerProxy::TaskManagerCommunicator::__write(::IceInternal::BasicStream* __os) const
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
mtsTaskManagerProxy::TaskManagerCommunicator::__read(::IceInternal::BasicStream* __is, bool __rid)
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
mtsTaskManagerProxy::TaskManagerCommunicator::__write(const ::Ice::OutputStreamPtr&) const
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type mtsTaskManagerProxy::TaskManagerCommunicator was not generated with stream support";
    throw ex;
}

void
mtsTaskManagerProxy::TaskManagerCommunicator::__read(const ::Ice::InputStreamPtr&, bool)
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type mtsTaskManagerProxy::TaskManagerCommunicator was not generated with stream support";
    throw ex;
}

void ICE_DECLSPEC_EXPORT 
mtsTaskManagerProxy::__patch__TaskManagerCommunicatorPtr(void* __addr, ::Ice::ObjectPtr& v)
{
    ::mtsTaskManagerProxy::TaskManagerCommunicatorPtr* p = static_cast< ::mtsTaskManagerProxy::TaskManagerCommunicatorPtr*>(__addr);
    assert(p);
    *p = ::mtsTaskManagerProxy::TaskManagerCommunicatorPtr::dynamicCast(v);
    if(v && !*p)
    {
        IceInternal::Ex::throwUOE(::mtsTaskManagerProxy::TaskManagerCommunicator::ice_staticId(), v->ice_id());
    }
}

bool
mtsTaskManagerProxy::operator==(const ::mtsTaskManagerProxy::TaskManagerCommunicator& l, const ::mtsTaskManagerProxy::TaskManagerCommunicator& r)
{
    return static_cast<const ::Ice::Object&>(l) == static_cast<const ::Ice::Object&>(r);
}

bool
mtsTaskManagerProxy::operator<(const ::mtsTaskManagerProxy::TaskManagerCommunicator& l, const ::mtsTaskManagerProxy::TaskManagerCommunicator& r)
{
    return static_cast<const ::Ice::Object&>(l) < static_cast<const ::Ice::Object&>(r);
}
