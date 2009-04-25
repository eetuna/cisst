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

static const ::std::string __mtsTaskManagerProxy__TaskManagerClient__ReceiveData_name = "ReceiveData";

static const ::std::string __mtsTaskManagerProxy__TaskManagerServer__AddClient_name = "AddClient";

static const ::std::string __mtsTaskManagerProxy__TaskManagerServer__AddTaskManager_name = "AddTaskManager";

::Ice::Object* IceInternal::upCast(::mtsTaskManagerProxy::TaskManagerClient* p) { return p; }
::IceProxy::Ice::Object* IceInternal::upCast(::IceProxy::mtsTaskManagerProxy::TaskManagerClient* p) { return p; }

::Ice::Object* IceInternal::upCast(::mtsTaskManagerProxy::TaskManagerServer* p) { return p; }
::IceProxy::Ice::Object* IceInternal::upCast(::IceProxy::mtsTaskManagerProxy::TaskManagerServer* p) { return p; }

void
mtsTaskManagerProxy::__read(::IceInternal::BasicStream* __is, ::mtsTaskManagerProxy::TaskManagerClientPrx& v)
{
    ::Ice::ObjectPrx proxy;
    __is->read(proxy);
    if(!proxy)
    {
        v = 0;
    }
    else
    {
        v = new ::IceProxy::mtsTaskManagerProxy::TaskManagerClient;
        v->__copyFrom(proxy);
    }
}

void
mtsTaskManagerProxy::__read(::IceInternal::BasicStream* __is, ::mtsTaskManagerProxy::TaskManagerServerPrx& v)
{
    ::Ice::ObjectPrx proxy;
    __is->read(proxy);
    if(!proxy)
    {
        v = 0;
    }
    else
    {
        v = new ::IceProxy::mtsTaskManagerProxy::TaskManagerServer;
        v->__copyFrom(proxy);
    }
}

bool
mtsTaskManagerProxy::TaskList::operator==(const TaskList& __rhs) const
{
    if(this == &__rhs)
    {
        return true;
    }
    if(taskManagerID != __rhs.taskManagerID)
    {
        return false;
    }
    if(taskNames != __rhs.taskNames)
    {
        return false;
    }
    return true;
}

bool
mtsTaskManagerProxy::TaskList::operator<(const TaskList& __rhs) const
{
    if(this == &__rhs)
    {
        return false;
    }
    if(taskManagerID < __rhs.taskManagerID)
    {
        return true;
    }
    else if(__rhs.taskManagerID < taskManagerID)
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
mtsTaskManagerProxy::TaskList::__write(::IceInternal::BasicStream* __os) const
{
    __os->write(taskManagerID);
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
mtsTaskManagerProxy::TaskList::__read(::IceInternal::BasicStream* __is)
{
    __is->read(taskManagerID);
    __is->read(taskNames);
}

void
IceProxy::mtsTaskManagerProxy::TaskManagerClient::ReceiveData(::Ice::Int num, const ::Ice::Context* __ctx)
{
    int __cnt = 0;
    while(true)
    {
        ::IceInternal::Handle< ::IceDelegate::Ice::Object> __delBase;
        try
        {
            __delBase = __getDelegate(false);
            ::IceDelegate::mtsTaskManagerProxy::TaskManagerClient* __del = dynamic_cast< ::IceDelegate::mtsTaskManagerProxy::TaskManagerClient*>(__delBase.get());
            __del->ReceiveData(num, __ctx);
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
IceProxy::mtsTaskManagerProxy::TaskManagerClient::ice_staticId()
{
    return ::mtsTaskManagerProxy::TaskManagerClient::ice_staticId();
}

::IceInternal::Handle< ::IceDelegateM::Ice::Object>
IceProxy::mtsTaskManagerProxy::TaskManagerClient::__createDelegateM()
{
    return ::IceInternal::Handle< ::IceDelegateM::Ice::Object>(new ::IceDelegateM::mtsTaskManagerProxy::TaskManagerClient);
}

::IceInternal::Handle< ::IceDelegateD::Ice::Object>
IceProxy::mtsTaskManagerProxy::TaskManagerClient::__createDelegateD()
{
    return ::IceInternal::Handle< ::IceDelegateD::Ice::Object>(new ::IceDelegateD::mtsTaskManagerProxy::TaskManagerClient);
}

::IceProxy::Ice::Object*
IceProxy::mtsTaskManagerProxy::TaskManagerClient::__newInstance() const
{
    return new TaskManagerClient;
}

void
IceProxy::mtsTaskManagerProxy::TaskManagerServer::AddClient(const ::Ice::Identity& ident, const ::Ice::Context* __ctx)
{
    int __cnt = 0;
    while(true)
    {
        ::IceInternal::Handle< ::IceDelegate::Ice::Object> __delBase;
        try
        {
            __delBase = __getDelegate(false);
            ::IceDelegate::mtsTaskManagerProxy::TaskManagerServer* __del = dynamic_cast< ::IceDelegate::mtsTaskManagerProxy::TaskManagerServer*>(__delBase.get());
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

void
IceProxy::mtsTaskManagerProxy::TaskManagerServer::AddTaskManager(const ::mtsTaskManagerProxy::TaskList& localTaskInfo, const ::Ice::Context* __ctx)
{
    int __cnt = 0;
    while(true)
    {
        ::IceInternal::Handle< ::IceDelegate::Ice::Object> __delBase;
        try
        {
            __delBase = __getDelegate(false);
            ::IceDelegate::mtsTaskManagerProxy::TaskManagerServer* __del = dynamic_cast< ::IceDelegate::mtsTaskManagerProxy::TaskManagerServer*>(__delBase.get());
            __del->AddTaskManager(localTaskInfo, __ctx);
            return;
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

const ::std::string&
IceProxy::mtsTaskManagerProxy::TaskManagerServer::ice_staticId()
{
    return ::mtsTaskManagerProxy::TaskManagerServer::ice_staticId();
}

::IceInternal::Handle< ::IceDelegateM::Ice::Object>
IceProxy::mtsTaskManagerProxy::TaskManagerServer::__createDelegateM()
{
    return ::IceInternal::Handle< ::IceDelegateM::Ice::Object>(new ::IceDelegateM::mtsTaskManagerProxy::TaskManagerServer);
}

::IceInternal::Handle< ::IceDelegateD::Ice::Object>
IceProxy::mtsTaskManagerProxy::TaskManagerServer::__createDelegateD()
{
    return ::IceInternal::Handle< ::IceDelegateD::Ice::Object>(new ::IceDelegateD::mtsTaskManagerProxy::TaskManagerServer);
}

::IceProxy::Ice::Object*
IceProxy::mtsTaskManagerProxy::TaskManagerServer::__newInstance() const
{
    return new TaskManagerServer;
}

void
IceDelegateM::mtsTaskManagerProxy::TaskManagerClient::ReceiveData(::Ice::Int num, const ::Ice::Context* __context)
{
    ::IceInternal::Outgoing __og(__handler.get(), __mtsTaskManagerProxy__TaskManagerClient__ReceiveData_name, ::Ice::Normal, __context);
    try
    {
        ::IceInternal::BasicStream* __os = __og.os();
        __os->write(num);
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
IceDelegateM::mtsTaskManagerProxy::TaskManagerServer::AddClient(const ::Ice::Identity& ident, const ::Ice::Context* __context)
{
    ::IceInternal::Outgoing __og(__handler.get(), __mtsTaskManagerProxy__TaskManagerServer__AddClient_name, ::Ice::Normal, __context);
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
IceDelegateM::mtsTaskManagerProxy::TaskManagerServer::AddTaskManager(const ::mtsTaskManagerProxy::TaskList& localTaskInfo, const ::Ice::Context* __context)
{
    ::IceInternal::Outgoing __og(__handler.get(), __mtsTaskManagerProxy__TaskManagerServer__AddTaskManager_name, ::Ice::Idempotent, __context);
    try
    {
        ::IceInternal::BasicStream* __os = __og.os();
        localTaskInfo.__write(__os);
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
IceDelegateD::mtsTaskManagerProxy::TaskManagerClient::ReceiveData(::Ice::Int num, const ::Ice::Context* __context)
{
    class _DirectI : public ::IceInternal::Direct
    {
    public:

        _DirectI(::Ice::Int num, const ::Ice::Current& __current) : 
            ::IceInternal::Direct(__current),
            _m_num(num)
        {
        }
        
        virtual ::Ice::DispatchStatus
        run(::Ice::Object* object)
        {
            ::mtsTaskManagerProxy::TaskManagerClient* servant = dynamic_cast< ::mtsTaskManagerProxy::TaskManagerClient*>(object);
            if(!servant)
            {
                throw ::Ice::OperationNotExistException(__FILE__, __LINE__, _current.id, _current.facet, _current.operation);
            }
            servant->ReceiveData(_m_num, _current);
            return ::Ice::DispatchOK;
        }
        
    private:
        
        ::Ice::Int _m_num;
    };
    
    ::Ice::Current __current;
    __initCurrent(__current, __mtsTaskManagerProxy__TaskManagerClient__ReceiveData_name, ::Ice::Normal, __context);
    try
    {
        _DirectI __direct(num, __current);
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
IceDelegateD::mtsTaskManagerProxy::TaskManagerServer::AddClient(const ::Ice::Identity& ident, const ::Ice::Context* __context)
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
            ::mtsTaskManagerProxy::TaskManagerServer* servant = dynamic_cast< ::mtsTaskManagerProxy::TaskManagerServer*>(object);
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
    __initCurrent(__current, __mtsTaskManagerProxy__TaskManagerServer__AddClient_name, ::Ice::Normal, __context);
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

void
IceDelegateD::mtsTaskManagerProxy::TaskManagerServer::AddTaskManager(const ::mtsTaskManagerProxy::TaskList& localTaskInfo, const ::Ice::Context* __context)
{
    class _DirectI : public ::IceInternal::Direct
    {
    public:

        _DirectI(const ::mtsTaskManagerProxy::TaskList& localTaskInfo, const ::Ice::Current& __current) : 
            ::IceInternal::Direct(__current),
            _m_localTaskInfo(localTaskInfo)
        {
        }
        
        virtual ::Ice::DispatchStatus
        run(::Ice::Object* object)
        {
            ::mtsTaskManagerProxy::TaskManagerServer* servant = dynamic_cast< ::mtsTaskManagerProxy::TaskManagerServer*>(object);
            if(!servant)
            {
                throw ::Ice::OperationNotExistException(__FILE__, __LINE__, _current.id, _current.facet, _current.operation);
            }
            servant->AddTaskManager(_m_localTaskInfo, _current);
            return ::Ice::DispatchOK;
        }
        
    private:
        
        const ::mtsTaskManagerProxy::TaskList& _m_localTaskInfo;
    };
    
    ::Ice::Current __current;
    __initCurrent(__current, __mtsTaskManagerProxy__TaskManagerServer__AddTaskManager_name, ::Ice::Idempotent, __context);
    try
    {
        _DirectI __direct(localTaskInfo, __current);
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
mtsTaskManagerProxy::TaskManagerClient::ice_clone() const
{
    throw ::Ice::CloneNotImplementedException(__FILE__, __LINE__);
    return 0; // to avoid a warning with some compilers
}

static const ::std::string __mtsTaskManagerProxy__TaskManagerClient_ids[2] =
{
    "::Ice::Object",
    "::mtsTaskManagerProxy::TaskManagerClient"
};

bool
mtsTaskManagerProxy::TaskManagerClient::ice_isA(const ::std::string& _s, const ::Ice::Current&) const
{
    return ::std::binary_search(__mtsTaskManagerProxy__TaskManagerClient_ids, __mtsTaskManagerProxy__TaskManagerClient_ids + 2, _s);
}

::std::vector< ::std::string>
mtsTaskManagerProxy::TaskManagerClient::ice_ids(const ::Ice::Current&) const
{
    return ::std::vector< ::std::string>(&__mtsTaskManagerProxy__TaskManagerClient_ids[0], &__mtsTaskManagerProxy__TaskManagerClient_ids[2]);
}

const ::std::string&
mtsTaskManagerProxy::TaskManagerClient::ice_id(const ::Ice::Current&) const
{
    return __mtsTaskManagerProxy__TaskManagerClient_ids[1];
}

const ::std::string&
mtsTaskManagerProxy::TaskManagerClient::ice_staticId()
{
    return __mtsTaskManagerProxy__TaskManagerClient_ids[1];
}

::Ice::DispatchStatus
mtsTaskManagerProxy::TaskManagerClient::___ReceiveData(::IceInternal::Incoming& __inS, const ::Ice::Current& __current)
{
    __checkMode(::Ice::Normal, __current.mode);
    ::IceInternal::BasicStream* __is = __inS.is();
    __is->startReadEncaps();
    ::Ice::Int num;
    __is->read(num);
    __is->endReadEncaps();
    ReceiveData(num, __current);
    return ::Ice::DispatchOK;
}

static ::std::string __mtsTaskManagerProxy__TaskManagerClient_all[] =
{
    "ReceiveData",
    "ice_id",
    "ice_ids",
    "ice_isA",
    "ice_ping"
};

::Ice::DispatchStatus
mtsTaskManagerProxy::TaskManagerClient::__dispatch(::IceInternal::Incoming& in, const ::Ice::Current& current)
{
    ::std::pair< ::std::string*, ::std::string*> r = ::std::equal_range(__mtsTaskManagerProxy__TaskManagerClient_all, __mtsTaskManagerProxy__TaskManagerClient_all + 5, current.operation);
    if(r.first == r.second)
    {
        throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
    }

    switch(r.first - __mtsTaskManagerProxy__TaskManagerClient_all)
    {
        case 0:
        {
            return ___ReceiveData(in, current);
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
mtsTaskManagerProxy::TaskManagerClient::__write(::IceInternal::BasicStream* __os) const
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
mtsTaskManagerProxy::TaskManagerClient::__read(::IceInternal::BasicStream* __is, bool __rid)
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
mtsTaskManagerProxy::TaskManagerClient::__write(const ::Ice::OutputStreamPtr&) const
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type mtsTaskManagerProxy::TaskManagerClient was not generated with stream support";
    throw ex;
}

void
mtsTaskManagerProxy::TaskManagerClient::__read(const ::Ice::InputStreamPtr&, bool)
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type mtsTaskManagerProxy::TaskManagerClient was not generated with stream support";
    throw ex;
}

void 
mtsTaskManagerProxy::__patch__TaskManagerClientPtr(void* __addr, ::Ice::ObjectPtr& v)
{
    ::mtsTaskManagerProxy::TaskManagerClientPtr* p = static_cast< ::mtsTaskManagerProxy::TaskManagerClientPtr*>(__addr);
    assert(p);
    *p = ::mtsTaskManagerProxy::TaskManagerClientPtr::dynamicCast(v);
    if(v && !*p)
    {
        IceInternal::Ex::throwUOE(::mtsTaskManagerProxy::TaskManagerClient::ice_staticId(), v->ice_id());
    }
}

bool
mtsTaskManagerProxy::operator==(const ::mtsTaskManagerProxy::TaskManagerClient& l, const ::mtsTaskManagerProxy::TaskManagerClient& r)
{
    return static_cast<const ::Ice::Object&>(l) == static_cast<const ::Ice::Object&>(r);
}

bool
mtsTaskManagerProxy::operator<(const ::mtsTaskManagerProxy::TaskManagerClient& l, const ::mtsTaskManagerProxy::TaskManagerClient& r)
{
    return static_cast<const ::Ice::Object&>(l) < static_cast<const ::Ice::Object&>(r);
}

::Ice::ObjectPtr
mtsTaskManagerProxy::TaskManagerServer::ice_clone() const
{
    throw ::Ice::CloneNotImplementedException(__FILE__, __LINE__);
    return 0; // to avoid a warning with some compilers
}

static const ::std::string __mtsTaskManagerProxy__TaskManagerServer_ids[2] =
{
    "::Ice::Object",
    "::mtsTaskManagerProxy::TaskManagerServer"
};

bool
mtsTaskManagerProxy::TaskManagerServer::ice_isA(const ::std::string& _s, const ::Ice::Current&) const
{
    return ::std::binary_search(__mtsTaskManagerProxy__TaskManagerServer_ids, __mtsTaskManagerProxy__TaskManagerServer_ids + 2, _s);
}

::std::vector< ::std::string>
mtsTaskManagerProxy::TaskManagerServer::ice_ids(const ::Ice::Current&) const
{
    return ::std::vector< ::std::string>(&__mtsTaskManagerProxy__TaskManagerServer_ids[0], &__mtsTaskManagerProxy__TaskManagerServer_ids[2]);
}

const ::std::string&
mtsTaskManagerProxy::TaskManagerServer::ice_id(const ::Ice::Current&) const
{
    return __mtsTaskManagerProxy__TaskManagerServer_ids[1];
}

const ::std::string&
mtsTaskManagerProxy::TaskManagerServer::ice_staticId()
{
    return __mtsTaskManagerProxy__TaskManagerServer_ids[1];
}

::Ice::DispatchStatus
mtsTaskManagerProxy::TaskManagerServer::___AddClient(::IceInternal::Incoming& __inS, const ::Ice::Current& __current)
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
mtsTaskManagerProxy::TaskManagerServer::___AddTaskManager(::IceInternal::Incoming& __inS, const ::Ice::Current& __current) const
{
    __checkMode(::Ice::Idempotent, __current.mode);
    ::IceInternal::BasicStream* __is = __inS.is();
    __is->startReadEncaps();
    ::mtsTaskManagerProxy::TaskList localTaskInfo;
    localTaskInfo.__read(__is);
    __is->endReadEncaps();
    AddTaskManager(localTaskInfo, __current);
    return ::Ice::DispatchOK;
}

static ::std::string __mtsTaskManagerProxy__TaskManagerServer_all[] =
{
    "AddClient",
    "AddTaskManager",
    "ice_id",
    "ice_ids",
    "ice_isA",
    "ice_ping"
};

::Ice::DispatchStatus
mtsTaskManagerProxy::TaskManagerServer::__dispatch(::IceInternal::Incoming& in, const ::Ice::Current& current)
{
    ::std::pair< ::std::string*, ::std::string*> r = ::std::equal_range(__mtsTaskManagerProxy__TaskManagerServer_all, __mtsTaskManagerProxy__TaskManagerServer_all + 6, current.operation);
    if(r.first == r.second)
    {
        throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
    }

    switch(r.first - __mtsTaskManagerProxy__TaskManagerServer_all)
    {
        case 0:
        {
            return ___AddClient(in, current);
        }
        case 1:
        {
            return ___AddTaskManager(in, current);
        }
        case 2:
        {
            return ___ice_id(in, current);
        }
        case 3:
        {
            return ___ice_ids(in, current);
        }
        case 4:
        {
            return ___ice_isA(in, current);
        }
        case 5:
        {
            return ___ice_ping(in, current);
        }
    }

    assert(false);
    throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
}

void
mtsTaskManagerProxy::TaskManagerServer::__write(::IceInternal::BasicStream* __os) const
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
mtsTaskManagerProxy::TaskManagerServer::__read(::IceInternal::BasicStream* __is, bool __rid)
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
mtsTaskManagerProxy::TaskManagerServer::__write(const ::Ice::OutputStreamPtr&) const
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type mtsTaskManagerProxy::TaskManagerServer was not generated with stream support";
    throw ex;
}

void
mtsTaskManagerProxy::TaskManagerServer::__read(const ::Ice::InputStreamPtr&, bool)
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type mtsTaskManagerProxy::TaskManagerServer was not generated with stream support";
    throw ex;
}

void 
mtsTaskManagerProxy::__patch__TaskManagerServerPtr(void* __addr, ::Ice::ObjectPtr& v)
{
    ::mtsTaskManagerProxy::TaskManagerServerPtr* p = static_cast< ::mtsTaskManagerProxy::TaskManagerServerPtr*>(__addr);
    assert(p);
    *p = ::mtsTaskManagerProxy::TaskManagerServerPtr::dynamicCast(v);
    if(v && !*p)
    {
        IceInternal::Ex::throwUOE(::mtsTaskManagerProxy::TaskManagerServer::ice_staticId(), v->ice_id());
    }
}

bool
mtsTaskManagerProxy::operator==(const ::mtsTaskManagerProxy::TaskManagerServer& l, const ::mtsTaskManagerProxy::TaskManagerServer& r)
{
    return static_cast<const ::Ice::Object&>(l) == static_cast<const ::Ice::Object&>(r);
}

bool
mtsTaskManagerProxy::operator<(const ::mtsTaskManagerProxy::TaskManagerServer& l, const ::mtsTaskManagerProxy::TaskManagerServer& r)
{
    return static_cast<const ::Ice::Object&>(l) < static_cast<const ::Ice::Object&>(r);
}
