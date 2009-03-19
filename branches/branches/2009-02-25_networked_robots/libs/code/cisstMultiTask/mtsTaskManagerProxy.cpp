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

static const ::std::string __mtsTaskManagerProxy__Printer__printString_name = "printString";

ICE_DECLSPEC_EXPORT ::Ice::Object* IceInternal::upCast(::mtsTaskManagerProxy::Printer* p) { return p; }
ICE_DECLSPEC_EXPORT ::IceProxy::Ice::Object* IceInternal::upCast(::IceProxy::mtsTaskManagerProxy::Printer* p) { return p; }

void
mtsTaskManagerProxy::__read(::IceInternal::BasicStream* __is, ::mtsTaskManagerProxy::PrinterPrx& v)
{
    ::Ice::ObjectPrx proxy;
    __is->read(proxy);
    if(!proxy)
    {
        v = 0;
    }
    else
    {
        v = new ::IceProxy::mtsTaskManagerProxy::Printer;
        v->__copyFrom(proxy);
    }
}
#ifdef __SUNPRO_CC
class ICE_DECLSPEC_EXPORT IceProxy::mtsTaskManagerProxy::Printer;
#endif

void
IceProxy::mtsTaskManagerProxy::Printer::printString(const ::std::string& s, const ::Ice::Context* __ctx)
{
    int __cnt = 0;
    while(true)
    {
        ::IceInternal::Handle< ::IceDelegate::Ice::Object> __delBase;
        try
        {
            __delBase = __getDelegate(false);
            ::IceDelegate::mtsTaskManagerProxy::Printer* __del = dynamic_cast< ::IceDelegate::mtsTaskManagerProxy::Printer*>(__delBase.get());
            __del->printString(s, __ctx);
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
IceProxy::mtsTaskManagerProxy::Printer::ice_staticId()
{
    return ::mtsTaskManagerProxy::Printer::ice_staticId();
}

::IceInternal::Handle< ::IceDelegateM::Ice::Object>
IceProxy::mtsTaskManagerProxy::Printer::__createDelegateM()
{
    return ::IceInternal::Handle< ::IceDelegateM::Ice::Object>(new ::IceDelegateM::mtsTaskManagerProxy::Printer);
}

::IceInternal::Handle< ::IceDelegateD::Ice::Object>
IceProxy::mtsTaskManagerProxy::Printer::__createDelegateD()
{
    return ::IceInternal::Handle< ::IceDelegateD::Ice::Object>(new ::IceDelegateD::mtsTaskManagerProxy::Printer);
}

::IceProxy::Ice::Object*
IceProxy::mtsTaskManagerProxy::Printer::__newInstance() const
{
    return new Printer;
}

void
IceDelegateM::mtsTaskManagerProxy::Printer::printString(const ::std::string& s, const ::Ice::Context* __context)
{
    ::IceInternal::Outgoing __og(__handler.get(), __mtsTaskManagerProxy__Printer__printString_name, ::Ice::Normal, __context);
    try
    {
        ::IceInternal::BasicStream* __os = __og.os();
        __os->write(s);
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
IceDelegateD::mtsTaskManagerProxy::Printer::printString(const ::std::string& s, const ::Ice::Context* __context)
{
    class _DirectI : public ::IceInternal::Direct
    {
    public:

        _DirectI(const ::std::string& s, const ::Ice::Current& __current) : 
            ::IceInternal::Direct(__current),
            _m_s(s)
        {
        }
        
        virtual ::Ice::DispatchStatus
        run(::Ice::Object* object)
        {
            ::mtsTaskManagerProxy::Printer* servant = dynamic_cast< ::mtsTaskManagerProxy::Printer*>(object);
            if(!servant)
            {
                throw ::Ice::OperationNotExistException(__FILE__, __LINE__, _current.id, _current.facet, _current.operation);
            }
            servant->printString(_m_s, _current);
            return ::Ice::DispatchOK;
        }
        
    private:
        
        const ::std::string& _m_s;
    };
    
    ::Ice::Current __current;
    __initCurrent(__current, __mtsTaskManagerProxy__Printer__printString_name, ::Ice::Normal, __context);
    try
    {
        _DirectI __direct(s, __current);
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
mtsTaskManagerProxy::Printer::ice_clone() const
{
    throw ::Ice::CloneNotImplementedException(__FILE__, __LINE__);
    return 0; // to avoid a warning with some compilers
}

static const ::std::string __mtsTaskManagerProxy__Printer_ids[2] =
{
    "::Ice::Object",
    "::mtsTaskManagerProxy::Printer"
};

bool
mtsTaskManagerProxy::Printer::ice_isA(const ::std::string& _s, const ::Ice::Current&) const
{
    return ::std::binary_search(__mtsTaskManagerProxy__Printer_ids, __mtsTaskManagerProxy__Printer_ids + 2, _s);
}

::std::vector< ::std::string>
mtsTaskManagerProxy::Printer::ice_ids(const ::Ice::Current&) const
{
    return ::std::vector< ::std::string>(&__mtsTaskManagerProxy__Printer_ids[0], &__mtsTaskManagerProxy__Printer_ids[2]);
}

const ::std::string&
mtsTaskManagerProxy::Printer::ice_id(const ::Ice::Current&) const
{
    return __mtsTaskManagerProxy__Printer_ids[1];
}

const ::std::string&
mtsTaskManagerProxy::Printer::ice_staticId()
{
    return __mtsTaskManagerProxy__Printer_ids[1];
}

::Ice::DispatchStatus
mtsTaskManagerProxy::Printer::___printString(::IceInternal::Incoming& __inS, const ::Ice::Current& __current)
{
    __checkMode(::Ice::Normal, __current.mode);
    ::IceInternal::BasicStream* __is = __inS.is();
    __is->startReadEncaps();
    ::std::string s;
    __is->read(s);
    __is->endReadEncaps();
    printString(s, __current);
    return ::Ice::DispatchOK;
}

static ::std::string __mtsTaskManagerProxy__Printer_all[] =
{
    "ice_id",
    "ice_ids",
    "ice_isA",
    "ice_ping",
    "printString"
};

::Ice::DispatchStatus
mtsTaskManagerProxy::Printer::__dispatch(::IceInternal::Incoming& in, const ::Ice::Current& current)
{
    ::std::pair< ::std::string*, ::std::string*> r = ::std::equal_range(__mtsTaskManagerProxy__Printer_all, __mtsTaskManagerProxy__Printer_all + 5, current.operation);
    if(r.first == r.second)
    {
        throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
    }

    switch(r.first - __mtsTaskManagerProxy__Printer_all)
    {
        case 0:
        {
            return ___ice_id(in, current);
        }
        case 1:
        {
            return ___ice_ids(in, current);
        }
        case 2:
        {
            return ___ice_isA(in, current);
        }
        case 3:
        {
            return ___ice_ping(in, current);
        }
        case 4:
        {
            return ___printString(in, current);
        }
    }

    assert(false);
    throw ::Ice::OperationNotExistException(__FILE__, __LINE__, current.id, current.facet, current.operation);
}

void
mtsTaskManagerProxy::Printer::__write(::IceInternal::BasicStream* __os) const
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
mtsTaskManagerProxy::Printer::__read(::IceInternal::BasicStream* __is, bool __rid)
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
mtsTaskManagerProxy::Printer::__write(const ::Ice::OutputStreamPtr&) const
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type mtsTaskManagerProxy::Printer was not generated with stream support";
    throw ex;
}

void
mtsTaskManagerProxy::Printer::__read(const ::Ice::InputStreamPtr&, bool)
{
    Ice::MarshalException ex(__FILE__, __LINE__);
    ex.reason = "type mtsTaskManagerProxy::Printer was not generated with stream support";
    throw ex;
}

void ICE_DECLSPEC_EXPORT 
mtsTaskManagerProxy::__patch__PrinterPtr(void* __addr, ::Ice::ObjectPtr& v)
{
    ::mtsTaskManagerProxy::PrinterPtr* p = static_cast< ::mtsTaskManagerProxy::PrinterPtr*>(__addr);
    assert(p);
    *p = ::mtsTaskManagerProxy::PrinterPtr::dynamicCast(v);
    if(v && !*p)
    {
        IceInternal::Ex::throwUOE(::mtsTaskManagerProxy::Printer::ice_staticId(), v->ice_id());
    }
}

bool
mtsTaskManagerProxy::operator==(const ::mtsTaskManagerProxy::Printer& l, const ::mtsTaskManagerProxy::Printer& r)
{
    return static_cast<const ::Ice::Object&>(l) == static_cast<const ::Ice::Object&>(r);
}

bool
mtsTaskManagerProxy::operator<(const ::mtsTaskManagerProxy::Printer& l, const ::mtsTaskManagerProxy::Printer& r)
{
    return static_cast<const ::Ice::Object&>(l) < static_cast<const ::Ice::Object&>(r);
}
