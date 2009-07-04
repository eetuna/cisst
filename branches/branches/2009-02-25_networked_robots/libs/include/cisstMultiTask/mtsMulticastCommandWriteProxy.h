/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsMulticastCommandWrite.h 475 2009-06-17 17:30:16Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-06-24

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


/*!
  \file
  \brief Defines a command with one argument sent to multiple interfaces
*/


#ifndef _mtsMulticastCommandWriteProxy_h
#define _mtsMulticastCommandWriteProxy_h

#include <cisstMultiTask/mtsMulticastCommandWriteBase.h>
#include <vector>

#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  mtsMulticastCommandWriteProxy is a proxy for mtsMulticastCommandWrite.  
 */
class mtsMulticastCommandWriteProxy : public mtsMulticastCommandWriteBase
{
public:
    typedef mtsMulticastCommandWriteBase BaseType;

protected:
    /*! Device interface proxy objects which execute a write command at 
        peer's memory space across networks. */
    mtsDeviceInterfaceProxyClient * ProvidedInterfaceProxy;
    mtsDeviceInterfaceProxyServer * RequiredInterfaceProxy;

    /*! CommandId is set as a pointer to a mtsFunctionWrite at peer's memory
        space which binds to an actual write command. */
    CommandProxyIdType CommandId;

public:
    /*! The constructor with a name. */
    mtsMulticastCommandWriteProxy(const CommandProxyIdType commandId,
                                  mtsDeviceInterfaceProxyClient * providedInterfaceProxy,
                                  const std::string & name) :
        BaseType(name),
        CommandId(commandId),
        ProvidedInterfaceProxy(providedInterfaceProxy),
        RequiredInterfaceProxy(NULL)
    {}

    mtsMulticastCommandWriteProxy(const CommandProxyIdType commandId,
                                  mtsDeviceInterfaceProxyServer * requiredInterfaceProxy,
                                  const std::string & name) :
        BaseType(name),
        CommandId(commandId),
        ProvidedInterfaceProxy(NULL),
        RequiredInterfaceProxy(requiredInterfaceProxy)
    {}
    
    /*! Default destructor. Does nothing. */
    ~mtsMulticastCommandWriteProxy() 
    {}
    
    /*! Execute all the commands in the composite. */
    virtual mtsCommandBase::ReturnType Execute(const mtsGenericObject & argument) {
        //
        // GOHOME
        //
        unsigned int index;        
        if (ProvidedInterfaceProxy) {
            for (index = 0; index < Commands.size(); ++index) {
                // TODO: implement this
                //
                //Commands[index]->Execute(*data);
                //ProvidedInterfaceProxy->SendExecuteCommandWriteSerialized(CommandId, argument);
            }
        } else {
            for (index = 0; index < Commands.size(); ++index) {
                //
                // TODO: implement this
                //
                //Commands[index]->Execute(*data);
                //RequiredInterfaceProxy->SendExecuteCommandWriteSerialized(CommandId, argument);
            }
        }

        return mtsCommandBase::DEV_OK;
    }

    /*! Return a pointer on the argument prototype.  Uses the first
      command added to find the argument prototype.  If no command is
      available, return 0 (null pointer) */
    const mtsGenericObject * GetArgumentPrototype(void) const {
        //
        // TODO: FIX THIS!!! UGLY
        //
        static mtsDouble TEMP_VAR_FIX_ME;
        //return reinterpret_cast<const mtsGenericObject *>(0x1122);
        return &TEMP_VAR_FIX_ME;
    }
};

#endif // _mtsMulticastCommandWriteProxy_h

///////////////////////////////////////////////////////////////////////////////////////

#ifndef _mtsMulticastCommandWriteProxy_h
#define _mtsMulticastCommandWriteProxy_h


#include <cisstMultiTask/mtsMulticastCommandWriteBase.h>
#include <vector>

// Always include last
#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  This class contains a vector of two or more command objects.
  The primary use of this class is to send events to all observers.
 */
class mtsMulticastCommandWriteProxy: public mtsMulticastCommandWriteBase
{
public:
    typedef mtsMulticastCommandWriteBase BaseType;
    //typedef _argumentType ArgumentType;

protected:
    mtsDeviceInterfaceProxyClient * ProvidedInterfaceProxy;

    /*! ID assigned by the server as a pointer to the actual command in server's
        memory space. */
    const int CommandSID;

    /*! Argument prototype */
    //ArgumentType ArgumentPrototype;
    
public:
    /*! Default constructor. Does nothing. */
    //mtsMulticastCommandWriteProxy(const std::string & name, const ArgumentType & argumentPrototype):
    //    BaseType(name),
    //    ArgumentPrototype(argumentPrototype)
    //{}
    //mtsMulticastCommandWriteProxy(
    //    const int commandSID, mtsDeviceInterfaceProxyClient * providedInterfaceProxy)
    //    : CommandSID(commandSID), ProvidedInterfaceProxy(providedInterfaceProxy), BaseType()
    //{}

    mtsMulticastCommandWriteProxy(
        const int commandSID, mtsDeviceInterfaceProxyClient * providedInterfaceProxy, const std::string & name)
        : CommandSID(commandSID), ProvidedInterfaceProxy(providedInterfaceProxy), BaseType(name)
    {}

    /*! Default destructor. Does nothing. */
    ~mtsMulticastCommandWriteProxy() {}
    
    /*! Execute all the commands in the composite. */
    virtual mtsCommandBase::ReturnType Execute(const cmnGenericObject & argument) {
        //// cast argument first
        //const ArgumentType * data = dynamic_cast< const ArgumentType * >(&argument);
        //if (data == NULL)
        //    return mtsCommandBase::BAD_INPUT;
        //// if cast succeeded call using actual type
        //unsigned int index;
        //for (index = 0; index < Commands.size(); index++) {
        //    Commands[index]->Execute(*data);
        //}
        //return mtsCommandBase::DEV_OK;
        static int cnt = 0;
        std::cout << "mtsMulticastCommandWriteProxy called (" << ++cnt << "): " << Name 
            << ", " << argument << std::endl;

        unsigned int index;
        for (index = 0; index < Commands.size(); index++) {
            //Commands[index]->Execute(*data);
            std::stringstream ss;
            cmnSerializer serialization(ss);
            serialization.Serialize(argument);
            std::string s = ss.str();

            //ProvidedInterfaceProxy->SendExecuteCommandWriteSerialized(CommandSID, s);
        }

        return mtsCommandBase::DEV_OK;
    }

    /*! Return a pointer on the argument prototype.  Uses the first
      command added to find the argument prototype.  If no command is
      available, return 0 (null pointer) */
    const cmnGenericObject * GetArgumentPrototype(void) const {
        return reinterpret_cast<const cmnGenericObject *>(0);// &ArgumentPrototype;
    }
};


#endif // _mtsMulticastCommandWriteProxy_h

