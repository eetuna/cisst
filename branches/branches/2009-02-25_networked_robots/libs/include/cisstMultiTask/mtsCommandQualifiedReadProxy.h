/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCommandQualifiedReadProxy.h 75 2009-02-24 16:47:20Z adeguet1 $

  Author(s):  Min Yang Jung
  Created on: 2009-04-29

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
  \brief Defines a command with one argument 
*/

#ifndef _mtsCommandQualifiedReadProxy_h
#define _mtsCommandQualifiedReadProxy_h

#include <cisstMultiTask/mtsCommandReadOrWriteBase.h>

/*!
  \ingroup cisstMultiTask
*/
class mtsCommandQualifiedReadProxy: public mtsCommandQualifiedReadBase {
public:
    typedef const cmnDouble Argument1Type;
    typedef cmnDouble Argument2Type;
    typedef mtsCommandQualifiedReadBase BaseType;

protected:
    mtsDeviceInterfaceProxyClient * ProvidedInterfaceProxy;

    /*! ID assigned by the server as a pointer to the actual command in server's
        memory space. */
    const int CommandSID;

public:
    mtsCommandQualifiedReadProxy(const int commandSID, 
                                 mtsDeviceInterfaceProxyClient * providedInterfaceProxy) 
        : CommandSID(commandSID), ProvidedInterfaceProxy(providedInterfaceProxy), BaseType()
    {}

    mtsCommandQualifiedReadProxy(const int commandSID,
                                 mtsDeviceInterfaceProxyClient * providedInterfaceProxy,
                                 const std::string & name)
                         //ArgumentPointerType argumentProtoType) :
        : CommandSID(commandSID), ProvidedInterfaceProxy(providedInterfaceProxy), BaseType(name)
        //, ArgumentPointerPrototype(argumentProtoType)
    {}

    /*! The destructor. Does nothing */
    virtual ~mtsCommandQualifiedReadProxy() {}

    /*! The execute method. */
    virtual mtsCommandBase::ReturnType Execute(const cmnGenericObject & argument1,
                                               cmnGenericObject & argument2) 
    {
        Argument1Type * data1 = dynamic_cast<Argument1Type *>(&argument1);
        if (data1 == NULL)
            return mtsCommandBase::BAD_INPUT;
        Argument2Type * data2 = dynamic_cast<Argument2Type *>(&argument2);
        if (data2 == NULL)
            return mtsCommandBase::BAD_INPUT;
        
        static int cnt = 0;
        std::cout << "mtsCommandQualifiedReadProxy called (" << ++cnt << "): " << *data1 << std::endl;

        ProvidedInterfaceProxy->InvokeExecuteCommandQualifiedRead(
            CommandSID, *data1, *data2);

        return mtsCommandBase::DEV_OK;
    }

    /*! For debugging. Generate a human QualifiedReadable output for the
      command object */
    void ToStream(std::ostream & outputStream) const {
        // TODO
    }

    /*! Return a pointer on the argument prototype */
    const cmnGenericObject * GetArgumentPrototype(void) const {
        //
        // TODO: FIX THIS
        //
        return reinterpret_cast<const cmnGenericObject *>(0x12345678);
    }
};

#endif // _mtsCommandQualifiedReadProxy_h
