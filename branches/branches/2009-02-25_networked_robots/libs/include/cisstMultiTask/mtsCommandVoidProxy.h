/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCommandVoidProxy.h 75 2009-02-24 16:47:20Z adeguet1 $

  Author(s):  Min Yang Jung
  Created on: 2009-04-28

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
  \brief Defines a base class for a command with no argument
 */

#ifndef _mtsCommandVoidProxy_h
#define _mtsCommandVoidProxy_h

#include <cisstMultiTask/mtsCommandVoidBase.h>
#include <cisstMultiTask/mtsDeviceInterfaceProxyClient.h>

/*!
  \ingroup cisstMultiTask
  
  TODO: add class description here
*/
class mtsCommandVoidProxy: public mtsCommandVoidBase
{
protected:
    mtsDeviceInterfaceProxyClient * ProvidedInterfaceProxy;

    /*! ID assigned by the server as a pointer to the actual command in server's
        memory space. */
    const int CommandId;

public:
    typedef mtsCommandVoidBase BaseType;
    
    /*! The constructor. Does nothing */
    mtsCommandVoidProxy(const int commandId, 
                        mtsDeviceInterfaceProxyClient * providedInterfaceProxy):
        BaseType(),
        ProvidedInterfaceProxy(providedInterfaceProxy),
        CommandId(commandId)
    {}
    
    /*! Constructor with a name. */
    mtsCommandVoidProxy(const int commandId,
                        mtsDeviceInterfaceProxyClient * providedInterfaceProxy,
                        const std::string & name):
        BaseType(name),
        ProvidedInterfaceProxy(providedInterfaceProxy),
        CommandId(commandId)
    {}
    
    /*! The destructor. Does nothing */
    ~mtsCommandVoidProxy() {}

    /*! The execute method. */
    BaseType::ReturnType Execute() {
        //!!!!!!!!!!
        //ProvidedInterfaceProxy->SendExecuteCommandVoid(CommandId);
        return BaseType::DEV_OK;
    }

    void ToStream(std::ostream & outputStream) const {
        outputStream << "mtsCommandVoidProxy: " << Name << ", " << CommandId << std::endl;
    }

    /*! Returns number of arguments (parameters) expected by Execute().
        Overloaded to return NULL. */
    unsigned int NumberOfArguments(void) const {
        return NULL;
    }
};

#endif // _mtsCommandVoidProxy_h

