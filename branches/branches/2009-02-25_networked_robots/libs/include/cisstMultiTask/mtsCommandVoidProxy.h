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

/*!
  \ingroup cisstMultiTask
  
  TODO: add class description here
*/
class mtsCommandVoidProxy: public mtsCommandVoidBase
{
public:
    typedef mtsCommandVoidBase BaseType;
    
    /*! The constructor. Does nothing */
    mtsCommandVoidProxy(void): BaseType() {}
    
    /*! Constructor with a name. */
    mtsCommandVoidProxy(const std::string & name): BaseType(name) {}
    
    /*! The destructor. Does nothing */
    ~mtsCommandVoidProxy() {}

    /*! The execute method. */
    BaseType::ReturnType Execute() {
        static int cnt = 0;
        std::cout << "mtsCommandVoidProxy called (" << ++cnt << "): " << Name << std::endl;
        return BaseType::DEV_OK;
    }

    void ToStream(std::ostream & outputStream) const {
        outputStream << "mtsCommandVoidProxy: " << Name << std::endl;
    }

    /*! Returns number of arguments (parameters) expected by Execute
      method.  Overloaded for mtsCommandVoidProxy to return 0. */
    unsigned int NumberOfArguments(void) const {
        return 0;
    }
};

#endif // _mtsCommandVoidProxy_h

