/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s): Anton Deguet
  Created on: 2010-09-30

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


/*!
  \file
  \brief Defines a base class for a command with one argument
*/

#ifndef _mtsCallableWriteBase_h
#define _mtsCallableWriteBase_h

#include <cisstMultiTask/mtsGenericObject.h>

// Always include last
#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  A base class of callable object with an execute method that takes
  one argument.  This pure virtual class is derived to support either
  global functions or methods with signature "void method(const
  mtsGenericObject &)" (non const) */
class mtsCallableWriteBase
{
public:
    enum { } ReturnType;

    /*! The constructor. Does nothing */
    mtsCallableWriteBase(void) {}

    /*! The destructor. Does nothing */
    virtual ~mtsCallableWriteBase() {}

    /*! The execute method. Abstract method to be implemented by
      derived classes to run the actual operation on the receiver
      \param obj The data passed to the operation method */
    virtual mtsExecutionResult Execute(const mtsGenericObject & argument) = 0;

    /*! Human readable description */
    virtual void ToStream(std::ostream & outputStream) const = 0;
};

/*! Stream out operator for all classes derived from
  mtsCallableWriteBase.  This operator uses the ToStream method so
  that the output can be different for each derived class. */
inline std::ostream & operator << (std::ostream & outputStream,
                                   const mtsCallableWriteBase & callable) {
    callable.ToStream(outputStream);
    return outputStream;
}

#endif // _mtsCallableWriteBase_h

