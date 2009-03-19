/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCollectorBuffer.h 2009-03-02 mjung5

  Author(s):  Min Yang Jung
  Created on: 2009-02-25

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
  \brief A data collection tool
*/

#ifndef _mtsCollectorBuffer_h
#define _mtsCollectorBuffer_h

#include <cisstCommon/cmnGenericObject.h>
#include <cisstCommon/cmnClassRegister.h>
//#include <cisstMultiTask/mtsTaskPeriodic.h>
//#include <cisstMultiTask/mtsTaskManager.h>
//#include <cisstOSAbstraction/osaStopwatch.h>
//
//#include <string>
//#include <map>
//#include <stdexcept>

// If the following line is commented out, C2491 error is generated.
#include <cisstMultiTask/mtsExport.h>

// Enable this macro for unit-test purposes only
#define	_OPEN_PRIVATE_FOR_UNIT_TEST_

/*!
\ingroup cisstMultiTask

This class implements an efficient and fast buffer dedicated to mtsCollector class.
*/
class CISST_EXPORT mtsCollectorBuffer : public cmnGenericObject
{
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION, 5);

//public:
//    class mtsCollectorBufferException : public std::runtime_error {
//    private:
//        std::string ExceptionDescription;    // exception descriptor
//
//    public:
//        mtsCollectorBufferException(std::string exceptionDescription) 
//            : ExceptionDescription(exceptionDescription),
//            std::runtime_error("mtsCollectorBufferException") {}
//
//        const std::string GetExceptionDescription(void) const { return ExceptionDescription; }    
//    };

#ifndef _OPEN_PRIVATE_FOR_UNIT_TEST_
protected:
#else
public:
#endif

public:
    mtsCollectorBuffer(void);
    ~mtsCollectorBuffer(void);

#ifndef _OPEN_PRIVATE_FOR_UNIT_TEST_
protected:
#else
public:
#endif

};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsCollectorBuffer)

#endif // _mtsCollectorBuffer_h
