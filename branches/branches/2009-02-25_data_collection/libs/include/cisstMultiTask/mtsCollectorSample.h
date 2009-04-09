/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCollectorSample.h 2009-03-20 mjung5

  Author(s):  Min Yang Jung
  Created on: 2009-03-24

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

#ifndef _mtsCollectorSample_h
#define _mtsCollectorSample_h

#include <cisstCommon/cmnUnits.h>
#include <cisstOSAbstraction/osaStopwatch.h>
#include <cisstMultiTask/mtsTaskPeriodic.h>
#include <cisstMultiTask/mtsExport.h>
#include <cisstMultiTask/mtsTaskManager.h>
#include <cisstMultiTask/mtsCollectorBase.h>

#include <string>
#include <map>
#include <stdexcept>

// If the following line is commented out, C2491 error is generated.
#include <cisstMultiTask/mtsExport.h>

/*!
\ingroup cisstMultiTask

This class provides a way to collect data from state table in real-time.
Collected data can be either saved as a log file or displayed in GUI like an oscilloscope.
*/
class CISST_EXPORT mtsCollectorSample : public mtsCollectorBase
{
    friend class mtsCollectorSampleTest;

    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

private:
    unsigned int NextReadIndex;

public:
    mtsCollectorSample(const std::string & collectorName,
                     double period = 100 * cmn_ms);
    virtual ~mtsCollectorSample(void);

    /*! Add a signal to the list. Currently 'format' argument is meaningless. */
    bool AddSignal(const std::string & taskName, 
                   const std::string & signalName, 
                   const std::string & format = "");

    /*! Called periodically */
    //void Collect(void);

    void Startup(void);
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsCollectorSample)

#endif // _mtsCollectorSample_h

