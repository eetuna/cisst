/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCollectorDump.h 2009-03-20 mjung5

  Author(s):  Min Yang Jung
  Created on: 2009-03-20

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

#ifndef _mtsCollectorDump_h
#define _mtsCollectorDump_h

#include <cisstMultiTask/mtsTaskPeriodic.h>
#include <cisstMultiTask/mtsCollectorBase.h>
#include <cisstMultiTask/mtsHistory.h>

#include <string>
#include <iostream>
#include <fstream>

#include <cisstMultiTask/mtsExport.h>

/*!
\ingroup cisstMultiTask

As a child class of mtsCollectorBase, this class provides the functionality of collecting
signals from a single task and dumping them into a log file in a specified format. (csv, 
txt, bin, etc.) 

TODO:

1) Serialization/Deseriliazation for binary output
2) Support for different output file format (csv, bin, etc.)

*/
class CISST_EXPORT mtsCollectorDump : public mtsCollectorBase
{
    friend class mtsCollectorDumpTest;

    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

private:
    /*! Data index which should be read at the next time */
    int LastReadIndex;

    /*! Local copy to reduce the number of reference in Collect() method. */
    unsigned int TableHistoryLength;

    /*! Local counter to support sampling-by-stride mode. */
    unsigned int SamplingStrideCounter;

    /*! Local counter to support sampling-by-time mode. */
    double LastToc;

    /*! Flag for PrintHeader() method. */
    bool IsThisFirstRunning;

    /*! If this flag is set, all signals are collected. */
    bool CollectAllSignal;

    /*! Output file stream */
    std::ofstream LogFile;

    /*! Initialization */
    void Initialize(void);

    /*! Fetch state table data */
    bool (mtsCollectorDump::*FetchStateTableData)(const mtsStateTable * table, 
                                const unsigned int startIdx, 
                                const unsigned int endIdx);

    bool FetchStateTableDataAll       (const mtsStateTable * table, 
                                       const unsigned int startIdx, 
                                       const unsigned int endIdx);
    bool FetchStateTableDataByTime    (const mtsStateTable * table, 
                                       const unsigned int startIdx, 
                                       const unsigned int endIdx);
    bool FetchStateTableDataByStride  (const mtsStateTable * table, 
                                       const unsigned int startIdx, 
                                       const unsigned int endIdx);

    /*! Print out the signal names which are being collected. */
    void PrintHeader(void);

    /*! Convert string to double. */
    inline double ConvertToDouble(const std::string& s)
    {
        return 0.0;
    }

public:
    /*! There are two ways of specifying the periodicity of mtsCollectorDump class.
        One is to explicitly specify it and the other one is to pass a pointer to the task 
        that you want to collect data from. In case of the latter, a period is automatically set.
    */
    mtsCollectorDump(const std::string & collectorName, const double periodicityInSeconds);
    mtsCollectorDump(const std::string & collectorName, const mtsTaskPeriodic * targetTask);
    ~mtsCollectorDump(void);

    /*! Add a signal to the list. Currently 'format' argument is meaningless. */
    bool AddSignal(const std::string & taskName,
                   const std::string & signalName,
                   const std::string & format = "");

    /*! Called periodically */
    void Collect(void);
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsCollectorDump)

#endif // _mtsCollectorDump_h

