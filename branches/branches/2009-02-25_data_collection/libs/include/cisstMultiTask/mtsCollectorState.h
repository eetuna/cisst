/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCollectorState.h 2009-03-20 mjung5

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

#ifndef _mtsCollectorState_h
#define _mtsCollectorState_h

#include <cisstMultiTask/mtsTaskPeriodic.h>
#include <cisstMultiTask/mtsCollectorBase.h>
#include <cisstMultiTask/mtsHistory.h>
#include <cisstMultiTask/mtsCommandVoid.h>
#include <cisstMultiTask/mtsStateTable.h>

#include <string>

#include <cisstMultiTask/mtsExport.h>

/*!
\ingroup cisstMultiTask

As a child class of mtsCollectorBase, this class provides the functionality of collecting
signals from a single task and dumping them into a log file in a specified format. (csv, 
txt, bin, etc.) 

TODO:

1) Serialization/Deserialiazation for binary output
2) Support for different output file format (csv, bin, etc.)

*/

// Enable this macro to measure the elapsed time for data collection
//#define COLLECTOR_OVERHEAD_MEASUREMENT

class CISST_EXPORT mtsCollectorState : public mtsCollectorBase
{
    friend class mtsCollectorStateTest;
    friend class mtsStateTable;

    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

    /*! Structure and container definition to manage the list of signals to be
        collected by this collector. */
    typedef struct {
        std::string Name;
        unsigned int ID;
    } SignalElement;

    typedef std::vector<SignalElement> RegisteredSignalElementType;
    RegisteredSignalElementType RegisteredSignalElements;

    /*! Offset */
    unsigned int OffsetForNextRead;

    /*! Data index which should be read at the next time */
    int LastReadIndex;

    /*! Local copy to reduce the number of reference in Collect() method. */
    unsigned int TableHistoryLength;

    /*! Local counter to support sampling-by-time mode. */
    double LastToc;

    /*! Flag for PrintHeader() method. */
    bool FirstRunningFlag;

    /*! True if a user want to collect data from all registered signals. */
    bool CollectAllSignal;

    /*! If this is unset, this collector thread wakes up. */
    bool WaitingForTrigger;

    /*! A stride value for data collector to skip several records. */
    unsigned int SamplingInterval;

    /*! Output file name. */
    std::string LogFileName;

    /*! Void command to enable a data thread's trigger. */
    mtsCommandVoidBase * DataCollectionTriggerResetCommand;

    /*! Performance measurement variables */
#ifdef COLLECTOR_OVERHEAD_MEASUREMENT
    double ElapsedTimeForProcessing;
    osaStopwatch StopWatch;
#endif

    /*! The names of target task and the target state table. */
    const std::string TargetTaskName;
    const std::string TargetStateTableName;

    /*! Pointers to the target task and the target state table. */
    mtsTask * TargetTask;
    mtsStateTable * TargetStateTable;

    /*! Thread-related Methods */
    void Run(void);

    void Startup(void);

    /*! Initialization */
    void Initialize(void);

    /*! Check if the signal specified by a user has been already registered. 
        This is to avoid duplicate signal registration. */
    bool IsRegisteredSignal(const std::string & signalName) const;

    /*! Add a signal element. Called internally by mtsCollectorState::AddSignal(). */
    bool AddSignalElement(const std::string & signalName, const unsigned int signalID);

    void SetDataCollectionTriggerResetCommand();

    /*! Fetch state table data */
    bool FetchStateTableData(const mtsStateTable * table, 
                             const unsigned int startIdx, 
                             const unsigned int endIdx);

    /*! Print out the signal names which are being collected. */
    void PrintHeader(void);
    
    /*! When this function is called (called by the data thread as a form of an event),
        bulk-fetch is performed and data is dumped to a log fie. */
    void DataCollectionEventHandler();

    /*! Fetch bulk data from StateTable. */
    void Collect(void);

public:
    mtsCollectorState(const std::string & targetTaskName,
                      const mtsCollectorBase::CollectorLogFormat collectorLogFormat = mtsCollectorBase::COLLECTOR_LOG_FORMAT_PLAIN_TEXT,
                      const std::string & targetStateTableName = STATE_TABLE_DEFAULT_NAME);
    mtsCollectorState(mtsTask * targetTask,
                      const mtsCollectorBase::CollectorLogFormat collectorLogFormat = mtsCollectorBase::COLLECTOR_LOG_FORMAT_PLAIN_TEXT,
                      const std::string & targetStateTableName = STATE_TABLE_DEFAULT_NAME);
    ~mtsCollectorState(void);

    /*! Add the signal specified to a list of registered signals. 
        Currently, 'format' argument is reserved. */
    bool AddSignal(const std::string & signalName = "",
                   const std::string & format = "");

    /*! Set a stride so that data collector can skip several values. */
    void SetSamplingInterval(const unsigned int samplingInterval) {
        SamplingInterval = (samplingInterval > 0 ? samplingInterval : 1);
    }

#ifdef COLLECTOR_OVERHEAD_MEASUREMENT
    inline const double GetElapsedTimeForProcessing() {
        double ret = ElapsedTimeForProcessing;
        ElapsedTimeForProcessing = 0.0;
        return ret;
    }
#endif
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsCollectorState)

#endif // _mtsCollectorState_h
