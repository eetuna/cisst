/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Min Yang Jung, Anton Deguet
  Created on: 2009-03-20

  (C) Copyright 2009-2011 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#ifndef _mtsCollectorState_h
#define _mtsCollectorState_h

#include <cisstMultiTask/mtsTaskPeriodic.h>
#include <cisstMultiTask/mtsCollectorBase.h>
#include <cisstMultiTask/mtsCommandVoid.h>
#include <cisstMultiTask/mtsStateTable.h>

#include <string>

// Always include last
#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  This class provides a way to collect data in the state table without
  loss and make a log file. The type of a log file can be plain text
  (ascii), csv, or binary.  A state table of which data is to be
  collected can be specified in the constructor.  This is intended for
  future usage where a task can have more than two state tables.
*/
class CISST_EXPORT mtsCollectorState : public mtsCollectorBase
{
    friend class mtsCollectorStateTest;
    friend class mtsStateTable;

    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_ALLOW_DEFAULT);

    /*! Structure and container definition to manage the list of signals to be
        collected by this collector. */
    typedef struct {
        std::string Name;
        unsigned int ID;
    } SignalElement;

    typedef std::vector<SignalElement> RegisteredSignalElementType;
    RegisteredSignalElementType RegisteredSignalElements;

    /*! Offset for a sampling stride. */
    size_t OffsetForNextRead;

    /*! Data index which should be read at the next time. */
    ptrdiff_t LastReadIndex;

    /*! Local copy to reduce the number of reference in Collect() method. */
    size_t TableHistoryLength;

    /*! A stride value for data collector to skip several records. */
    size_t SamplingInterval;

    /*! Pointers to the target component and the target state table. */
    mtsComponent * TargetComponent;
    mtsStateTable * TargetStateTable;

    /*! Thread-related methods */
    void Run(void);

    void Startup(void);

    /*! Initialization */
    void Initialize(void);

    // documented in base class
    virtual std::string GetDefaultOutputName(void);

    mtsFunctionWrite StateTableStartCollection;
    mtsFunctionWrite StateTableStopCollection;

    /*! Check if the signal specified by a user has been already registered.
        This is to avoid duplicate signal registration. */
    bool IsRegisteredSignal(const std::string & signalName) const;

    /*! Add a signal element. Called internally by mtsCollectorState::AddSignal(). */
    bool AddSignalElement(const std::string & signalName, const unsigned int signalID);

    /*! Fetch state table data */
    bool FetchStateTableData(const mtsStateTable * table,
                             const size_t startIdx,
                             const size_t endIdx);

    /*! Print out the signal names which are being collected. */
    void PrintHeader(const CollectorFileFormat & fileFormat);

    /*! Mark the end of the header. Called in case of binary log file. */
    void MarkHeaderEnd(std::ostream & logFile);

    /*! Check if the given buffer contains the header mark. */
    static bool IsHeaderEndMark(const char * buffer);

    /*! When this function is called (called by the data thread as a form of an event),
        bulk-fetch is performed and data is dumped to a log fie. */
    void BatchReadyHandler(const mtsStateTable::IndexRange & range);

    /*! Event handler for collection started.  The original event is
      generated by the state table and this handler (not a queued
      event) simply generate an event for the provided interface. */
    void CollectionStartedHandler(void);

    /*! Event handler for collection stopped.  The original event is
      generated by the state table and this handler (not a queued
      event) simply generate an event for the provided interface. */
    void CollectionStoppedHandler(const mtsUInt & count);

    /*! Event handler for progress.  The original event is
      generated by the state table and this handler (not a queued
      event) simply generate an event for the provided interface. */
    void ProgressHandler(const mtsUInt & count);

    /*! Fetch bulk data from StateTable. */
    void BatchCollect(const mtsStateTable::IndexRange & range);

public:
    /*! Constructor using the component name and table name. */
    mtsCollectorState(const std::string & collectorName);

    /*! Constructor using a component pointer and table name. */
    mtsCollectorState(const std::string & targetComponentName,
                      const std::string & targetStateTableName,
                      const CollectorFileFormat fileFormat);

    ~mtsCollectorState(void);

    /*! Defines which table to collect data from.  This is defined by
        the component name and the table name.  If the table name is not
        provided, the collector will use the default component's state
        table. */
    bool SetStateTable(const std::string & componentName,
                       const std::string & stateTableName = "");

    /*! Add the signal specified to a list of registered signals. */
    bool AddSignal(const std::string & signalName = "");

    /*! Set a sampling interval so that data collector can skip
      several values.  This is useful when a frequency of the component is
      somewhat high and you don't want to collect ALL data from it. */
    void SetSamplingInterval(const unsigned int samplingInterval) {
        SamplingInterval = (samplingInterval > 0 ? samplingInterval : 1);
    }

    /*! Connect.  Once the state collector has been configured,
      i.e. the methods SetStateTable and SetOutput have been use,
      the collector should be added to the manager and then the
      Connect method should be called. */
    bool Connect(void);

    /*! Disconnect.  Attempt to disconnect from the observed
      component. */
    bool Disconnect(void);

    /*! Convert a binary log file into a plain text one. */
    static bool ConvertBinaryToText(const std::string sourceBinaryLogFileName,
                                    const std::string targetPlainTextLogFileName,
                                    const char delimiter = ',');

    /*! Methods defined as virtual in base class to control stop/start
      collection with delay.  For the state table collection, these
      methods are mainly pass through, i.e. they call the
      corresponding commands from the state table component. */
    //@{
    void StartCollection(const mtsDouble & delayInSeconds);
    void StopCollection(const mtsDouble & delayInSeconds);
    //@}
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsCollectorState)

#endif // _mtsCollectorState_h
