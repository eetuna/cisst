/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCollectorState.cpp 188 2009-03-20 17:07:32Z mjung5 $

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

#include <cisstMultiTask/mtsCollectorState.h>
#include <cisstMultiTask/mtsTaskManager.h>
#include <cisstOSAbstraction/osaGetTime.h>
#include <cisstCommon/cmnThrow.h>

#include <iostream>
#include <fstream>

#define END_OF_HEADER_SIZE 5
#define END_OF_HEADER      {0,1,2,1,0}

static char EndOfHeader[END_OF_HEADER_SIZE] = END_OF_HEADER;

CMN_IMPLEMENT_SERVICES(mtsCollectorState)

//-------------------------------------------------------
//	Constructor, Destructor, and Initializer
//-------------------------------------------------------
mtsCollectorState::mtsCollectorState(const std::string & targetTaskName,
                                     const mtsCollectorBase::CollectorLogFormat collectorLogFormat,
                                     const std::string & targetStateTableName)
    : mtsCollectorBase("mtsCollectorState", collectorLogFormat),
      TargetTaskName(targetTaskName), TargetTask(NULL),
      TargetStateTableName(targetStateTableName), TargetStateTable(NULL),
      Serializer(NULL)
#ifdef COLLECTOR_OVERHEAD_MEASUREMENT
      , ElapsedTimeForProcessing(0.0)
#endif
{
    // Check if there is the specified task and the specified state table.    
    TargetTask = taskManager->GetTask(TargetTaskName);
    if (!TargetTask) {
        cmnThrow(std::runtime_error("mtsCollectorState::Initialize(): No such task exists."));
    }

    Initialize();
}

mtsCollectorState::mtsCollectorState(mtsTask * targetTask,
                                     const mtsCollectorBase::CollectorLogFormat collectorLogFormat,
                                     const std::string & targetStateTableName)
    : mtsCollectorBase("mtsCollectorState", collectorLogFormat),
      TargetTaskName(TargetTask->GetName()), TargetTask(targetTask),
      TargetStateTableName(targetStateTableName), TargetStateTable(NULL)
#ifdef COLLECTOR_OVERHEAD_MEASUREMENT
      , ElapsedTimeForProcessing(0.0)
#endif
{
    Initialize();
}

mtsCollectorState::~mtsCollectorState()
{
    LogFile.close();

#define DELETE_OBJECT(_object) if (_object) delete _object;
    DELETE_OBJECT(DataCollectionTriggerResetCommand);
    DELETE_OBJECT(Serializer);
#undef DELETE_OBJECT
}

void mtsCollectorState::Initialize()
{
    LastReadIndex = -1;
    TableHistoryLength = 0;
    SamplingInterval = 1;
    LastToc = 0;
    OffsetForNextRead = 0;

    WaitingForTrigger = true;
    CollectAllSignal = false;
    FirstRunningFlag = true;

    TargetStateTable = TargetTask->GetStateTable(TargetStateTableName);
    if (!TargetStateTable) {
        cmnThrow(std::runtime_error("mtsCollectorState::Initialize(): No such state table exists."));
    }

    // Bind a command and an event.
    // Command (Collector -> Target task) : Create a void command to enable the state table's data collection trigger.
    SetDataCollectionTriggerResetCommand();

    // Event (Target task -> Collector) : Create an event handler to wake up this thread.
    TargetStateTable->SetDataCollectionEventHandler(this);

    // Determine a ratio to generate a data collection event.
    //
    // TODO: to determine the size of a state table adaptively considering an adaptive scaling feature according to 'sizeStateTable' might be useful.
    //
    TargetStateTable->SetDataCollectionEventTriggeringRatio(0.3);

    // Initialize serializer
    if (LogFormat == COLLECTOR_LOG_FORMAT_BINARY) {
        Serializer = new cmnSerializer(StringStreamBufferForSerialization);        
    }
}

void mtsCollectorState::SetDataCollectionTriggerResetCommand()
{
    DataCollectionTriggerResetCommand = new mtsCommandVoidMethod<mtsStateTable>(
        &mtsStateTable::ResetDataCollectionTrigger, TargetStateTable, TargetStateTable->GetStateTableName());
}

void mtsCollectorState::DataCollectionEventHandler()
{
    WaitingForTrigger = false;

    Wakeup();
}

//-------------------------------------------------------
//	Thread Management
//-------------------------------------------------------
void mtsCollectorState::Startup(void)
{
    DataCollectionTriggerResetCommand->Execute();
}

void mtsCollectorState::Run(void)
{
    mtsCollectorBase::Run();

    if (!IsRunnable) return;

    DataCollectionTriggerResetCommand->Execute();

    WaitingForTrigger = true;
    while (WaitingForTrigger) {
        WaitForWakeup();
    }
    
    // Collect data
    Collect();
}

//-------------------------------------------------------
//	Signal Management
//-------------------------------------------------------
bool mtsCollectorState::AddSignal(const std::string & signalName, 
                                  const std::string & format)
{	
    // Check if a user wants to collect all signals
    CollectAllSignal = (signalName.length() == 0);

    if (!CollectAllSignal) {
        // Check if the specified signal does exist in the state table.
        int StateVectorID = TargetStateTable->GetStateVectorID(signalName); // 0: Toc, 1: Tic, 2: Period, >=3: user
        if (StateVectorID == -1) {  // 0: Toc, 1: Tic, 2: Period, >3: user
            CMN_LOG_CLASS(5) << "Cannot find: " << signalName << std::endl;

            //throw mtsCollectorState::mtsCollectorBaseException(
            //    "Cannot find: signal name = " + signalName);
            return false;
        }

        // Add a signal
        if (!AddSignalElement(signalName, StateVectorID)) {
            CMN_LOG_CLASS(5) << "Already registered signal: " << signalName << std::endl;

            //throw mtsCollectorState::mtsCollectorBaseException(
            //    "Already collecting signal: " + signalName);
            return false;
        }
    } else {        
        // Add all signals in the state table
        for (unsigned int i = 0; i < TargetStateTable->StateVectorDataNames.size(); ++i) {
            if (!AddSignalElement(TargetStateTable->StateVectorDataNames[i], i)) {
                CMN_LOG_CLASS(5) << "Already registered signal: " << TargetStateTable->StateVectorDataNames[i] << std::endl;
                return false;
            }
        }

    }

    // To reduce reference counter in the mtsCollectorState::Collect() method.
    TableHistoryLength = TargetStateTable->HistoryLength;

    return true;
}

bool mtsCollectorState::IsRegisteredSignal(const std::string & signalName) const
{
    RegisteredSignalElementType::const_iterator it = RegisteredSignalElements.begin();
    for (; it != RegisteredSignalElements.end(); ++it) {
        if (it->Name == signalName) return true;
    }

    return false;
}

bool mtsCollectorState::AddSignalElement(const std::string & signalName, const unsigned int signalID)
{
    // Prevent duplicate signal registration
    if (IsRegisteredSignal(signalName)) {
        return false;
    }

    SignalElement element;
    element.Name = signalName;
    element.ID = signalID;

    RegisteredSignalElements.push_back(element);

    CMN_LOG_CLASS(5) << "Signal added: " << signalName << std::endl;

    return true;
}

//-------------------------------------------------------
//	Collecting Data
//-------------------------------------------------------
void mtsCollectorState::Collect(void)
{
    if (RegisteredSignalElements.size() == 0) return;

    // If this method is called for the first time, print out some information.
    if (FirstRunningFlag) {
        PrintHeader();
    }

    const unsigned int StartIndex = (LastReadIndex + 1) % TableHistoryLength;
    {    
        // state data validity check
        if (TargetStateTable->Ticks[(StartIndex + 1) % TableHistoryLength] - 
            TargetStateTable->Ticks[StartIndex] != 1) 
        {
            return;
        }
    }
    const unsigned int EndIndex = TargetStateTable->IndexReader;
   
    if (StartIndex < EndIndex) {
        // normal case
        if (FetchStateTableData(TargetStateTable, StartIndex, EndIndex)) {
            LastReadIndex = (EndIndex + (OffsetForNextRead - 1)) % TableHistoryLength;
        }
    } else if (StartIndex == EndIndex) {
        // No data to be read. Wait for the next run
    } else {
        // Wrap-around case
        // first part: from the last read index to the bottom of the array
        if (FetchStateTableData(TargetStateTable, StartIndex, TableHistoryLength - 1)) {
            // second part: from the top of the array to the IndexReader
            const unsigned int indexReader = TargetStateTable->IndexReader;
            if (FetchStateTableData(TargetStateTable, 0, indexReader)) {
                LastReadIndex = (indexReader + (OffsetForNextRead - 1)) % TableHistoryLength;
            }
        }
    }
}

void mtsCollectorState::PrintHeader(void)
{
    std::string currentDateTime; osaGetDateTimeString(currentDateTime);
    
    LogFileName = "DataCollection_" + TargetTask->GetName() + "_" + 
        TargetStateTable->GetStateTableName() + "_" + currentDateTime + ".txt";

    // Create a log file and don't close. File closing is handled by destructor.
    LogFile.open(LogFileName.c_str(), std::ios::out);

    // Print out some information on the state table.

    // All lines in the header should be preceded by '#' which represents 
    // the line contains header information rather than collected data.
    LogFile << "#------------------------------------------------------------------------------" << std::endl;
    LogFile << "# Task Name          : " << TargetTask->GetName() << std::endl;
    LogFile << "# Date & Time        : " << currentDateTime << std::endl;
    LogFile << "# Total signal count : " << RegisteredSignalElements.size() << std::endl;
    LogFile << "# Data format        : ";
    if (LogFormat == COLLECTOR_LOG_FORMAT_PLAIN_TEXT) {
        LogFile << "Text";
    } else if (LogFormat == COLLECTOR_LOG_FORMAT_CSV) {
        LogFile << "Text (CSV)";
    } else {
        LogFile << "Binary";
    }
    LogFile << std::endl;
    LogFile << "#------------------------------------------------------------------------------" << std::endl;
    LogFile << "#" << std::endl;

    LogFile << "# Ticks ";

    RegisteredSignalElementType::const_iterator it = RegisteredSignalElements.begin();
    for (; it != RegisteredSignalElements.end(); ++it) {
        LogFile << TargetStateTable->StateVectorDataNames[it->ID] << " ";
    }

    LogFile << std::endl;
    LogFile << "#-------------------------------------------------------------------------------" << std::endl;

    // In case of using binary format
    if (LogFormat == COLLECTOR_LOG_FORMAT_BINARY) {
        // Mark the end of the header.
        MarkHeaderEnd(LogFile);


        // Remember the number of registered signals.
        cmnULong cmnULongTotalSignalCount;
        cmnULongTotalSignalCount.Data = RegisteredSignalElements.size();
        StringStreamBufferForSerialization.str("");
        Serializer->Serialize(cmnULongTotalSignalCount);
        LogFile << StringStreamBufferForSerialization.str();
    }

    LogFile.flush();

    FirstRunningFlag = false;
}

void mtsCollectorState::MarkHeaderEnd(std::ofstream & logFile)
{
    for (int i = 0; i < END_OF_HEADER_SIZE; ++i) {
        logFile << EndOfHeader[i];
    }

    logFile << std::endl;    
}

bool mtsCollectorState::IsHeaderEndMark(const char * buf) const
{
    for (int i = 0; i < END_OF_HEADER_SIZE; ++i) {
        if (buf[i] != EndOfHeader[i]) return false;
    }
    
    return true;
}

bool mtsCollectorState::FetchStateTableData(const mtsStateTable * table, 
                                           const unsigned int startIndex, 
                                           const unsigned int endIndex)
{
    // Performance measurement
#ifdef COLLECTOR_OVERHEAD_MEASUREMENT
    StopWatch.Reset();
    StopWatch.Start();
#endif
    
    if (LogFormat == COLLECTOR_LOG_FORMAT_BINARY) {
        cmnDouble cmnDoubleTick;
        unsigned int i;
        for (i = startIndex; i <= endIndex; i += SamplingInterval) {
            StringStreamBufferForSerialization.str("");
            cmnDoubleTick.Data = TargetStateTable->Ticks[i];
            Serializer->Serialize(cmnDoubleTick);
            LogFile << StringStreamBufferForSerialization.str();

            for (unsigned int j = 0; j < RegisteredSignalElements.size(); ++j) {
                StringStreamBufferForSerialization.str("");
                Serializer->Serialize((*table->StateVector[RegisteredSignalElements[j].ID])[i]);
                LogFile << StringStreamBufferForSerialization.str();
            }
        }

        OffsetForNextRead = (i - endIndex == 0 ? SamplingInterval : i - endIndex);
        LogFile.flush();
    } else {
        unsigned int i;
        for (i = startIndex; i <= endIndex; i += SamplingInterval) {
            LogFile << TargetStateTable->Ticks[i] << " ";
            for (unsigned int j = 0; j < RegisteredSignalElements.size(); ++j) {
                LogFile << (*table->StateVector[RegisteredSignalElements[j].ID])[i] << " ";
            }
            LogFile << std::endl;
        }
        
        OffsetForNextRead = (i - endIndex == 0 ? SamplingInterval : i - endIndex);
        LogFile.flush();
    }

#ifdef COLLECTOR_OVERHEAD_MEASUREMENT
    StopWatch.Stop();
    ElapsedTimeForProcessing += StopWatch.GetElapsedTime();
#endif

    return true;
}

bool mtsCollectorState::ConvertBinaryLogFileIntoPlainText(
    const std::string sourceBinaryLogFileName, const std::string targetPlainTextLogFileName)
{
    // Try to open a binary log file (source).
    std::ifstream inFile(sourceBinaryLogFileName.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    if (!inFile.is_open()) {
        CMN_LOG_CLASS(5) << "Unable to open binary log file: " << sourceBinaryLogFileName << std::endl;
        return false;
    }

    // Get the total size of the log file in bytes.
    std::ifstream::pos_type inFileTotalSize = inFile.tellg();
    inFile.seekg(0, std::ios::beg);

    // Read the first character in a line; if it is '#', skip the line.
    char line[256];
    while(true) {
        inFile.getline(line, 256);

        // All header lines begins with '#'.
        if (line[0] == '#') continue;

        break;
    }

    // Check the end of header.
    if (!IsHeaderEndMark(line)) {
        CMN_LOG_CLASS(5) << "Corrupted header." << std::endl;
        inFile.close();
        return false;
    }

    // Prepare output log file with plain text format.
    std::ofstream outFile(targetPlainTextLogFileName.c_str(), std::ios::out);
    if (!outFile.is_open()) {
        CMN_LOG_CLASS(5) << "Unable to create text log file: " << targetPlainTextLogFileName << std::endl;
        inFile.close();
        return false;
    }

    cmnDeSerializer DeSerializer(inFile);

    // Deserialize to get the total number of recorded signals.
    cmnGenericObject * element = DeSerializer.DeSerialize();
    cmnULong * totalSignalCountObject = dynamic_cast<cmnULong *>(element);
    if (!totalSignalCountObject) {
        CMN_LOG_CLASS(5) << "Corrupted header." << std::endl;
        inFile.close();
        outFile.close();
        return false;
    }

    unsigned int totalSignalCount = totalSignalCountObject->Data;

    int columnCount = 0;
    std::ifstream::pos_type currentPos = inFile.tellg();

    while (currentPos < inFileTotalSize) {
        element = DeSerializer.DeSerialize();
        if (!element) {
            CMN_LOG_CLASS(5) << "Unexpected termination: " << 
                currentPos << " / " << inFileTotalSize << std::endl;
            break;
        }

        element->ToStream(outFile);
        if (++columnCount == totalSignalCount + 1) { // +1 due to 'Ticks' field.
            outFile << std::endl;
            columnCount = 0;
        } else {
            outFile << " ";
        }

        currentPos = inFile.tellg();
    }

    CMN_LOG_CLASS(5) << "Conversion completed: " << targetPlainTextLogFileName  << std::endl;

    outFile.close();
    inFile.close();
    
    return true;
}
