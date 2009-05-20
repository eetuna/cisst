/*
  $Id: mtsCollectorState.cpp 2009-03-02 mjung5

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

#include <fstream>

CMN_IMPLEMENT_SERVICES(mtsCollectorState)

//-------------------------------------------------------
//	Constructor, Destructor, and Initializer
//-------------------------------------------------------
mtsCollectorState::mtsCollectorState(const std::string & targetTaskName,
                                     const mtsCollectorBase::CollectorLogFormat collectorLogFormat,
                                     const std::string & targetStateTableName)
    : mtsCollectorBase("mtsCollectorState", collectorLogFormat),
      LastReadIndex(-1), TableHistoryLength(0), SamplingInterval(1), LastToc(0),
      WaitingForTrigger(true), CollectAllSignal(false), FirstRunningFlag(true),
      TargetTaskName(targetTaskName), TargetStateTableName(targetStateTableName),
      TargetTask(NULL), TargetStateTable(NULL)
#ifdef COLLECTOR_OVERHEAD_MEASUREMENT
      , ElapsedTimeForProcessing(0.0)
#endif
{
    Initialize();
}

mtsCollectorState::~mtsCollectorState()
{
    if (DataCollectionTriggerResetCommand) {
        delete DataCollectionTriggerResetCommand;
    }
}

void mtsCollectorState::Initialize()
{
    // Check if there is the specified task and the specified state table.
    mtsTaskManager * TaskManager = mtsTaskManager::GetInstance();
    TargetTask = TaskManager->GetTask(TargetTaskName);
    if (!TargetTask) {
        cmnThrow(std::runtime_error("mtsCollectorState::Initialize(): No such task exists."));
    }

    TargetStateTable = TargetTask->GetStateTable(TargetStateTableName);
    if (!TargetStateTable) {
        cmnThrow(std::runtime_error("mtsCollectorState::Initialize(): No such state table exists."));
    }

    CMN_ASSERT(TargetTask);
    CMN_ASSERT(TargetStateTable);

    // Bind a command and an event.
    // Command (Collector -> Target task) : Create a void command to enable the state table's data collection trigger.
    SetDataCollectionTriggerResetCommand();

    // Event (Target task -> Collector) : Create an event handler to wake up this thread.
    TargetStateTable->SetDataCollectionEventHandler(this);

    // Determine a ratio to generate a data collection event.
    //
    // TODO: an adaptive scaling feature according to 'sizeStateTable' might be useful.
    //
    TargetStateTable->SetDataCollectionEventTriggeringRatio(0.3);
}

void mtsCollectorState::SetDataCollectionTriggerResetCommand()
{
    CMN_ASSERT(TargetStateTable);

    DataCollectionTriggerResetCommand = new mtsCommandVoidMethod<mtsStateTable>(
        &mtsStateTable::ResetDataCollectionTrigger, TargetStateTable, TargetStateTable->GetStateTableName());
    
    CMN_ASSERT(DataCollectionTriggerResetCommand);
}

void mtsCollectorState::DataCollectionEventHandler()
{
    CMN_LOG(5) << "DCEvent has arrived." << std::endl;

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
    CMN_ASSERT(taskManager);

    // Check if a user wants to collect all signals
    CollectAllSignal = (signalName.length() == 0);

    if (!CollectAllSignal) {
        // Prevent duplicate signal registration
        //
        //  TODO: (Re)Move FindSignal() method to the mtsCollector class.
        //
        //if (FindSignal(taskName, signalName)) {
        if (IsRegisteredSignal(signalName)) {
            CMN_LOG_CLASS(5) << "Already registered signal: " << signalName << std::endl;

            throw mtsCollectorState::mtsCollectorBaseException(
                "Already collecting signal: " + signalName);
        }

        // Check if the specified signal does exist in the state table.
        CMN_ASSERT(TargetTask);
        int StateVectorID = TargetStateTable->GetStateVectorID(signalName); // 0: Toc, 1: Tic, 2: Period, >=3: user
        if (StateVectorID == -1) {  // 0: Toc, 1: Tic, 2: Period, >3: user
            CMN_LOG_CLASS(5) << "Cannot find: " << signalName << std::endl;

            throw mtsCollectorState::mtsCollectorBaseException(
                "Cannot find: task name = " + signalName);
        }

        // Add a signal
        AddSignalElement(signalName, StateVectorID);
    } else {        
        // Add all signals in the state table
        CMN_ASSERT(TargetTask);
        for (unsigned int i = 0; i < TargetStateTable->StateVectorDataNames.size(); ++i) {
            AddSignalElement(TargetStateTable->StateVectorDataNames[i], i);
        }

    }

    // To reduce reference counter in the mtsCollectorState::Collect() method.
    TableHistoryLength = TargetStateTable->HistoryLength;

    return true;
}

const bool mtsCollectorState::IsRegisteredSignal(const std::string & signalName) const
{
    RegisteredSignalElementType::const_iterator it = RegisteredSignalElements.begin();
    for (; it != RegisteredSignalElements.end(); ++it) {
        if (it->Name == signalName) return true;
    }

    return false;
}

void mtsCollectorState::AddSignalElement(const std::string & signalName, const unsigned int signalID)
{
    CMN_ASSERT(!IsRegisteredSignal(signalName));

    SignalElement element;
    element.Name = signalName;
    element.ID = signalID;

    RegisteredSignalElements.push_back(element);

    CMN_LOG_CLASS(5) << "Signal added: " << signalName << std::endl;
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
   
    //
    //  TODO:
    //  1. output format control (CSV, tab, binary/txt, etc.)
    //  2. using an iterator instead of loop? (Peter's idea from GetHistory())
    //    
    if (StartIndex < EndIndex) {
        // normal case
        if (FetchStateTableData(TargetStateTable, StartIndex, EndIndex)) {
            LastReadIndex = EndIndex;
        }
    } else if (StartIndex == EndIndex) {
        // No data to be read. Wait for the next run
    } else {
        // Wrap-around case
        // First part: from the last read index to the bottom of the array
        if (FetchStateTableData(TargetStateTable, StartIndex, TableHistoryLength - 1)) {
            // Second part: from the top of the array to the IndexReader
            const unsigned int indexReader = TargetStateTable->IndexReader;
            if (FetchStateTableData(TargetStateTable, 0, indexReader)) {
                LastReadIndex = indexReader;
            }
        }
    }
}

void mtsCollectorState::PrintHeader(void)
{
    std::string currentDateTime; osaGetDateTimeString(currentDateTime);
    
    LogFileName = "DataCollection_" + TargetTask->GetName() + "_" + 
        TargetStateTable->GetStateTableName() + "_" + currentDateTime + ".txt";

    //
    // TODO: Currently, only COLLECTOR_LOG_FORMAT_PLAIN_TEXT is considered.
    //
    std::ofstream LogFile;
    LogFile.open(LogFileName.c_str(), std::ios::out);
    {
        // Print out some information on the state table.
        LogFile << "-------------------------------------------------------------------------------" << std::endl;
        LogFile << "Task Name          : " << TargetTask->GetName() << std::endl;
        LogFile << "Date & Time        : " << currentDateTime << std::endl;
        LogFile << "Total signal count : " << RegisteredSignalElements.size() << std::endl;
        LogFile << "-------------------------------------------------------------------------------" << std::endl;
        LogFile << std::endl;

        LogFile << "Ticks ";

        RegisteredSignalElementType::const_iterator it = RegisteredSignalElements.begin();
        for (; it != RegisteredSignalElements.end(); ++it) {
            LogFile << TargetStateTable->StateVectorDataNames[it->ID] << " ";
        }

        LogFile << std::endl;
    }
    LogFile.close();

    FirstRunningFlag = false;
}

bool mtsCollectorState::FetchStateTableData(const mtsStateTable * table, 
                                           const unsigned int startIdx, 
                                           const unsigned int endIdx)
{
    /*
    int idx = 0, signalIndex = 0;
    std::ostringstream line;
    std::vector<std::string> vecLines;
    
    SignalMap::const_iterator itr;

    // Performance measurement  
#ifdef COLLECTOR_OVERHEAD_MEASUREMENT
    StopWatch.Reset();
    StopWatch.Start();
#endif

    for (itr = taskMap.begin()->second->begin(); 
        itr != taskMap.begin()->second->end(); ++itr) 
    {        
        idx = 0;
        for (unsigned int i = startIdx; i <= endIdx; ++i) {
            line.str("");
            line << table->Ticks[i] << " ";
            vecLines.push_back(line.str());
        }

        if (!CollectAllSignal) {
            signalIndex = table->GetStateVectorID(itr->first);
            if (signalIndex == -1) continue;
            
            idx = 0;
            for (unsigned int i = startIdx; i <= endIdx; ++i) {
                line.str("");
                line << (*table->StateVector[signalIndex])[i] << " ";
                vecLines[idx++].append(line.str());
            }
        } else {            
            for (unsigned int j = 0; j < table->StateVector.size(); ++j) {
                idx = 0;
                for (unsigned int i = startIdx; i <= endIdx; ++i) {
                    line.str("");
                    line << (*table->StateVector[j])[i] << " ";
                    vecLines[idx++].append(line.str());
                }
            }
        }
    }

#ifdef COLLECTOR_OVERHEAD_MEASUREMENT
    StopWatch.Stop();
    ElapsedTimeForProcessing += StopWatch.GetElapsedTime();
#endif
    
    idx = 0;
    std::vector<std::string>::const_iterator it = vecLines.begin();

#ifdef COLLECTOR_OVERHEAD_MEASUREMENT
    StopWatch.Reset();
    StopWatch.Start();
#endif
    for (; it != vecLines.end(); ++it) {
        LogFile << vecLines[idx++];
        LogFile << std::endl;
    }
#ifdef COLLECTOR_OVERHEAD_MEASUREMENT
    StopWatch.Stop();
    ElapsedTimeForFileIO += StopWatch.GetElapsedTime();
#endif

    return true;
    */

    // Performance measurement
#ifdef COLLECTOR_OVERHEAD_MEASUREMENT
    StopWatch.Reset();
    StopWatch.Start();
#endif

    std::ofstream LogFile;
    LogFile.open(LogFileName.c_str(), std::ios::app);
    {
        int signalIndex = 0;
        unsigned int counter = 0;
        for (unsigned int i = startIdx; i <= endIdx; ++i) {
            // Skip some records, if specified.
            if (SamplingInterval > 1) {
                if (++counter == SamplingInterval) {
                    counter = 0;
                } else {
                    continue;
                }
            }

            LogFile << TargetStateTable->Ticks[i] << " ";
            {
                for (unsigned int j = 0; j < RegisteredSignalElements.size(); ++j) {
                    LogFile << (*table->StateVector[RegisteredSignalElements[j].ID])[i] << " ";
                }
            }
            LogFile << std::endl;
        }
    }
    LogFile.close();

#ifdef COLLECTOR_OVERHEAD_MEASUREMENT
    StopWatch.Stop();
    ElapsedTimeForProcessing += StopWatch.GetElapsedTime();
#endif

    return true;
}
