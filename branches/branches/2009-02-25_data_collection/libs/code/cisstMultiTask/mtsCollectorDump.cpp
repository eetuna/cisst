/*
  $Id: mtsCollectorDump.cpp 2009-03-02 mjung5

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

#include <cisstMultiTask/mtsCollectorDump.h>
#include <cisstOSAbstraction/osaGetTime.h>

CMN_IMPLEMENT_SERVICES(mtsCollectorDump)

//-------------------------------------------------------
//	Constructor, Destructor, and Initializer
//-------------------------------------------------------
mtsCollectorDump::mtsCollectorDump(const std::string & collectorName) 
    : mtsCollectorBase(collectorName), WaitingForTrigger(true),
      ElapsedTimeForProcessing(0.0), ElapsedTimeForFileIO(0.0)
{
    Initialize();    
}

mtsCollectorDump::~mtsCollectorDump()
{
    LogFile.close();
}

void mtsCollectorDump::Initialize()
{
    LastReadIndex = -1;
    TableHistoryLength = 0;
    //SamplingStrideCounter = 0;
    LastToc = 0.0;
    
    IsThisFirstRunning = true;
    CollectAllSignal = false;

    // Create a required interface to listen an event trigerred by another task(s).
    mtsRequiredInterface * requiredInterface = 
        AddRequiredInterface(GetDataCollectorRequiredInterfaceName());

    if (requiredInterface) {
        // Create a void command to enable a data thread's trigger.
        requiredInterface->AddFunction(
            GetDataCollectorResetEventName(), DataCollectionEventReset);

        // Create an event handler to wake up this thread.
        requiredInterface->AddEventHandlerVoid(
            &mtsCollectorDump::DataCollectionEventHandler, this,
            GetDataCollectorEventName(), false);
    }
}

void mtsCollectorDump::DataCollectionEventHandler()
{
    //CMN_LOG(5) << "DCEvent has arrived." << std::endl;

    WaitingForTrigger = false;

    Wakeup();
}

//-------------------------------------------------------
//	Thread Management
//-------------------------------------------------------
void mtsCollectorDump::Startup(void)
{
    WaitingForTrigger = false;

    DataCollectionEventReset();
}

void mtsCollectorDump::Run(void)
{
    mtsCollectorBase::Run();

    if (!IsRunnable) return;

    DataCollectionEventReset();

    WaitingForTrigger = true;
    while (WaitingForTrigger) {
        WaitForWakeup();
    }
    
    // Collect data
    Collect();

    ProcessQueuedEvents();
}

//-------------------------------------------------------
//	Signal Management
//-------------------------------------------------------
bool mtsCollectorDump::AddSignal(const std::string & taskName, 
                                 const std::string & signalName, 
                                 const std::string & format)
{	
    CMN_ASSERT(taskManager);

    // Ensure that the specified task is under the control of the task manager.
    mtsTask * task = taskManager->GetTask(taskName);
    if (task == NULL) {
        CMN_LOG_CLASS(5) << "Unknown task: " << taskName << std::endl;
        return false;
    }

    // Only one task can be registered.
    if (taskMap.size() == 1) {
        if (taskMap.begin()->first != taskName) {
            CMN_LOG_CLASS(5) << "Cannot track more than one task." << std::endl;
            return false;
        }
    }

    if (signalName == "") {
        CollectAllSignal = true;
    }

    if (!CollectAllSignal) {
        // Prevent duplicate signal registration
        if (FindSignal(taskName, signalName)) {
            CMN_LOG_CLASS(5) << "Already registered signal: " << taskName 
                << ", " << signalName << std::endl;

            throw mtsCollectorDump::mtsCollectorBaseException(
                "Already collecting signal: " + taskName + ", " + signalName);
        }

        // Check if the specified signal does exist in the state table.
        if (task->GetStateVectorID(signalName) == -1) {  // 0: Toc, 1: Tic, 2: Period, >3: user
            CMN_LOG_CLASS(5) << "Cannot find: task name = " << taskName
                << ", signal name = " << signalName << std::endl;

            throw mtsCollectorDump::mtsCollectorBaseException(
                "Cannot find: task name = " + taskName + ", signal name = " + signalName);
        }
    }

    // Add a signal
    SignalMapElement element;
    element.Task = task;
    element.History = NULL; // This type of collector does not use this.

    TaskMap::iterator itr = taskMap.find(taskName);
    if (itr == taskMap.end()) {    // If the task is new
        // Create a new instance of SignalMap
        SignalMap * signalMap = new SignalMap();
        CMN_ASSERT(signalMap);
        
        signalMap->insert(make_pair(signalName, element));
        taskMap.insert(make_pair(taskName, signalMap));

        TableHistoryLength = task->StateTable.HistoryLength;
    } else {    // If the task has one or more signals being collected
        itr->second->insert(make_pair(signalName, element));
    }

    if (!CollectAllSignal) {
        CMN_LOG_CLASS(5) << "Signal added: task name = " << taskName
                << ", signal name = " << signalName << std::endl;
    } else {
        // Actually no task has been added; however, this is handled by FetchStateTableData()
        // later with the help of CollectAllSignal flag.
        CMN_LOG_CLASS(5) << "All signals added: task name = " << taskName << std::endl;
    }

    return true;
}

//-------------------------------------------------------
//	Collecting Data
//-------------------------------------------------------
void mtsCollectorDump::Collect(void)
{
    // If this is the first time calling, print out brief information on the target task. 
    if (IsThisFirstRunning) {
        PrintHeader();

        // Active data collectors in all registered tasks.
        TaskMap::const_iterator it = taskMap.begin();
        for (; it != taskMap.end(); ++it) {
            it->second->begin()->second.Task->ActivateDataCollection(true);
        }
    }

    if (taskMap.empty()) return;

    mtsTask * task = taskMap.begin()->second->begin()->second.Task;
    CMN_ASSERT(task);
    mtsStateTable * table = &task->StateTable;
    CMN_ASSERT(table);

    const unsigned int StartIndex = (LastReadIndex + 1) % TableHistoryLength;
    {    
        // state data validity check
        if (table->Ticks[(StartIndex + 1) % TableHistoryLength] - 
            table->Ticks[StartIndex] != 1) {
                return;
        }
    }
    const unsigned int EndIndex = table->IndexReader;
   
    //
    //  TODO:
    //  1. output format control (CSV, tab, binary/txt, etc.)
    //  2. using an iterator instead of loop? (Peter's idea from GetHistory())
    //    
    if (StartIndex < EndIndex) {
        // normal case
        if (FetchStateTableData(table, StartIndex, EndIndex)) {
            LastReadIndex = EndIndex;
        }
    } else if (StartIndex == EndIndex) {
        // No data to be read. Wait for the next run
    } else {
        // Wrap-around case
        // First part: from the last read index to the bottom of the array
        if (FetchStateTableData(table, StartIndex, table->HistoryLength - 1)) {
            // Second part: from the top of the array to the IndexReader
            const unsigned int indexReader = table->IndexReader;
            if (FetchStateTableData(table, 0, indexReader)) {
                LastReadIndex = indexReader;
            }
        }
    }
}

bool mtsCollectorDump::FetchStateTableData(const mtsStateTable * table, 
                                           const unsigned int startIdx, 
                                           const unsigned int endIdx)
{
    int idx = 0, signalIndex = 0;
    std::ostringstream line;
    std::vector<std::string> vecLines;
    
    SignalMap::const_iterator itr;

    // Performance measurement
    osaStopwatch stopWatch;
    stopWatch.Reset();
    stopWatch.Start();

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

    stopWatch.Stop();
    ElapsedTimeForProcessing += stopWatch.GetElapsedTime();
    
    idx = 0;
    std::vector<std::string>::const_iterator it = vecLines.begin();

    stopWatch.Reset();
    stopWatch.Start();
    for (; it != vecLines.end(); ++it) {
        LogFile << vecLines[idx++];
        LogFile << std::endl;
    }
    stopWatch.Stop();
    ElapsedTimeForFileIO += stopWatch.GetElapsedTime();

    return true;
}

void mtsCollectorDump::PrintHeader(void)
{
    std::string currentDateTime; osaGetDateTimeString(currentDateTime);
    std::string taskName = taskMap.begin()->first;
    std::string fileName = "DataCollection_" + taskName + "_" + currentDateTime;

    LogFile.open(fileName.c_str(), std::ios::out);// | std::ios::app);

    // Collect signal names
    mtsTask * task = taskMap.begin()->second->begin()->second.Task;
    CMN_ASSERT(task);

    mtsStateTable * table = &task->StateTable;
    CMN_ASSERT(table);

    // Print out some information on the task being collected.
    LogFile << "-------------------------------------------------------------------------------" << std::endl;
    LogFile << "Task Name          : " << taskName << std::endl;
    LogFile << "Date & Time        : " << currentDateTime << std::endl;
    LogFile << "Total signal count : " << table->NumberStateData << std::endl;
    LogFile << "-------------------------------------------------------------------------------" << std::endl;
    LogFile << std::endl;

    if (CollectAllSignal) {
        LogFile << "Ticks ";
        for (unsigned int i = 0; i < table->StateVector.size(); ++i) {
            LogFile << table->StateVectorDataNames[i] << " ";
        }
    } else {
        SignalMap::const_iterator itr = taskMap.begin()->second->begin();
        for ( ; itr != taskMap.begin()->second->end(); ++itr) {
            if (itr == taskMap.begin()->second->begin()) {
                LogFile << "Ticks " << itr->first;                
            } else {
                LogFile << " " << itr->first;
            }
        }
    }
    LogFile << std::endl;

    IsThisFirstRunning = false;
}