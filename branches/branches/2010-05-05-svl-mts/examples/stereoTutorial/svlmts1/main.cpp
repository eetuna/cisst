/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
 $Id: $
 
 Author(s):  Balazs Vagvolgyi
 Created on: 2010
 
 (C) Copyright 2006-2010 Johns Hopkins University (JHU), All Rights
 Reserved.
 
 --- begin cisst license - do not edit ---
 
 This software is provided "as is" under an open source license, with
 no warranty.  The complete license can be found in license.txt and
 http://www.cisst.org/cisst/license.txt.
 
 --- end cisst license ---
 
 */

#include <cisstCommon.h>
#include <cisstOSAbstraction.h>
#include <cisstMultiTask.h>
#include <cisstStereoVision.h>

#include "testFilter.h"
#include "displayTask.h"

using namespace std;

int main(void)
{
    // log configuration
    cmnLogger::SetLoD(CMN_LOG_LOD_VERY_VERBOSE);
    cmnLogger::GetMultiplexer()->AddChannel(cout, CMN_LOG_LOD_VERY_VERBOSE);
    // add a log per thread
    osaThreadedLogFile threadedLog("svlExMultitask1-");
    cmnLogger::GetMultiplexer()->AddChannel(threadedLog, CMN_LOG_LOD_VERY_VERBOSE);
    // specify a higher, more verbose log level for these classes
    cmnClassRegister::SetLoD("taskFilter", CMN_LOG_LOD_VERY_VERBOSE);
    cmnClassRegister::SetLoD("displayTask", CMN_LOG_LOD_VERY_VERBOSE);
    cmnClassRegister::SetLoD("mtsInterfaceProvided", CMN_LOG_LOD_VERY_VERBOSE);
    cmnClassRegister::SetLoD("mtsTaskManager", CMN_LOG_LOD_VERY_VERBOSE);

    svlInitialize();

    // create video stream
    svlStreamManager stream;

    svlFilterSourceVideoFile stream_source;
    stream_source.SetName("SRC");

    svlFilterTest stream_testfilter;
    stream_testfilter.SetName("FILT");

    svlFilterImageWindow stream_window;

    // create our task
    const double PeriodDisplay = 50 * cmn_ms; // in milliseconds
    mtsTaskManager * taskManager = mtsTaskManager::GetInstance();
    displayTask * displayTaskObject = new displayTask("DISP", PeriodDisplay);
    displayTaskObject->Configure();

    // add the task and the filter to the task manager
    taskManager->AddTask(displayTaskObject);
    taskManager->AddComponent(&stream_source);
    taskManager->AddComponent(&stream_testfilter);

//    taskManager->Connect("SRC", "Output", "FILT", "Input");
//    taskManager->Connect("FILT", "Output", "WIN", "Input");

    // connect the task with the component, task.RequiresInterface -> component.ProvidesInterface
    taskManager->Connect("DISP", "SourceConfig", "SRC", "Settings");
    taskManager->Connect("DISP", "FilterParams", "FILT", "Parameters");

    // create the tasks, i.e. find the commands
    taskManager->CreateAll();
    // start the periodic Run
    taskManager->StartAll();

    // connect filters and start streaming
    stream.SetSourceFilter(&stream_source);
    stream_source.GetOutput()->Connect(stream_testfilter.GetInput());
    stream_testfilter.GetOutput()->Connect(stream_window.GetInput());
    
    // start video pipeline
    stream.Start();

    int ch = 0;
    while (ch != 'q') {
        ch = cmnGetChar();
    }

    // cleanup
    taskManager->KillAll();
    taskManager->Cleanup();

    stream.Release();

    return 0;
}

