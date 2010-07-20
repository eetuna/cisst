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
    cmnClassRegister::SetLoDForAllClasses(CMN_LOG_LOD_VERY_VERBOSE);

    svlInitialize();

    // create our task
    const double PeriodDisplay = 50 * cmn_ms; // in milliseconds
    mtsTaskManager * taskManager = mtsTaskManager::GetInstance();

    // create video stream
    svlStreamManager stream;
    stream.SetName("Stream");
    // taskManager->AddComponent(&stream);

    svlFilterSourceVideoFile stream_source(1);
    stream_source.SetName("StreamSource");
    taskManager->AddComponent(&stream_source);

    svlFilterTest stream_testfilter;
    stream_testfilter.SetName("TestFilter");
    taskManager->AddComponent(&stream_testfilter);

    svlFilterImageWindow stream_window;
    stream_window.SetName("Window");
    taskManager->AddComponent(&stream_window);


    displayTask * displayTaskObject = new displayTask("TestComponent", PeriodDisplay);
    displayTaskObject->Configure();

    // add the task and the filter to the task manager
    taskManager->AddComponent(displayTaskObject);

    // taskManager->Connect("StreamSource", "input", "Stream", "output");
    stream.SetSourceFilter(&stream_source);
    taskManager->Connect("TestFilter", "input", "StreamSource", "output");
    taskManager->Connect("Window", "input", "TestFilter", "output");

    // connect the task with thex component, task.RequiresInterface -> component.ProvidesInterface
    taskManager->Connect("TestComponent", "SourceConfig", "StreamSource", "Settings");
    taskManager->Connect("TestComponent", "FilterParams", "TestFilter", "Parameters");

    // create the tasks, i.e. find the commands
    taskManager->CreateAll();
    // start the periodic Run
    taskManager->StartAll();

    osaSleep(2 * cmn_s);
    // start video pipeline
    stream.Start();


    // connect filters and start streaming
    // stream.SetSourceFilter(&stream_source);
    // stream_source.GetOutput()->Connect(stream_testfilter.GetInput());
    // stream_testfilter.GetOutput()->Connect(stream_window.GetInput());

    int ch = 0;
    while (ch != 'q') {
        ch = cmnGetChar();
    }

    // cleanup
    taskManager->KillAll();
    taskManager->Cleanup();

    // stream.Release();

    return 0;
}

