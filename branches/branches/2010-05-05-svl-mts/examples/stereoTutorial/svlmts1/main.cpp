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

#include "exampleFilter.h"
#include "exampleComponent.h"

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
    const double PeriodComponent = 50 * cmn_ms; // in milliseconds
    mtsTaskManager * taskManager = mtsTaskManager::GetInstance();

    // create video stream
    svlStreamManager stream;
    stream.SetName("Stream");
    taskManager->AddComponent(&stream);

    svlFilterSourceVideoFile stream_source(1);
    stream_source.SetName("StreamSource");
    taskManager->AddComponent(&stream_source);

    exampleFilter stream_examplefilter;
    stream_examplefilter.SetName("ExampleFilter");
    taskManager->AddComponent(&stream_examplefilter);

    svlFilterImageWindow stream_window;
    stream_window.SetName("Window");
    taskManager->AddComponent(&stream_window);

    exampleComponent * exampleComponentObject = new exampleComponent("ExampleComponent", PeriodComponent);
    exampleComponentObject->Configure();

    // add the task and the filter to the task manager
    taskManager->AddComponent(exampleComponentObject);

    // taskManager->Connect("StreamSource", "input", "Stream", "output");
    stream.SetSourceFilter(&stream_source);
    taskManager->Connect("ExampleFilter", "input", "StreamSource", "output");
    taskManager->Connect("Window", "input", "ExampleFilter", "output");

    // connect the task with thex component, task.RequiresInterface -> component.ProvidesInterface
    taskManager->Connect("ExampleComponent", "StreamControl", "Stream", "Control");
    taskManager->Connect("ExampleComponent", "SourceConfig", "StreamSource", "Settings");
    taskManager->Connect("ExampleComponent", "FilterParams", "ExampleFilter", "Parameters");

    // create the tasks, i.e. find the commands
    taskManager->CreateAll();
    // start the periodic Run
    taskManager->StartAll();

    osaSleep(2.0 * cmn_s);

    int ch = 0;
    while (ch != 'q') {
        ch = cmnGetChar();
    }

    // cleanup
    taskManager->KillAll();
    taskManager->Cleanup();

    return 0;
}

