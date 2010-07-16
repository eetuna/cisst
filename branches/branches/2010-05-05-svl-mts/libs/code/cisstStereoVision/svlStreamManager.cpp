/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$
  
  Author(s):  Balazs Vagvolgyi
  Created on: 2006 

  (C) Copyright 2006-2007 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#include <cisstStereoVision/svlStreamManager.h>
#include <cisstStereoVision/svlTypes.h>
#include <cisstStereoVision/svlSyncPoint.h>
#include <cisstStereoVision/svlFilterBase.h>
#include <cisstStereoVision/svlFilterSourceBase.h>
#include <cisstStereoVision/svlStreamProc.h>

#include <cisstOSAbstraction/osaSleep.h>
#include <cisstOSAbstraction/osaThread.h>
#include <cisstOSAbstraction/osaCriticalSection.h>


/*************************************/
/*** svlStreamManager class **********/
/*************************************/

svlStreamManager::svlStreamManager() :
    ThreadCount(1),
    SyncPoint(0),
    CS(0),
    StreamSource(0),
    Initialized(false),
    Running(false),
    StreamStatus(SVL_STREAM_CREATED)
{
}

svlStreamManager::svlStreamManager(unsigned int threadcount) :
    ThreadCount(std::max(1u, threadcount)),
    SyncPoint(0),
    CS(0),
    StreamSource(0),
    Initialized(false),
    Running(false),
    StreamStatus(SVL_STREAM_CREATED)
{
    // To do: autodetect the number of available processor cores
}

svlStreamManager::~svlStreamManager()
{
    Release();
}

int svlStreamManager::SetSourceFilter(svlFilterSourceBase* source)
{
    if (source == 0) return SVL_FAIL;
    if (source == StreamSource) return SVL_OK;
    if (Initialized) return SVL_ALREADY_INITIALIZED;

    StreamSource = source;

    return SVL_OK;
}

int svlStreamManager::Initialize()
{
    if (Initialized) return SVL_ALREADY_INITIALIZED;

    svlSample *inputsample, *outputsample = 0;
    svlFilterSourceBase *source = StreamSource;
    svlFilterBase *prevfilter, *filter;
    svlFilterBase::_Outputs::iterator iteroutputs;
    svlFilterOutput* output;
    svlFilterInput* input;
    int err;

    if (source == 0) return SVL_NO_SOURCE_IN_LIST;

    // Initialize the stream, starting from the stream source
    err = source->Initialize(outputsample);
    if (err != SVL_OK) {
        Release();
        return err;
    }
    source->Initialized = true;

    // Initialize non-trunk filter outputs
    for (iteroutputs = source->Outputs.begin(); iteroutputs != source->Outputs.end(); iteroutputs ++) {
        if (!iteroutputs->second->IsTrunk() && iteroutputs->second->Stream) {
            err = iteroutputs->second->Stream->Initialize();
            if (err != SVL_OK) {
                Release();
                return err;
            }
        }
    }

    prevfilter = source;

    // Get next filter in the trunk
    output = source->GetOutput();
    filter = 0;
    // Check if trunk output exists
    if (output) {
        input = output->Connection;
        // Check if trunk output is connected to a trunk input
        if (input && input->Trunk) filter = input->Filter;
    }

    // Going downstream filter by filter
    while (filter != 0) {

        // Pass samples downstream
        inputsample = outputsample; outputsample = 0;

        // Check if the previous output is valid input for the next filter
        err = filter->IsDataValid(filter->GetInput()->GetType(), inputsample);
        if (err != SVL_OK) {
            Release();
            return err;
        }
        err = filter->Initialize(inputsample, outputsample);
        if (err != SVL_OK) {
            Release();
            return err;
        }
        filter->Initialized = true;

        // Initialize non-trunk filter outputs
        for (iteroutputs = filter->Outputs.begin(); iteroutputs != filter->Outputs.end(); iteroutputs ++) {
            if (!iteroutputs->second->IsTrunk() && iteroutputs->second->Stream) {
                err = iteroutputs->second->Stream->Initialize();
                if (err != SVL_OK) {
                    Release();
                    return err;
                }
            }
        }

        prevfilter = filter;

        // Get next filter in the trunk
        output = filter->GetOutput();
        filter = 0;
        // Check if trunk output exists
        if (output) {
            input = output->Connection;
            // Check if trunk output is connected to a trunk input
            if (input && input->Trunk) filter = input->Filter;
        }
    }

    Initialized = true;

    StreamStatus = SVL_STREAM_INITIALIZED;

    return SVL_OK;
}

void svlStreamManager::Release()
{
    if (!Initialized) return;

    Stop();

    unsigned int i;
    svlFilterBase::_Outputs::iterator iteroutputs;
    svlFilterOutput* output;
    svlFilterInput* input;

    // There might be a thread object still open (in case of an internal shutdown)
    for (i = 0; i < StreamProcInstance.size(); i ++) {
        if (StreamProcInstance[i]) {
            delete StreamProcInstance[i];
            StreamProcInstance[i] = 0;
        }
    }
    for (i = 0; i < StreamProcThread.size(); i ++) {
        if (StreamProcThread[i]) {
            delete StreamProcThread[i];
            StreamProcThread[i] = 0;
        }
    }

    // Release the stream, starting from the stream source
    svlFilterBase *filter = StreamSource;
    while (filter) {

        // Release filter
        if (filter->Initialized) {
            filter->Release();
            filter->Initialized = false;
        }

        // Release non-trunk filter outputs
        for (iteroutputs = filter->Outputs.begin(); iteroutputs != filter->Outputs.end(); iteroutputs ++) {
            if (!iteroutputs->second->IsTrunk() && iteroutputs->second->Stream) {
                iteroutputs->second->Stream->Release();
            }
        }

        // Get next filter in the trunk
        output = filter->GetOutput();
        filter = 0;
        // Check if trunk output exists
        if (output) {
            input = output->Connection;
            // Check if trunk output is connected to a trunk input
            if (input && input->Trunk) filter = input->Filter;
        }
    }

    Initialized = false;

    StreamStatus = SVL_STREAM_RELEASED;
}

bool svlStreamManager::IsInitialized()
{
    return Initialized;
}

int svlStreamManager::Start()
{
    if (Running) return SVL_ALREADY_RUNNING;

    int err;
    unsigned int i;
    svlFilterBase::_Outputs::iterator iteroutputs;
    svlFilterOutput* output;
    svlFilterInput* input;

    if (!Initialized) {
        // Try to initialize it if it hasn't been done before
        err = Initialize();
        if (err != SVL_OK) return err;
    }

    Running = true;

    // Call OnStart for all filters in the trunk
    svlFilterBase *filter = StreamSource;
    while (filter) {
        filter->Running = true;
        if (filter->OnStart(ThreadCount) != SVL_OK) {
            Stop();
            return SVL_FAIL;
        }

        // Get next filter in the trunk
        output = filter->GetOutput();
        filter = 0;
        // Check if trunk output exists
        if (output) {
            input = output->Connection;
            // Check if trunk output is connected to a trunk input
            if (input && input->Trunk) filter = input->Filter;
        }
    }

    // There might be a thread object still open (in case of an internal shutdown)
    for (i = 0; i < StreamProcInstance.size(); i ++) {
        if (StreamProcInstance[i]) {
            delete StreamProcInstance[i];
            StreamProcInstance[i] = 0;
        }
    }
    for (i = 0; i < StreamProcThread.size(); i ++) {
        if (StreamProcThread[i]) {
            delete StreamProcThread[i];
            StreamProcThread[i] = 0;
        }
    }

    // Allocate new thread control object array
    StreamProcInstance.SetSize(ThreadCount);
    StreamProcThread.SetSize(ThreadCount);

    // Create thread synchronization object
    if (ThreadCount > 1) {
        SyncPoint = new svlSyncPoint;
        SyncPoint->Count(ThreadCount);
        CS = new osaCriticalSection;
    }

    StopThread = false;
    StreamStatus = SVL_STREAM_RUNNING;

    // Initialize media control events
    if (StreamSource->PlayCounter != 0) StreamSource->PauseAtFrameID = -1;
    else StreamSource->PauseAtFrameID = 0;

    for (i = 0; i < ThreadCount; i ++) {
        // Starting multi thread processing
        StreamProcInstance[i] = new svlStreamProc(ThreadCount, i);
        StreamProcThread[i] = new osaThread;
        StreamProcThread[i]->Create<svlStreamProc, svlStreamManager*>(StreamProcInstance[i], &svlStreamProc::Proc, this);
    }

    // Start all filter outputs recursively, if any
    filter = StreamSource;
    while (filter != 0) {

        // Start non-trunk filter outputs
        for (iteroutputs = filter->Outputs.begin(); iteroutputs != filter->Outputs.end(); iteroutputs ++) {
            if (!iteroutputs->second->IsTrunk() && iteroutputs->second->Stream) {
                err = iteroutputs->second->Stream->Start();
                if (err != SVL_OK) {
                    Release();
                    return err;
                }
            }
        }

        // Get next filter in the trunk
        output = filter->GetOutput();
        filter = 0;
        // Check if trunk output exists
        if (output) {
            input = output->Connection;
            // Check if trunk output is connected to a trunk input
            if (input && input->Trunk) filter = input->Filter;
        }
    }

    StreamStatus = SVL_STREAM_RUNNING;

    return SVL_OK;
}

void svlStreamManager::Stop()
{
    if (!Running) return;

    svlFilterBase::_Outputs::iterator iteroutputs;
    svlFilterOutput* output;
    svlFilterInput* input;

    // Stop all filter outputs recursively, if any
    svlFilterBase *filter = StreamSource;
    while (filter != 0) {

        // Stop non-trunk filter outputs
        for (iteroutputs = filter->Outputs.begin(); iteroutputs != filter->Outputs.end(); iteroutputs ++) {
            if (!iteroutputs->second->IsTrunk() && iteroutputs->second->Stream) {
                iteroutputs->second->Stream->Stop();
            }
        }

        // Get next filter in the trunk
        output = filter->GetOutput();
        filter = 0;
        // Check if trunk output exists
        if (output) {
            input = output->Connection;
            // Check if trunk output is connected to a trunk input
            if (input && input->Trunk) filter = input->Filter;
        }
    }

    // Set running flags to false
    filter = StreamSource;
    while (filter) {
        filter->Running = false;

        // Get next filter in the trunk
        output = filter->GetOutput();
        filter = 0;
        // Check if trunk output exists
        if (output) {
            input = output->Connection;
            // Check if trunk output is connected to a trunk input
            if (input && input->Trunk) filter = input->Filter;
        }
    }

    Running = false;
    StopThread = true;

    // Stopping multi thread processing and delete thread objects
    for (unsigned int i = 0; i < ThreadCount; i ++) {
        if (StreamProcThread[i]) {
            StreamProcThread[i]->Wait();
            delete StreamProcThread[i];
            StreamProcThread[i] = 0;
        }
        if (StreamProcInstance[i]) {
            delete StreamProcInstance[i];
            StreamProcInstance[i] = 0;
        }
    }

    // Release thread control arrays and objects
    StreamProcThread.SetSize(0);
    StreamProcInstance.SetSize(0);
    if (SyncPoint) {
        delete SyncPoint;
        SyncPoint = 0;
    }
    if (CS) {
        delete CS;
        CS = 0;
    }

    // Call OnStop for all filters in the trunk
    filter = StreamSource;
    while (filter) {
        filter->OnStop();

        // Get next filter in the trunk
        output = filter->GetOutput();
        filter = 0;
        // Check if trunk output exists
        if (output) {
            input = output->Connection;
            // Check if trunk output is connected to a trunk input
            if (input && input->Trunk) filter = input->Filter;
        }
    }

    StreamStatus = SVL_STREAM_STOPPED;
}

void svlStreamManager::InternalStop(unsigned int callingthreadID)
{
    if (!Running) return;

    svlFilterBase::_Outputs::iterator iteroutputs;
    svlFilterOutput* output;
    svlFilterInput* input;

    // Stop all filter outputs recursively, if any
    svlFilterBase *filter = StreamSource;
    while (filter != 0) {

        // Stop non-trunk filter outputs
        for (iteroutputs = filter->Outputs.begin(); iteroutputs != filter->Outputs.end(); iteroutputs ++) {
            if (!iteroutputs->second->IsTrunk() && iteroutputs->second->Stream) {
                iteroutputs->second->Stream->Stop();
            }
        }

        // Get next filter in the trunk
        output = filter->GetOutput();
        filter = 0;
        // Check if trunk output exists
        if (output) {
            input = output->Connection;
            // Check if trunk output is connected to a trunk input
            if (input && input->Trunk) filter = input->Filter;
        }
    }

    filter = StreamSource;
    while (filter) {
        filter->Running = false;

        // Get next filter in the trunk
        output = filter->GetOutput();
        filter = 0;
        // Check if trunk output exists
        if (output) {
            input = output->Connection;
            // Check if trunk output is connected to a trunk input
            if (input && input->Trunk) filter = input->Filter;
        }
    }

    Running = false;
    StopThread = true;

    // Stopping multi thread processing and delete thread objects
    for (unsigned int i = 0; i < ThreadCount; i ++) {
        if (i != callingthreadID) {
            if (StreamProcThread[i]) {
                StreamProcThread[i]->Wait();
                delete StreamProcThread[i];
                StreamProcThread[i] = 0;
            }
            if (StreamProcInstance[i]) {
                delete StreamProcInstance[i];
                StreamProcInstance[i] = 0;
            }
        }
        else {
            // Skip calling thread!
            // That will be deleted at next Start() or Release()
        }
    }

    // Release thread control arrays and objects
    if (SyncPoint) {
        delete SyncPoint;
        SyncPoint = 0;
    }
    if (CS) {
        delete CS;
        CS = 0;
    }

    // Call OnStop for all filters in the trunk
    filter = StreamSource;
    while (filter) {
        filter->OnStop();

        // Get next filter in the trunk
        output = filter->GetOutput();
        filter = 0;
        // Check if trunk output exists
        if (output) {
            input = output->Connection;
            // Check if trunk output is connected to a trunk input
            if (input && input->Trunk) filter = input->Filter;
        }
    }
}

bool svlStreamManager::IsRunning()
{
    return Running;
}

int svlStreamManager::WaitForStop(double timeout)
{
    if (timeout < 0.0) timeout = 1000000000.0;
    while (Running && timeout > 0.0) {
        timeout -= 0.2;
        osaSleep(0.2);
    }
    if (!Running) return SVL_OK;
    else return SVL_WAIT_TIMEOUT;
}

int svlStreamManager::GetStreamStatus()
{
    return StreamStatus;
}

