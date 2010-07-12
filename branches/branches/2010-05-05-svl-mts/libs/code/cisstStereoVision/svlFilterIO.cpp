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

#include <cisstStereoVision/svlFilterIO.h>
#include <cisstStereoVision/svlFilterBase.h>
#include <cisstStereoVision/svlStreamManager.h>
#include <cisstStereoVision/svlStreamBranchSource.h>


/*************************************/
/*** svlFilterInput class ************/
/*************************************/

svlFilterInput::svlFilterInput(svlFilterBase* filter, bool trunk, const std::string &name) :
    Filter(filter),
    Trunk(trunk),
    Name(name),
    Connected(false),
    Connection(0),
    ConnectedFilter(0),
    Type(svlTypeInvalid),
    Buffer(0),
    Timestamp(-1.0)
{
}

svlFilterInput::~svlFilterInput()
{
    if (Buffer) delete Buffer;
}

bool svlFilterInput::IsTrunk() const
{
    return Trunk;
}

svlStreamType svlFilterInput::GetType() const
{
    return Type;
}

const std::string& svlFilterInput::GetName() const
{
    return Name;
}

svlFilterBase* svlFilterInput::GetFilter()
{
    return Filter;
}

svlFilterBase* svlFilterInput::GetConnectedFilter()
{
    return ConnectedFilter;
}

int svlFilterInput::AddType(svlStreamType type)
{
    if (!Filter || Filter->IsInitialized()) return SVL_FAIL;

    unsigned int size = SupportedTypes.size();
    SupportedTypes.resize(size + 1);
    SupportedTypes[size] = type;
    return SVL_OK;
}

bool svlFilterInput::IsConnected() const
{
    return Connected;
}

bool svlFilterInput::IsTypeSupported(svlStreamType type)
{
    const unsigned int size = SupportedTypes.size();
    for (unsigned int i = 0; i < size; i ++) {
        if (SupportedTypes[i] == type) return true;
    }
    return false;
}

svlFilterOutput* svlFilterInput::GetConnection()
{
    return Connection;
}

int svlFilterInput::Disconnect()
{
    // TO DO
    return SVL_FAIL;
}

int svlFilterInput::PushSample(const svlSample* sample)
{
    if (!sample || !Filter || Trunk || Connected) return SVL_FAIL;

    svlStreamType type = sample->GetType();
    if (Type != svlTypeInvalid && Type != type) return SVL_FAIL;

    if (Filter->AutoType) {
        // Automatic setup
        if (!IsTypeSupported(type)) return SVL_FAIL;
        Type = type;
    }
    else {
        // Manual setup
        Type = type;
        if (Filter->UpdateTypes(*this, Type) != SVL_OK) return SVL_FAIL;
    }

    if (!Buffer) Buffer = new svlBufferSample(Type);

    // Store timestamp
    Timestamp = sample->GetTimestamp();

    return Buffer->Push(sample);
}

svlSample* svlFilterInput::PullSample(bool waitfornew, double timeout)
{
    if (!Filter || !Filter->IsInitialized() || !Buffer) return 0;
    return Buffer->Pull(waitfornew, timeout);
}

double svlFilterInput::GetTimestamp()
{
    return Timestamp;
}


/*************************************/
/*** svlFilterOutput class ***********/
/*************************************/

svlFilterOutput::svlFilterOutput(svlFilterBase* filter, bool trunk, const std::string &name) :
    Filter(filter),
    Trunk(trunk),
    Name(name),
    Connected(false),
    Connection(0),
    ConnectedFilter(0),
    Type(svlTypeInvalid),
    ThreadCount(2),
    BufferSize(3),
    Blocked(false),
    Stream(0),
    BranchSource(0),
    Timestamp(-1.0)
{
}

svlFilterOutput::~svlFilterOutput()
{
    if (Stream) delete Stream;
    if (BranchSource) delete BranchSource;
}

bool svlFilterOutput::IsTrunk() const
{
    return Trunk;
}

svlStreamType svlFilterOutput::GetType() const
{
    return Type;
}

const std::string& svlFilterOutput::GetName() const
{
    return Name;
}

svlFilterBase* svlFilterOutput::GetFilter()
{
    return Filter;
}

svlFilterBase* svlFilterOutput::GetConnectedFilter()
{
    return ConnectedFilter;
}

bool svlFilterOutput::IsConnected() const
{
    return Connected;
}

int svlFilterOutput::SetType(svlStreamType type)
{
    if (!Filter || Filter->Initialized) return SVL_FAIL;
    Type = type;
    return SVL_OK;
}

svlFilterInput* svlFilterOutput::GetConnection()
{
    return Connection;
}

int svlFilterOutput::GetDroppedSampleCount()
{
    if (!BranchSource) return SVL_FAIL;
    return static_cast<int>(BranchSource->GetDroppedSampleCount());
}

int svlFilterOutput::GetBufferUsage()
{
    if (!BranchSource) return SVL_FAIL;
    return BranchSource->GetBufferUsage();
}

double svlFilterOutput::GetBufferUsageRatio()
{
    if (!BranchSource) return -1.0;
    return BranchSource->GetBufferUsageRatio();
}

int svlFilterOutput::SetThreadCount(unsigned int threadcount)
{
    if (!Filter || Filter->Initialized) return SVL_FAIL;

    // Thread count is inherited on the trunk
    if (Trunk || threadcount < 1) return SVL_FAIL;
    ThreadCount = threadcount;
    return SVL_OK;
}

int svlFilterOutput::SetBufferSize(unsigned int buffersize)
{
    if (!Filter || Filter->Initialized) return SVL_FAIL;

    // There is no buffering on the trunk
    if (Trunk || buffersize < 3) return SVL_FAIL;
    BufferSize = buffersize;
    return SVL_OK;
}

int svlFilterOutput::SetBlock(bool block)
{
    if (!Filter) return SVL_FAIL;

    // Trunk output cannot be blocked
    if (Trunk) return SVL_FAIL;
    Blocked = block;
    return SVL_OK;
}

int svlFilterOutput::Connect(svlFilterInput *input)
{
    if (!Filter ||
        !input || !input->Filter ||
        Connected || input->Connected ||
        Filter->Initialized || input->Filter->Initialized) return SVL_FAIL;

    // Setup output types in the connected filter
    if (input->Trunk && input->Filter->AutoType) {
        // Automatic setup
        if (!input->IsTypeSupported(Type)) return SVL_FAIL;
        svlFilterOutput* output = input->Filter->GetOutput();
        if (output) output->SetType(Type);
    }
    else {
        // Manual setup
        if (input->Filter->UpdateTypes(*input, Type) != SVL_OK) return SVL_FAIL;
    }

    if (!Trunk && input->Trunk) {
        // Create stream branch if not trunk
        Stream = new svlStreamManager(ThreadCount);
        BranchSource = new svlStreamBranchSource(Type, BufferSize);
        Stream->SetSourceFilter(BranchSource);

        // Connect filters
        svlFilterOutput* output = BranchSource->GetOutput();
        if (output) output->Connect(input);
    }
    else {
        if (!input->Trunk) {
            input->Buffer = new svlBufferSample(Type);
        }

        // Connect filters
        input->Connected = true;
        input->Connection = this;
        input->Type = Type;
        input->ConnectedFilter = Filter;
    }

    Connection = input;
    ConnectedFilter = input->Filter;
    Connected = true;

    return SVL_OK;
}

int svlFilterOutput::Disconnect()
{
    // TO DO
    return SVL_FAIL;
}

void svlFilterOutput::SetupSample(svlSample* sample)
{
    if (sample &&
        Filter && !Filter->Initialized &&
        !Trunk &&
        Connected) {

        if (Connection->Trunk) BranchSource->SetInput(sample);
    }
}

void svlFilterOutput::PushSample(const svlSample* sample)
{
    if (sample &&
        Filter && Filter->Initialized && 
        !Trunk && Connected && !Blocked) {

        if (Connection->Trunk) BranchSource->PushSample(sample);
        else if (Connection->Buffer) Connection->Buffer->Push(sample);

        // Store timestamp
        Timestamp = sample->GetTimestamp();
    }
}

double svlFilterOutput::GetTimestamp()
{
    return Timestamp;
}

