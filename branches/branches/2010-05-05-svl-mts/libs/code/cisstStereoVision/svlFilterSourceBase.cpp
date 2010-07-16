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

#include <cisstStereoVision/svlFilterSourceBase.h>
#include <cisstOSAbstraction/osaSleep.h>


#ifdef _MSC_VER
    // Quick fix for Visual Studio Intellisense:
    // The Intellisense parser can't handle the CMN_UNUSED macro
    // correctly if defined in cmnPortability.h, thus
    // we should redefine it here for it.
    // Removing this part of the code will not effect compilation
    // in any way, on any platforms.
    #undef CMN_UNUSED
    #define CMN_UNUSED(argument) argument
#endif


/*************************************/
/*** svlFilterSourceBase class *******/
/*************************************/

svlFilterSourceBase::svlFilterSourceBase() :
    svlFilterBase(),
    TargetFrequency(-1.0),
    LoopFlag(true),
    AutoTimestamp(true),
    TargetStartTime(0.0),
    TargetFrameTime(0.0),
    PlaybackFrameOffset(0),
    PlayCounter(-1)
{
}

svlFilterSourceBase::svlFilterSourceBase(bool autotimestamps) :
    svlFilterBase(),
    TargetFrequency(-1.0),
    LoopFlag(true),
    AutoTimestamp(autotimestamps),
    TargetStartTime(0.0),
    TargetFrameTime(0.0),
    PlaybackFrameOffset(0),
    PlayCounter(-1)
{
}

svlFilterSourceBase::~svlFilterSourceBase()
{
}

double svlFilterSourceBase::GetTargetFrequency()
{
    return TargetFrequency;
}

int svlFilterSourceBase::SetTargetFrequency(double hertz)
{
    TargetFrequency = hertz;
    return SVL_OK;
}

void svlFilterSourceBase::SetLoop(bool loop)
{
    LoopFlag = loop;
}

bool svlFilterSourceBase::GetLoop()
{
    return LoopFlag;
}

void svlFilterSourceBase::Pause()
{
    Play(0);
}

void svlFilterSourceBase::Play()
{
    Play(-1);
}

void svlFilterSourceBase::Play(const int frames)
{
    PlayCounter = frames;

    if (frames != 0) {
        OnResetTimer();
        // Reset target timer
        if (TargetTimer.IsRunning()) PlaybackFrameOffset = FrameCounter;
    }
}

int svlFilterSourceBase::Initialize(svlSample* &CMN_UNUSED(syncOutput))
{
    return SVL_OK;
}

int svlFilterSourceBase::OnStart(unsigned int CMN_UNUSED(procCount))
{
    RestartTargetTimer();
    return SVL_OK;
}

void svlFilterSourceBase::OnStop()
{
    StopTargetTimer();
}

int svlFilterSourceBase::Release()
{
    return SVL_OK;
}

void svlFilterSourceBase::OnResetTimer()
{
    // Default implementation does nothing
}

int svlFilterSourceBase::RestartTargetTimer()
{
    if (TargetFrequency >= 0.1) {
        TargetFrameTime = 1.0 / TargetFrequency;
        TargetTimer.Reset();
        TargetTimer.Start();
        return SVL_OK;
    }
    return SVL_FAIL;
}

int svlFilterSourceBase::StopTargetTimer()
{
    if (TargetTimer.IsRunning()) {
        TargetTimer.Stop();
        return SVL_OK;
    }
    return SVL_FAIL;
}

int svlFilterSourceBase::WaitForTargetTimer()
{
    if (TargetTimer.IsRunning()) {
        if (FrameCounter == 0) FrameCounter = PlaybackFrameOffset;
        if (FrameCounter > PlaybackFrameOffset) {
            double time = TargetTimer.GetElapsedTime();
            double t1 = TargetFrameTime * (FrameCounter - PlaybackFrameOffset);
            double t2 = time - TargetStartTime;
            if (t1 > t2) osaSleep(t1 - t2);
        }
        else {
            TargetStartTime = TargetTimer.GetElapsedTime();
        }
        return SVL_OK;
    }
    return SVL_FAIL;
}

bool svlFilterSourceBase::IsTargetTimerRunning()
{
    return TargetTimer.IsRunning();
}

int svlFilterSourceBase::Initialize(svlSample* CMN_UNUSED(syncInput), svlSample* &syncOutput)
{
    return Initialize(syncOutput);
}

int svlFilterSourceBase::Process(svlProcInfo* procInfo, svlSample* CMN_UNUSED(syncInput), svlSample* &syncOutput)
{
    return Process(procInfo, syncOutput);
}

int svlFilterSourceBase::AddInput(const std::string & CMN_UNUSED(inputname), bool CMN_UNUSED(trunk))
{
    return SVL_FAIL;
}

int svlFilterSourceBase::AddInputType(const std::string & CMN_UNUSED(inputname), svlStreamType CMN_UNUSED(type))
{
    return SVL_FAIL;
}

int svlFilterSourceBase::UpdateTypes(svlFilterInput & CMN_UNUSED(input), svlStreamType CMN_UNUSED(type))
{
    return SVL_FAIL;
}


/******************************************/
/*** svlFilterSourceImageBase class *******/
/******************************************/

svlFilterSourceImageBase::svlFilterSourceImageBase() : svlFilterSourceBase()
{
}

svlFilterSourceImageBase::svlFilterSourceImageBase(bool autotimestamps) : svlFilterSourceBase(autotimestamps)
{
}

