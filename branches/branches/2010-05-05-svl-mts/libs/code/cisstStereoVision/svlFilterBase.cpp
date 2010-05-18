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

#include <cisstStereoVision/svlFilterBase.h>
#include <cisstStereoVision/svlStreamManager.h>

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
/*** svlFilterBase class *************/
/*************************************/

svlFilterBase::svlFilterBase() :
    FrameCounter(0),
    Initialized(false),
    Running(false),
    PrevInputTimestamp(-1.0),
    AutoType(false)
{
}

svlFilterBase::~svlFilterBase()
{
    for (_Inputs::iterator iterinputs = Inputs.begin(); iterinputs != Inputs.end(); iterinputs ++) {
        if (iterinputs->second) delete iterinputs->second;
    }
    for (_Outputs::iterator iteroutputs = Outputs.begin(); iteroutputs != Outputs.end(); iteroutputs ++) {
        if (iteroutputs->second) delete iteroutputs->second;
    }
}

bool svlFilterBase::IsInitialized()
{
    return Initialized;
}

bool svlFilterBase::IsRunning()
{
    return Running;
}

unsigned int svlFilterBase::GetFrameCounter()
{
    return FrameCounter;
}

svlFilterInput* svlFilterBase::GetInput()
{
    for (_Inputs::iterator iterinputs = Inputs.begin(); iterinputs != Inputs.end(); iterinputs ++) {
        if (iterinputs->second->Trunk) return iterinputs->second;
    }
    return 0;
}

svlFilterOutput* svlFilterBase::GetOutput()
{
    for (_Outputs::iterator iteroutputs = Outputs.begin(); iteroutputs != Outputs.end(); iteroutputs ++) {
        if (iteroutputs->second->Trunk) return iteroutputs->second;
    }
    return 0;
}

svlFilterInput* svlFilterBase::GetInput(const std::string &inputname)
{
    _Inputs::iterator iterinputs = Inputs.find(inputname);
    if (iterinputs == Inputs.end()) return 0;
    return iterinputs->second;
}

svlFilterOutput* svlFilterBase::GetOutput(const std::string &outputname)
{
    _Outputs::iterator iteroutputs = Outputs.find(outputname);
    if (iteroutputs == Outputs.end()) return 0;
    return iteroutputs->second;
}

svlFilterInput* svlFilterBase::AddInput(const std::string &inputname, bool trunk)
{
    if (trunk) {
        // Check if there is already a trunk input
        for (_Inputs::iterator iterinputs = Inputs.begin(); iterinputs != Inputs.end(); iterinputs ++) {
            if (iterinputs->second->Trunk) return 0;
        }
    }

    svlFilterInput* input = new svlFilterInput(this, trunk, inputname);
    Inputs[inputname] = input;
    return input;
}

svlFilterOutput* svlFilterBase::AddOutput(const std::string &outputname, bool trunk)
{
    if (trunk) {
        // Check if there is already a trunk output
        for (_Outputs::iterator iteroutputs = Outputs.begin(); iteroutputs != Outputs.end(); iteroutputs ++) {
            if (iteroutputs->second->Trunk) return 0;
        }
    }

    svlFilterOutput* output = new svlFilterOutput(this, trunk, outputname);
    Outputs[outputname] = output;
    return output;
}

int svlFilterBase::AddInputType(const std::string &inputname, svlStreamType type)
{
    _Inputs::iterator iterinputs = Inputs.find(inputname);
    if (iterinputs == Inputs.end()) return SVL_FAIL;
    return iterinputs->second->AddType(type);
}

int svlFilterBase::SetOutputType(const std::string &outputname, svlStreamType type)
{
    _Outputs::iterator iteroutputs = Outputs.find(outputname);
    if (iteroutputs == Outputs.end()) return 0;
    return iteroutputs->second->SetType(type);
}

void svlFilterBase::SetAutomaticOutputType(bool autotype)
{
    AutoType = autotype;
}

int svlFilterBase::UpdateTypes(svlFilterInput & CMN_UNUSED(input), svlStreamType CMN_UNUSED(type))
{
    // Needs to be overloaded to handle manual type setup
    return SVL_OK;
}

int svlFilterBase::Initialize(svlSample* CMN_UNUSED(syncInput), svlSample* &CMN_UNUSED(syncOutput))
{
    return SVL_OK;
}

int svlFilterBase::OnStart(unsigned int CMN_UNUSED(procCount))
{
    return SVL_OK;
}

int svlFilterBase::Process(svlProcInfo* CMN_UNUSED(procInfo), svlSample* CMN_UNUSED(syncInput), svlSample* &CMN_UNUSED(syncOutput))
{
    return SVL_OK;
}

void svlFilterBase::OnStop()
{
}

int svlFilterBase::Release()
{
    return SVL_OK;
}

int svlFilterBase::IsDataValid(svlStreamType type, svlSample* data)
{
    if (data == 0) return SVL_NO_INPUT_DATA;
    if (type != data->GetType()) return SVL_INVALID_INPUT_TYPE;
    if (data->IsInitialized() == false) return SVL_NO_INPUT_DATA;
    return SVL_OK;
}

bool svlFilterBase::IsNewSample(svlSample* sample)
{
    return (sample && sample->GetTimestamp() > PrevInputTimestamp) ? true : false;
}

