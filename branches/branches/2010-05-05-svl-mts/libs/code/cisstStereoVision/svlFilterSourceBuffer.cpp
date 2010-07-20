/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: devMicronTrackerToolQDevice.cpp 1307 2010-03-18 20:34:00Z auneri1 $

  Author(s):  Ali Uneri
  Created on: 2010-05-27

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstStereoVision/svlFilterSourceBuffer.h>
#include <string.h>
#include "time.h"


/*************************************/
/*** svlFilterSourceBuffer class ******/
/*************************************/

CMN_IMPLEMENT_SERVICES(svlFilterSourceBuffer)

svlFilterSourceBuffer::svlFilterSourceBuffer() :
    svlFilterSourceBase(),
    Width(0),
    Height(0),
    Buffer(0)
{
    OutputImage = 0;
    SetType(svlTypeImageRGB);
}

svlFilterSourceBuffer::svlFilterSourceBuffer(svlStreamType type) :
    svlFilterSourceBase(),
    Width(0),
    Height(0),
    Buffer(0)
{
    OutputImage = 0;
    SetType(type);
}

svlFilterSourceBuffer::svlFilterSourceBuffer(const svlSampleImage & image) :
    svlFilterSourceBase(),
    Width(0),
    Height(0),
    Buffer(0)
{
    OutputImage = 0;
    SetImage(image);
}

svlFilterSourceBuffer::~svlFilterSourceBuffer()
{
    if (OutputImage) {
        delete OutputImage;
    }
}

int svlFilterSourceBuffer::SetType(svlStreamType type)
{
    if (IsInitialized() == true) {
        return SVL_ALREADY_INITIALIZED;
    }

    // Other types may be added in the future
    if (type != svlTypeImageRGB && type != svlTypeImageRGBStereo) return SVL_FAIL;

    if (OutputImage && OutputImage->GetType() != type) {
        delete OutputImage;
        OutputImage = svlSample::GetNewFromType(type);
    } else if (!OutputImage) {
        OutputImage = svlSample::GetNewFromType(type);
    }

    if (Width > 0 && Height > 0) {
        dynamic_cast<svlSampleImage *>(OutputImage)->SetSize(Width, Height);
    }
    SetOutputType("Output", type);

    return SVL_OK;
}

int svlFilterSourceBuffer::SetImage(const svlSampleImage & image)
{
    if (IsInitialized() == true) {
        return SVL_ALREADY_INITIALIZED;
    }

    svlStreamType type = image.GetType();

    // Other types may be added in the future
    if (type != svlTypeImageRGB && type != svlTypeImageRGBStereo) return SVL_FAIL;

    if (OutputImage && OutputImage->GetType() != type) {
        delete OutputImage;
        OutputImage = svlSample::GetNewFromType(type);
    } else if (!OutputImage) {
        OutputImage = svlSample::GetNewFromType(type);
    }

    OutputImage->CopyOf(image);
    SetOutputType("Output", type);

    return SVL_OK;
}

int svlFilterSourceBuffer::SetDimensions(unsigned int width, unsigned int height)
{
    if (IsInitialized() == true) {
        return SVL_ALREADY_INITIALIZED;
    }
    Width = width;
    Height = height;

    if (OutputImage) {
        dynamic_cast<svlSampleImage *>(OutputImage)->SetSize(width, height);
    }
    return SVL_OK;
}

int svlFilterSourceBuffer::SetBuffer(svlBufferImage * buffer)
{
    Buffer = buffer;
    return SVL_OK;
}

int svlFilterSourceBuffer::Initialize()
{
    if (OutputImage == 0 || Buffer == 0) {
        return SVL_FAIL;
    }
    srand(static_cast<unsigned int>(time(0)));
    return SVL_OK;
}

int svlFilterSourceBuffer::ProcessFrame(svlProcInfo* procInfo)
{
    svlSampleImage * outputImage = dynamic_cast<svlSampleImage *>(OutputImage);
    svlImageRGB * inputImage;

    for (unsigned int ch = 0; ch < outputImage->GetVideoChannels(); ch++) {
        inputImage = Buffer->Pull(true);
        if (inputImage != 0) {
            memcpy(outputImage->GetUCharPointer(ch), inputImage->Pointer(), inputImage->size());
        }
    }
    return SVL_OK;
}
