/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$
  
  Author(s):  Balazs Vagvolgyi
  Created on: 2007 

  (C) Copyright 2006-2007 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#include <cisstStereoVision/svlFilterImageCropper.h>


/******************************************/
/*** svlFilterImageCropper class **********/
/******************************************/

CMN_IMPLEMENT_SERVICES(svlFilterImageCropper)

svlFilterImageCropper::svlFilterImageCropper() :
    svlFilterBase(),
    cmnGenericObject(),
    OutputImage(0)
{
    AddInput("input", true);
    AddInputType("input", svlTypeImageRGB);
    AddInputType("input", svlTypeImageRGBA);
    AddInputType("input", svlTypeImageMono8);
    AddInputType("input", svlTypeImageMono16);
    AddInputType("input", svlTypeImageRGBStereo);
    AddInputType("input", svlTypeImageRGBAStereo);
    AddInputType("input", svlTypeImageMono8Stereo);
    AddInputType("input", svlTypeImageMono16Stereo);
    AddInputType("input", svlTypeImageMonoFloat);
    AddInputType("input", svlTypeImage3DMap);

    AddOutput("output", true);
    SetAutomaticOutputType(true);

    Enabled.SetAll(false);
}

svlFilterImageCropper::~svlFilterImageCropper()
{
    Release();
}

int svlFilterImageCropper::SetRectangle(const int left, const int top, const int right, const int bottom, unsigned int videoch)
{
    return SetRectangle(svlRect(left, top, right, bottom), videoch);
}

int svlFilterImageCropper::SetRectangle(const svlRect & rect, unsigned int videoch)
{
    if (IsInitialized() || videoch >= Rectangles.size()) return SVL_FAIL;

    svlRect r(rect);
    r.Normalize();
    if ((r.right - r.left) < 1 || (r.right - r.left) > 4096 ||
        (r.bottom - r.top) < 1 || (r.bottom - r.top) > 4096) return SVL_FAIL;

    Rectangles[videoch].Assign(r);
    Enabled[videoch] = true;

    return SVL_OK;
}

int svlFilterImageCropper::SetCorner(const int x, const int y, unsigned int videoch)
{
    if (videoch >= Rectangles.size() || !Enabled[videoch]) return SVL_FAIL;
    Rectangles[videoch].right = x + Rectangles[videoch].right - Rectangles[videoch].left;
    Rectangles[videoch].bottom = y + Rectangles[videoch].bottom - Rectangles[videoch].top;
    Rectangles[videoch].left = x;
    Rectangles[videoch].top = y;
    return SVL_OK;
}

int svlFilterImageCropper::SetCenter(const int x, const int y, unsigned int videoch)
{
    if (videoch >= Rectangles.size() || !Enabled[videoch]) return SVL_FAIL;
    return SetCorner(x - (Rectangles[videoch].right - Rectangles[videoch].left) / 2,
                     y - (Rectangles[videoch].bottom - Rectangles[videoch].top) / 2,
                     videoch);
}

int svlFilterImageCropper::Initialize(svlSample* syncInput, svlSample* &syncOutput)
{
    Release();

    OutputImage = dynamic_cast<svlSampleImage*>(syncInput->GetNewInstance());

    svlSampleImage* input = dynamic_cast<svlSampleImage*>(syncInput);

    for (unsigned int i = 0; i < input->GetVideoChannels(); i ++) {
        if (Enabled[i]) {
            OutputImage->SetSize(i, Rectangles[i].right - Rectangles[i].left, Rectangles[i].bottom - Rectangles[i].top); 
        }
        else {
            OutputImage->SetSize(i, input->GetWidth(i), input->GetHeight(i));
        }
    }

    syncOutput = OutputImage;

    return SVL_OK;
}

int svlFilterImageCropper::Process(svlProcInfo* procInfo, svlSample* syncInput, svlSample* &syncOutput)
{
    syncOutput = OutputImage;
    _SkipIfAlreadyProcessed(syncInput, syncOutput);

    svlSampleImage* in_image = dynamic_cast<svlSampleImage*>(syncInput);
    unsigned int videochannels = in_image->GetVideoChannels();
    unsigned int idx;
    int i, wi, hi, wo, ho, xi, yi, xo, yo, copylen, linecount;
    unsigned char *in_data, *out_data;

    _ParallelLoop(procInfo, idx, videochannels)
    {
        // Set background to black
        memset(OutputImage->GetUCharPointer(idx), 0, OutputImage->GetDataSize(idx));

        // Prepare for data copy
        wi = static_cast<int>(in_image->GetWidth(idx) * in_image->GetBPP());
        hi = static_cast<int>(in_image->GetHeight(idx));
        wo = static_cast<int>(OutputImage->GetWidth(idx) * OutputImage->GetBPP());
        ho = static_cast<int>(OutputImage->GetHeight(idx));

        copylen = wo;
        linecount = ho;
        xi = Rectangles[idx].left * in_image->GetBPP();
        yi = Rectangles[idx].top;
        xo = yo = 0;

        // If cropping rectangle reaches out on the left
        if (xi < 0) {
            copylen += xi;
            xo -= xi;
            xi = 0;
        }
        // If cropping rectangle reaches out on the right
        if ((xi + copylen) > wi) {
            copylen += wi - (xi + copylen);
        }
        // If cropping rectangle is outside of the image boundaries
        if (copylen <= 0) continue;

        // If cropping rectangle reaches out on the top
        if (yi < 0) {
            linecount += yi;
            yo -= yi;
            yi = 0;
        }
        // If cropping rectangle reaches out on the bottom
        if ((yi + linecount) > hi) {
            linecount += hi - (yi + linecount);
        }
        // If cropping rectangle is outside of the image boundaries
        if (linecount <= 0) continue;

        in_data = in_image->GetUCharPointer(idx) + (yi * wi) + xi;
        out_data = OutputImage->GetUCharPointer(idx) + (yo * wo) + xo;

        for (i = 0; i < linecount; i ++) {
            memcpy(out_data, in_data, copylen);
            in_data += wi;
            out_data += wo;
        }
    }

    return SVL_OK;
}

int svlFilterImageCropper::Release()
{
    if (OutputImage) {
        delete OutputImage;
        OutputImage = 0;
    }

    return SVL_OK;
}

