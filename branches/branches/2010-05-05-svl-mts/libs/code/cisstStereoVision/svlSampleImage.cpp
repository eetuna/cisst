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

#include <cisstStereoVision/svlTypes.h>
#include <cisstStereoVision/svlConverters.h>


/*****************************/
/*** svlSampleImage class ****/
/*****************************/

svlSampleImage::svlSampleImage() :
    svlSample()
{
    SetEncoder("bmp", 0);
}

svlSampleImage::~svlSampleImage()
{
}

int svlSampleImage::ConvertFrom(const svlSampleImage* image)
{
    if (image == 0) return SVL_FAIL;
    return ConvertFrom(*image);
}

int svlSampleImage::ConvertFrom(const svlSampleImage& image)
{
    const unsigned int channels = image.GetVideoChannels();
    if (channels != GetVideoChannels()) return SVL_FAIL;

    int ret = SVL_OK;

    for (unsigned int i = 0; i < channels; i ++) {
        if (ConvertFrom(image, i, i) != SVL_OK) ret = SVL_FAIL;
    }

    return ret;
}

int svlSampleImage::ConvertFrom(const svlSampleImage* image, const unsigned int src_channel, const unsigned int dest_channel)
{
    if (image == 0) return SVL_FAIL;
    return ConvertFrom(*image, src_channel, dest_channel);
}

int svlSampleImage::ConvertFrom(const svlSampleImage& image, const unsigned int src_channel, const unsigned int dest_channel)
{
    if (src_channel  >= image.GetVideoChannels() || dest_channel >= GetVideoChannels()) return SVL_FAIL;

    SetSize(dest_channel, image.GetWidth(src_channel), image.GetHeight(src_channel));

    if (image.GetPixelType() != GetPixelType()) {
        return svlConverter::ConvertImage(&image, src_channel, this, dest_channel);
    }

    // Same pixel type: copy instead of convert
    memcpy(GetUCharPointer(dest_channel), image.GetUCharPointer(src_channel), GetDataSize(dest_channel));
    return SVL_OK;
}

