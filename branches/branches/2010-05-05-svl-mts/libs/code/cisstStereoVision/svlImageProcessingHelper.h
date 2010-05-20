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

#ifndef _svlImageProcessingHelper_h
#define _svlImageProcessingHelper_h


namespace svlImageProcessingHelper
{
    //////////////
    // Resizing //
    //////////////

    void ResampleMono8(unsigned char* src, const unsigned int srcwidth, const unsigned int srcheight,
                       unsigned char* dst, const unsigned int dstwidth, const unsigned int dstheight);
    void ResampleAndInterpolateHMono8(unsigned char* src, const unsigned int srcwidth,
                                      unsigned char* dst, const unsigned int dstwidth,
                                      const unsigned int height);
    void ResampleAndInterpolateVMono8(unsigned char* src, const unsigned int srcheight,
                                      unsigned char* dst, const unsigned int dstheight,
                                      const unsigned int width);
    void ResampleRGB24(unsigned char* src, const unsigned int srcwidth, const unsigned int srcheight,
                       unsigned char* dst, const unsigned int dstwidth, const unsigned int dstheight);
    void ResampleAndInterpolateHRGB24(unsigned char* src, const unsigned int srcwidth,
                                      unsigned char* dst, const unsigned int dstwidth,
                                      const unsigned int height);
    void ResampleAndInterpolateVRGB24(unsigned char* src, const unsigned int srcheight,
                                      unsigned char* dst, const unsigned int dstheight,
                                      const unsigned int width);

    ///////////////////
    // Deinterlacing //
    ///////////////////

    void DeinterlaceBlending(unsigned char* buffer, const unsigned int width, const unsigned int height);
    void DeinterlaceDiscarding(unsigned char* buffer, const unsigned int width, const unsigned int height);
    void DeinterlaceAdaptiveBlending(unsigned char* buffer, const unsigned int width, const unsigned int height);
    void DeinterlaceAdaptiveDiscarding(unsigned char* buffer, const unsigned int width, const unsigned int height);
};

#endif // _svlImageProcessingHelper_h

