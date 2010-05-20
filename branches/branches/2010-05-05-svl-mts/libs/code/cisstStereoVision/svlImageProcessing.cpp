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

#include <cisstStereoVision/svlImageProcessing.h>
#include "svlImageProcessingHelper.h"


/************************************/
/*** svlImageProcessing namespace ***/
/************************************/

int svlImageProcessing::Resize(svlSampleImage* src_img, unsigned int src_videoch,
                               svlSampleImage* dst_img, unsigned int dst_videoch,
                               bool interpolation)
{
    // Please note if OpenCV is NOT enabled:
    //     This is slow because it may reallocate the work buffer every time it gets called.
    //     If possible, use the other implementation and provide a work buffer externally.
    // If OpenCV is enabled this warning does not apply.
    vctDynamicVector<unsigned char> internals;
    return Resize(src_img, src_videoch, dst_img, dst_videoch, interpolation, internals);
}

#if (CISST_SVL_HAS_OPENCV == ON)
int svlImageProcessing::Resize(svlSampleImage* src_img, unsigned int src_videoch,
                               svlSampleImage* dst_img, unsigned int dst_videoch,
                               bool interpolation,
                               vctDynamicVector<unsigned char>& CMN_UNUSED(internals))
#else // CISST_SVL_HAS_OPENCV
int svlImageProcessing::Resize(svlSampleImage* src_img, unsigned int src_videoch,
                               svlSampleImage* dst_img, unsigned int dst_videoch,
                               bool interpolation,
                               vctDynamicVector<unsigned char>& internals)
#endif // CISST_SVL_HAS_OPENCV
{
    if (!src_img || !dst_img ||                       // source or destination is zero
        src_img->GetVideoChannels() <= src_videoch || // source has no such video channel
        dst_img->GetVideoChannels() <= dst_videoch || // destination has no such video channel
        src_img->GetBPP() != dst_img->GetBPP() ||     // image type mismatch
        (src_img->GetBPP() != 1 &&                    // pixel type is not Mono8
         src_img->GetBPP() != 3)) {                   // pixel type is not RGB
        return SVL_FAIL;
    }

    const unsigned int src_width  = src_img->GetWidth(src_videoch);
    const unsigned int src_height = src_img->GetHeight(src_videoch);
    const unsigned int dst_width  = dst_img->GetWidth(dst_videoch);
    const unsigned int dst_height = dst_img->GetHeight(dst_videoch);
    const bool weq = (src_width  == dst_width);
    const bool heq = (src_height == dst_height);

    if (weq && heq) {
        memcpy(dst_img->GetUCharPointer(dst_videoch), src_img->GetUCharPointer(src_videoch), src_img->GetDataSize(src_videoch));
        return SVL_OK;
    }

#if (CISST_SVL_HAS_OPENCV == ON)

    if (interpolation) cvResize(src_img->IplImageRef(src_videoch), dst_img->IplImageRef(dst_videoch), CV_INTER_LINEAR);
    else cvResize(src_img->IplImageRef(src_videoch), dst_img->IplImageRef(dst_videoch), CV_INTER_NN);

#else // CISST_SVL_HAS_OPENCV

    if (src_img->GetBPP() == 3) { // RGB
        if (interpolation) {
            if (weq) {
                svlImageProcessingHelper::ResampleAndInterpolateVRGB24(src_img->GetUCharPointer(src_videoch),
                                                                       src_height,
                                                                       dst_img->GetUCharPointer(dst_videoch),
                                                                       dst_height,
                                                                       dst_width);
            }
            else if (heq) {
                svlImageProcessingHelper::ResampleAndInterpolateHRGB24(src_img->GetUCharPointer(src_videoch),
                                                                       src_width,
                                                                       dst_img->GetUCharPointer(dst_videoch),
                                                                       dst_width,
                                                                       src_height);
            }
            else {
                // Reallocate internal work buffer if needed
                const unsigned int internals_size = dst_width * src_height * 3;
                if (internals.size() < internals_size) internals.SetSize(internals_size);

                svlImageProcessingHelper::ResampleAndInterpolateHRGB24(src_img->GetUCharPointer(src_videoch),
                                                                       src_width,
                                                                       internals.Pointer(),
                                                                       dst_width,
                                                                       src_height);
                svlImageProcessingHelper::ResampleAndInterpolateVRGB24(internals.Pointer(),
                                                                       src_height,
                                                                       dst_img->GetUCharPointer(dst_videoch),
                                                                       dst_height,
                                                                       dst_width);
            }
        }
        else {
            svlImageProcessingHelper::ResampleRGB24(src_img->GetUCharPointer(src_videoch),
                                                    src_width,
                                                    src_height,
                                                    dst_img->GetUCharPointer(dst_videoch),
                                                    dst_width,
                                                    dst_height);
        }
    }
    else { // Mono8
        if (interpolation) {
            if (weq) {
                svlImageProcessingHelper::ResampleAndInterpolateVMono8(src_img->GetUCharPointer(src_videoch),
                                                                       src_height,
                                                                       dst_img->GetUCharPointer(dst_videoch),
                                                                       dst_height,
                                                                       dst_width);
            }
            else if (heq) {
                svlImageProcessingHelper::ResampleAndInterpolateHMono8(src_img->GetUCharPointer(src_videoch),
                                                                       src_width,
                                                                       dst_img->GetUCharPointer(dst_videoch),
                                                                       dst_width,
                                                                       src_height);
            }
            else {
                // Reallocate internal work buffer if needed
                const unsigned int internals_size = dst_width * src_height;
                if (internals.size() < internals_size) internals.SetSize(internals_size);

                svlImageProcessingHelper::ResampleAndInterpolateHMono8(src_img->GetUCharPointer(src_videoch),
                                                                       src_width,
                                                                       internals.Pointer(),
                                                                       dst_width,
                                                                       src_height);
                svlImageProcessingHelper::ResampleAndInterpolateVMono8(internals.Pointer(),
                                                                       src_height,
                                                                       dst_img->GetUCharPointer(dst_videoch),
                                                                       dst_height,
                                                                       dst_width);
            }
        }
        else {
            svlImageProcessingHelper::ResampleMono8(src_img->GetUCharPointer(src_videoch),
                                                    src_width,
                                                    src_height,
                                                    dst_img->GetUCharPointer(dst_videoch),
                                                    dst_width,
                                                    dst_height);
        }
    }

#endif // CISST_SVL_HAS_OPENCV

    return SVL_OK;
}

int svlImageProcessing::Deinterlace(svlSampleImage* image, unsigned int videoch, svlImageProcessing::DI_Algorithm algorithm)
{
    if (!image || image->GetVideoChannels() <= videoch || image->GetBPP() != 3) return SVL_FAIL;

    switch (algorithm) {
        case DI_None:
            // NOP
        break;

        case DI_Blending:
            svlImageProcessingHelper::DeinterlaceBlending(image->GetUCharPointer(videoch),
                                                          static_cast<int>(image->GetWidth(videoch)),
                                                          static_cast<int>(image->GetHeight(videoch)));
        break;

        case DI_Discarding:
            svlImageProcessingHelper::DeinterlaceDiscarding(image->GetUCharPointer(videoch),
                                                            static_cast<int>(image->GetWidth(videoch)),
                                                            static_cast<int>(image->GetHeight(videoch)));
        break;

        case DI_AdaptiveBlending:
            svlImageProcessingHelper::DeinterlaceAdaptiveBlending(image->GetUCharPointer(videoch),
                                                                  static_cast<int>(image->GetWidth(videoch)),
                                                                  static_cast<int>(image->GetHeight(videoch)));
        break;

        case DI_AdaptiveDiscarding:
            svlImageProcessingHelper::DeinterlaceAdaptiveDiscarding(image->GetUCharPointer(videoch),
                                                                    static_cast<int>(image->GetWidth(videoch)),
                                                                    static_cast<int>(image->GetHeight(videoch)));
        break;
    }

    return SVL_OK;
}

