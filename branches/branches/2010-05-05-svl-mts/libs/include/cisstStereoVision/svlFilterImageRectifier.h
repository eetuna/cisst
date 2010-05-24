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

#ifndef _svlFilterImageRectifier_h
#define _svlFilterImageRectifier_h

#include <cisstStereoVision/svlFilterBase.h>

// Always include last!
#include <cisstStereoVision/svlExport.h>


class CISST_EXPORT svlFilterImageRectifier : public svlFilterBase
{
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

    typedef struct _RectificationLUT {
        unsigned int Width;
        unsigned int Height;
        unsigned int* idxDest;
        int idxDestSize;
        unsigned int* idxSrc1;
        int idxSrc1Size;
        unsigned int* idxSrc2;
        int idxSrc2Size;
        unsigned int* idxSrc3;
        int idxSrc3Size;
        unsigned int* idxSrc4;
        int idxSrc4Size;
        unsigned char* blendSrc1;
        int blendSrc1Size;
        unsigned char* blendSrc2;
        int blendSrc2Size;
        unsigned char* blendSrc3;
        int blendSrc3Size;
        unsigned char* blendSrc4;
        int blendSrc4Size;
    } RectificationLUT;

public:
    svlFilterImageRectifier();
    virtual ~svlFilterImageRectifier();

    int LoadTable(const std::string &filepath, unsigned int videoch = SVL_LEFT, int exponentlen = 3);
    void EnableInterpolation(bool enable = true);

protected:
    virtual int Initialize(svlSample* syncInput, svlSample* &syncOutput);
    virtual int Process(svlProcInfo* procInfo, svlSample* syncInput, svlSample* &syncOutput);

private:
    svlSampleImage* OutputImage;

    vctFixedSizeVector<RectificationLUT*, SVL_MAX_CHANNELS>  RectifLUT;
    bool InterpolationEnabled;

    int LoadRectificationData(RectificationLUT* rectdata, const std::string &filepath, int explen);
    int LoadLine(std::ifstream &file, double* dblbuf, char* chbuf, unsigned int size, int explen);
    void TransposeLUTArray(unsigned int* index, unsigned int size, unsigned int width, unsigned int height);
    void Rectify(RectificationLUT* rectdata, unsigned char* src, unsigned char* dest, bool interpolation = true);
    void ResetLUT(RectificationLUT* lut);
    void ReleaseLUT(RectificationLUT* lut);
};

CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlFilterImageRectifier)

#endif // _svlFilterImageRectifier_h

