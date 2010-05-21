/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$
  
  Author(s):  Balazs Vagvolgyi
  Created on: 2009 

  (C) Copyright 2006-2007 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstStereoVision/svlFilterDisparityMapToSurface.h>
#include <cisstStereoVision/svlImageProcessing.h>


/********************************************/
/*** svlFilterDisparityMapToSurface class ***/
/********************************************/

CMN_IMPLEMENT_SERVICES(svlFilterDisparityMapToSurface)

svlFilterDisparityMapToSurface::svlFilterDisparityMapToSurface() :
    svlFilterBase(),
    cmnGenericObject()
{
    AddInput("input", true);
    AddInputType("input", svlTypeImageMonoFloat);

    AddOutput("output", true);
    SetAutomaticOutputType(false);
    GetOutput()->SetType(svlTypeImage3DMap);
}

int svlFilterDisparityMapToSurface::Initialize(svlSample* syncInput, svlSample* &syncOutput)
{
    OutputSurface.SetSize(syncInput);

    svlSampleImage* image = dynamic_cast<svlSampleImage*>(syncInput);

    ROI.Normalize();
    ROI.Trim(0, image->GetWidth() - 1, 0, image->GetHeight() - 1);

    syncOutput = &OutputSurface;

    return SVL_OK;
}

int svlFilterDisparityMapToSurface::Process(svlProcInfo* procInfo, svlSample* syncInput, svlSample* &syncOutput)
{
    syncOutput = &OutputSurface;
    _SkipIfAlreadyProcessed(syncInput, syncOutput);

    _OnSingleThread(procInfo)
    {
        svlImageProcessing::DisparityMapToSurface(dynamic_cast<svlSampleImageMonoFloat*>(syncInput),
                                                  &OutputSurface,
                                                  Geometry,
                                                  ROI);
    }

    return SVL_OK;
}

int svlFilterDisparityMapToSurface::SetCameraGeometry(const svlCameraGeometry & geometry)
{
    if (IsInitialized()) return SVL_FAIL;
    Geometry = geometry;
    return SVL_OK;
}

int svlFilterDisparityMapToSurface::SetROI(const svlRect & rect)
{
    if (IsInitialized()) return SVL_FAIL;
    ROI = rect;
    return SVL_OK;
}

int svlFilterDisparityMapToSurface::SetROI(int left, int top, int right, int bottom)
{
    return SetROI(svlRect(left, top, right, bottom));
}

