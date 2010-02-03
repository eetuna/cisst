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

#include <cisstStereoVision/svlStreamManager.h>
#include "ftInitializer.h"

#include "ftImageBMP.h"
#include "ftImagePPM.h"

#if (CISST_SVL_HAS_JPEG == ON)
#include "ftImageJPEG.h"
#endif // CISST_SVL_HAS_JPEG

#if (CISST_SVL_HAS_PNG == ON)
#include "ftImagePNG.h"
#endif // CISST_SVL_HAS_PNG


void svlInitializeImageCodecs()
{
#ifdef _ftImageBMP_h
    delete new ftImageBMP;
#endif // _ftImageBMP_h

#ifdef _ftImagePPM_h
    delete new ftImagePPM;
#endif // _ftImagePPM_h

#ifdef _ftImageJPEG_h
    delete new ftImageJPEG;
#endif // _ftImageJPEG_h

#ifdef _ftImagePNG_h
    delete new ftImagePNG;
#endif // _ftImagePNG_h
}

