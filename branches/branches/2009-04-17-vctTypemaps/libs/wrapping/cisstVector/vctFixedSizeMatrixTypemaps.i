/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: $

  Author(s):  Daniel Li, Anton Deguet
  Created on: 2009-05-20

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/


/**************************************************************************
*                           Applying Typemaps
**************************************************************************/

%define VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES_ONE_SIZE(elementType, rows, cols)
%apply vctDynamicMatrix         {vctFixedSizeMatrix<elementType, rows, cols>};
%apply vctDynamicMatrix &       {vctFixedSizeMatrix<elementType, rows, cols> &};
%apply const vctDynamicMatrix & {const vctFixedSizeMatrix<elementType, rows, cols> &};
%enddef

%define VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES(elementType)
VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES_ONE_SIZE(elementType, 2, 2);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES_ONE_SIZE(elementType, 3, 3);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES_ONE_SIZE(elementType, 4, 4);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES_ONE_SIZE(elementType, 5, 5);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES_ONE_SIZE(elementType, 6, 6);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES_ONE_SIZE(elementType, 7, 7);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES_ONE_SIZE(elementType, 8, 8);
%enddef

VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES(int);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES(unsigned int);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES(double);
