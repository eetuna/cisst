/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: $

  Author(s):  Anton Deguet
  Created on: 2010-01-10

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/


#ifndef _vctFrame_i
#define _vctFrame_i

#include "cisstVector/vctFrameBase.h"

%ignore Inverse;
%ignore ApplyTo;
%ignore ApplyInverseTo;
%ignore Equal;
%ignore AlmostEqual;
%ignore Equivalent;
%ignore AlmostEquivalent;
%ignore ProductOf;
%ignore operator==;

// instantiate the templated base class
%include "cisstVector/vctFrameBase.h"
%include "cisstVector/vctTransformationTypes.h"

// to get access to the translation data member
%include "cisstVector/vctDynamicVectorTypemaps.i"
%apply vctDynamicVector         {vctFixedSizeVector< vctMatrixRotation3< double,true >::value_type,vctFrameBase< vctMatrixRotation3< double,true > >::DIMENSION >};
%apply vctDynamicVector &       {vctFixedSizeVector< vctMatrixRotation3< double,true >::value_type,vctFrameBase< vctMatrixRotation3< double,true > >::DIMENSION > &};
%apply const vctDynamicVector & {const vctFixedSizeVector< vctMatrixRotation3< double,true >::value_type,vctFrameBase< vctMatrixRotation3< double,true > >::DIMENSION > &};

// to get access to the rotation data member
%include "cisstVector/vctDynamicMatrixTypemaps.i"
%apply vctDynamicMatrix         {vctFrameBase< vctMatrixRotation3< double,true > >::RotationType};
%apply vctDynamicMatrix &       {vctFrameBase< vctMatrixRotation3< double,true > >::RotationType &};
%apply const vctDynamicMatrix & {const vctFrameBase< vctMatrixRotation3< double,true > >::RotationType &};

%template(vctFrm3) vctFrameBase<vctRot3 >;

// type declarations for SWIG
%{
    typedef vctFrameBase<vctRot3> vctFrm3;
%}

typedef vctFrameBase<vctRot3 > vctFrm3;

%types(vctFrm3 *);


// clean-up to avoid side effects of %apply
%clear vctFixedSizeVector< vctMatrixRotation3< double,true >::value_type,vctFrameBase< vctMatrixRotation3< double,true > >::DIMENSION >;
%clear vctFixedSizeVector< vctMatrixRotation3< double,true >::value_type,vctFrameBase< vctMatrixRotation3< double,true > >::DIMENSION > &;
%clear const vctFixedSizeVector< vctMatrixRotation3< double,true >::value_type,vctFrameBase< vctMatrixRotation3< double,true > >::DIMENSION > &;

%clear vctFrameBase< vctMatrixRotation3< double,true > >::RotationType;
%clear vctFrameBase< vctMatrixRotation3< double,true > >::RotationType &;
%clear const vctFrameBase< vctMatrixRotation3< double,true > >::RotationType &;

#endif // _vctFrame_i
