/******************************************************************************
 Authors: Daniel Li, Anton Deguet
******************************************************************************/


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
