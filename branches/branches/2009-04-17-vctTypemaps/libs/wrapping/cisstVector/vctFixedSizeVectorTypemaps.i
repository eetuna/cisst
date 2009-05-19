/******************************************************************************
 Authors: Daniel Li, Anton Deguet
******************************************************************************/


/**************************************************************************
*                          Applying Typemaps
**************************************************************************/

%define VCT_TYPEMAPS_APPLY_FIXED_SIZE_VECTORS_ONE_SIZE(elementType, size)
%apply vctDynamicVector         {vctFixedSizeVector<elementType, size>};
%apply vctDynamicVector &       {vctFixedSizeVector<elementType, size> &};
%apply const vctDynamicVector & {const vctFixedSizeVector<elementType, size> &};
%enddef

%define VCT_TYPEMAPS_APPLY_FIXED_SIZE_VECTORS(elementType)
VCT_TYPEMAPS_APPLY_FIXED_SIZE_VECTORS_ONE_SIZE(elementType, 2);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_VECTORS_ONE_SIZE(elementType, 3);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_VECTORS_ONE_SIZE(elementType, 4);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_VECTORS_ONE_SIZE(elementType, 5);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_VECTORS_ONE_SIZE(elementType, 6);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_VECTORS_ONE_SIZE(elementType, 7);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_VECTORS_ONE_SIZE(elementType, 8);
%enddef

VCT_TYPEMAPS_APPLY_FIXED_SIZE_VECTORS(int);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_VECTORS(unsigned int);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_VECTORS(double);
