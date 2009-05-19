/******************************************************************************
 Authors: Daniel Li, Anton Deguet
******************************************************************************/

/*****************************************************************************
 PLACEHOLDER STRINGS TO LOOK FOR:

   TODO
*****************************************************************************/

#define CISST_EXPORT
#define CISST_DEPRECATED

%include "std_streambuf.i"
%include "std_iostream.i"

%rename(__str__) ToString;


%init %{
        import_array()   /* Initial function for NumPy */
%}


/******************************************************************************
  TYPEMAPS (in, out) FOR vctFixedSizeVector
******************************************************************************/


%typemap(in) vctFixedSizeVector
{
    /*****************************************************************************
    *   %typemap(in) vctFixedSizeVector
    *   Passing a vctFixedSizeVector by copy
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    *****************************************************************************/

    /*****************************************************************************
     CHECK IF THE PYTHON OBJECT (NAMED `$input') THAT WAS PASSED TO THIS TYPE MAP
     IS A PYARRAY, IS OF THE CORRECT DTYPE, AND IS ONE-DIMENSIONAL
    *****************************************************************************/

    if (!(   vctThrowUnlessIsPyArray($input)
          && vctThrowUnlessIsSameTypeArray<$1_ltype::value_type>($input)
          && vctThrowUnlessDimension1($input)
          && vctThrowUnlessCorrectVectorSize($input, $1))
        ) {
          return NULL;
    }

    /*****************************************************************************
     COPY THE DATA OF THE PYARRAY (NAMED `$input') TO THE vctFixedSizeVector
    *****************************************************************************/

    // Create a temporary vctDynamicVectorRef container
    const npy_intp size = PyArray_DIM($input, 0);
    const npy_intp stride = PyArray_STRIDE($input, 0) / sizeof($1_ltype::value_type);
    const $1_ltype::pointer data = reinterpret_cast<$1_ltype::pointer>(PyArray_DATA($input));

    const vctDynamicVectorRef<$1_ltype::value_type> tempContainer(size, data, stride);

    // Copy the data from the temporary container to the vctFixedSizeVector
    $1.Assign(tempContainer);
}


%typemap(out) vctFixedSizeVector
{
    /*****************************************************************************
    *   %typemap(out) vctFixedSizeVector
    *   Returning a vctFixedSizeVector
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    *****************************************************************************/

    /*****************************************************************************
     CREATE A NEW PYARRAY OBJECT
    *****************************************************************************/

    //Create a new PyArray and set its size
    npy_intp* sizes = PyDimMem_NEW(1);
    sizes[0] = $1.size();
    int type = vctPythonType<$1_ltype::value_type>();
    $result = PyArray_SimpleNew(1, sizes, type);

    /*****************************************************************************
     COPY THE DATA FROM THE vctFixedSizeConstVectorRef TO THE PYARRAY
    *****************************************************************************/

    // Create a temporary vctDynamicVectorRef container
    const npy_intp size = $1.size();
    const npy_intp stride = 1;
    const $1_ltype::pointer data = reinterpret_cast<$1_ltype::pointer>(PyArray_DATA($result));

    vctDynamicVectorRef<$1_ltype::value_type> tempContainer(size, data, stride);

    // Copy the data from the vctFixedSizeConstVectorRef to the temporary container
    tempContainer.Assign($1);
}


/******************************************************************************
  TYPEMAPS (in, argout, out) FOR vctFixedSizeVector &
******************************************************************************/

%typemap(in) vctFixedSizeVector &
{
    /*****************************************************************************
    *   %typemap(in) vctFixedSizeVector &
    *   Passing a vctFixedSizeVector by reference
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this typemap.
    *****************************************************************************/

    /*****************************************************************************
     CHECK IF THE PYTHON OBJECT (NAMED `$input') THAT WAS PASSED TO THIS TYPE MAP
     IS A PYARRAY, IS OF THE CORRECT DTYPE, IS ONE-DIMENSIONAL, AND IS WRITABLE
    *****************************************************************************/

    if (!(   vctThrowUnlessIsPyArray($input)
          && vctThrowUnlessIsSameTypeArray<$*1_ltype::value_type>($input)
          && vctThrowUnlessDimension1($input)
          && vctThrowUnlessCorrectVectorSize($input, *($1))
          && vctThrowUnlessIsWritable($input)
          && vctThrowUnlessOwnsData($input, *($1))
          && vctThrowUnlessNotReferenced($input, *($1)))
        ) {
          return NULL;
    }

    /*****************************************************************************
     COPY THE DATA OF THE PYARRAY (NAMED `$input') TO THE vctFixedSizeVector
    *****************************************************************************/

    // Create a temporary vctDynamicVectorRef container
    const npy_intp size = PyArray_DIM($input, 0);
    const npy_intp stride = PyArray_STRIDE($input, 0) / sizeof($*1_ltype::value_type);
    const $*1_ltype::pointer data = reinterpret_cast<$*1_ltype::pointer>(PyArray_DATA($input));

    const vctDynamicVectorRef<$*1_ltype::value_type> tempContainer(size, data, stride);

    // Create the vctFixedSizeVector
    $1 = new $*1_ltype(tempContainer);
}


%typemap(argout) vctFixedSizeVector &
{
    /*****************************************************************************
    *   %typemap(argout) vctFixedSizeVector &
    *   Passing a vctFixedSizeVector by reference
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    *****************************************************************************/

    /*************************************************************************
     COPY THE DATA TO THE PYARRAY
    *************************************************************************/

    // Create a temporary vctDynamicVectorRef container
    const npy_intp size = PyArray_DIM($input, 0);
    const npy_intp stride = PyArray_STRIDE($input, 0) / sizeof($*1_ltype::value_type);
    const $*1_ltype::pointer data = reinterpret_cast<$*1_ltype::pointer>(PyArray_DATA($input));

    vctDynamicVectorRef<$*1_ltype::value_type> tempContainer(size, data, stride);

    // Copy the data from the temporary container to the vctFixedSizeVector
    tempContainer.Assign(*($1));

    /*************************************************************************
     CLEAN UP
    *************************************************************************/

    // Don't forget to free the memory we allocated in the `in' typemap
    delete $1;
}


%typemap(out) vctFixedSizeVector &
{
    /* Return vector by reference
       Using: %typemap(out) vctFixedSizeVector &
     */

    //Create new size array and set size
    npy_intp* sizeOfReturnedVector = PyDimMem_NEW(1);
    sizeOfReturnedVector[0] = $1->size();

    //create a new PyArray from the reference returned by the C function
    int type = vctPythonType<$*1_ltype::value_type>();
    $result = PyArray_SimpleNewFromData(1, sizeOfReturnedVector, type,  $1->Pointer() );
}


/******************************************************************************
  TYPEMAPS (in, argout, out) FOR const vctFixedSizeVector &
******************************************************************************/


%typemap(in) const vctFixedSizeVector &
{
    /*****************************************************************************
    *   %typemap(in) const vctFixedSizeVector &
    *   Passing a vctFixedSizeVector by const &
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    *****************************************************************************/

    /*****************************************************************************
     CHECK IF THE PYTHON OBJECT (NAMED `$input') THAT WAS PASSED TO THIS TYPE MAP
     IS A PYARRAY, IS OF THE CORRECT DTYPE, AND IS ONE-DIMENSIONAL
    *****************************************************************************/

    if (!(   vctThrowUnlessIsPyArray($input)
          && vctThrowUnlessIsSameTypeArray<$*1_ltype::value_type>($input)
          && vctThrowUnlessDimension1($input)
          && vctThrowUnlessCorrectVectorSize($input, *($1)))
        ) {
          return NULL;
    }

    /*****************************************************************************
     COPY THE DATA OF THE PYARRAY (NAMED `$input') TO THE vctFixedSizeVector
    *****************************************************************************/

    // Create a temporary vctDynamicVectorRef container
    const npy_intp size = PyArray_DIM($input, 0);
    const npy_intp stride = PyArray_STRIDE($input, 0) / sizeof($*1_ltype::value_type);
    const $*1_ltype::pointer data = reinterpret_cast<$*1_ltype::pointer>(PyArray_DATA($input));

    const vctDynamicVectorRef<$*1_ltype::value_type> tempContainer(size, data, stride);

    // Create the vctFixedSizeVector
    $1 = new $*1_ltype(tempContainer);
}


%typemap(argout) const vctFixedSizeVector &
{
    /**************************************************************************
    *   %typemap(argout) const vctFixedSizeVector &
    *   Passing a vctFixedSizeVector by const reference
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    **************************************************************************/

    /**************************************************************************
     CLEAN UP
    **************************************************************************/

    // Don't forget to free the memory we allocated in the `in' typemap
    delete $1;
}


%typemap(out) const vctFixedSizeVector &
{
    /* Return vector by const reference
       Using: %typemap(out) const vctFixedSizeVector &
     */

    /* To imitate const functionality, set the writable flag to false */

    //Create new size array and set size
    npy_intp* sizes = PyDimMem_NEW(1);
    sizes[0] = $1->size();

    //NPY_CARRAY_RO = set flags for a C Array that is Read Only (i.e. const)
    int type = vctPythonType<$*1_ltype::value_type>();
    $result = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(type), 1, sizes, NULL, $1->Pointer(), NPY_CARRAY_RO, NULL);
}


/**************************************************************************
*                          Applying Typemaps
**************************************************************************/

%define VCT_TYPEMAPS_APPLY_FIXED_SIZE_VECTORS_ONE_SIZE(elementType, size)
%apply vctFixedSizeVector         {vctFixedSizeVector<elementType, size>};
%apply vctFixedSizeVector &       {vctFixedSizeVector<elementType, size> &};
%apply const vctFixedSizeVector & {const vctFixedSizeVector<elementType, size> &};
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
