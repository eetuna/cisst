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
  TYPEMAPS (in, out) FOR vctFixedSizeMatrix
******************************************************************************/


%typemap(in) vctFixedSizeMatrix
{
    /*****************************************************************************
    *   %typemap(in) vctFixedSizeMatrix
    *   Passing a vctFixedSizeMatrix by copy
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    *****************************************************************************/

    /*****************************************************************************
     CHECK IF THE PYTHON OBJECT (NAMED `$input') THAT WAS PASSED TO THIS TYPE MAP
     IS A PYARRAY, IS OF THE CORRECT DTYPE, AND IS TWO-DIMENSIONAL
    *****************************************************************************/

    if (!(   vctThrowUnlessIsPyArray($input)
          && vctThrowUnlessIsSameTypeArray<$1_ltype::value_type>($input)
          && vctThrowUnlessDimension2($input)
          && vctThrowUnlessCorrectMatrixSize($input, $1))
        ) {
          return NULL;
    }

    /*****************************************************************************
     COPY THE DATA OF THE PYARRAY (NAMED `$input') TO THE vctFixedSizeMatrix
    *****************************************************************************/

    // Create a temporary vctDynamicMatrixRef container
    const npy_intp size0 = PyArray_DIM($input, 0);
    const npy_intp size1 = PyArray_DIM($input, 1);
    const npy_intp stride0 = PyArray_STRIDE($input, 0) / sizeof($1_ltype::value_type);
    const npy_intp stride1 = PyArray_STRIDE($input, 1) / sizeof($1_ltype::value_type);
    const $1_ltype::pointer data = reinterpret_cast<$1_ltype::pointer>(PyArray_DATA($input));

    const vctDynamicMatrixRef<$1_ltype::value_type> tempContainer(size0, size1, stride0, stride1, data);

    // Copy the data from the temporary container to the vctFixedSizeMatrix
    $1.Assign(tempContainer);
}


%typemap(out) vctFixedSizeMatrix
{
    /*****************************************************************************
    *   %typemap(out) vctFixedSizeMatrix
    *   Return a vctFixedSizeMatrix by copy
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    *****************************************************************************/

    /*****************************************************************************
     CREATE A NEW PYARRAY OBJECT
    *****************************************************************************/

    npy_intp *sizes = PyDimMem_NEW(2);
    sizes[0] = $1.rows();
    sizes[1] = $1.cols();
    int type = vctPythonType<$1_ltype::value_type>();
    $result = PyArray_SimpleNew(2, sizes, type);

    /*****************************************************************************
     COPY THE DATA FROM THE vctDynamicConstMatrixRef TO THE PYARRAY
    *****************************************************************************/

    // Create a temporary vctDynamicMatrixRef container
    const npy_intp size0 = PyArray_DIM($result, 0);
    const npy_intp size1 = PyArray_DIM($result, 1);
    const npy_intp stride0 = PyArray_STRIDE($result, 0) / sizeof($1_ltype::value_type);
    const npy_intp stride1 = PyArray_STRIDE($result, 1) / sizeof($1_ltype::value_type);
    const $1_ltype::pointer data = reinterpret_cast<$1_ltype::pointer>(PyArray_DATA($result));

    vctDynamicMatrixRef<$1_ltype::value_type> tempContainer(size0, size1, stride0, stride1, data);

    // Copy the data from the vctDynamicConstMatrixRef to the temporary container
    tempContainer.Assign($1);
}


/******************************************************************************
  TYPEMAPS (in, argout, out) FOR vctFixedSizeMatrix &
******************************************************************************/


%typemap(in) vctFixedSizeMatrix &
{
    /*****************************************************************************
    *   %typemap(in) vctFixedSizeMatrix &
    *   Passing a vctFixedSizeMatrix by reference
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
          && vctThrowUnlessDimension2($input)
          && vctThrowUnlessCorrectMatrixSize($input, *($1))
          && vctThrowUnlessIsWritable($input)
          && vctThrowUnlessOwnsData($input, *($1))
          && vctThrowUnlessNotReferenced($input, *($1)))
        ) {
          return NULL;
    }

    /*****************************************************************************
     COPY THE DATA OF THE PYARRAY (NAMED `$input') TO THE vctFixedSizeMatrix
    *****************************************************************************/

    // Create a temporary vctDynamicMatrixRef container
    const npy_intp size0 = PyArray_DIM($input, 0);
    const npy_intp size1 = PyArray_DIM($input, 1);
    const npy_intp stride0 = PyArray_STRIDE($input, 0) / sizeof($*1_ltype::value_type);
    const npy_intp stride1 = PyArray_STRIDE($input, 1) / sizeof($*1_ltype::value_type);
    const $*1_ltype::pointer data = reinterpret_cast<$*1_ltype::pointer>(PyArray_DATA($input));

    const vctDynamicMatrixRef<$*1_ltype::value_type> tempContainer(size0, size1, stride0, stride1, data);

    // Create the vctFixedSizeMatrix
    $1 = new $*1_ltype(tempContainer);
}


%typemap(argout) vctFixedSizeMatrix &
{
    /*****************************************************************************
    *   %typemap(argout) vctFixedSizeMatrix &
    *   Passing a vctFixedSizeMatrix by reference
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    *****************************************************************************/

    /*************************************************************************
     COPY THE DATA TO THE PYARRAY
    *************************************************************************/

    // Create a temporary vctDynamicMatrixRef container
    const npy_intp size0 = PyArray_DIM($input, 0);
    const npy_intp size1 = PyArray_DIM($input, 1);
    const npy_intp stride0 = PyArray_STRIDE($input, 0) / sizeof($*1_ltype::value_type);
    const npy_intp stride1 = PyArray_STRIDE($input, 1) / sizeof($*1_ltype::value_type);
    const $*1_ltype::pointer data = reinterpret_cast<$*1_ltype::pointer>(PyArray_DATA($input));

    vctDynamicMatrixRef<$*1_ltype::value_type> tempContainer(size0, size1, stride0, stride1, data);

    // Copy the data from the temporary container to the vctFixedSizeMatrix
    tempContainer.Assign(*($1));

    /*************************************************************************
     CLEAN UP
    *************************************************************************/

    // Don't forget to free the memory we allocated in the `in' typemap
    delete $1;
}


%typemap(out) vctFixedSizeMatrix &
{
    /* Return vector by reference
       Using: %typemap(out) vctFixedSizeMatrix &
     */

    // Create new size array and set size
    npy_intp *sizes = PyDimMem_NEW(2);
    sizes[0] = $1->rows();
    sizes[1] = $1->cols();

    // NPY_CARRAY = set flags for a C Array that is non-Read Only
    int type = vctPythonType<$*1_ltype::value_type>();
    $result = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(type), 2, sizes, NULL, $1->Pointer(), NPY_CARRAY, NULL);
}


/******************************************************************************
  TYPEMAPS (in, argout, out) FOR const vctFixedSizeMatrix &
******************************************************************************/


%typemap(in) const vctFixedSizeMatrix &
{
    /*****************************************************************************
    *   %typemap(in) const vctFixedSizeMatrix &
    *   Passing a vctFixedSizeMatrix by const &
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
          && vctThrowUnlessDimension2($input)
          && vctThrowUnlessCorrectMatrixSize($input, *($1)))
        ) {
          return NULL;
    }

    /*****************************************************************************
     COPY THE DATA OF THE PYARRAY (NAMED `$input') TO THE vctFixedSizeMatrix
    *****************************************************************************/

    // Create a temporary vctDynamicMatrixRef container
    const npy_intp size0 = PyArray_DIM($input, 0);
    const npy_intp size1 = PyArray_DIM($input, 1);
    const npy_intp stride0 = PyArray_STRIDE($input, 0) / sizeof($*1_ltype::value_type);
    const npy_intp stride1 = PyArray_STRIDE($input, 1) / sizeof($*1_ltype::value_type);
    const $*1_ltype::pointer data = reinterpret_cast<$*1_ltype::pointer>(PyArray_DATA($input));

    const vctDynamicMatrixRef<$*1_ltype::value_type> tempContainer(size0, size1, stride0, stride1, data);

    // Create the vctFixedSizeMatrix
    $1 = new $*1_ltype(tempContainer);
}


%typemap(argout) const vctFixedSizeMatrix &
{
    /**************************************************************************
    *   %typemap(argout) const vctFixedSizeMatrix &
    *   Passing a vctFixedSizeMatrix by const reference
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


%typemap(out) const vctFixedSizeMatrix &
{
    /* Return vector by const reference
       Using: %typemap(out) const vctFixedSizeMatrix &
     */

    // Create new size array and set size
    npy_intp *sizes = PyDimMem_NEW(2);
    sizes[0] = $1->rows();
    sizes[1] = $1->cols();

    // To imitate const functionality, set the writable flag to false
    // NPY_CARRAY_RO = set flags for a C Array that is Read Only (i.e. const)
    int type = vctPythonType<$*1_ltype::value_type>();
    $result = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(type), 2, sizes, NULL, $1->Pointer(), NPY_CARRAY_RO, NULL);
}


/**************************************************************************
*                           Applying Typemaps
**************************************************************************/

%define VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES_ONE_SIZE(elementType, rows, cols)
%apply vctFixedSizeMatrix         {vctFixedSizeMatrix<elementType, rows, cols>};
%apply vctFixedSizeMatrix &       {vctFixedSizeMatrix<elementType, rows, cols> &};
%apply const vctFixedSizeMatrix & {const vctFixedSizeMatrix<elementType, rows, cols> &};
%enddef

%define VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES(elementType)
VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES_ONE_SIZE(elementType, 2, 2);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES_ONE_SIZE(elementType, 3, 3);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES_ONE_SIZE(elementType, 4, 4);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES_ONE_SIZE(elementType, 5, 5);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES_ONE_SIZE(elementType, 6, 6);
%enddef

VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES(int);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES(unsigned int);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_MATRICES(double);
