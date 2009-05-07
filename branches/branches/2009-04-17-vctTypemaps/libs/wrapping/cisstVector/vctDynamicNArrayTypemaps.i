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
  TYPEMAPS (in, out) FOR vctDynamicNArray
******************************************************************************/


%typemap(in) vctDynamicNArray
{
    /*****************************************************************************
    *   %typemap(in) vctDynamicNArray
    *   Passing a vctDynamicNArray by copy
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    *****************************************************************************/

    /*****************************************************************************
     CHECK IF THE PYTHON OBJECT (NAMED `$input') THAT WAS PASSED TO THIS TYPE MAP
     IS A PYARRAY, IS OF NUMPY DTYPE int, AND IS TWO-DIMENSIONAL
    *****************************************************************************/

    if (!(   vctThrowUnlessIsPyArray($input)
          && vctThrowUnlessIsSameTypeArray<$1_ltype::value_type>($input)
          && vctThrowUnlessDimensionN<$1_ltype>($input))
        ) {
          return NULL;
    }

    /*****************************************************************************
     COPY THE DATA OF THE PYARRAY (NAMED `$input') TO THE vctDynamicNArray
    *****************************************************************************/

    // Create a temporary vctDynamicNArrayRef container

    // sizes
    npy_intp *_sizes = PyArray_DIMS($input);
    vctFixedSizeConstVectorRef<npy_intp, $1_ltype::DIMENSION, 1> _sizesRef(_sizes);
    vctFixedSizeVector<unsigned int, $1_ltype::DIMENSION> sizes;
    sizes.Assign(_sizesRef);

    // strides
    npy_intp *_strides = PyArray_STRIDES($input);
    vctFixedSizeConstVectorRef<npy_intp, $1_ltype::DIMENSION, 1> _stridesRef(_strides);
    vctFixedSizeVector<int, $1_ltype::DIMENSION> strides;
    strides.Assign(_stridesRef);
    strides.Divide(sizeof($1_ltype::value_type));

    // data pointer
    const $1_ltype::pointer data = reinterpret_cast<$1_ltype::pointer>(PyArray_DATA($input));

    const vctDynamicNArrayRef<$1_ltype::value_type, $1_ltype::DIMENSION> tempContainer(data, sizes, strides);

    // Copy the data from the temporary container to the vctDynamicNArray
    $1.SetSize(sizes);
    $1.Assign(tempContainer);
}


// TODO: Search for other uses of %typemap(out, optimal="1")


%typemap(out) vctDynamicNArray
{
    /* Return vector by copy
       Using: %typemap(out) vctDynamicNArray
     */

    // Create a new PyArray and set its size
    npy_intp *sizes = PyDimMem_NEW(2);
    sizes[0] = $1.rows();
    sizes[1] = $1.cols();

    // TODO: Understand what this does
    $result = PyArray_SimpleNew(2, sizes, vctPythonType<$1_ltype::value_type>());

    // TODO: Understand what this does
    // copy data returned by C function into new PyArray
    memcpy(PyArray_DATA($result), $1.Pointer(), $1.size() * sizeof($1_ltype::value_type));
}


/******************************************************************************
  TYPEMAPS (in, argout, out) FOR vctDynamicNArray &
******************************************************************************/


%typemap(in) vctDynamicNArray &
{
    /*****************************************************************************
    *   %typemap(in) vctDynamicNArray &
    *   Passing a vctDynamicNArray by reference
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this typemap.
    *****************************************************************************/

    /*****************************************************************************
     CHECK IF THE PYTHON OBJECT (NAMED `$input') THAT WAS PASSED TO THIS TYPE MAP
     IS A PYARRAY, IS OF NUMPY DTYPE int, IS ONE-DIMENSIONAL, AND IS WRITABLE
    *****************************************************************************/

    if (!(   vctThrowUnlessIsPyArray($input)
          && vctThrowUnlessIsSameTypeArray<$*1_ltype::value_type>($input)
          && vctThrowUnlessDimensionN<$*1_ltype>($input)
          && vctThrowUnlessIsWritable($input)
          && vctThrowUnlessOwnsData($input)
          && vctThrowUnlessNotReferenced($input))
        ) {
          return NULL;
    }

    /*****************************************************************************
     COPY THE DATA OF THE PYARRAY (NAMED `$input') TO THE vctDynamicNArray
    *****************************************************************************/

    // TODO: Since the PyArray is guaranteed to be contiguous, should we use memcpy instead
    // of copying using a vctDynamicNArrayRef?

    // Create a temporary vctDynamicNArrayRef container

    // sizes
    npy_intp *_sizes = PyArray_DIMS($input);
    vctFixedSizeConstVectorRef<npy_intp, $*1_ltype::DIMENSION, 1> _sizesRef(_sizes);
    vctFixedSizeVector<unsigned int, $*1_ltype::DIMENSION> sizes;
    sizes.Assign(_sizesRef);

    // strides
    npy_intp *_strides = PyArray_STRIDES($input);
    vctFixedSizeConstVectorRef<npy_intp, $*1_ltype::DIMENSION, 1> _stridesRef(_strides);
    vctFixedSizeVector<int, $*1_ltype::DIMENSION> strides;
    strides.Assign(_stridesRef);
    strides.Divide(sizeof($*1_ltype::value_type));

    // data pointer
    const $*1_ltype::pointer data = reinterpret_cast<$*1_ltype::pointer>(PyArray_DATA($input));

    const vctDynamicNArrayRef<$*1_ltype::value_type, $*1_ltype::DIMENSION> tempContainer(data, sizes, strides);

    // Create the vctDynamicNArray
    $1 = new $*1_ltype(tempContainer);
}


%typemap(argout) vctDynamicNArray &
{
    /*****************************************************************************
    *   %typemap(argout) vctDynamicNArray &
    *   Passing a vctDynamicNArray by reference
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    *****************************************************************************/

    /*************************************************************************
     CHECK IF THE CONTAINER HAS BEEN RESIZED
    *************************************************************************/

    // input sizes
    npy_intp *_input_sizes = PyArray_DIMS($input);
    vctFixedSizeConstVectorRef<npy_intp, $*1_ltype::DIMENSION, 1> _input_sizesRef(_input_sizes);
    vctFixedSizeVector<unsigned int, $*1_ltype::DIMENSION> input_sizes;
    input_sizes.Assign(_input_sizesRef);

    // output sizes
    vctFixedSizeVector<unsigned int, $*1_ltype::DIMENSION> output_sizes;
    output_sizes.Assign($1->sizes());

    if (input_sizes.Equal(output_sizes)) {
        // Resize the PyArray by:
        //  1)  Creating an array containing the new size
        //  2)  Pass that array to the resizing function given by NumPy API

        // create an array of sizes
        //unsigned int sz = $*1_ltype::DIMENSION;
        npy_intp *sizes = PyDimMem_NEW(4);

        // set the sizes
        for (unsigned int i = 0; i < $*1_ltype::DIMENSION; i++) {
            sizes[i] = output_sizes.at(i);
        }

        // create a PyArray_Dims object to hand to PyArray_Resize()
        PyArray_Dims dims;
        dims.ptr = sizes;
        dims.len = $*1_ltype::DIMENSION;
        PyArray_Resize((PyArrayObject *) $input, &dims, 0, NPY_CORDER);
    }

    /*************************************************************************
     COPY THE DATA TO THE PYARRAY
    *************************************************************************/

    // TODO: Since the PyArray is guaranteed to be contiguous, should we use memcpy instead
    // of copying using a vctDynamicNArrayRef?

    // Create a temporary vctDynamicNArrayRef container

    // sizes
    npy_intp *_sizes = PyArray_DIMS($input);
    vctFixedSizeConstVectorRef<npy_intp, $*1_ltype::DIMENSION, 1> _sizesRef(_sizes);
    vctFixedSizeVector<unsigned int, $*1_ltype::DIMENSION> sizes;
    sizes.Assign(_sizesRef);

    // strides
    npy_intp *_strides = PyArray_STRIDES($input);
    vctFixedSizeConstVectorRef<npy_intp, $*1_ltype::DIMENSION, 1> _stridesRef(_strides);
    vctFixedSizeVector<int, $*1_ltype::DIMENSION> strides;
    strides.Assign(_stridesRef);
    strides.Divide(sizeof($*1_ltype::value_type));

    // data pointer
    const $*1_ltype::pointer data = reinterpret_cast<$*1_ltype::pointer>(PyArray_DATA($input));

    // temporary container
    vctDynamicNArrayRef<$*1_ltype::value_type, $*1_ltype::DIMENSION> tempContainer(data, sizes, strides);

    // Copy the data from the temporary container to the vctDynamicNArray
    tempContainer.Assign($1->Pointer());

    /*************************************************************************
     CLEAN UP
    *************************************************************************/

    // Don't forget to free the memory we allocated in the `in' typemap
    delete $1;
}


%typemap(out) vctDynamicNArray &
{
    /* Return vector by reference
       Using: %typemap(out) vctDynamicNArray &
     */

    // Create new size array and set size
    npy_intp *sizes = PyDimMem_NEW(2);
    sizes[0] = $1->rows();
    sizes[1] = $1->cols();

    // TODO: Understand what this does
    // NPY_CARRAY = set flags for a C Array that is non-Read Only
    $result = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(vctPythonType<$*1_ltype::value_type>()), 2, sizes, NULL, $1->Pointer(), NPY_CARRAY, NULL);
}


/******************************************************************************
  TYPEMAPS (in, argout, out) FOR const vctDynamicNArray &
******************************************************************************/


%typemap(in) const vctDynamicNArray &
{
    /*****************************************************************************
    *   %typemap(in) const vctDynamicNArray &
    *   Passing a vctDynamicNArray by const &
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    *****************************************************************************/

    /*****************************************************************************
     CHECK IF THE PYTHON OBJECT (NAMED `$input') THAT WAS PASSED TO THIS TYPE MAP
     IS A PYARRAY, IS OF NUMPY DTYPE int, AND IS ONE-DIMENSIONAL
    *****************************************************************************/

    if (!(vctThrowUnlessIsPyArray($input)
          && vctThrowUnlessIsSameTypeArray<$*1_ltype::value_type>($input)
          && vctThrowUnlessDimensionN<$*1_ltype>($input))
        ) {
          return NULL;
    }

    /*****************************************************************************
     COPY THE DATA OF THE PYARRAY (NAMED `$input') TO THE vctDynamicNArray
    *****************************************************************************/

    // Create a temporary vctDynamicNArrayRef container

    // sizes
    npy_intp *_sizes = PyArray_DIMS($input);
    vctFixedSizeConstVectorRef<npy_intp, $*1_ltype::DIMENSION, 1> _sizesRef(_sizes);
    vctFixedSizeVector<unsigned int, $*1_ltype::DIMENSION> sizes;
    sizes.Assign(_sizesRef);

    // strides
    npy_intp *_strides = PyArray_STRIDES($input);
    vctFixedSizeConstVectorRef<npy_intp, $*1_ltype::DIMENSION, 1> _stridesRef(_strides);
    vctFixedSizeVector<int, $*1_ltype::DIMENSION> strides;
    strides.Assign(_stridesRef);
    strides.Divide(sizeof($*1_ltype::value_type));

    // data pointer
    const $*1_ltype::pointer data = reinterpret_cast<$*1_ltype::pointer>(PyArray_DATA($input));

    const vctDynamicNArrayRef<$*1_ltype::value_type, $*1_ltype::DIMENSION> tempContainer(data, sizes, strides);

    // Create the vctDynamicNArray
    $1 = new $*1_ltype(tempContainer);
}


%typemap(argout) const vctDynamicNArray &
{
    /**************************************************************************
    *   %typemap(argout) const vctDynamicNArray &
    *   Passing a vctDynamicNArray by const reference
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


%typemap(out) const vctDynamicNArray &
{
    /* Return vector by const reference
       Using: %typemap(out) const vctDynamicNArray &
     */

    // Create new size array and set size
    npy_intp *sizes = PyDimMem_NEW(2);
    sizes[0] = $1->rows();
    sizes[1] = $1->cols();

    // TODO: Understand what this does
    // To imitate const functionality, set the writable flag to false
    // NPY_CARRAY_RO = set flags for a C Array that is Read Only (i.e. const)
    $result = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(vctPythonType<$*1_ltype::value_type>()), 2, sizes, NULL, $1->Pointer(), NPY_CARRAY_RO, NULL);
}


/******************************************************************************
  TYPEMAPS (in, out) FOR vctDynamicNArrayRef
******************************************************************************/


%typemap(in) vctDynamicNArrayRef
{
    /*************************************************************************
    *   %typemap(in) vctDynamicNArrayRef
    *   Passing a vctDynamicNArrayRef by copy
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic
    *   behind this typemap.
    *************************************************************************/

    /*************************************************************************
     CHECK IF THE PYTHON OBJECT (NAMED `$input') THAT WAS PASSED TO THIS
     TYPEMAP IS A PYARRAY, IS OF NUMPY DTYPE int, IS ONE-DIMENSIONAL, AND
     IS WRITABLE
    *************************************************************************/

    if (!(   vctThrowUnlessIsPyArray($input)
          && vctThrowUnlessIsSameTypeArray<$1_ltype::value_type>($input)
          && vctThrowUnlessDimensionN<$1_ltype>($input)
          && vctThrowUnlessIsWritable($input))
        ) {
          return NULL;
    }

    /*************************************************************************
     SET THE SIZE, STRIDE AND DATA POINTER OF THE vctDynamicNArrayRef
     OBJECT (NAMED `$1') TO MATCH THAT OF THE PYARRAY (NAMED `$input')
    *************************************************************************/

    // sizes
    npy_intp *_sizes = PyArray_DIMS($input);
    vctFixedSizeConstVectorRef<npy_intp, $1_ltype::DIMENSION, 1> _sizesRef(_sizes);
    vctFixedSizeVector<unsigned int, $1_ltype::DIMENSION> sizes;
    sizes.Assign(_sizesRef);

    // strides
    npy_intp *_strides = PyArray_STRIDES($input);
    vctFixedSizeConstVectorRef<npy_intp, $1_ltype::DIMENSION, 1> _stridesRef(_strides);
    vctFixedSizeVector<int, $1_ltype::DIMENSION> strides;
    strides.Assign(_stridesRef);
    strides.Divide(sizeof($1_ltype::value_type));

    // data pointer
    const $1_ltype::pointer data = reinterpret_cast<$1_ltype::pointer>(PyArray_DATA($input));

    $1.SetRef(data, sizes, strides);
}


%typemap(out, optimal="1") vctDynamicNArrayRef
{
    /*****************************************************************************
    *   %typemap(out, optimal="1") vctDynamicNArrayRef
    *   Returning a vctDynamicNArrayRef
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
    $result = PyArray_SimpleNew(2, sizes, vctPythonType<$1_ltype::value_type>());

    /*****************************************************************************
     COPY THE DATA FROM THE vctDynamicNArrayRef TO THE PYARRAY
    *****************************************************************************/

    // Create a temporary vctDynamicNArrayRef container
    const npy_intp size0 = $1.rows();
    const npy_intp size1 = $1.cols();
    const npy_intp stride0 = size1;     // TODO: Why does this work?  Shouldn't it be:  stride0 = size1 * sizeof($1_ltype::value_type)
    const npy_intp stride1 = 1;
    const $1_ltype::pointer data = reinterpret_cast<$1_ltype::pointer>(PyArray_DATA($result));

    vctDynamicNArrayRef<$1_ltype::value_type> tempContainer(size0, size1, stride0, stride1, data);

    // Copy the data from the vctDynamicNArrayRef to the temporary container
    tempContainer.Assign($1);
}


/******************************************************************************
  TYPEMAPS (in, argout) FOR const vctDynamicNArrayRef &
******************************************************************************/


%typemap(in) const vctDynamicNArrayRef &
{
    /**************************************************************************
    *   %typemap(in) const vctDynamicNArrayRef &
    *   Passing a vctDynamicNArrayRef by const reference
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    **************************************************************************/

    /*****************************************************************************
     CHECK IF THE PYTHON OBJECT (NAMED `$input') THAT WAS PASSED TO THIS TYPE MAP
     IS A PYARRAY, IS OF NUMPY DTYPE int, AND IS ONE-DIMENSIONAL
    *****************************************************************************/

    if (!(   vctThrowUnlessIsPyArray($input)
          && vctThrowUnlessIsSameTypeArray<$*1_ltype::value_type>($input)
          && vctThrowUnlessDimensionN<$*1_ltype>($input))
        ) {
          return NULL;
    }

    /*****************************************************************************
     CREATE A vctDynamicNArrayRef TO POINT TO THE DATA OF THE PYARRAY
    *****************************************************************************/

    // Create the vctDynamicNArrayRef

    // dimensions
    const unsigned int ndim = PyArray_NDIM($input);

    // sizes
    npy_intp *_sizes = PyArray_DIMS($input);
    vctFixedSizeConstVectorRef<npy_intp, $*1_ltype::DIMENSION, 1> _sizesRef(_sizes);
    vctFixedSizeVector<unsigned int, $*1_ltype::DIMENSION> sizes;
    sizes.Assign(_sizesRef);

    // strides
    npy_intp *_strides = PyArray_STRIDES($input);
    vctFixedSizeConstVectorRef<npy_intp, $*1_ltype::DIMENSION, 1> _stridesRef(_strides);
    vctFixedSizeVector<int, $*1_ltype::DIMENSION> strides;
    strides.Assign(_stridesRef);
    strides.Divide(sizeof($*1_ltype::value_type));

    // data pointer
    const $*1_ltype::pointer data = reinterpret_cast<$*1_ltype::pointer>(PyArray_DATA($input));

    $1 = new $*1_ltype(data, sizes, strides);
}


%typemap(argout) const vctDynamicNArrayRef &
{
    /**************************************************************************
    *   %typemap(argout) const vctDynamicNArrayRef &
    *   Passing a vctDynamicNArrayRef by const reference
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


/******************************************************************************
  TYPEMAPS (in, out) FOR vctDynamicConstNArrayRef
******************************************************************************/


%typemap(in) vctDynamicConstNArrayRef
{
    /*****************************************************************************
    *   %typemap(in) vctDynamicConstNArrayRef
    *   Passing a vctDynamicConstNArrayRef by copy
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    *****************************************************************************/

    /*****************************************************************************
     CHECK IF THE PYTHON OBJECT (NAMED `$input') THAT WAS PASSED TO THIS TYPE MAP
     IS A PYARRAY, IS OF NUMPY DTYPE int, AND IS N-DIMENSIONAL
    *****************************************************************************/

    if (!(   vctThrowUnlessIsPyArray($input)
          && vctThrowUnlessIsSameTypeArray<$1_ltype::value_type>($input)
          && vctThrowUnlessDimensionN<$1_ltype>($input))
        ) {
          return NULL;
    }

    /*****************************************************************************
     SET THE SIZE, STRIDE, AND DATA POINTER OF THE
     vctDynamicConstNArrayRef OBJECT (NAMED `$1') TO MATCH THAT OF THE
     PYARRAY (NAMED `$input')
    *****************************************************************************/

    // sizes
    npy_intp *_sizes = PyArray_DIMS($input);
    vctFixedSizeConstVectorRef<npy_intp, $1_ltype::DIMENSION, 1> _sizesRef(_sizes);
    vctFixedSizeVector<unsigned int, $1_ltype::DIMENSION> sizes;
    sizes.Assign(_sizesRef);

    // strides
    npy_intp *_strides = PyArray_STRIDES($input);
    vctFixedSizeConstVectorRef<npy_intp, $1_ltype::DIMENSION, 1> _stridesRef(_strides);
    vctFixedSizeVector<int, $1_ltype::DIMENSION> strides;
    strides.Assign(_stridesRef);
    strides.Divide(sizeof($1_ltype::value_type));

    // data pointer
    const $1_ltype::pointer data = reinterpret_cast<$1_ltype::pointer>(PyArray_DATA($input));

    $1.SetRef(data, sizes, strides);
}


// TODO: Why does this out typemap work without the ``optimal'' flag?
%typemap(out) vctDynamicConstNArrayRef
{
    /*****************************************************************************
    *   %typemap(out) vctDynamicConstNArrayRef
    *   Returning a vctDynamicConstNArrayRef
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
    //$result = PyArray_SimpleNew(2, sizes, vctPythonType<$1_ltype::value_type>());
    // Look at the NumPy C API to see how these lines work: http://projects.scipy.org/numpy/wiki/NumPyCAPI
    PyArray_Descr *descr = PyArray_DescrFromType(vctPythonType<$1_ltype::value_type>());
    $result = PyArray_NewFromDescr(&PyArray_Type, descr, 2, sizes, NULL, NULL, NPY_CONTIGUOUS | NPY_OWNDATA | NPY_ALIGNED, NULL);

    /*****************************************************************************
     COPY THE DATA FROM THE vctDynamicConstNArrayRef TO THE PYARRAY
    *****************************************************************************/

    // Create a temporary vctDynamicNArrayRef container
    const npy_intp size0 = PyArray_DIM($result, 0);
    const npy_intp size1 = PyArray_DIM($result, 1);
    const npy_intp stride0 = PyArray_STRIDE($result, 0) / sizeof($1_ltype::value_type);
    const npy_intp stride1 = PyArray_STRIDE($result, 1) / sizeof($1_ltype::value_type);
    const $1_ltype::pointer data = reinterpret_cast<$1_ltype::pointer>(PyArray_DATA($result));

    vctDynamicNArrayRef<$1_ltype::value_type> tempContainer(size0, size1, stride0, stride1, data);

    // Copy the data from the vctDynamicConstNArrayRef to the temporary container
    tempContainer.Assign($1);
}


/******************************************************************************
  TYPEMAPS (in, argout) FOR const vctDynamicConstNArrayRef &
******************************************************************************/


%typemap(in) const vctDynamicConstNArrayRef &
{
    /**************************************************************************
    *   %typemap(in) const vctDynamicConstNArrayRef &
    *   Passing a vctDynamicConstNArrayRef by const reference
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    **************************************************************************/

    /*****************************************************************************
     CHECK IF THE PYTHON OBJECT (NAMED `$input') THAT WAS PASSED TO THIS TYPE MAP
     IS A PYARRAY, IS OF NUMPY DTYPE int, AND IS ONE-DIMENSIONAL
    *****************************************************************************/

    if (!(   vctThrowUnlessIsPyArray($input)
          && vctThrowUnlessIsSameTypeArray<$*1_ltype::value_type>($input)
          && vctThrowUnlessDimensionN<$*1_ltype>($input))
        ) {
          return NULL;
    }

    /*****************************************************************************
     CREATE A vctDynamicConstNArrayRef TO POINT TO THE DATA OF THE PYARRAY
    *****************************************************************************/

    // Create the vctDynamicConstNArrayRef

    // dimensions
    const unsigned int ndim = PyArray_NDIM($input);

    // sizes
    npy_intp *_sizes = PyArray_DIMS($input);
    vctFixedSizeConstVectorRef<npy_intp, $*1_ltype::DIMENSION, 1> _sizesRef(_sizes);
    vctFixedSizeVector<unsigned int, $*1_ltype::DIMENSION> sizes;
    sizes.Assign(_sizesRef);

    // strides
    npy_intp *_strides = PyArray_STRIDES($input);
    vctFixedSizeConstVectorRef<npy_intp, $*1_ltype::DIMENSION, 1> _stridesRef(_strides);
    vctFixedSizeVector<int, $*1_ltype::DIMENSION> strides;
    strides.Assign(_stridesRef);
    strides.Divide(sizeof($*1_ltype::value_type));

    // data pointer
    const $*1_ltype::pointer data = reinterpret_cast<$*1_ltype::pointer>(PyArray_DATA($input));

    $1 = new $*1_ltype(data, sizes, strides);
}


%typemap(argout) const vctDynamicConstNArrayRef &
{
    /**************************************************************************
    *   %typemap(argout) const vctDynamicConstNArrayRef &
    *   Passing a vctDynamicConstNArrayRef by const reference
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


/**************************************************************************
*                    Applying Typemaps to Other Types
**************************************************************************/


%define VCT_TYPEMAPS_APPLY_DYNAMIC_NARRAYS_ONE_DIMENSION(elementType, ndim)
%apply vctDynamicNArray         {vctDynamicNArray<elementType, ndim>};
%apply vctDynamicNArray &       {vctDynamicNArray<elementType, ndim> &};
%apply const vctDynamicNArray & {const vctDynamicNArray<elementType, ndim> &};

%apply vctDynamicNArrayRef         {vctDynamicNArrayRef<elementType, ndim>};
%apply const vctDynamicNArrayRef & {const vctDynamicNArrayRef<elementType, ndim> &};

%apply vctDynamicConstNArrayRef         {vctDynamicConstNArrayRef<elementType, ndim>};
%apply const vctDynamicConstNArrayRef & {const vctDynamicConstNArrayRef<elementType, ndim> &};
%enddef


%define VCT_TYPEMAPS_APPLY_DYNAMIC_NARRAYS(elementType)
VCT_TYPEMAPS_APPLY_DYNAMIC_NARRAYS_ONE_DIMENSION(elementType, 3);
VCT_TYPEMAPS_APPLY_DYNAMIC_NARRAYS_ONE_DIMENSION(elementType, 4);
%enddef


VCT_TYPEMAPS_APPLY_DYNAMIC_NARRAYS(int);
