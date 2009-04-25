/******************************************************************************
 Authors: Daniel Li, Mitch Williams
******************************************************************************/

/*****************************************************************************
 PLACEHOLDER STRINGS TO LOOK FOR:

   PythonException      Placeholder for a Python exception message
   @mystery@            Bookmark for strange code behavior
   TODO                 todo
*****************************************************************************/

%header %{
#include <Python.h>
#include <arrayobject.h>
#include <math.h>
%}

#define CISST_EXPORT
#define CISST_DEPRECATED

%include "std_streambuf.i"
%include "std_iostream.i"

%rename(__str__) ToString;


%init %{
        import_array()   /* Initial function for NumPy */
%}

%header %{

    #include <cisstCommon/cmnAssert.h>
    #include <cisstVector/vctFixedSizeConstVectorBase.h>
    #include <cisstVector/vctDynamicConstVectorBase.h>

    bool vctPythonTestIsPyArray(PyObject * input)
    {
        if (!PyArray_Check(input)) {
            PyErr_SetString(PyExc_TypeError, "Not a NumPy array");
            return false;
        }
        return true;
    }

    template <class _elementType>
    bool vctPythonTestIsSameTypeArray(PyObject * input)
    {
        PyErr_SetString(PyExc_ValueError, "Unsupported data type");
        return false;
    }

    template <>
    bool vctPythonTestIsSameTypeArray<int>(PyObject * input)
    {
        if (PyArray_ObjectType(input, 0) != NPY_INT32) {
            PyErr_SetString(PyExc_ValueError, "Array must be of type int");
            return false;
        }

        return true;
    }

    template <class _elementType>
    int vctPythonTestPythonType(void)
    {
        return NPY_NOTYPE; // unsupported type
    }

    template <>
    int vctPythonTestPythonType<int>(void)
    {
        return NPY_INT32;
    }

    bool vctPythonTestIs1DArray(PyObject * input)
    {
        if (PyArray_NDIM(input) != 1) {
            PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
            return false;
        }

        return true;
    }

    bool vctPythonTestIs2DArray(PyObject * input)
    {
        if (PyArray_NDIM(input) != 2) {
            PyErr_SetString(PyExc_ValueError, "Array must be of dimension two (matrix)");
            return false;
        }
        return true;
    }

    template <unsigned int _size, int _stride, class _elementType, class _dataPtrType>
    bool vctPythonTestVectorSize(const vctFixedSizeConstVectorBase<_size, _stride, _elementType, _dataPtrType> & input,
                                 unsigned int desiredSize)
    {
        if (input.size() != desiredSize) {
            PyErr_SetString(PyExc_ValueError, "Input vector's size must match fixed size's one");
            return false;
        }
        return true;
    }

    template <class _vectorOwnerType, typename _elementType>
    bool vctPythonTestVectorSize(const vctDynamicConstVectorBase<_vectorOwnerType, _elementType> & input,
                                 unsigned int desiredSize)
    {
        return true;
    }


    bool vctPythonTestIsWritable(PyObject *input)
    {
        int flags = PyArray_FLAGS(input);
        if(!(flags & NPY_WRITEABLE)) {
            PyErr_SetString(PyExc_ValueError, "Array must be writable");
            return false;
        }
        return true;
    }


    bool vctPythonTestOwnsData(PyObject * input)
    {
        int flags = PyArray_FLAGS(input);
        if(!(flags & NPY_OWNDATA)){
            PyErr_SetString(PyExc_ValueError, "Array must own its data");
            return false;
        }
        return true;
    }


    bool vctPythonTestNotReferenced(PyObject *input)
    {
        if (PyArray_REFCOUNT(input) > 5) {      // TODO: what is the correct value to test against?  4?
            PyErr_SetString(PyExc_ValueError, "You have tried to resize the array.  The array must not be referenced by other objects.  Try making a deep copy of the array and call the function again.");
            return false;
        }
        return true;
    }
%}


/******************************************************************************
  TYPEMAPS (in, out) FOR vctDynamicVector
******************************************************************************/


%typemap(in) vctDynamicVector
{
    /*****************************************************************************
    *   %typemap(in) vctDynamicVector
    *   Passing a vctDynamicVector by copy
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    *****************************************************************************/

    /*****************************************************************************
     CHECK IF THE PYTHON OBJECT (NAMED `$input') THAT WAS PASSED TO THIS TYPE MAP
     IS A PYARRAY, IS OF NUMPY DTYPE int, AND IS ONE-DIMENSIONAL
    *****************************************************************************/

    if (!(vctPythonTestIsPyArray($input)
          && vctPythonTestIsSameTypeArray<$1_ltype::value_type>($input)
          && vctPythonTestIs1DArray($input))
        ) {
          // PyErr_SetString(PyExc_TypeError, "PythonException");
          return NULL;
    }

    /*****************************************************************************
     COPY THE DATA OF THE PYARRAY (NAMED `$input') TO THE vctDynamicVector
    *****************************************************************************/

    // Create a temporary vctDynamicVectorRef container
    const npy_intp size = PyArray_DIM($input, 0);
    const npy_intp stride = PyArray_STRIDE($input, 0) /
                                sizeof($1_ltype::value_type);
    const $1_ltype::pointer data =
        reinterpret_cast<$1_ltype::pointer>(PyArray_DATA($input));
    const vctDynamicVectorRef<$1_ltype::value_type> tempContainer =
        vctDynamicVectorRef<$1_ltype::value_type>(size, data, stride);

    // Copy the data from the temporary container to the vctDynamicVector
    $1 = tempContainer;
}


%typemap(out) vctDynamicVector
{
    /* Return vector by copy
       Using: %typemap(out) vctDynamicVector
     */

    //Create a new PyArray and set its size
    npy_intp* sizes = PyDimMem_NEW(1);
    sizes[0] = $1.size();

    $result = PyArray_SimpleNew(1, sizes, vctPythonTestPythonType<$1_ltype::value_type>());

    // copy data returned by C function into new PyArray
    memcpy(PyArray_DATA($result), $1.Pointer(), $1.size() * sizeof($1_ltype::value_type) );
}


/******************************************************************************
  TYPEMAPS (in, argout, out) FOR vctDynamicVector &
******************************************************************************/


%typemap(in) vctDynamicVector &
{
    /*****************************************************************************
    *   %typemap(in) vctDynamicVector &
    *   Passing a vctDynamicVector by reference
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this typemap.
    *****************************************************************************/

    /*****************************************************************************
     CHECK IF THE PYTHON OBJECT (NAMED `$input') THAT WAS PASSED TO THIS TYPE MAP
     IS A PYARRAY, IS OF NUMPY DTYPE int, IS ONE-DIMENSIONAL, AND IS WRITABLE
    *****************************************************************************/

    if (!(   vctPythonTestIsPyArray($input)
          && vctPythonTestIsSameTypeArray<$*1_ltype::value_type>($input)
          && vctPythonTestIs1DArray($input)
          && vctPythonTestIsWritable($input)
          && vctPythonTestOwnsData($input)
          && vctPythonTestNotReferenced($input))
        ) {
          // PyErr_SetString(PyExc_TypeError, "PythonException");
          return NULL;
    }

    /*****************************************************************************
     COPY THE DATA OF THE PYARRAY (NAMED `$input') TO THE vctDynamicVector
    *****************************************************************************/

    // TODO: Since the PyArray is guaranteed to be contiguous, should we use memcpy instead
    // of copying using a vctDynamicVectorRef?

    // Create a temporary vctDynamicVectorRef container
    const npy_intp size = PyArray_DIM($input, 0);
    const npy_intp stride = PyArray_STRIDE($input, 0) /
                                sizeof($*1_ltype::value_type);
    const $*1_ltype::pointer data =
        reinterpret_cast<$*1_ltype::pointer>(PyArray_DATA($input));
    const vctDynamicVectorRef<$*1_ltype::value_type> tempContainer =
        vctDynamicVectorRef<$*1_ltype::value_type>(size, data, stride);

    // Create the vctDynamicVector
    $1 = new $*1_ltype(tempContainer);
}


%typemap(argout) vctDynamicVector &
{
    /*****************************************************************************
    *   %typemap(argout) vctDynamicVector &
    *   Passing a vctDynamicVector by reference
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    *****************************************************************************/

    /*************************************************************************
     CHECK IF THE CONTAINER HAS BEEN RESIZED
    *************************************************************************/

    const $*1_ltype::size_type input_size = PyArray_DIM($input, 0);
    const $*1_ltype::size_type output_size = $1->size();

    if (input_size != output_size) {
        // Resize the PyArray by:
        //  1)  Creating an array containing the new size
        //  2)  Pass that array to the resizing function given by NumPy API

        npy_intp *sizes = PyDimMem_NEW(1);              // create an array of sizes; dimension 1
        sizes[0] = output_size;                         // set the size
        PyArray_Dims dims;                              // create a PyArray_Dims object to hand to PyArray_Resize
        dims.ptr = sizes;
        dims.len = 1;
        PyArray_Resize((PyArrayObject *) $input, &dims, 0, NPY_CORDER);   // resize the PyArray
                                                                          // @mystery@
                                                                          // Why does setting the third parameter to be 1
                                                                          // result in Python errors during unit testing?
                                                                          // I've already checked that there are no extraneous
                                                                          // references, but the Python errors are telling me
                                                                          // otherwise!
    }

    /*************************************************************************
     COPY THE DATA TO THE PYARRAY
    *************************************************************************/

    // TODO: Since the PyArray is guaranteed to be contiguous, should we use memcpy instead
    // of copying using a vctDynamicVectorRef?

    // Create a temporary vctDynamicVectorRef container
    const npy_intp size = PyArray_DIM($input, 0);
    const npy_intp stride = PyArray_STRIDE($input, 0) /
                                sizeof($*1_ltype::value_type);
    const $*1_ltype::pointer data =
        reinterpret_cast<$*1_ltype::pointer>(PyArray_DATA($input));
    vctDynamicVectorRef<$*1_ltype::value_type> tempContainer =
        vctDynamicVectorRef<$*1_ltype::value_type>(size, data, stride);

    // Copy the data from the temporary container to the vctDynamicVector
    tempContainer.Assign($1->Pointer());    // @mystery@
                                            // For some reason, .Assign(*($1))
                                            // results in the unit tests crashing,
                                            // so we use .Assign($1->Pointer()) instead

    /*************************************************************************
     CLEAN UP
    *************************************************************************/

    // Don't forget to free the memory we allocated in the `in' typemap
    delete $1;
}


%typemap(out) vctDynamicVector &
{
    /* Return vector by reference
       Using: %typemap(out) vctDynamicVector &
     */

    //Create new size array and set size
    npy_intp* sizeOfReturnedVector = PyDimMem_NEW(1);
    sizeOfReturnedVector[0] = $1->size();

    //create a new PyArray from the reference returned by the C function
    $result = PyArray_SimpleNewFromData(1, sizeOfReturnedVector, vctPythonTestPythonType<$*1_ltype::value_type>(),  $1->Pointer() );
}


/******************************************************************************
  TYPEMAPS (in, argout, out) FOR const vctDynamicVector &
******************************************************************************/


%typemap(in) const vctDynamicVector &
{
    /*****************************************************************************
    *   %typemap(in) const vctDynamicVector &
    *   Passing a vctDynamicVector by const &
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    *****************************************************************************/

    /*****************************************************************************
     CHECK IF THE PYTHON OBJECT (NAMED `$input') THAT WAS PASSED TO THIS TYPE MAP
     IS A PYARRAY, IS OF NUMPY DTYPE int, AND IS ONE-DIMENSIONAL
    *****************************************************************************/

    if (!(vctPythonTestIsPyArray($input)
          && vctPythonTestIsSameTypeArray<$*1_ltype::value_type>($input)
          && vctPythonTestIs1DArray($input))
        ) {
          // PyErr_SetString(PyExc_TypeError, "PythonException");
          return NULL;
    }

    /*****************************************************************************
     COPY THE DATA OF THE PYARRAY (NAMED `$input') TO THE vctDynamicVector
    *****************************************************************************/

    // Create a temporary vctDynamicVectorRef container
    const npy_intp size = PyArray_DIM($input, 0);
    const npy_intp stride = PyArray_STRIDE($input, 0) /
                                sizeof($*1_ltype::value_type);
    const $*1_ltype::pointer data =
        reinterpret_cast<$*1_ltype::pointer>(PyArray_DATA($input));
    const vctDynamicVectorRef<$*1_ltype::value_type> tempContainer =
        vctDynamicVectorRef<$*1_ltype::value_type>(size, data, stride);

    // Create the vctDynamicVector
    $1 = new $*1_ltype(tempContainer);
}


%typemap(argout) const vctDynamicVector &
{
    /**************************************************************************
    *   %typemap(argout) const vctDynamicVector &
    *   Passing a vctDynamicVector by const reference
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


%typemap(out) const vctDynamicVector &
{
    /* Return vector by const reference
       Using: %typemap(out) const vctDynamicVector &
     */

    /* To imitate const functionality, set the writable flag to false */

    //Create new size array and set size
    npy_intp* sizes = PyDimMem_NEW(1);
    sizes[0] = $1->size();

    //NPY_CARRAY_RO = set flags for a C Array that is Read Only (i.e. const)
    $result = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(vctPythonTestPythonType<$*1_ltype::value_type>()), 1, sizes, NULL, $1->Pointer(), NPY_CARRAY_RO, NULL);
}


/******************************************************************************
  TYPEMAPS (in, out) FOR vctDynamicVectorRef
******************************************************************************/


%typemap(in) vctDynamicVectorRef
{
    /*************************************************************************
    *   %typemap(in) vctDynamicVectorRef
    *   Passing a vctDynamicVectorRef by copy
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic
    *   behind this typemap.
    *************************************************************************/

    /*************************************************************************
     CHECK IF THE PYTHON OBJECT (NAMED `$input') THAT WAS PASSED TO THIS
     TYPEMAP IS A PYARRAY, IS OF NUMPY DTYPE int, IS ONE-DIMENSIONAL, AND
     IS WRITABLE
    *************************************************************************/

    if (!(vctPythonTestIsPyArray($input)
          && vctPythonTestIsSameTypeArray<$1_ltype::value_type>($input)
          && vctPythonTestIs1DArray($input)
          && vctPythonTestIsWritable($input))
        ) {
          // PyErr_SetString(PyExc_TypeError, "PythonException");
          return NULL;
    }

    /*************************************************************************
     SET THE SIZE, STRIDE AND DATA POINTER OF THE vctDynamicVectorRef
     OBJECT (NAMED `$1') TO MATCH THAT OF THE PYARRAY (NAMED `$input')
    *************************************************************************/

    const npy_intp size = PyArray_DIM($input, 0);
    const npy_intp stride = PyArray_STRIDE($input, 0) /
                                sizeof($1_ltype::value_type);
    const $1_ltype::pointer data =
        reinterpret_cast<$1_ltype::pointer>(PyArray_DATA($input));

    $1.SetRef(size, data, stride);
}


%typemap(out) vctDynamicVectorRef
{
    /*****************************************************************************
    *   %typemap(out) vctDynamicVectorRef
    *   Returning a vctDynamicVectorRef
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    *****************************************************************************/

    /*****************************************************************************
     CREATE A NEW PYARRAY OBJECT
    *****************************************************************************/

    npy_intp *sizes = PyDimMem_NEW(1);
    sizes[0] = $1.size();
    $result = PyArray_SimpleNew(1, sizes, vctPythonTestPythonType<$1_ltype::value_type>());  // TODO: clean this up

    /*****************************************************************************
     COPY THE DATA FROM THE vctDynamicVectorRef TO THE PYARRAY
    *****************************************************************************/

    // Create a temporary vctDynamicVectorRef container
    const npy_intp size = PyArray_DIM($result, 0);
    const npy_intp stride = PyArray_STRIDE($result, 0) /
                                sizeof($1_ltype::value_type);
    const $1_ltype::pointer data =
        reinterpret_cast<$1_ltype::pointer>(PyArray_DATA($result));
    const vctDynamicVectorRef<$1_ltype::value_type> tempContainer =
        vctDynamicVectorRef<$1_ltype::value_type>(size, data, stride);

    // Copy the data from the vctDynamicVectorRef to the temporary container
    tempContainer = $1;
}


/******************************************************************************
  TYPEMAPS (in, argout) FOR const vctDynamicVectorRef &
******************************************************************************/


%typemap(in) const vctDynamicVectorRef &
{
    /**************************************************************************
    *   %typemap(in) const vctDynamicVectorRef &
    *   Passing a vctDynamicVectorRef by const reference
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    **************************************************************************/

    /*****************************************************************************
     CHECK IF THE PYTHON OBJECT (NAMED `$input') THAT WAS PASSED TO THIS TYPE MAP
     IS A PYARRAY, IS OF NUMPY DTYPE int, AND IS ONE-DIMENSIONAL
    *****************************************************************************/

    if (!(vctPythonTestIsPyArray($input)
          && vctPythonTestIsSameTypeArray<$*1_ltype::value_type>($input)
          && vctPythonTestIs1DArray($input))
        ) {
          // PyErr_SetString(PyExc_TypeError, "PythonException");
          return NULL;
    }

    /*****************************************************************************
     CREATE A vctDynamicVectorRef TO POINT TO THE DATA OF THE PYARRAY
    *****************************************************************************/

    // Create the vctDynamicVectorRef
    const npy_intp size = PyArray_DIM($input, 0);
    const npy_intp stride = PyArray_STRIDE($input, 0) /
                                sizeof($*1_ltype::value_type);
    const $*1_ltype::pointer data =
        reinterpret_cast<$*1_ltype::pointer>(PyArray_DATA($input));
    $1 = new $*1_ltype(size, data, stride);
}


%typemap(argout) const vctDynamicVectorRef &
{
    /**************************************************************************
    *   %typemap(argout) const vctDynamicVectorRef &
    *   Passing a vctDynamicVectorRef by const reference
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
  TYPEMAPS (in, out) FOR vctDynamicConstVectorRef
******************************************************************************/


%typemap(in) vctDynamicConstVectorRef
{
    /*****************************************************************************
    *   %typemap(in) vctDynamicConstVectorRef
    *   Passing a vctDynamicConstVectorRef by copy
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    *****************************************************************************/

    /*****************************************************************************
     CHECK IF THE PYTHON OBJECT (NAMED `$input') THAT WAS PASSED TO THIS TYPE MAP
     IS A PYARRAY, IS OF NUMPY DTYPE int, AND IS ONE-DIMENSIONAL
    *****************************************************************************/

    if (!(vctPythonTestIsPyArray($input)
          && vctPythonTestIsSameTypeArray<$1_ltype::value_type>($input)
          && vctPythonTestIs1DArray($input))
        ) {
          // PyErr_SetString(PyExc_TypeError, "PythonException");
          return NULL;
    }

    /*****************************************************************************
     SET THE SIZE, STRIDE, AND DATA POINTER OF THE
     vctDynamicConstVectorRef OBJECT (NAMED `$1') TO MATCH THAT OF THE
     PYARRAY (NAMED `$input')
    *****************************************************************************/

    const npy_intp size = PyArray_DIM($input, 0);
    const npy_intp stride = PyArray_STRIDE($input, 0) /
                                sizeof($1_ltype::value_type);
    const $1_ltype::pointer data =
        reinterpret_cast<$1_ltype::pointer>(PyArray_DATA($input));

    $1.SetRef(size, data, stride);
}


%typemap(out) vctDynamicConstVectorRef
{
    /*****************************************************************************
    *   %typemap(out) vctDynamicConstVectorRef
    *   Returning a vctDynamicConstVectorRef
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    *****************************************************************************/

    /*****************************************************************************
     CREATE A NEW PYARRAY OBJECT
    *****************************************************************************/

    npy_intp *sizes = PyDimMem_NEW(1);
    sizes[0] = $1.size();
    $result = PyArray_SimpleNew(1, sizes, vctPythonTestPythonType<$1_ltype::value_type>());  // TODO: clean this up

    /*****************************************************************************
     COPY THE DATA FROM THE vctDynamicConstVectorRef TO THE PYARRAY
    *****************************************************************************/

    // Create a temporary vctDynamicVectorRef container
    const npy_intp size = PyArray_DIM($result, 0);
    const npy_intp stride = PyArray_STRIDE($result, 0) /
                                sizeof($1_ltype::value_type);
    const $1_ltype::pointer data =
        reinterpret_cast<$1_ltype::pointer>(PyArray_DATA($result));
    const vctDynamicVectorRef<$1_ltype::value_type> tempContainer =
        vctDynamicVectorRef<$1_ltype::value_type>(size, data, stride);

    // Copy the data from the vctDynamicConstVectorRef to the temporary container
    tempContainer = $1;
}


/******************************************************************************
  TYPEMAPS (in, argout) FOR const vctDynamicConstVectorRef &
******************************************************************************/


%typemap(in) const vctDynamicConstVectorRef &
{
    /**************************************************************************
    *   %typemap(in) const vctDynamicConstVectorRef &
    *   Passing a vctDynamicConstVectorRef by const reference
    *
    *   See the documentation ``Developer's Guide to Writing Typemaps'' for documentation on the logic behind
    *   this type map.
    **************************************************************************/

    /*****************************************************************************
     CHECK IF THE PYTHON OBJECT (NAMED `$input') THAT WAS PASSED TO THIS TYPE MAP
     IS A PYARRAY, IS OF NUMPY DTYPE int, AND IS ONE-DIMENSIONAL
    *****************************************************************************/

    if (!(vctPythonTestIsPyArray($input)
          && vctPythonTestIsSameTypeArray<$*1_ltype::value_type>($input)
          && vctPythonTestIs1DArray($input))
        ) {
          // PyErr_SetString(PyExc_TypeError, "PythonException");
          return NULL;
    }

    /*****************************************************************************
     CREATE A vctDynamicConstVectorRef TO POINT TO THE DATA OF THE PYARRAY
    *****************************************************************************/

    // Create the vctDynamicConstVectorRef
    const npy_intp size = PyArray_DIM($input, 0);
    const npy_intp stride = PyArray_STRIDE($input, 0) /
                                sizeof($*1_ltype::value_type);
    const $*1_ltype::pointer data =
        reinterpret_cast<$*1_ltype::pointer>(PyArray_DATA($input));

    $1 = new $*1_ltype(size, data, stride);
}


%typemap(argout) const vctDynamicConstVectorRef &
{
    /**************************************************************************
    *   %typemap(argout) const vctDynamicConstVectorRef &
    *   Passing a vctDynamicConstVectorRef by const reference
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


%define VCT_TYPEMAPS_APPLY_FIXED_SIZE_VECTORS_ONE_SIZE(elementType, size)
%apply vctDynamicVector {vctFixedSizeVector<elementType, size>};
%apply vctDynamicVector & {vctFixedSizeVector<elementType, size> &};
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

%define VCT_TYPEMAPS_APPLY_DYNAMIC_VECTORS(elementType)
%apply vctDynamicVector         {vctDynamicVector<elementType>};
%apply vctDynamicVector &       {vctDynamicVector<elementType> &};
%apply const vctDynamicVector & {const vctDynamicVector<elementType> &};

%apply vctDynamicVectorRef         {vctDynamicVectorRef<elementType>};
%apply const vctDynamicVectorRef & {const vctDynamicVectorRef<elementType> &};

%apply vctDynamicConstVectorRef         {vctDynamicConstVectorRef<elementType>};
%apply const vctDynamicConstVectorRef & {const vctDynamicConstVectorRef<elementType> &};
%enddef

%import <cisstVector/vctFixedSizeVectorTypes.h>
%import <cisstVector/vctDynamicVectorTypes.h>

VCT_TYPEMAPS_APPLY_DYNAMIC_VECTORS(int);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_VECTORS(int);
VCT_TYPEMAPS_APPLY_DYNAMIC_VECTORS(double);
VCT_TYPEMAPS_APPLY_FIXED_SIZE_VECTORS(double);