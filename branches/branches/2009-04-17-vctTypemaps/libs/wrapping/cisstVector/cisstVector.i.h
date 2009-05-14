/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Anton Deguet
  Created on: 2005-08-21

  (C) Copyright 2005-2007 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/


/* This file is to be used only for the generation of SWIG wrappers.
   It includes all the regular header files from the libraries as well
   as some header files created only for the wrapping process
   (e.g. vctDynamicMatrixRotation3.h).

   For any wrapper using %import "cisstVector.i", the file
   cisstVector.i.h should be included in the %header %{ ... %} section
   of the interface file. */


#ifndef _cisstVector_i_h
#define _cisstVector_i_h


/* Put header files here */
#include <Python.h>
#include <arrayobject.h>
#include <math.h>
#include <cisstCommon/cmnAssert.h>
#include <cisstVector/vctFixedSizeConstVectorBase.h>
#include <cisstVector/vctDynamicConstVectorBase.h>
#include <cisstVector/vctFixedSizeConstMatrixBase.h>
#include <cisstVector/vctDynamicConstMatrixBase.h>
#include <cisstVector/vctDynamicConstNArrayBase.h>

bool vctThrowUnlessIsPyArray(PyObject * input)
{
    if (!PyArray_Check(input)) {
        PyErr_SetString(PyExc_TypeError, "Object must be a NumPy array");
        return false;
    }
    return true;
}

template <class _elementType>
bool vctThrowUnlessIsSameTypeArray(PyObject * input)
{
    PyErr_SetString(PyExc_ValueError, "Unsupported data type");
    return false;
}

template <>
bool vctThrowUnlessIsSameTypeArray<int>(PyObject * input)
{
    if (PyArray_ObjectType(input, 0) != NPY_INT32) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type int");
        return false;
    }

    return true;
}

template <>
bool vctThrowUnlessIsSameTypeArray<double>(PyObject * input)
{
    if (PyArray_ObjectType(input, 0) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type double");
        return false;
    }

    return true;
}

template <>
bool vctThrowUnlessIsSameTypeArray<unsigned int>(PyObject * input)
{
    if (PyArray_ObjectType(input, 0) != NPY_UINT32) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type unsigned int");
        return false;
    }

    return true;
}

template <class _elementType>
int vctPythonType(void)
{
    return NPY_NOTYPE; // unsupported type
}

template <>
int vctPythonType<int>(void)
{
    return NPY_INT32;
}

template <>
int vctPythonType<double>(void)
{
    return NPY_DOUBLE;
}

template <>
int vctPythonType<unsigned int>(void)
{
    return NPY_UINT32;
}

bool vctThrowUnlessDimension1(PyObject * input)
{
    if (PyArray_NDIM(input) != 1) {
        PyErr_SetString(PyExc_ValueError, "Array must be 1D (vector)");
        return false;
    }

    return true;
}

bool vctThrowUnlessDimension2(PyObject * input)
{
    if (PyArray_NDIM(input) != 2) {
        PyErr_SetString(PyExc_ValueError, "Array must be 2D (matrix)");
        return false;
    }

    return true;
}

template <class _containerType>
bool vctThrowUnlessDimensionN(PyObject * input)
{
    if (PyArray_NDIM(input) != _containerType::DIMENSION) {
        std::stringstream stream;
        stream << "Array must have " << _containerType::DIMENSION << " dimension(s)";
        std::string msg = stream.str();
        PyErr_SetString(PyExc_ValueError, msg.c_str());
        return false;
    }

    return true;
}

bool vctThrowUnlessIsWritable(PyObject *input)
{
    int flags = PyArray_FLAGS(input);
    if(!(flags & NPY_WRITEABLE)) {
        PyErr_SetString(PyExc_ValueError, "Array must be writable");
        return false;
    }
    return true;
}


// TODO: Make sure this is correct for FixedSize and Dynamic Vectors and Matrices
template <unsigned int _size, int _stride, class _elementType, class _dataPtrType>
bool vctThrowUnlessCorrectVectorSize(PyObject *input,
                                     const vctFixedSizeConstVectorBase<_size, _stride, _elementType, _dataPtrType> &target)
{
    unsigned int inputSize = PyArray_DIM(input, 0);
    unsigned int targetSize = target.size();

    if (inputSize != targetSize) {
        std::stringstream stream;
        stream << "Input vector's size must be " << targetSize;
        std::string msg = stream.str();
        PyErr_SetString(PyExc_ValueError, msg.c_str());
        return false;
    }
    return true;
}

template <class _vectorOwnerType, typename _elementType>
bool vctThrowUnlessCorrectVectorSize(const vctDynamicConstVectorBase<_vectorOwnerType, _elementType> & input,
                             unsigned int desiredSize)
{
    return true;
}


template <unsigned int _rows, unsigned int _cols, int _rowStride, int _colStride, class _elementType, class _dataPtrType>
bool vctThrowUnlessCorrectMatrixSize(PyObject *input,
                                     const vctFixedSizeConstMatrixBase<_rows, _cols, _rowStride, _colStride, _elementType, _dataPtrType> &target)
{
    unsigned int inputRows = PyArray_DIM(input, 0);
    unsigned int inputCols = PyArray_DIM(input, 1);
    unsigned int targetRows = target.rows();
    unsigned int targetCols = target.cols();

    if (   inputRows != targetRows
        || inputCols != targetCols) {
        std::stringstream stream;
        stream << "Input matrix's size must be " << targetRows << " rows by " << targetCols << " columns";
        std::string msg = stream.str();
        PyErr_SetString(PyExc_ValueError, msg.c_str());
        return false;
    }
    return true;
}


template <class _vectorOwnerType, typename _elementType>
bool vctThrowUnlessCorrectMatrixSize(const vctDynamicConstMatrixBase<_vectorOwnerType, _elementType> & input,
                                     unsigned int desiredRows, unsigned int desiredCols)
{
    return true;
}



bool vctThrowUnlessOwnsData(PyObject * input)
{
    int flags = PyArray_FLAGS(input);
    if(!(flags & NPY_OWNDATA)) {
        PyErr_SetString(PyExc_ValueError, "Array must own its data");
        return false;
    }
    return true;
}


bool vctThrowUnlessNotReferenced(PyObject *input)
{
    std::cout << "--- THE REFCOUNT IS SET AT 5 ---" << std::endl;
    if (PyArray_REFCOUNT(input) > 5) {      // TODO: what is the correct value to test against?  4?
        PyErr_SetString(PyExc_ValueError, "Array must not be referenced by other objects.  Try making a deep copy of the array and call the function again.");
        return false;
    }
    return true;
}



#endif // _cisstVector_i_h

