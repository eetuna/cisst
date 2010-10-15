/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$
  
  Author(s):	Ofri Sadowsky, Anton Deguet
  Created on: 2004-07-01

  (C) Copyright 2004-2007 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


#ifndef _vctDynamicVectorRefOwner_h
#define _vctDynamicVectorRefOwner_h

/*!
  \file
  \brief Declaration of vctDynamicVectorRefOwner
*/


#include <cisstVector/vctVarStrideVectorIterator.h>

/*! 
  \ingroup cisstVector

  This templated class stores a pointer, a size, and a stride, and
  allows element access, but does not provide any other operations,
  and does not own the data */
template<class _elementType>
class vctDynamicVectorRefOwner
{
public:
    /* define most types from vctContainerTraits */
    VCT_CONTAINER_TRAITS_TYPEDEFS(_elementType);

    enum {TYPE_SIZE = sizeof(value_type)}; 

    /*! The type of this owner. */
    typedef vctDynamicVectorRefOwner<_elementType> ThisType;

    /* iterators are container specific */    
    typedef vctVarStrideVectorConstIterator<value_type> const_iterator;
    typedef vctVarStrideVectorIterator<value_type> iterator;
    typedef vctVarStrideVectorConstIterator<value_type> const_reverse_iterator;
    typedef vctVarStrideVectorIterator<value_type> reverse_iterator;


    vctDynamicVectorRefOwner():
        Size(0),
        ByteStride(TYPE_SIZE),
        Data(0)
    {}

    vctDynamicVectorRefOwner(size_type size, pointer data, stride_type stride = 1):
        Size(size),
        ByteStride(stride * TYPE_SIZE),
        Data(reinterpret_cast<byte_pointer>(data))
    {}

    void SetRef(size_type size, pointer data, stride_type stride = 1)
    {
        this->Size = size;
        this->ByteStride = stride * TYPE_SIZE;
        this->Data = reinterpret_cast<byte_pointer>(data);
    }

    size_type size(void) const {
        return this->Size;
    }

    stride_type stride(void) const {
        return this->ByteStride / TYPE_SIZE;
    }

    stride_type byte_stride(void) const {
        return this->ByteStride;
    }
    
    pointer Pointer(index_type index = 0) {
        return reinterpret_cast<pointer>(this->Data + this->ByteStride * index);
    }

    const_pointer Pointer(index_type index = 0) const {
        return reinterpret_cast<pointer>(this->Data + this->ByteStride * index);
    }
    
    const_iterator begin(void) const {
        return const_iterator(reinterpret_cast<const_pointer>(this->Data),
                              this->stride());
    }
     
    const_iterator end(void) const {
        return const_iterator(reinterpret_cast<const_pointer>(this->Data + this->Size * this->ByteStride),
                              this->stride());
    }

    iterator begin(void) {
        return iterator(reinterpret_cast<pointer>(this->Data), this->stride());
    }
     
    iterator end(void) {
        return iterator(reinterpret_cast<pointer>(this->Data + this->Size * this->ByteStride),
                        this->stride());
    }

    const_reverse_iterator rbegin(void) const {
        return const_reverse_iterator(reinterpret_cast<const_pointer>(this->Data + (this->Size-1) * this->ByteStride),
                                      -this->stride());
    }
     
    const_reverse_iterator rend(void) const {
        return const_reverse_iterator(reinterpret_cast<const_pointer>(this->Data - this->ByteStride),
                                      -this->stride());
    }

    reverse_iterator rbegin(void) {
        return reverse_iterator(reinterpret_cast<pointer>(this->Data + (this->Size-1) * this->ByteStride),
                                -this->stride());
    }
     
    reverse_iterator rend(void) {
        return reverse_iterator(reinterpret_cast<pointer>(this->Data - this->ByteStride),
                                -this->stride());
    }

protected:
    size_type Size;
    stride_type ByteStride;
    byte_pointer Data;
    
private:
    // copy constructor private to prevent any call
    vctDynamicVectorRefOwner(const ThisType & other) {};

};


#endif // _vctDynamicVectorRefOwner_h

