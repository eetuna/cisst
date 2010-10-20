/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):	Daniel Li
  Created on:	2006-07-05

  (C) Copyright 2006-2007 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#pragma once
#ifndef _vctDynamicNArrayLoopEngines_h
#define _vctDynamicNArrayLoopEngines_h

/*!
  \file
  \brief Declaration of vctDynamicNArrayLoopEngines
 */

#include <cisstCommon/cmnPortability.h>
#include <cisstCommon/cmnThrow.h>

#include <cisstVector/vctFixedSizeVector.h>
#include <cisstVector/vctContainerTraits.h>
#include <cisstVector/vctDynamicCompactLoopEngines.h>

/*!
  We temporarily define two compilation control flags:
    * OFRI_NEW_NARRY_LOOP_CONTROL set to 1 uses the loop control mechanism
      with metaprogramming recursion. Set to 0 it uses the old, iteration-basd
      mechanism.
    * CHAR_TARGET set to 1 turns the "current" and "target" pointers in the
      loop control mechansim to "char *", which may or may not be faster than
      using native "_elementType *" pointers.  Set to 0, the mechanism uses
      native pointers.  So far, the native pointers seem slightly faster (for
      non-compact arrays) than the char pointers.

  Notice the #undef of the flags at the end of this file.  When finalizing the
  version, reove the both #defines and #undefs.
*/
#define OFRI_NEW_NARRY_LOOP_CONTROL 1
#if OFRI_NEW_NARRY_LOOP_CONTROL
#  define CHAR_TARGET 0
#endif

#if OFRI_NEW_NARRY_LOOP_CONTROL
/*! The vctNArrayLoopControl class wraps the loop control mechanism, which
  is defined as static methods.  When OFRI_NEW_NARRY_LOOP_CONTROL is 0,
  equivalent static functions are defined within the scope of
  vctDynamicNArrayLoopEngines.

  The class is not essential for the functioning of the mechanism.  It is
  useful for comparing the performance of native vs. char target pointers,
  as the target type is passed as a template parameter. 

  Notice that when the mechanism is used with input-only targets (as in SoNi,
  for example) the target type must be a *const* value, as the input object
  can only provide a const element pointer.
*/
template<vct::size_type _dimension, class _targetType>
class vctNArrayLoopControl
{
public:
    /* define types */
    typedef vct::size_type size_type;
    typedef vct::stride_type stride_type;
    typedef vct::difference_type difference_type;
    typedef vct::index_type index_type;

    VCT_NARRAY_TRAITS_TYPEDEFS(_dimension);

	typedef _targetType * target_ptr_type;
	typedef const _targetType * ctarget_ptr_type;
	typedef vctFixedSizeVector<target_ptr_type, _dimension> target_vec_type;

	/*! Helper function to calculate the initial targets.  Notice that
     this function uses a return value rather than referenced output,
     which is how the "old" mechanism worked.
    */
    static target_vec_type InitializeTargets(const nsize_type & sizes,
		const nstride_type & strides, target_ptr_type basePtr)
    {
		nstride_type offsets(sizes);
		offsets.ElementwiseMultiply(strides);
#if !CHAR_TARGET
		const stride_type outputElementStrideForCharPtr = (stride_type)( ((target_ptr_type)(0)) + 1 );
		offsets.Multiply(outputElementStrideForCharPtr);
#endif
		offsets.Add( (stride_type)(basePtr) );
		return target_vec_type(offsets);
    }

    /*! Helper function to calculate the strides to next dimension. */
	static nstride_type CalculateDimWrapCharStrides(const nsize_type & sizes, const nstride_type & strides)
	{
		nstride_type result(0);
		nstride_type::ConstSubvector<_dimension-1>::Type suffixStrides(strides.Pointer(1));
		nstride_type sizesAsStrides(sizes);
		nstride_type::Subvector<_dimension-1>::Type suffixSizes(sizesAsStrides.Pointer(1));
		nstride_type::Subvector<_dimension-1>::Type prefixResult(result.Pointer(0));
		prefixResult.ElementwiseProductOf(suffixSizes, suffixStrides);
		result.NegationSelf();
		result.Add(strides);

		return result;
	}

    /*! Helper function to calculate the pointer offsets to the next dimension. */
    inline static nstride_type CalculateOTND(const nstride_type & stnd)
    {
		nstride_type result(0);
		int i = _dimension-1;
		stride_type curOfs = 0;
		for (; i >= 0; --i)
		{
			curOfs += stnd[i];
			result[i] = curOfs;
		}
		return result;
	}

    /*! Helper function to synchronize the given nArray's current pointer
      with the master nArray's current pointer. */
    inline static ctarget_ptr_type SyncCurrentPointer(ctarget_ptr_type currentPointer,
		const nstride_type & otnd, difference_type numberOfWrappedDimensions)
    {
		if (numberOfWrappedDimensions < _dimension)
			return currentPointer + otnd[_dimension - numberOfWrappedDimensions - 1];
		return currentPointer;
    }

    /*! Helper function to increment the current pointer and any necessary
      target pointers. */
	// Consider the following structure.
	//
	// initial target setup :    // performed before engine loop is begun, outside of this function
	//       currPtr = base;
	//       for each k
	//           target[k] = currPtr + size[k] * stride[k];  // This should be invariant after every dimension wrap.  See InitializeTargets(...)
	//           stnd[k] = stride[k-1] - size[k] * stride[k];   // This is constant through the run.  See CalculateDimWrapCharStrides(...)
	//
	// wrap(nw)  :      // nw is the number of dimensions wrapped
	//       currPtr += sum_{i =(d-nw)}^{d-1}( stnd[i] );  // obsolete if currPtr already determined; see below
	//       for k = (d-nw) to (d-1)
	//           target[k] = currPtr + size[k] * stride[k];
	//
	// check_wrap :     // this updates currentPtr (see above) and finds out nw (number of wrapped dimensions)
	//       k = d-1;
	//       nw = 0;
	//       currPtr += stride[k];
	//       while (currPtr == target[k]) {
	//           currPtr += stnd[k];
	//           --k;
	//           ++nw;
	//       }
	//       return nw;
	//
	// NOTE:  The invariant condition above and the initialization of stnd leads to
	//       size[k] * stride[k] = stride[k-1] - stnd[k];
	// We take advantage of this as we update the targets.
	//
	// Below is a "recursive" implementation of the check_wrap operation, combined with target update
	template<int _depth>
	inline static dimension_type IncrementPointers(
		target_vec_type & targets, target_ptr_type & currentPointer,
		const nstride_type & strides, const nstride_type & stnd)
	{
		enum {index = _depth - 1};
		currentPointer += stnd[index];
		if (currentPointer != targets[index])
			return _dimension - _depth;
		dimension_type numberOfWrappedDimensions =
			IncrementPointers<_depth-1>(targets, currentPointer, strides, stnd);
		targets[index] = currentPointer + strides[index-1] - stnd[index-1];
		return numberOfWrappedDimensions;
	}

	template<>
	inline static dimension_type IncrementPointers<1>(
		target_vec_type & targets, target_ptr_type & currentPointer,
		const nstride_type & strides, const nstride_type & stnd)
	{
		enum {index = 0};
		currentPointer += stnd[index];
		if (currentPointer != targets[index])
			return _dimension - 1;
		return _dimension;
	}

	template<>
	inline static dimension_type IncrementPointers<0>(
		target_vec_type & targets, target_ptr_type & currentPointer,
		const nstride_type & strides, const nstride_type & stnd)
	{
		return 0;
	}

};
#endif  // OFRI_NEW_NARRY_LOOP_CONTROL


/*!
  \brief Container class for the dynamic nArray engines.

  \sa SoNi SoNiNi SoNiSi NoNiNi NoNiSi NoSiNi NioSi NioNi NoNi Nio NioSiNi MinAndMax
*/
template <vct::size_type _dimension>
class vctDynamicNArrayLoopEngines
{
public:
    /* define types */
    typedef vct::size_type size_type;
    typedef vct::stride_type stride_type;
    typedef vct::difference_type difference_type;
    typedef vct::index_type index_type;

    VCT_NARRAY_TRAITS_TYPEDEFS(_dimension);

    /*! Helper function to throw an exception whenever sizes mismatch.
      This enforces that a standard message is sent. */
    inline static void ThrowSizeMismatchException(void) throw(std::runtime_error)
    {
        cmnThrow(std::runtime_error("vctDynamicNArrayLoopEngines: Sizes of nArrays don't match"));
    }

#if !OFRI_NEW_NARRY_LOOP_CONTROL
    /*! Helper function to calculate the strides to next dimension. */
    inline static void CalculateSTND(nstride_type & stnd,
                                     const nsize_type & sizes,
                                     const nstride_type & strides)
    {
        // set up iterators
        typename nsize_type::const_iterator sizesIter = sizes.begin();
        typename nstride_type::const_iterator stridesIter = strides.begin();
        typename nstride_type::iterator stndIter = stnd.begin();
        const typename nstride_type::const_iterator stndIterEnd = stnd.end();

        *stndIter = 0;
        ++sizesIter;
        ++stridesIter;
        ++stndIter;

        stride_type skippedStrides;
        for (;
             stndIter != stndIterEnd;
             ++sizesIter, ++stridesIter, ++stndIter)
        {
            skippedStrides = static_cast<stride_type>(*sizesIter) * (*stridesIter);
            *stndIter = *(stridesIter-1) - skippedStrides;
        }
    }


    /*! Helper function to calculate the pointer offsets to the next dimension. */
    inline static void CalculateOTND(nstride_type & otnd,
                                     const nstride_type & strides,
                                     const nstride_type & stnd)
    {
        // set up iterators
        stride_type previousOTND = *(strides.rbegin());
        typename nstride_type::const_reverse_iterator stndIter = stnd.rbegin();
        typename nstride_type::reverse_iterator otndIter = otnd.rbegin();
        const typename nstride_type::const_reverse_iterator otnd_rend = otnd.rend();

        *otndIter = previousOTND;
        ++otndIter;

        for (;
             otndIter != otnd_rend;
             ++otndIter, ++stndIter)
        {
            *otndIter = *stndIter + previousOTND;
            previousOTND = *otndIter;
        }
    }


    /*! Helper function to calculate the initial targets. */
    template <class _elementType>
    inline static void InitializeTargets(vctFixedSizeVector<const _elementType *, _dimension> & targets,
                                         const nsize_type & sizes,
                                         const nstride_type & strides,
                                         const _elementType * basePtr)
    {
        typedef _elementType value_type;

        // set up iterators
        typename nsize_type::const_iterator sizesIter = sizes.begin();
        typename nstride_type::const_iterator stridesIter = strides.begin();
        typename vctFixedSizeVector<const value_type *, _dimension>::iterator targetsIter = targets.begin();
        const typename vctFixedSizeVector<const value_type *, _dimension>::const_iterator targetsIterEnd = targets.end();

        stride_type offset;
        for (;
             targetsIter != targetsIterEnd;
             ++targetsIter, ++sizesIter, ++stridesIter)
        {
            offset = static_cast<stride_type>(*sizesIter) * (*stridesIter);
            *targetsIter = basePtr + offset;
        }
    }


    /*! Helper function to synchronize the given nArray's current pointer
      with the master nArray's current pointer. */
    template <class _elementType>
    inline static void SyncCurrentPointer(const _elementType * & currentPointer,
                                          const nstride_type & otnd,
                                          difference_type numberOfWrappedDimensions)
    {
        const typename nstride_type::const_reverse_iterator otndBegin = otnd.rbegin();
        currentPointer += otndBegin[numberOfWrappedDimensions];
    }


    /*! Helper function to increment the current pointer and any necessary
      target pointers. */
    template <class _elementType, class _pointerType>
    inline static dimension_type IncrementPointers(vctFixedSizeVector<const _elementType *, _dimension> & targets,
                                                    _pointerType & currentPointer,
                                                    const nstride_type & strides,
                                                    const nstride_type & stnd)
    {
        typedef _elementType value_type;

        // set up iterators
        typename vctFixedSizeVector<const value_type *, _dimension>::reverse_iterator targetsIter = targets.rbegin();
        const typename vctFixedSizeVector<const value_type *, _dimension>::const_reverse_iterator targets_rbeg = targets.rbegin();
        // typename vctFixedSizeVector<const value_type *, _dimension>::reverse_iterator targets_innerIter;
        typename nstride_type::const_reverse_iterator stridesIter = strides.rbegin();
        // const typename nstride_type::const_reverse_iterator strides_rend = strides.rend();
        typename nstride_type::const_reverse_iterator stndIter = stnd.rbegin();
        // typename nstride_type::const_reverse_iterator stnd_innerIter;
        dimension_type numberOfWrappedDimensions = 0;

        //* Below is Ofri's code
        // Consider the following structure.
        //
        // initial target setup :    // performed before engine loop is begun, outside of this function
        //       currPtr = base;
        //       for each k
        //           target[k] = currPtr + size[k] * stride[k];  // This should be invariant after every dimension wrap
        //           stnd[k] = stride[k-1] - size[k] * stride[k];   // This is constant through the run
        //
        // wrap(nw)  :      // nw is the number of dimensions wrapped
        //       currPtr += sum_{i =(d-nw)}^{d-1}( stnd[i] );  // obsolete if currPtr already determined; see below
        //       for k = (d-nw) to (d-1)
        //           target[k] = currPtr + size[k] * stride[k];
        //
        // check_wrap :     // this updates currentPtr (see above) and finds out nw (number of wrapped dimensions)
        //       k = d-1;
        //       nw = 0;
        //       currPtr += stride[k];
        //       while (currPtr == target[k]) {
        //           currPtr += stnd[k];
        //           --k;
        //           ++nw;
        //       }
        //       return nw;
        //
        // NOTE:  The invariant condition above and the initialization of stnd leads to
        //       size[k] * stride[k] = stride[k-1] - stnd[k];

        currentPointer += *stridesIter;
        while (currentPointer == *targetsIter) {
            currentPointer += *stndIter;
            ++targetsIter;
            ++stndIter;
            ++numberOfWrappedDimensions;
            if (numberOfWrappedDimensions == _dimension)
                return numberOfWrappedDimensions;
        }

        if (numberOfWrappedDimensions == 0)
            return numberOfWrappedDimensions;

        stridesIter += numberOfWrappedDimensions;
        difference_type targetOffset;
        do {
            --targetsIter;
            --stndIter;
            targetOffset = *stridesIter - *stndIter;
            *targetsIter = currentPointer + targetOffset;
            --stridesIter;
        } while (targetsIter != targets_rbeg);
        return numberOfWrappedDimensions;
    }
#endif  // !OFRI_NEW_NARRY_LOOP_CONTROL

    template <class _incrementalOperationType, class _elementOperationType>
    class SoNi
    {
    public:
        typedef typename _incrementalOperationType::OutputType OutputType;

        template <class _inputNArrayType>
        static OutputType Run(const _inputNArrayType & inputNArray)
        {
            typedef _inputNArrayType InputNArrayType;
            typedef typename InputNArrayType::OwnerType InputOwnerType;
            typedef typename InputOwnerType::const_pointer InputPointerType;

            // retrieve owners
            const InputOwnerType & inputOwner = inputNArray.Owner();

            // if compact
            if (inputOwner.IsCompact()) {
                return vctDynamicCompactLoopEngines::SoCi<_incrementalOperationType, _elementOperationType>::Run(inputOwner);
            } else {
#if OFRI_NEW_NARRY_LOOP_CONTROL
				// declare all variables used for inputOwner
                const nsize_type & inputSizes = inputOwner.sizes();
                const nstride_type & inputStrides = inputOwner.strides();
				const stride_type inputElementStrideForCharPtr = (stride_type)( ((typename InputNArrayType::value_type *)(0)) + 1 );
#  if CHAR_TARGET
				typedef const char inputTargetType;
				const nstride_type inputCharStrides(inputStrides * inputElementStrideForCharPtr);
#  else
				typedef const InputNArrayType::value_type inputTargetType;
				const nstride_type inputCharStrides(inputStrides);
#  endif
				typedef vctNArrayLoopControl<_dimension, inputTargetType> inputLoopControl;
				typedef typename inputLoopControl::target_vec_type target_vec_type;

				const nstride_type inputSTND = inputLoopControl::CalculateDimWrapCharStrides(inputSizes, inputCharStrides);
				inputTargetType * inputPointer = reinterpret_cast<const inputTargetType *>(inputOwner.Pointer());
                target_vec_type inputTargets =
					inputLoopControl::InitializeTargets(inputSizes, inputCharStrides, inputPointer);

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = inputOwner.dimension();

                OutputType incrementalResult = _incrementalOperationType::NeutralElement();

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    incrementalResult =
                        _incrementalOperationType::Operate(incrementalResult,
                            _elementOperationType::Operate(* reinterpret_cast<InputPointerType>(inputPointer) ) );

                    numberOfWrappedDimensions =
						inputLoopControl::IncrementPointers<_dimension>(inputTargets, inputPointer, inputCharStrides, inputSTND);
                }
                return incrementalResult;
#else
				// declare all variables used for inputOwner
                const nsize_type & inputSizes = inputOwner.sizes();
                const nstride_type & inputStrides = inputOwner.strides();
                nstride_type inputSTND;
                vctFixedSizeVector<InputPointerType, _dimension> inputTargets;
                InputPointerType inputPointer = inputOwner.Pointer();

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = inputOwner.dimension();

                CalculateSTND(inputSTND, inputSizes, inputStrides);
                InitializeTargets(inputTargets, inputSizes, inputStrides, inputPointer);

                OutputType incrementalResult = _incrementalOperationType::NeutralElement();

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    incrementalResult =
                        _incrementalOperationType::Operate(incrementalResult,
                                                           _elementOperationType::Operate(*inputPointer) );

                    numberOfWrappedDimensions =
                        IncrementPointers(inputTargets, inputPointer, inputStrides, inputSTND);
                }
                return incrementalResult;
#endif
            }
        }   // Run method
    };  // SoNi class


    template <class _incrementalOperationType, class _elementOperationType>
    class SoNiNi
    {
    public:
        typedef typename _incrementalOperationType::OutputType OutputType;

        template <class _input1NArrayType, class _input2NArrayType>
        static OutputType Run(const _input1NArrayType & input1NArray,
                              const _input2NArrayType & input2NArray)
        {
            typedef _input1NArrayType Input1NArrayType;
            typedef typename Input1NArrayType::OwnerType Input1OwnerType;
            typedef typename Input1OwnerType::const_pointer Input1PointerType;

            typedef _input2NArrayType Input2NArrayType;
            typedef typename Input2NArrayType::OwnerType Input2OwnerType;
            typedef typename Input2OwnerType::const_pointer Input2PointerType;

            // retrieve owners
            const Input1OwnerType & input1Owner = input1NArray.Owner();
            const Input2OwnerType & input2Owner = input2NArray.Owner();

            // check sizes
            const nsize_type & input1Sizes = input1Owner.sizes();
            const nsize_type & input2Sizes = input2Owner.sizes();
            if (input1Sizes.NotEqual(input2Sizes)) {
                ThrowSizeMismatchException();
            }

            // if compact and same strides
            const nstride_type & input1Strides = input1Owner.strides();
            const nstride_type & input2Strides = input2Owner.strides();

            if (input1Owner.IsCompact() && input2Owner.IsCompact()
                && (input1Owner.strides() == input2Owner.strides())) {
                return vctDynamicCompactLoopEngines::SoCiCi<_incrementalOperationType, _elementOperationType>::Run(input1Owner, input2Owner);
            } else {
#if OFRI_NEW_NARRY_LOOP_CONTROL
				const stride_type input1ElementStrideForCharPtr = (stride_type)( ((typename Input1NArrayType::value_type *)(0)) + 1 );
				const stride_type input2ElementStrideForCharPtr = (stride_type)( ((typename Input2NArrayType::value_type *)(0)) + 1 );
#  if CHAR_TARGET
				typedef const char input1TargetType;
				typedef const char input2TargetType;
				const nstride_type input1CharStrides(input1Strides * input1ElementStrideForCharPtr);
				const nstride_type input2CharStrides(input2Strides * input2ElementStrideForCharPtr);
#  else
				typedef const Input1NArrayType::value_type input1TargetType;
				typedef const Input2NArrayType::value_type input2TargetType;
				const nstride_type input1CharStrides(input1Strides);
				const nstride_type input2CharStrides(input2Strides);
#endif

				typedef vctNArrayLoopControl<_dimension, input1TargetType> input1LoopControl;
				typedef typename input1LoopControl::target_vec_type target_vec_type;

                // declare all variables used for input1Owner
                const nstride_type input1STND = input1LoopControl::CalculateDimWrapCharStrides(input1Sizes, input1CharStrides);
				const input1TargetType * input1Pointer = reinterpret_cast<const input1TargetType *>(input1Owner.Pointer());
                target_vec_type input1Targets =
					input1LoopControl::InitializeTargets(input1Sizes, input1CharStrides, input1Pointer);

                // declare all variables used for input2Owner
				typedef vctNArrayLoopControl<_dimension, input2TargetType> input2LoopControl;
                const nstride_type input2STND = input2LoopControl::CalculateDimWrapCharStrides(input2Sizes, input2CharStrides);
                const nstride_type input2OTND = input2LoopControl::CalculateOTND(input2STND);
				const input2TargetType * input2Pointer = reinterpret_cast<const input2TargetType *>(input2Owner.Pointer());

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = input1Owner.dimension();

                OutputType incrementalResult = _incrementalOperationType::NeutralElement();

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    incrementalResult =
                        _incrementalOperationType::Operate(
						    incrementalResult, _elementOperationType::Operate(
							    *reinterpret_cast<Input1PointerType>(input1Pointer),
								*reinterpret_cast<Input2PointerType>(input2Pointer) ) );

                    numberOfWrappedDimensions =
                        input1LoopControl::IncrementPointers<_dimension>(input1Targets, input1Pointer, input1CharStrides, input1STND);

                    input2Pointer = input2LoopControl::SyncCurrentPointer(input2Pointer, input2OTND, numberOfWrappedDimensions);
                }
                return incrementalResult;
#else
                // declare all variables used for input1Owner
                nstride_type input1STND;
                vctFixedSizeVector<Input1PointerType, _dimension> input1Targets;
                Input1PointerType input1Pointer = input1Owner.Pointer();

                // declare all variables used for input2Owner
                nstride_type input2STND;
                nstride_type input2OTND;
                Input2PointerType input2Pointer = input2Owner.Pointer();

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = input1Owner.dimension();

                CalculateSTND(input1STND, input1Sizes, input1Strides);
                CalculateSTND(input2STND, input2Sizes, input2Strides);
                CalculateOTND(input2OTND, input2Strides, input2STND);
                InitializeTargets(input1Targets, input1Sizes, input1Strides, input1Pointer);

                OutputType incrementalResult = _incrementalOperationType::NeutralElement();

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    incrementalResult =
                        _incrementalOperationType::Operate(incrementalResult,
                                                           _elementOperationType::Operate(*input1Pointer, *input2Pointer) );

                    numberOfWrappedDimensions =
                        IncrementPointers(input1Targets, input1Pointer, input1Strides, input1STND);

                    SyncCurrentPointer(input2Pointer, input2OTND, numberOfWrappedDimensions);
                }
                return incrementalResult;
#endif
            }
        }   // Run method
    };  // SoNiNi class


    template <class _incrementalOperationType, class _elementOperationType>
    class SoNiSi
    {
    public:
        typedef typename _incrementalOperationType::OutputType OutputType;

        template <class _inputNArrayType, class _inputScalarType>
        static OutputType Run(const _inputNArrayType & inputNArray,
                              const _inputScalarType inputScalar)
        {
            typedef _inputNArrayType InputNArrayType;
            typedef typename InputNArrayType::OwnerType InputOwnerType;
            typedef typename InputOwnerType::const_pointer InputPointerType;

            // retrieve owners
            const InputOwnerType & inputOwner = inputNArray.Owner();

            // if compact
            if (inputOwner.IsCompact()) {
                return vctDynamicCompactLoopEngines::SoCiSi<_incrementalOperationType, _elementOperationType>::Run(inputOwner, inputScalar);
            } else {
#if OFRI_NEW_NARRY_LOOP_CONTROL
                // declare all variables used for inputOwner
                const nsize_type & inputSizes = inputOwner.sizes();
                const nstride_type & inputStrides = inputOwner.strides();

				const stride_type inputElementStrideForCharPtr = (stride_type)( ((typename InputNArrayType::value_type *)(0)) + 1 );
#  if CHAR_TARGET
				typedef const char inputTargetType;
				const nstride_type inputCharStrides(inputStrides * inputElementStrideForCharPtr);
#  else
				typedef const InputNArrayType::value_type inputTargetType;
				const nstride_type inputCharStrides(inputStrides);
#endif

				typedef vctNArrayLoopControl<_dimension, inputTargetType> inputLoopControl;
				typedef typename inputLoopControl::target_vec_type target_vec_type;
				const nstride_type inputSTND = inputLoopControl::CalculateDimWrapCharStrides(inputSizes, inputCharStrides);
				inputTargetType * inputPointer = reinterpret_cast<const inputTargetType *>(inputOwner.Pointer());
                target_vec_type inputTargets =
					inputLoopControl::InitializeTargets(inputSizes, inputCharStrides, inputPointer);

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = inputOwner.dimension();

                OutputType incrementalResult = _incrementalOperationType::NeutralElement();

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    incrementalResult =
                        _incrementalOperationType::Operate(incrementalResult,
						_elementOperationType::Operate(*reinterpret_cast<const inputTargetType *>(inputPointer), inputScalar) );

                    numberOfWrappedDimensions =
                        inputLoopControl::IncrementPointers<_dimension>(inputTargets, inputPointer, inputCharStrides, inputSTND);
                }
                return incrementalResult;
#else
                // declare all variables used for inputOwner
                const nsize_type & inputSizes = inputOwner.sizes();
                const nstride_type & inputStrides = inputOwner.strides();
                nstride_type inputSTND;
                vctFixedSizeVector<InputPointerType, _dimension> inputTargets;
                InputPointerType inputPointer = inputOwner.Pointer();

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = inputOwner.dimension();

                CalculateSTND(inputSTND, inputSizes, inputStrides);
                InitializeTargets(inputTargets, inputSizes, inputStrides, inputPointer);

                OutputType incrementalResult = _incrementalOperationType::NeutralElement();

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    incrementalResult =
                        _incrementalOperationType::Operate(incrementalResult,
                                                           _elementOperationType::Operate(*inputPointer, inputScalar) );

                    numberOfWrappedDimensions =
                        IncrementPointers(inputTargets, inputPointer, inputStrides, inputSTND);
                }
                return incrementalResult;
#endif
            }
        }   // Run method
    };  // SoNiSi class


    template <class _elementOperationType>
    class NoNiNi
    {
    public:
        template <class _outputNArrayType, class _input1NArrayType, class _input2NArrayType>
        static void Run(_outputNArrayType & outputNArray,
                        const _input1NArrayType & input1NArray,
                        const _input2NArrayType & input2NArray)
        {
            typedef _outputNArrayType OutputNArrayType;
            typedef typename OutputNArrayType::OwnerType OutputOwnerType;
            typedef typename OutputOwnerType::pointer OutputPointerType;
            typedef typename OutputOwnerType::const_pointer OutputConstPointerType;

            typedef _input1NArrayType Input1NArrayType;
            typedef typename Input1NArrayType::OwnerType Input1OwnerType;
            typedef typename Input1OwnerType::const_pointer Input1PointerType;

            typedef _input2NArrayType Input2NArrayType;
            typedef typename Input2NArrayType::OwnerType Input2OwnerType;
            typedef typename Input2OwnerType::const_pointer Input2PointerType;

            // retrieve owners
            OutputOwnerType & outputOwner = outputNArray.Owner();
            const Input1OwnerType & input1Owner = input1NArray.Owner();
            const Input2OwnerType & input2Owner = input2NArray.Owner();

            // check sizes
            const nsize_type & outputSizes = outputOwner.sizes();
            const nsize_type & input1Sizes = input1Owner.sizes();
            const nsize_type & input2Sizes = input2Owner.sizes();
            if (outputSizes.NotEqual(input1Sizes) || outputSizes.NotEqual(input2Sizes)) {
                ThrowSizeMismatchException();
            }

            // if compact and same strides
            const nstride_type & outputStrides = outputOwner.strides();
            const nstride_type & input1Strides = input1Owner.strides();
            const nstride_type & input2Strides = input2Owner.strides();

            if (outputOwner.IsCompact() && input1Owner.IsCompact() && input2Owner.IsCompact()
                && (outputOwner.strides() == input1Owner.strides())
                && (outputOwner.strides() == input2Owner.strides())) {
                vctDynamicCompactLoopEngines::CoCiCi<_elementOperationType>::Run(outputOwner, input1Owner, input2Owner);
            } else {
#if OFRI_NEW_NARRY_LOOP_CONTROL
				const stride_type outputElementStrideForCharPtr = (stride_type)( ((typename OutputNArrayType::value_type *)(0)) + 1 );
				const stride_type input1ElementStrideForCharPtr = (stride_type)( ((typename Input1NArrayType::value_type *)(0)) + 1 );
				const stride_type input2ElementStrideForCharPtr = (stride_type)( ((typename Input2NArrayType::value_type *)(0)) + 1 );

#  if CHAR_TARGET
				typedef char outputTargetType;
				typedef char input1TargetType;
				typedef char input2TargetType;
				const nstride_type outputCharStrides(outputStrides * outputElementStrideForCharPtr);
				const nstride_type input1CharStrides(input1Strides * input1ElementStrideForCharPtr);
				const nstride_type input2CharStrides(input2Strides * input2ElementStrideForCharPtr);
#  else
				typedef OutputNArrayType::value_type outputTargetType;
				typedef Input1NArrayType::value_type input1TargetType;
				typedef Input2NArrayType::value_type input2TargetType;
				const nstride_type outputCharStrides(outputStrides);
				const nstride_type input1CharStrides(input1Strides);
				const nstride_type input2CharStrides(input2Strides);
#endif

                // declare all variables used for outputOwner
				typedef vctNArrayLoopControl<_dimension, outputTargetType> outputLoopControl;
				typedef typename outputLoopControl::target_vec_type target_vec_type1;
                const nstride_type outputSTND = outputLoopControl::CalculateDimWrapCharStrides(outputSizes, outputCharStrides);
                outputTargetType * outputPointer = reinterpret_cast<outputTargetType *>(outputOwner.Pointer());
				target_vec_type1 outputTargets =
					outputLoopControl::InitializeTargets(outputSizes, outputCharStrides, outputPointer);

                // declare all variables used for input1Owner
				typedef vctNArrayLoopControl<_dimension, input1TargetType> input1LoopControl;
				const nstride_type input1STND = input1LoopControl::CalculateDimWrapCharStrides(input1Sizes, input1CharStrides);
                const nstride_type input1OTND = input1LoopControl::CalculateOTND(input1STND);
				const input1TargetType * input1Pointer = reinterpret_cast<const input1TargetType *>(input1Owner.Pointer());

                // declare all variables used for input2Owner
				typedef vctNArrayLoopControl<_dimension, input2TargetType> input2LoopControl;
				const nstride_type input2STND = input2LoopControl::CalculateDimWrapCharStrides(input2Sizes, input2CharStrides);
                const nstride_type input2OTND = input2LoopControl::CalculateOTND(input2STND);
				const input2TargetType * input2Pointer = reinterpret_cast<const input2TargetType *>(input2Owner.Pointer());

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = outputOwner.dimension();

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    *reinterpret_cast<OutputPointerType>(outputPointer) =
						_elementOperationType::Operate(*reinterpret_cast<Input1PointerType>(input1Pointer),
						*reinterpret_cast<Input2PointerType>(input2Pointer));

                    numberOfWrappedDimensions =
						outputLoopControl::IncrementPointers<_dimension>(outputTargets, outputPointer, outputCharStrides, outputSTND);

					input1Pointer = input1LoopControl::SyncCurrentPointer(input1Pointer, input1OTND, numberOfWrappedDimensions);
					input2Pointer = input2LoopControl::SyncCurrentPointer(input2Pointer, input2OTND, numberOfWrappedDimensions);
                }
#else
                // declare all variables used for outputOwner
                nstride_type outputSTND;
                vctFixedSizeVector<OutputConstPointerType, _dimension> outputTargets;
                OutputPointerType outputPointer = outputOwner.Pointer();

                // declare all variables used for input1Owner
                nstride_type input1STND;
                nstride_type input1OTND;
                Input1PointerType input1Pointer = input1Owner.Pointer();

                // declare all variables used for input2Owner
                const nsize_type & input2Sizes = input2Owner.sizes();
                nstride_type input2STND;
                nstride_type input2OTND;
                Input2PointerType input2Pointer = input2Owner.Pointer();

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = outputOwner.dimension();

                CalculateSTND(outputSTND, outputSizes, outputStrides);
                CalculateSTND(input1STND, input1Sizes, input1Strides);
                CalculateSTND(input2STND, input2Sizes, input2Strides);
                CalculateOTND(input1OTND, input1Strides, input1STND);
                CalculateOTND(input2OTND, input2Strides, input2STND);
                InitializeTargets(outputTargets, outputSizes, outputStrides, outputPointer);

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    *outputPointer = _elementOperationType::Operate(*input1Pointer, *input2Pointer);

                    numberOfWrappedDimensions =
                        IncrementPointers(outputTargets, outputPointer, outputStrides, outputSTND);

                    SyncCurrentPointer(input1Pointer, input1OTND, numberOfWrappedDimensions);
                    SyncCurrentPointer(input2Pointer, input2OTND, numberOfWrappedDimensions);
				}
#endif
            }
        }   // Run method
    };  // NoNiNi class


    template <class _elementOperationType>
    class NoNiSi
    {
    public:
        template <class _outputNArrayType, class _inputNArrayType, class _inputScalarType>
        static void Run(_outputNArrayType & outputNArray,
                        const _inputNArrayType & inputNArray,
                        const _inputScalarType inputScalar)
        {
            typedef _outputNArrayType OutputNArrayType;
            typedef typename OutputNArrayType::OwnerType OutputOwnerType;
            typedef typename OutputOwnerType::pointer OutputPointerType;
            typedef typename OutputOwnerType::const_pointer OutputConstPointerType;

            typedef _inputNArrayType InputNArrayType;
            typedef typename InputNArrayType::OwnerType InputOwnerType;
            typedef typename InputOwnerType::const_pointer InputPointerType;

            // retrieve owners
            OutputOwnerType & outputOwner = outputNArray.Owner();
            const InputOwnerType & inputOwner = inputNArray.Owner();

            // check sizes
            const nsize_type & outputSizes = outputOwner.sizes();
            const nsize_type & inputSizes = inputOwner.sizes();
            if (outputSizes.NotEqual(inputSizes)) {
                ThrowSizeMismatchException();
            }

            // if compact and same strides
            const nstride_type & outputStrides = outputOwner.strides();
            const nstride_type & inputStrides = inputOwner.strides();

            if (outputOwner.IsCompact() && inputOwner.IsCompact()
                && (outputOwner.strides() == inputOwner.strides())) {
                vctDynamicCompactLoopEngines::CoCiSi<_elementOperationType>::Run(outputOwner, inputOwner, inputScalar);
            } else {
#if OFRI_NEW_NARRY_LOOP_CONTROL
				const stride_type outputElementStrideForCharPtr = (stride_type)( ((typename OutputNArrayType::value_type *)(0)) + 1 );
				const stride_type inputElementStrideForCharPtr = (stride_type)( ((typename InputNArrayType::value_type *)(0)) + 1 );
#  if CHAR_TARGET
				typedef char outputTargetType;
				typedef char inputTargetType;
				const nstride_type outputCharStrides(outputStrides * outputElementStrideForCharPtr);
				const nstride_type inputCharStrides(inputStrides * inputElementStrideForCharPtr);
#  else
				typedef OutputNArrayType::value_type outputTargetType;
				typedef InputNArrayType::value_type inputTargetType;
				const nstride_type outputCharStrides(outputStrides);
				const nstride_type inputCharStrides(inputStrides);
#endif

                // declare all variables used for outputOwner
				typedef vctNArrayLoopControl<_dimension, outputTargetType> outputLoopControl;
				typedef typename outputLoopControl::target_vec_type target_vec_type;

				const nstride_type outputSTND = outputLoopControl::CalculateDimWrapCharStrides(outputSizes, outputCharStrides);
                outputTargetType * outputPointer = reinterpret_cast<outputTargetType *>(outputOwner.Pointer());

				target_vec_type outputTargets =
					outputLoopControl::InitializeTargets(outputSizes, outputCharStrides, outputPointer);

                // declare all variables used for inputOwner
				typedef vctNArrayLoopControl<_dimension, inputTargetType> inputLoopControl;

				const nstride_type inputSTND = inputLoopControl::CalculateDimWrapCharStrides(inputSizes, inputCharStrides);
                const nstride_type inputOTND = inputLoopControl::CalculateOTND(inputSTND);
				const inputTargetType * inputPointer = reinterpret_cast<const inputTargetType *>(inputOwner.Pointer());

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = outputOwner.dimension();

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    *reinterpret_cast<OutputPointerType>(outputPointer) =
						_elementOperationType::Operate(*reinterpret_cast<InputPointerType>(inputPointer), inputScalar);

                    numberOfWrappedDimensions =
						outputLoopControl::IncrementPointers<_dimension>(outputTargets, outputPointer, outputCharStrides, outputSTND);

					inputPointer = inputLoopControl::SyncCurrentPointer(inputPointer, inputOTND, numberOfWrappedDimensions);
                }
#else
                // otherwise
                // declare all variables used for outputOwner
                nstride_type outputSTND;
                vctFixedSizeVector<OutputConstPointerType, _dimension> outputTargets;
                OutputPointerType outputPointer = outputOwner.Pointer();

                // declare all variables used for inputOwner
                nstride_type inputSTND;
                nstride_type inputOTND;
                InputPointerType inputPointer = inputOwner.Pointer();

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = outputOwner.dimension();

                CalculateSTND(outputSTND, outputSizes, outputStrides);
                CalculateSTND(inputSTND, inputSizes, inputStrides);
                CalculateOTND(inputOTND, inputStrides, inputSTND);
                InitializeTargets(outputTargets, outputSizes, outputStrides, outputPointer);

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    *outputPointer = _elementOperationType::Operate(*inputPointer, inputScalar);

                    numberOfWrappedDimensions =
                        IncrementPointers(outputTargets, outputPointer, outputStrides, outputSTND);

                    SyncCurrentPointer(inputPointer, inputOTND, numberOfWrappedDimensions);
                }
#endif
            }
        }   // Run method
    };  // NoNiSi class


    template <class _elementOperationType>
    class NoSiNi
    {
    public:
        template <class _outputNArrayType, class _inputScalarType, class _inputNArrayType>
        static void Run(_outputNArrayType & outputNArray,
                        const _inputScalarType inputScalar,
                        const _inputNArrayType & inputNArray)
        {
            typedef _outputNArrayType OutputNArrayType;
            typedef typename OutputNArrayType::OwnerType OutputOwnerType;
            typedef typename OutputOwnerType::pointer OutputPointerType;
            typedef typename OutputOwnerType::const_pointer OutputConstPointerType;

            typedef _inputNArrayType InputNArrayType;
            typedef typename InputNArrayType::OwnerType InputOwnerType;
            typedef typename InputOwnerType::const_pointer InputPointerType;

            // retrieve owners
            OutputOwnerType & outputOwner = outputNArray.Owner();
            const InputOwnerType & inputOwner = inputNArray.Owner();

            // check sizes
            const nsize_type & outputSizes = outputOwner.sizes();
            const nsize_type & inputSizes = inputOwner.sizes();
            if (outputSizes.NotEqual(inputSizes)) {
                ThrowSizeMismatchException();
            }

            // if compact and same strides
            const nstride_type & outputStrides = outputOwner.strides();
            const nstride_type & inputStrides = inputOwner.strides();

            if (outputOwner.IsCompact() && inputOwner.IsCompact()
                && (outputOwner.strides() == inputOwner.strides())) {
                vctDynamicCompactLoopEngines::CoSiCi<_elementOperationType>::Run(outputOwner, inputScalar, inputOwner);
            } else {
#if OFRI_NEW_NARRY_LOOP_CONTROL
				const stride_type outputElementStrideForCharPtr = (stride_type)( ((typename OutputNArrayType::value_type *)(0)) + 1 );
				const stride_type inputElementStrideForCharPtr = (stride_type)( ((typename InputNArrayType::value_type *)(0)) + 1 );
#  if CHAR_TARGET
				typedef char outputTargetType;
				typedef char inputTargetType;
				const nstride_type outputCharStrides(outputStrides * outputElementStrideForCharPtr);
				const nstride_type inputCharStrides(inputStrides * inputElementStrideForCharPtr);
#  else
				typedef OutputNArrayType::value_type outputTargetType;
				typedef InputNArrayType::value_type inputTargetType;
				const nstride_type outputCharStrides(outputStrides);
				const nstride_type inputCharStrides(inputStrides);
#endif

                // declare all variables used for outputOwner
				typedef vctNArrayLoopControl<_dimension, outputTargetType> outputLoopControl;
				typedef typename outputLoopControl::target_vec_type target_vec_type;

				const nstride_type outputSTND = outputLoopControl::CalculateDimWrapCharStrides(outputSizes, outputCharStrides);
                outputTargetType * outputPointer = reinterpret_cast<outputTargetType *>(outputOwner.Pointer());

				target_vec_type outputTargets =
					outputLoopControl::InitializeTargets(outputSizes, outputCharStrides, outputPointer);

                // declare all variables used for inputOwner
				typedef vctNArrayLoopControl<_dimension, inputTargetType> inputLoopControl;

				const nstride_type inputSTND = inputLoopControl::CalculateDimWrapCharStrides(inputSizes, inputCharStrides);
                const nstride_type inputOTND = inputLoopControl::CalculateOTND(inputSTND);
				const inputTargetType * inputPointer = reinterpret_cast<const inputTargetType *>(inputOwner.Pointer());

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = outputOwner.dimension();

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    *reinterpret_cast<OutputPointerType>(outputPointer) =
						_elementOperationType::Operate(inputScalar, *reinterpret_cast<InputPointerType>(inputPointer));

                    numberOfWrappedDimensions =
						outputLoopControl::IncrementPointers<_dimension>(outputTargets, outputPointer, outputCharStrides, outputSTND);

					inputPointer = inputLoopControl::SyncCurrentPointer(inputPointer, inputOTND, numberOfWrappedDimensions);
                }
#else
                // otherwise
                // declare all variables used for outputNArray
                nstride_type outputSTND;
                vctFixedSizeVector<OutputConstPointerType, _dimension> outputTargets;
                OutputPointerType outputPointer = outputNArray.Pointer();

                // declare all variables used for inputNArray
                nstride_type inputSTND;
                nstride_type inputOTND;
                InputPointerType inputPointer = inputNArray.Pointer();

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = outputNArray.dimension();

                CalculateSTND(outputSTND, outputSizes, outputStrides);
                CalculateSTND(inputSTND, inputSizes, inputStrides);
                CalculateOTND(inputOTND, inputStrides, inputSTND);
                InitializeTargets(outputTargets, outputSizes, outputStrides, outputPointer);

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    *outputPointer = _elementOperationType::Operate(inputScalar, *inputPointer);

                    numberOfWrappedDimensions =
                        IncrementPointers(outputTargets, outputPointer, outputStrides, outputSTND);

                    SyncCurrentPointer(inputPointer, inputOTND, numberOfWrappedDimensions);
                }
#endif
            }
        }   // Run method
    };  // NoSiNi class


    template <class _elementOperationType>
    class NioSi
    {
    public:
        template <class _inputOutputNArrayType, class _inputScalarType>
        static void Run(_inputOutputNArrayType & inputOutputNArray,
                        const _inputScalarType inputScalar)
        {
            typedef _inputOutputNArrayType InputOutputNArrayType;
            typedef typename InputOutputNArrayType::OwnerType InputOutputOwnerType;
            typedef typename InputOutputOwnerType::const_pointer InputOutputConstPointerType;
            typedef typename InputOutputOwnerType::pointer InputOutputPointerType;

            // retrieve owners
            InputOutputOwnerType & inputOutputOwner = inputOutputNArray.Owner();

            // if compact
            if (inputOutputOwner.IsCompact()) {
                vctDynamicCompactLoopEngines::CioSi<_elementOperationType>::Run(inputOutputOwner, inputScalar);
            } else {
#if OFRI_NEW_NARRY_LOOP_CONTROL
                const nsize_type & inputOutputSizes = inputOutputOwner.sizes();
                const nstride_type & inputOutputStrides = inputOutputOwner.strides();
				const stride_type inOutElementStrideForCharPtr = (stride_type)( ((typename InputOutputNArrayType::value_type *)(0)) + 1 );
#  if CHAR_TARGET
				typedef char inOutTargetType;
				const nstride_type inOutCharStrides(inputOutputStrides * inOutElementStrideForCharPtr);
#  else
				typedef InputOutputNArrayType::value_type inOutTargetType;
				const nstride_type inOutCharStrides(inputOutputStrides);
#endif
				typedef vctNArrayLoopControl<_dimension, inOutTargetType> inOutLoopControl;
				typedef typename inOutLoopControl::target_vec_type target_vec_type;

				const nstride_type inOutSTND = inOutLoopControl::CalculateDimWrapCharStrides(inputOutputSizes, inOutCharStrides);
                inOutTargetType * inputOutputPointer = reinterpret_cast<inOutTargetType *>(inputOutputOwner.Pointer());

				target_vec_type inOutTargets =
					inOutLoopControl::InitializeTargets(inputOutputSizes, inOutCharStrides, inputOutputPointer);

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = inputOutputOwner.dimension();

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    _elementOperationType::Operate(
						*reinterpret_cast<InputOutputPointerType>(inputOutputPointer), inputScalar);

                    numberOfWrappedDimensions =
						inOutLoopControl::IncrementPointers<_dimension>(inOutTargets, inputOutputPointer, inOutCharStrides, inOutSTND);
                }
#else
                // declare all variables used for inputOwner
                const nsize_type & inputOutputSizes = inputOutputOwner.sizes();
                const nstride_type & inputOutputStrides = inputOutputOwner.strides();
                nstride_type inputOutputSTND;
                vctFixedSizeVector<InputOutputConstPointerType, _dimension> inputOutputTargets;
                InputOutputPointerType inputOutputPointer = inputOutputOwner.Pointer();

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = inputOutputOwner.dimension();

                CalculateSTND(inputOutputSTND, inputOutputSizes, inputOutputStrides);
                InitializeTargets(inputOutputTargets, inputOutputSizes, inputOutputStrides, inputOutputPointer);

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    _elementOperationType::Operate(*inputOutputPointer, inputScalar);

                    numberOfWrappedDimensions =
                        IncrementPointers(inputOutputTargets, inputOutputPointer, inputOutputStrides, inputOutputSTND);
                }
#endif
            }
        }   // Run method
    };  // NioSi class


    template <class _elementOperationType>
    class NioNi
    {
    public:
        template <class _inputOutputNArrayType, class _inputNArrayType>
        static void Run(_inputOutputNArrayType & inputOutputNArray,
                        const _inputNArrayType & inputNArray)
        {
            typedef _inputOutputNArrayType InputOutputNArrayType;
            typedef typename InputOutputNArrayType::OwnerType InputOutputOwnerType;
            typedef typename InputOutputOwnerType::const_pointer InputOutputConstPointerType;
            typedef typename InputOutputOwnerType::pointer InputOutputPointerType;

            typedef _inputNArrayType InputNArrayType;
            typedef typename InputNArrayType::OwnerType InputOwnerType;
            typedef typename InputOwnerType::const_pointer InputPointerType;

            // retrieve owners
            InputOutputOwnerType & inputOutputOwner = inputOutputNArray.Owner();
            const InputOwnerType & inputOwner = inputNArray.Owner();

            // check sizes
            const nsize_type & inputOutputSizes = inputOutputOwner.sizes();
            const nsize_type & inputSizes = inputOwner.sizes();
            if (inputOutputSizes.NotEqual(inputSizes)) {
                ThrowSizeMismatchException();
            }

            // if compact and same strides
            const nstride_type & inputOutputStrides = inputOutputOwner.strides();
            const nstride_type & inputStrides = inputOwner.strides();

            if (inputOutputOwner.IsCompact() && inputOwner.IsCompact()
                && (inputOutputOwner.strides() == inputOwner.strides())) {
                vctDynamicCompactLoopEngines::CioCi<_elementOperationType>::Run(inputOutputOwner, inputOwner);
            } else {
#if OFRI_NEW_NARRY_LOOP_CONTROL
				const stride_type inOutElementStrideForCharPtr = (stride_type)( ((typename InputOutputNArrayType::value_type *)(0)) + 1 );
				const stride_type inputElementStrideForCharPtr = (stride_type)( ((typename InputNArrayType::value_type *)(0)) + 1 );
#  if CHAR_TARGET
				typedef char inOutTargetType;
				typedef char inputTargetType;
				const nstride_type inOutCharStrides(inputOutputStrides * inOutElementStrideForCharPtr);
				const nstride_type inputCharStrides(inputStrides * inputElementStrideForCharPtr);
#  else
				typedef InputOutputNArrayType::value_type inOutTargetType;
				typedef InputNArrayType::value_type inputTargetType;
				const nstride_type inOutCharStrides(inputOutputStrides);
				const nstride_type inputCharStrides(inputStrides);
#  endif
				typedef vctNArrayLoopControl<_dimension, inOutTargetType> inOutLoopControl;
				typedef typename inOutLoopControl::target_vec_type target_vec_type1;

				const nstride_type inOutSTND = inOutLoopControl::CalculateDimWrapCharStrides(inputOutputSizes, inOutCharStrides);
                inOutTargetType * inOutPointer = reinterpret_cast<inOutTargetType *>(inputOutputOwner.Pointer());

				target_vec_type1 inOutTargets =
					inOutLoopControl::InitializeTargets(inputOutputSizes, inOutCharStrides, inOutPointer);

				typedef vctNArrayLoopControl<_dimension, inputTargetType> inputLoopControl;

				const nstride_type inputSTND = inputLoopControl::CalculateDimWrapCharStrides(inputSizes, inputCharStrides);
                const nstride_type inputOTND = inputLoopControl::CalculateOTND(inputSTND);
				const inputTargetType * inputPointer = reinterpret_cast<const inputTargetType *>(inputOwner.Pointer());

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = inputOutputOwner.dimension();

                while (numberOfWrappedDimensions < maxWrappedDimensions) {
                    _elementOperationType::Operate(
						*(reinterpret_cast<InputOutputPointerType>(inOutPointer)),
						*(reinterpret_cast<InputPointerType>(inputPointer)) );

                    numberOfWrappedDimensions =
						inOutLoopControl::IncrementPointers<_dimension>(inOutTargets, inOutPointer, inOutCharStrides, inOutSTND);
					inputPointer = 
						inputLoopControl::SyncCurrentPointer(inputPointer, inputOTND, numberOfWrappedDimensions);
                }
#else
                // otherwise
                // declare all variables used for inputOutputOwner
                nstride_type inputOutputSTND;
                vctFixedSizeVector<InputOutputConstPointerType, _dimension> inputOutputTargets;
                InputOutputPointerType inputOutputPointer = inputOutputOwner.Pointer();

                // declare all variables used for inputOwner
                nstride_type inputSTND;
                nstride_type inputOTND;
                InputPointerType inputPointer = inputOwner.Pointer();

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = inputOutputOwner.dimension();

                CalculateSTND(inputOutputSTND, inputOutputSizes, inputOutputStrides);
                CalculateSTND(inputSTND, inputSizes, inputStrides);
                CalculateOTND(inputOTND, inputStrides, inputSTND);
                InitializeTargets(inputOutputTargets, inputOutputSizes, inputOutputStrides, inputOutputPointer);

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    _elementOperationType::Operate(*inputOutputPointer, *inputPointer);

                    numberOfWrappedDimensions =
                        IncrementPointers(inputOutputTargets, inputOutputPointer, inputOutputStrides, inputOutputSTND);

                    SyncCurrentPointer(inputPointer, inputOTND, numberOfWrappedDimensions);
                }
#endif
            }
        }   // Run method
    };  // NioNi class


    template <class _elementOperationType>
    class NoNi
    {
    public:
        template <class _outputNArrayType, class _inputNArrayType>
        static inline void Run(_outputNArrayType & outputNArray,
                               const _inputNArrayType & inputNArray)
        {
            typedef _outputNArrayType OutputNArrayType;
            typedef typename OutputNArrayType::OwnerType OutputOwnerType;
            typedef typename OutputOwnerType::const_pointer OutputConstPointerType;
            typedef typename OutputOwnerType::pointer OutputPointerType;

            typedef _inputNArrayType InputNArrayType;
            typedef typename InputNArrayType::OwnerType InputOwnerType;
            typedef typename InputOwnerType::const_pointer InputPointerType;

            // retrieve owners
            OutputOwnerType & outputOwner = outputNArray.Owner();
            const InputOwnerType & inputOwner = inputNArray.Owner();

            // check sizes
            const nsize_type & outputSizes = outputOwner.sizes();
            const nsize_type & inputSizes = inputOwner.sizes();
            if (inputSizes.NotEqual(outputSizes)) {
                ThrowSizeMismatchException();
            }

            // if compact and same strides
			const nstride_type & outputStrides = outputOwner.strides();
			const nstride_type & inputStrides = inputOwner.strides();
            if (outputOwner.IsCompact() && inputOwner.IsCompact()
                && (outputOwner.strides() == inputOwner.strides())) {
                vctDynamicCompactLoopEngines::CoCi<_elementOperationType>::Run(outputOwner, inputOwner);
            } else {
#if OFRI_NEW_NARRY_LOOP_CONTROL
				const stride_type outputElementStrideForCharPtr = (stride_type)( ((typename OutputNArrayType::value_type *)(0)) + 1 );
				const stride_type inputElementStrideForCharPtr = (stride_type)( ((typename InputNArrayType::value_type *)(0)) + 1 );
#  if CHAR_TARGET
				typedef char outputTargetType;
				typedef char inputTargetType;
				const nstride_type outputCharStrides(outputStrides * outputElementStrideForCharPtr);
				const nstride_type inputCharStrides(inputStrides * inputElementStrideForCharPtr);
#  else
				typedef OutputNArrayType::value_type outputTargetType;
				typedef InputNArrayType::value_type inputTargetType;
				const nstride_type outputCharStrides(outputStrides);
				const nstride_type inputCharStrides(inputStrides);
#endif
                // declare all variables used for outputNArray
				typedef vctNArrayLoopControl<_dimension, outputTargetType> outputLoopControl;
				typedef typename outputLoopControl::target_vec_type target_vec_type;

				const nstride_type outputSTND = outputLoopControl::CalculateDimWrapCharStrides(outputSizes, outputCharStrides);
                outputTargetType * outputPointer = reinterpret_cast<outputTargetType *>(outputOwner.Pointer());

				target_vec_type outputTargets =
					outputLoopControl::InitializeTargets(outputSizes, outputCharStrides, outputPointer);


                // declare all variables used for inputNArray
				typedef vctNArrayLoopControl<_dimension, inputTargetType> inputLoopControl;

				const nstride_type inputSTND = inputLoopControl::CalculateDimWrapCharStrides(inputSizes, inputCharStrides);
                const nstride_type inputOTND = inputLoopControl::CalculateOTND(inputSTND);
				const inputTargetType * inputPointer = reinterpret_cast<const inputTargetType *>(inputOwner.Pointer());

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = outputOwner.dimension();

                while (numberOfWrappedDimensions < maxWrappedDimensions) {
                    *(reinterpret_cast<OutputPointerType>(outputPointer)) =
						_elementOperationType::Operate( *(reinterpret_cast<InputPointerType>(inputPointer)) );

                    numberOfWrappedDimensions =
						outputLoopControl::IncrementPointers<_dimension>(outputTargets, outputPointer, outputCharStrides, outputSTND);
					inputPointer =
						inputLoopControl::SyncCurrentPointer(inputPointer, inputOTND, numberOfWrappedDimensions);
                }
#else
                // declare all variables used for outputNArray
                nstride_type outputSTND;
                vctFixedSizeVector<OutputConstPointerType, _dimension> outputTargets;
                OutputPointerType outputPointer = outputOwner.Pointer();

                // declare all variables used for inputNArray
                nstride_type inputSTND;
                nstride_type inputOTND;
                InputPointerType inputPointer = inputOwner.Pointer();

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = outputOwner.dimension();

                CalculateSTND(outputSTND, outputSizes, outputStrides);
                CalculateSTND(inputSTND, inputSizes, inputStrides);
                CalculateOTND(inputOTND, inputStrides, inputSTND);
                InitializeTargets(outputTargets, outputSizes, outputStrides, outputPointer);

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    *outputPointer = _elementOperationType::Operate(*inputPointer);

                    numberOfWrappedDimensions =
                        IncrementPointers(outputTargets, outputPointer, outputStrides, outputSTND);

                    SyncCurrentPointer(inputPointer, inputOTND, numberOfWrappedDimensions);
                }
#endif
            }
        }   // Run method
    };  // NoNi class

    template <class _elementOperationType>
    class Nio
    {
    public:
        template <class _inputOutputNArrayType>
        static void Run(_inputOutputNArrayType & inputOutputNArray)
        {
            typedef _inputOutputNArrayType InputOutputNArrayType;
            typedef typename InputOutputNArrayType::OwnerType InputOutputOwnerType;
            typedef typename InputOutputOwnerType::const_pointer InputOutputConstPointerType;
            typedef typename InputOutputOwnerType::pointer InputOutputPointerType;

            // retrieve owners
            InputOutputOwnerType & inputOutputOwner = inputOutputNArray.Owner();

            // if compact
            if (inputOutputOwner.IsCompact()) {
                vctDynamicCompactLoopEngines::Cio<_elementOperationType>::Run(inputOutputOwner);
            } else {
#if OFRI_NEW_NARRY_LOOP_CONTROL
                const nsize_type & inputOutputSizes = inputOutputOwner.sizes();
                const nstride_type & inputOutputStrides = inputOutputOwner.strides();
				const stride_type inOutElementStrideForCharPtr = (stride_type)( ((typename InputOutputNArrayType::value_type *)(0)) + 1 );
#  if CHAR_TARGET
				typedef char inOutTargetType;
				const nstride_type inOutCharStrides(inputOutputStrides * inOutElementStrideForCharPtr);
#  else
				typedef InputOutputNArrayType::value_type inOutTargetType;
				const nstride_type inOutCharStrides(inputOutputStrides);
#endif
				typedef vctNArrayLoopControl<_dimension, inOutTargetType> inOutLoopControl;
				typedef typename inOutLoopControl::target_vec_type target_vec_type;

				const nstride_type inOutSTND = inOutLoopControl::CalculateDimWrapCharStrides(inputOutputSizes, inOutCharStrides);
                inOutTargetType * inputOutputPointer = reinterpret_cast<inOutTargetType *>(inputOutputOwner.Pointer());

				target_vec_type inOutTargets =
					inOutLoopControl::InitializeTargets(inputOutputSizes, inOutCharStrides, inputOutputPointer);

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = inputOutputOwner.dimension();

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    _elementOperationType::Operate(
						*reinterpret_cast<InputOutputPointerType>(inputOutputPointer));

                    numberOfWrappedDimensions =
						inOutLoopControl::IncrementPointers<_dimension>(inOutTargets, inputOutputPointer, inOutCharStrides, inOutSTND);
                }
#else
                // otherwise
                const nsize_type & inputOutputSizes = inputOutputOwner.sizes();
                const nstride_type & inputOutputStrides = inputOutputOwner.strides();
                nstride_type inputOutputSTND;
                vctFixedSizeVector<InputOutputConstPointerType, _dimension> inputOutputTargets;
                InputOutputPointerType inputOutputPointer = inputOutputOwner.Pointer();

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = inputOutputOwner.dimension();

                CalculateSTND(inputOutputSTND, inputOutputSizes, inputOutputStrides);
                InitializeTargets(inputOutputTargets, inputOutputSizes, inputOutputStrides, inputOutputPointer);

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    _elementOperationType::Operate(*inputOutputPointer);

                    numberOfWrappedDimensions =
                        IncrementPointers(inputOutputTargets, inputOutputPointer, inputOutputStrides, inputOutputSTND);
                }
#endif
            }
        }   // Run method
    };  // Nio class

    template <class _inputOutputElementOperationType, class _scalarNArrayElementOperationType>
    class NioSiNi
    {
    public:
        template <class _inputOutputNArrayType, class _inputScalarType, class _inputNArrayType>
        static void Run(_inputOutputNArrayType & inputOutputNArray,
                        const _inputScalarType inputScalar,
                        const _inputNArrayType & inputNArray)
        {
            typedef _inputOutputNArrayType InputOutputNArrayType;
            typedef typename InputOutputNArrayType::OwnerType InputOutputOwnerType;
            typedef typename InputOutputOwnerType::const_pointer InputOutputConstPointerType;
            typedef typename InputOutputOwnerType::pointer InputOutputPointerType;

            typedef _inputNArrayType InputNArrayType;
            typedef typename InputNArrayType::OwnerType InputOwnerType;
            typedef typename InputOwnerType::const_pointer InputPointerType;

            // retrieve owners
            InputOutputOwnerType & inputOutputOwner = inputOutputNArray.Owner();
            const InputOwnerType & inputOwner = inputNArray.Owner();

            // check sizes
            const nsize_type & inputOutputSizes = inputOutputOwner.sizes();
            const nsize_type & inputSizes = inputOwner.sizes();
            if (inputOutputSizes.NotEqual(inputSizes)) {
                ThrowSizeMismatchException();
            }

            // if compact and same strides
            const nstride_type & inputOutputStrides = inputOutputOwner.strides();
            const nstride_type & inputStrides = inputOwner.strides();

            if (inputOutputOwner.IsCompact() && inputOwner.IsCompact()
                && (inputOutputOwner.strides() == inputOwner.strides())) {
                vctDynamicCompactLoopEngines::CioSiCi<_inputOutputElementOperationType, _scalarNArrayElementOperationType>::Run(inputOutputOwner, inputScalar, inputOwner);
            } else {
#if OFRI_NEW_NARRY_LOOP_CONTROL
				const stride_type inOutElementStrideForCharPtr = (stride_type)( ((typename InputOutputNArrayType::value_type *)(0)) + 1 );
				const stride_type inputElementStrideForCharPtr = (stride_type)( ((typename InputNArrayType::value_type *)(0)) + 1 );
#  if CHAR_TARGET
				typedef char inOutTargetType;
				typedef char inputTargetType;
				const nstride_type inOutCharStrides(inputOutputStrides * inOutElementStrideForCharPtr);
				const nstride_type inputCharStrides(inputStrides * inputElementStrideForCharPtr);
#  else
				typedef InputOutputNArrayType::value_type inOutTargetType;
				typedef InputNArrayType::value_type inputTargetType;
				const nstride_type inOutCharStrides(inputOutputStrides);
				const nstride_type inputCharStrides(inputStrides);
#  endif
				typedef vctNArrayLoopControl<_dimension, inOutTargetType> inOutLoopControl;
				typedef typename inOutLoopControl::target_vec_type target_vec_type1;

				const nstride_type inOutSTND = inOutLoopControl::CalculateDimWrapCharStrides(inputOutputSizes, inOutCharStrides);
                inOutTargetType * inOutPointer = reinterpret_cast<inOutTargetType *>(inputOutputOwner.Pointer());

				target_vec_type1 inOutTargets =
					inOutLoopControl::InitializeTargets(inputOutputSizes, inOutCharStrides, inOutPointer);

				typedef vctNArrayLoopControl<_dimension, inputTargetType> inputLoopControl;

				const nstride_type inputSTND = inputLoopControl::CalculateDimWrapCharStrides(inputSizes, inputCharStrides);
                const nstride_type inputOTND = inputLoopControl::CalculateOTND(inputSTND);
				const inputTargetType * inputPointer = reinterpret_cast<const inputTargetType *>(inputOwner.Pointer());

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = inputOutputOwner.dimension();

                while (numberOfWrappedDimensions < maxWrappedDimensions) {
                    _inputOutputElementOperationType::Operate(
						*reinterpret_cast<InputOutputPointerType>(inOutPointer),
						_scalarNArrayElementOperationType::Operate(inputScalar, *reinterpret_cast<InputPointerType>(inputPointer)) );

                    numberOfWrappedDimensions =
						inOutLoopControl::IncrementPointers<_dimension>(inOutTargets, inOutPointer, inOutCharStrides, inOutSTND);
					inputPointer = 
						inputLoopControl::SyncCurrentPointer(inputPointer, inputOTND, numberOfWrappedDimensions);
                }
#else
                // otherwise
                // declare all variables used for inputOutputNArray
                nstride_type inputOutputSTND;
                vctFixedSizeVector<InputOutputConstPointerType, _dimension> inputOutputTargets;
                InputOutputPointerType inputOutputPointer = inputOutputNArray.Pointer();

                // declare all variables used for inputNArray
                nstride_type inputSTND;
                nstride_type inputOTND;
                InputPointerType inputPointer = inputNArray.Pointer();

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = inputOutputNArray.dimension();

                CalculateSTND(inputOutputSTND, inputOutputSizes, inputOutputStrides);
                CalculateSTND(inputSTND, inputSizes, inputStrides);
                CalculateOTND(inputOTND, inputStrides, inputSTND);
                InitializeTargets(inputOutputTargets, inputOutputSizes, inputOutputStrides, inputOutputPointer);

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    _inputOutputElementOperationType::Operate(*inputOutputPointer,
                                                              _scalarNArrayElementOperationType::Operate(inputScalar, *inputPointer) );

                    numberOfWrappedDimensions =
                        IncrementPointers(inputOutputTargets, inputOutputPointer, inputOutputStrides, inputOutputSTND);

                    SyncCurrentPointer(inputPointer, inputOTND, numberOfWrappedDimensions);
                }
#endif
            }
        }   // Run method

    };  // NioSiNi class


    class MinAndMax
    {
    public:
        template <class _inputNArrayType>
        static void Run(const _inputNArrayType & inputNArray,
                        typename _inputNArrayType::value_type & minValue,
                        typename _inputNArrayType::value_type & maxValue)
        {
            typedef _inputNArrayType InputNArrayType;
            typedef typename InputNArrayType::OwnerType InputOwnerType;
            typedef typename InputOwnerType::value_type value_type;
            typedef typename InputOwnerType::const_pointer InputPointerType;

            // retrieve owner
            const InputOwnerType & inputOwner = inputNArray.Owner();
            InputPointerType inputPointer = inputOwner.Pointer();

            if (inputPointer == 0)
                return;

            // if compact
            if (inputOwner.IsCompact()) {
                vctDynamicCompactLoopEngines::MinAndMax::Run(inputOwner, minValue, maxValue);
            } else {
#if OFRI_NEW_NARRY_LOOP_CONTROL
                const nsize_type & inputSizes = inputOwner.sizes();
                const nstride_type & inputStrides = inputOwner.strides();
				const stride_type inputElementStrideForCharPtr = (stride_type)( ((typename InputNArrayType::value_type *)(0)) + 1 );
#  if CHAR_TARGET
				typedef const char inputTargetType;
				const nstride_type inputCharStrides(inputStrides * inputElementStrideForCharPtr);
#  else
				typedef const InputNArrayType::value_type inputTargetType;
				const nstride_type inputCharStrides(inputStrides);
#  endif
				typedef vctNArrayLoopControl<_dimension, inputTargetType> inputLoopControl;
				typedef typename inputLoopControl::target_vec_type target_vec_type;

				const nstride_type inputSTND = inputLoopControl::CalculateDimWrapCharStrides(inputSizes, inputCharStrides);
				inputTargetType * inputPointer = reinterpret_cast<const inputTargetType *>(inputOwner.Pointer());
                target_vec_type inputTargets =
					inputLoopControl::InitializeTargets(inputSizes, inputCharStrides, inputPointer);

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = inputOwner.dimension();

                value_type minElement, maxElement, inputElement;
                minElement = maxElement = *reinterpret_cast<InputPointerType>(inputPointer);

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    inputElement = *inputPointer;

                    if (inputElement < minElement) {
                        minElement = inputElement;
                    } else if (inputElement > maxElement) {
                        maxElement = inputElement;
                    }

                    numberOfWrappedDimensions =
						inputLoopControl::IncrementPointers<_dimension>(inputTargets, inputPointer, inputCharStrides, inputSTND);
                }
                minValue = minElement;
                maxValue = maxElement;
#else
                // otherwise
                const nsize_type & inputSizes = inputOwner.sizes();
                const nstride_type & inputStrides = inputOwner.strides();
                nstride_type inputSTND;
                vctFixedSizeVector<InputPointerType, _dimension> inputTargets;

                dimension_type numberOfWrappedDimensions = 0;
                const dimension_type maxWrappedDimensions = inputOwner.dimension();

                CalculateSTND(inputSTND, inputSizes, inputStrides);
                InitializeTargets(inputTargets, inputSizes, inputStrides, inputPointer);

                value_type minElement, maxElement, inputElement;
                minElement = maxElement = *inputPointer;

                while (numberOfWrappedDimensions != maxWrappedDimensions) {
                    inputElement = *inputPointer;

                    if (inputElement < minElement) {
                        minElement = inputElement;
                    } else if (inputElement > maxElement) {
                        maxElement = inputElement;
                    }

                    numberOfWrappedDimensions =
                        IncrementPointers(inputTargets, inputPointer, inputStrides, inputSTND);
                }
                minValue = minElement;
                maxValue = maxElement;
#endif
            }
        }   // Run method
    };  // MinAndMax class


};  // vctDynamicNArrayLoopEngines

#ifndef OFRI_NEW_NARRY_LOOP_CONTROL
#error Remove #undef directives at the end of vctDynamicNArrayLoopEngines.h
#endif
#if OFRI_NEW_NARRY_LOOP_CONTROL
#undef CHAR_TARGET
#endif
#undef OFRI_NEW_NARRY_LOOP_CONTROL

#endif  // _vctDynamicNArrayLoopEngines_h

