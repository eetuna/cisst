/********************************
 PLACEHOLDER STRINGS TO LOOK FOR:

 TODO       todo
********************************/

#include <cisstVector.h>
#include <iostream>

// TODO: Check if the `delete $1' statements are actually part of the .cxx file;
//       i.e. that the argout typemaps are actually being used

typedef unsigned int size_type;

template <class _elementType, size_type _rows, size_type _cols>
class vctFixedSizeMatrixTypemapsTest
{

protected:

    vctFixedSizeMatrix<_elementType, _rows, _cols> copy;

public:

    vctFixedSizeMatrixTypemapsTest()
    {}

    void in_vctFixedSizeMatrix(vctFixedSizeMatrix<_elementType, _rows, _cols> param) {
        copy.Assign(param);
    }

    vctFixedSizeMatrix<_elementType, _rows, _cols> out_vctFixedSizeMatrix(void) {
        _elementType min = 0;
        _elementType max = 10;
        vctRandom(copy, min, max);     // TODO: this is actually not random!
        return copy;
    }

    void in_argout_vctFixedSizeMatrix_ref(vctFixedSizeMatrix<_elementType, _rows, _cols> &param) {
        copy.Assign(param);
        param += 1;
    }

    vctFixedSizeMatrix<_elementType, _rows, _cols> &out_vctFixedSizeMatrix_ref(void) {
        _elementType min = 0;
        _elementType max = 0;
        vctRandom(copy, min, max);     // TODO: this is actually not random!
        return copy;
    }

    void in_argout_const_vctFixedSizeMatrix_ref(const vctFixedSizeMatrix<_elementType, _rows, _cols> &param) {
        copy.Assign(param);
    }

    const vctFixedSizeMatrix<_elementType, _rows, _cols> &out_const_vctFixedSizeMatrix_ref(void) {
        _elementType min = 0;
        _elementType max = 0;
        vctRandom(copy, min, max);     // TODO: this is actually not random!
        return copy;
    }

    inline _elementType GetItem(size_type rowIndex, size_type colIndex) const
    throw(std::out_of_range) {
        return copy.at(rowIndex, colIndex);
    }

    inline void SetItem(size_type rowIndex, size_type colIndex, _elementType value)
    throw(std::out_of_range) {
        copy.at(rowIndex, colIndex) = value;
    }

    inline size_type rows(void) const {
        return copy.rows();
    }

    inline size_type cols(void) const {
        return copy.cols();
    }

    inline vctFixedSizeVector<size_type, 2> sizes(void) const {
        return copy.sizes();
    }
};