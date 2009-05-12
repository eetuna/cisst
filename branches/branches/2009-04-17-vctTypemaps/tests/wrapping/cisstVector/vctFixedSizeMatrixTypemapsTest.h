/********************************
 PLACEHOLDER STRINGS TO LOOK FOR:

 TODO       todo
********************************/

#include <cisstVector.h>
#include <iostream>

// TODO: Check if the `delete $1' statements are actually part of the .cxx file;
//       i.e. that the argout typemaps are actually being used

class vctFixedSizeMatrixTypemapsTest
{

protected:

    vctFixedSizeMatrix<unsigned int, 4, 4> copy;

public:

    vctFixedSizeMatrixTypemapsTest()
    {}

    void in_vctFixedSizeMatrix(vctFixedSizeMatrix<unsigned int, 4, 4> param) {
        copy.Assign(param);
    }

    vctFixedSizeMatrix<unsigned int, 4, 4> out_vctFixedSizeMatrix(void) {
        unsigned int min = 0;
        unsigned int max = 10;
        vctRandom(copy, min, max);     // TODO: this is actually not random!
        return copy;
    }

    void in_argout_vctFixedSizeMatrix_ref(vctFixedSizeMatrix<unsigned int, 4, 4> &param) {
        copy.Assign(param);
        param += 1;
    }

    vctFixedSizeMatrix<unsigned int, 4, 4> &out_vctFixedSizeMatrix_ref(void) {
        unsigned int min = 0;
        unsigned int max = 0;
        vctRandom(copy, min, max);     // TODO: this is actually not random!
        return copy;
    }

    void in_argout_const_vctFixedSizeMatrix_ref(const vctFixedSizeMatrix<unsigned int, 4, 4> &param) {
        copy.Assign(param);
    }

    const vctFixedSizeMatrix<unsigned int, 4, 4> &out_const_vctFixedSizeMatrix_ref(void) {
        unsigned int min = 0;
        unsigned int max = 0;
        vctRandom(copy, min, max);     // TODO: this is actually not random!
        return copy;
    }

    inline unsigned int GetItem(unsigned int rowIndex, unsigned int colIndex) const
    throw(std::out_of_range) {
        return copy.at(rowIndex, colIndex);
    }

    inline void SetItem(unsigned int rowIndex, unsigned int colIndex, unsigned int value)
    throw(std::out_of_range) {
        copy.at(rowIndex, colIndex) = value;
    }

    inline unsigned int rows(void) const {
        return copy.rows();
    }

    inline unsigned int cols(void) const {
        return copy.cols();
    }

    inline vctFixedSizeVector<unsigned int, 2> sizes(void) const {
        return copy.sizes();
    }
};