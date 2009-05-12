/********************************
 PLACEHOLDER STRINGS TO LOOK FOR:

 TODO       todo
********************************/

#include <cisstVector.h>
#include <iostream>

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