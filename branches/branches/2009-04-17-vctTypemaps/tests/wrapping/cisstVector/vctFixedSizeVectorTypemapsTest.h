/********************************
 PLACEHOLDER STRINGS TO LOOK FOR:

 TODO       todo
********************************/

#include <cisstVector.h>
#include <iostream>

class vctFixedSizeVectorTypemapsTest
{

protected:

    vctFixedSizeVector<unsigned int, 4> copy;

public:

    vctFixedSizeVectorTypemapsTest()
    {}

    void in_vctFixedSizeVector(vctFixedSizeVector<unsigned int, 4> param, unsigned int dummy) {
        copy.Assign(param);
    }

    vctFixedSizeVector<unsigned int, 4> out_vctFixedSizeVector(void) {
        unsigned int min = 0;
        unsigned int max = 10;
        vctRandom(copy, min, max);
        return copy;
    }

    inline unsigned int __getitem__(unsigned int index) const
    throw(std::out_of_range) {
        return copy.at(index);
    }

    inline void __setitem__(unsigned int index, unsigned int value)
    throw(std::out_of_range) {
        copy.at(index) = value;
    }

    inline unsigned int size() const {
        return copy.size();
    }
};