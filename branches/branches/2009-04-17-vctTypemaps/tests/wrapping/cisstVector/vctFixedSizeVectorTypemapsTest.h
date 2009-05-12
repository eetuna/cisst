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

    // TODO: Should we support returning a FixedSize vector of different length; in other words, emulating resizing?
    void in_argout_vctFixedSizeVector_ref(vctFixedSizeVector<unsigned int, 4> &param) {
        copy.Assign(param);
        param += 1;
    }

    vctFixedSizeVector<unsigned int, 4> &out_vctFixedSizeVector_ref(void) {
        unsigned int min = 0;
        unsigned int max = 10;
        vctRandom(copy, min, max);
        return copy;
    }

    void in_vctFixedSizeVector(vctFixedSizeVector<unsigned int, 4> param) {
        copy.Assign(param);
    }

    vctFixedSizeVector<unsigned int, 4> out_vctFixedSizeVector(void) {
        unsigned int min = 0;
        unsigned int max = 10;
        vctRandom(copy, min, max);
        return copy;
    }

    void in_argout_const_vctFixedSizeVector_ref(const vctFixedSizeVector<unsigned int, 4> &param) {
        copy.Assign(param);
    }

    const vctFixedSizeVector<unsigned int, 4> &out_const_vctFixedSizeVector_ref(void) {
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