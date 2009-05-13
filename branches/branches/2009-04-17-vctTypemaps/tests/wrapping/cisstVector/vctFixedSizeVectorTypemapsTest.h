/********************************
 PLACEHOLDER STRINGS TO LOOK FOR:

 TODO       todo
********************************/

#include <cisstVector.h>
#include <iostream>

typedef unsigned int size_type;

template <class _elementType, size_type _size>
class vctFixedSizeVectorTypemapsTest
{

protected:

    vctFixedSizeVector<_elementType, _size> copy;

public:

    vctFixedSizeVectorTypemapsTest()
    {}

    // TODO: Should we support returning a FixedSize vector of different length; in other words, emulating resizing?
    void in_argout_vctFixedSizeVector_ref(vctFixedSizeVector<_elementType, _size> &param) {
        copy.Assign(param);
        param += 1;
    }

    vctFixedSizeVector<_elementType, _size> &out_vctFixedSizeVector_ref(void) {
        _elementType min = 0;
        _elementType max = 10;
        vctRandom(copy, min, max);
        return copy;
    }

    void in_vctFixedSizeVector(vctFixedSizeVector<_elementType, _size> param) {
        copy.Assign(param);
    }

    vctFixedSizeVector<_elementType, _size> out_vctFixedSizeVector(void) {
        _elementType min = 0;
        _elementType max = 10;
        vctRandom(copy, min, max);
        return copy;
    }

    void in_argout_const_vctFixedSizeVector_ref(const vctFixedSizeVector<_elementType, _size> &param) {
        copy.Assign(param);
    }

    const vctFixedSizeVector<_elementType, _size> &out_const_vctFixedSizeVector_ref(void) {
        _elementType min = 0;
        _elementType max = 10;
        vctRandom(copy, min, max);
        return copy;
    }

    inline _elementType __getitem__(size_type index) const
    throw(std::out_of_range) {
        return copy.at(index);
    }

    inline void __setitem__(size_type index, _elementType value)
    throw(std::out_of_range) {
        copy.at(index) = value;
    }

    inline size_type size() const {
        return copy.size();
    }
};