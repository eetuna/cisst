/********************************
 PLACEHOLDER STRINGS TO LOOK FOR:

 TODO       todo
********************************/

#include <cisstVector.h>
#include <iostream>

typedef unsigned int size_type;

template <class _elementType>
class vctDynamicVectorTypemapsTest
{

protected:

    vctDynamicVector<_elementType> copy;

public:

    vctDynamicVectorTypemapsTest()
    {}

    void in_argout_vctDynamicVector_ref(vctDynamicVector<_elementType> &param, size_type sizeFactor) {
        copy.SetSize(param.size());
        copy.Assign(param);
        param += 1;

        if (sizeFactor != 0) {
            size_type size = param.size();
            size_type newsize = size * sizeFactor;
            param.resize(newsize);

            // TODO: is there a better way to do this?
            for (size_type i = size; i < newsize; i++) {
                param[i] = param[i % size];
            }
        }
    }

    void in_vctDynamicVectorRef(vctDynamicVectorRef<_elementType> param, size_type dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
        param += 1;
    }

    void in_vctDynamicConstVectorRef(vctDynamicConstVectorRef<_elementType> param, size_type dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicConstVectorRef_ref(const vctDynamicConstVectorRef<_elementType> &param, size_type dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicVectorRef_ref(const vctDynamicVectorRef<_elementType> &param, size_type dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }

    void in_vctDynamicVector(vctDynamicVector<_elementType> param, size_type dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicVector_ref(const vctDynamicVector<_elementType> &param, size_type dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }

    vctDynamicVector<_elementType> out_vctDynamicVector(size_type size) {
        copy.SetSize(size);
        size_type min = 0;
        size_type max = 10;
        vctRandom(copy, 0, 10);
        return copy;
    }

    vctDynamicVector<_elementType> &out_vctDynamicVector_ref(size_type size) {
        copy.SetSize(size);
        size_type min = 0;
        size_type max = 10;
        vctRandom(copy, 0, 10);
        return copy;
    }

    const vctDynamicVector<_elementType> &out_const_vctDynamicVector_ref(size_type size) {
        copy.SetSize(size);
        size_type min = 0;
        size_type max = 10;
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return copy;
    }

    vctDynamicVectorRef<_elementType> out_vctDynamicVectorRef(size_type size) {
        copy.SetSize(size);
        size_type min = 0;
        size_type max = 10;
        vctRandom(copy, 0, 10);
        return vctDynamicVectorRef<_elementType>(copy);
    }

    vctDynamicConstVectorRef<_elementType> out_vctDynamicConstVectorRef(size_type size) {
        copy.SetSize(size);
        size_type min = 0;
        size_type max = 10;
        vctRandom(copy, 0, 10);
        return vctDynamicConstVectorRef<_elementType>(copy);
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