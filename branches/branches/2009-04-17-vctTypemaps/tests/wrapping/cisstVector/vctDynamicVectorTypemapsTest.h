/********************************
 PLACEHOLDER STRINGS TO LOOK FOR:

 TODO       todo
********************************/

#include <cisstVector.h>
#include <iostream>

template <class _elementType>
class vctDynamicVectorTypemapsTest
{

protected:

    vctDynamicVector<_elementType> copy;

public:

    vctDynamicVectorTypemapsTest()
    {}

    void in_argout_vctDynamicVector_ref(vctDynamicVector<_elementType> &param, unsigned int sizeFactor) {
        copy.SetSize(param.size());
        copy.Assign(param);
        param += 1;

        if (sizeFactor != 0) {
            unsigned int size = param.size();
            unsigned int newsize = size * sizeFactor;
            param.resize(newsize);

            // TODO: is there a better way to do this?
            for (unsigned int i = size; i < newsize; i++) {
                param[i] = param[i % size];
            }
        }
    }

    void in_vctDynamicVectorRef(vctDynamicVectorRef<_elementType> param, unsigned int dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
        param += 1;
    }

    void in_vctDynamicConstVectorRef(vctDynamicConstVectorRef<_elementType> param, unsigned int dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicConstVectorRef_ref(const vctDynamicConstVectorRef<_elementType> &param, unsigned int dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicVectorRef_ref(const vctDynamicVectorRef<_elementType> &param, unsigned int dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }

    void in_vctDynamicVector(vctDynamicVector<_elementType> param, unsigned int dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicVector_ref(const vctDynamicVector<_elementType> &param, unsigned int dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }

    vctDynamicVector<_elementType> out_vctDynamicVector(unsigned int size) {
        copy.SetSize(size);
        vctRandom(copy, 0, 10);
        return copy;
    }

    vctDynamicVector<_elementType> &out_vctDynamicVector_ref(unsigned int size) {
        copy.SetSize(size);
        vctRandom(copy, 0, 10);
        return copy;
    }

    const vctDynamicVector<_elementType> &out_const_vctDynamicVector_ref(unsigned int size) {
        copy.SetSize(size);
        vctRandom(copy, 0, 10);
        return copy;
    }

    vctDynamicVectorRef<_elementType> out_vctDynamicVectorRef(unsigned int size) {
        copy.SetSize(size);
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return vctDynamicVectorRef<_elementType>(copy);
    }

    vctDynamicConstVectorRef<_elementType> out_vctDynamicConstVectorRef(unsigned int size) {
        copy.SetSize(size);
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return vctDynamicConstVectorRef<_elementType>(copy);
    }

    inline _elementType __getitem__(unsigned int index) const
    throw(std::out_of_range) {
        return copy.at(index);
    }

    inline void __setitem__(unsigned int index, _elementType value)
    throw(std::out_of_range) {
        copy.at(index) = value;
    }

    inline unsigned int size() const {
        return copy.size();
    }
};