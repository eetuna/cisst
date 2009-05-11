/********************************
 PLACEHOLDER STRINGS TO LOOK FOR:

 TODO       todo
********************************/

#include <cisstVector.h>
#include <iostream>

template <class _elementType, unsigned int _dimension>
class vctDynamicNArrayTypemapsTest
{

public:

    // enum {MY_DIM = 4};     // SWIG cannot understand enums
    typedef vctDynamicNArray<_elementType, _dimension> ArrayType;
    //typedef ArrayType::nsize_type nsize_type;

protected:

    ArrayType copy;

public:

    vctDynamicNArrayTypemapsTest()
    {}

    void in_argout_vctDynamicNArray_ref(vctDynamicNArray<_elementType, _dimension> &param, unsigned int sizeFactor) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
        param += 1;

        if (sizeFactor != 0) {
            const vctFixedSizeVector<unsigned int, _dimension> sizesOld(param.sizes());
            const unsigned int sizeOld = sizesOld.ProductOfElements();
            const vctFixedSizeVector<unsigned int, _dimension> sizesNew(sizesOld * sizeFactor);
            const unsigned int sizeNew = sizesNew.ProductOfElements();
            param.SetSize(sizesNew);

            // Fill all elements with a non-zero value
            for (unsigned int i = 0; i < sizeNew; i++) {
                param.at(i) = 17;       // TODO: modify this so it only fills the NEW elements, not ALL elements, with 17
            }
        }
    }

    void in_vctDynamicNArrayRef(vctDynamicNArrayRef<_elementType, _dimension> param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
        param += 1;
    }

    void in_vctDynamicConstNArrayRef(vctDynamicConstNArrayRef<_elementType, _dimension> param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicConstNArrayRef_ref(const vctDynamicConstNArrayRef<_elementType, _dimension> &param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicNArrayRef_ref(const vctDynamicNArrayRef<_elementType, _dimension> &param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

    void in_vctDynamicNArray(vctDynamicNArray<_elementType, _dimension> param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicNArray_ref(const vctDynamicNArray<_elementType, _dimension> &param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

    vctDynamicNArray<_elementType, _dimension> out_vctDynamicNArray(/*vctFixedSizeVector<unsigned int, _dimension> sizes*/) {     // TODO: I think there's something wrong with the vctFixedSizeVector in typemap
        vctFixedSizeVector<unsigned int, _dimension> sizes;
        sizes.SetAll(5);
        copy.SetSize(sizes);
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return copy;
    }

    vctDynamicNArray<_elementType, _dimension> &out_vctDynamicNArray_ref(/*vctFixedSizeVector<unsigned int, _dimension> sizes*/) {     // TODO: I think there's something wrong with the vctFixedSizeVector in typemap
        vctFixedSizeVector<unsigned int, _dimension> sizes;
        sizes.SetAll(5);
        copy.SetSize(sizes);
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return copy;
    }

    const vctDynamicNArray<_elementType, _dimension> &out_const_vctDynamicNArray_ref(/*vctFixedSizeVector<unsigned int, _dimension> sizes*/) {     // TODO: I think there's something wrong with the vctFixedSizeVector in typemap
        vctFixedSizeVector<unsigned int, _dimension> sizes;
        sizes.SetAll(5);
        copy.SetSize(sizes);
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return copy;
    }

    vctDynamicNArrayRef<_elementType, _dimension> out_vctDynamicNArrayRef(/*vctFixedSizeVector<unsigned int, _dimension> sizes*/) {     // TODO: I think there's something wrong with the vctFixedSizeVector in typemap
        vctFixedSizeVector<unsigned int, _dimension> sizes;
        sizes.SetAll(5);
        copy.SetSize(sizes);
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return copy;
    }

    vctDynamicConstNArrayRef<_elementType, _dimension> out_vctDynamicConstNArrayRef(/*vctFixedSizeVector<unsigned int, _dimension> sizes*/) {     // TODO: I think there's something wrong with the vctFixedSizeVector in typemap
        vctFixedSizeVector<unsigned int, _dimension> sizes;
        sizes.SetAll(5);
        copy.SetSize(sizes);
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return copy;
    }

    inline unsigned int Dim(void) const {
        return _dimension;
    }

    inline _elementType GetItem(const unsigned int metaIndex) const
    throw(std::out_of_range) {
        return copy.at(metaIndex);
    }

    inline void SetItem(const unsigned int metaIndex, _elementType value)
    throw(std::out_of_range) {
        copy.at(metaIndex) = value;
    }

    inline void sizes(vctFixedSizeVector<_elementType, _dimension> &shape) const {
        shape.Assign(copy.sizes());
    }
};