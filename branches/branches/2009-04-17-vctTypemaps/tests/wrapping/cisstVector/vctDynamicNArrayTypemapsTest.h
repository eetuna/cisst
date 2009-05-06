/********************************
 PLACEHOLDER STRINGS TO LOOK FOR:

 TODO       todo
********************************/

#include <cisstVector.h>
#include <iostream>

#define MY_DIM 3

class vctDynamicNArrayTypemapsTest
{

public:

    // enum {MY_DIM = 3};
    typedef vctDynamicNArray<int, MY_DIM> ArrayType;
    //typedef ArrayType::nsize_type nsize_type;

protected:

    ArrayType copy;

public:

    vctDynamicNArrayTypemapsTest()
    {}

#if 0
    void in_argout_vctDynamicNArray_ref(vctDynamicNArray<int, NDIMS> &param, unsigned int sizeFactor) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
        param += 1;

        if (sizeFactor != 0) {
            unsigned int rowsOld = param.rows();
            unsigned int colsOld = param.cols();
            unsigned int rowsNew = rowsOld * sizeFactor;
            unsigned int colsNew = colsOld * sizeFactor;
            param.resize(rowsNew, colsNew);

            // TODO: is there a better way to do this?
            for (unsigned int r = 0; r < rowsNew; r++) {
                for (unsigned int c = 0; c < colsNew; c++) {
                    param.at(r, c) = param.at(r % rowsOld, c % colsOld);
                }
            }
        }
    }

    void in_vctDynamicNArrayRef(vctDynamicNArrayRef<int, NDIMS> param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
        param += 1;
    }
#endif

    void in_vctDynamicConstNArrayRef(vctDynamicConstNArrayRef<int, MY_DIM> param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

#if 0
    void in_argout_const_vctDynamicConstNArrayRef_ref(const vctDynamicConstNArrayRef<int, NDIMS> &param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicNArrayRef_ref(const vctDynamicNArrayRef<int, NDIMS> &param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

    void in_vctDynamicNArray(vctDynamicNArray<int, NDIMS> param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicNArray_ref(const vctDynamicNArray<int, NDIMS> &param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

    vctDynamicNArray<int, NDIMS> out_vctDynamicNArray(unsigned int rows, unsigned int cols) {
        copy.SetSize(rows, cols);
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return copy;
    }

    vctDynamicNArray<int, NDIMS> &out_vctDynamicNArray_ref(unsigned int rows, unsigned int cols) {
        copy.SetSize(rows, cols);
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return copy;
    }

    const vctDynamicNArray<int, NDIMS> &out_const_vctDynamicNArray_ref(unsigned int rows, unsigned int cols) {
        copy.SetSize(rows, cols);
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return copy;
    }

    vctDynamicNArrayRef<int, NDIMS> out_vctDynamicNArrayRef(unsigned int rows, unsigned int cols) {
        copy.SetSize(rows, cols);
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return copy;
    }

    vctDynamicConstNArrayRef<int, NDIMS> out_vctDynamicConstNArrayRef(unsigned int rows, unsigned int cols) {
        copy.SetSize(rows, cols);
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return copy;
    }
#endif

    inline int Dim(void) const {
        return MY_DIM;
    }

    inline int GetItem(const unsigned int metaIndex) const
    throw(std::out_of_range) {
        return copy.at(metaIndex);
    }

    inline void SetItem(const unsigned int metaIndex, int value)
    throw(std::out_of_range) {
        copy.at(metaIndex) = value;
    }

    inline void sizes(vctFixedSizeVector<int, MY_DIM> &shape) const {
        shape.Assign(copy.sizes());
    }
};