/********************************
 PLACEHOLDER STRINGS TO LOOK FOR:

 TODO       todo
********************************/

#include <cisstVector.h>
#include <iostream>

class vctDynamicNArrayTypemapsTest
{

protected:

    static const int NDIMS = 5;
    vctDynamicNArray<int, NDIMS> copy;

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

    void in_vctDynamicConstNArrayRef(vctDynamicConstNArrayRef<int, NDIMS> param, unsigned int dummy) {
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

    inline int GetItem(unsigned int rowIndex, unsigned int colIndex) const
    throw(std::out_of_range) {
        return copy.at(rowIndex, colIndex);
    }

    inline void SetItem(unsigned int rowIndex, unsigned int colIndex, int value)
    throw(std::out_of_range) {
        copy.at(rowIndex, colIndex) = value;
    }

    inline unsigned int rows() const {
        return copy.rows();
    }

    inline unsigned int cols() const {
        return copy.cols();
    }
#endif
};