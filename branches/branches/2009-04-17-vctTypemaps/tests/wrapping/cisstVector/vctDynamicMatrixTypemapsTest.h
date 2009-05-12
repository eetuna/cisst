/********************************
 PLACEHOLDER STRINGS TO LOOK FOR:

 TODO       todo
********************************/

#include <cisstVector.h>
#include <iostream>

// TODO: fix the CMakeLists.txt file

template <class _elementType>
class vctDynamicMatrixTypemapsTest
{

protected:

    vctDynamicMatrix<_elementType> copy;

public:

    vctDynamicMatrixTypemapsTest()
    {}

    void in_argout_vctDynamicMatrix_ref(vctDynamicMatrix<_elementType> &param, unsigned int sizeFactor) {
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

    void in_vctDynamicMatrixRef(vctDynamicMatrixRef<_elementType> param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
        param += 1;
    }

    void in_vctDynamicConstMatrixRef(vctDynamicConstMatrixRef<_elementType> param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicConstMatrixRef_ref(const vctDynamicConstMatrixRef<_elementType> &param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicMatrixRef_ref(const vctDynamicMatrixRef<_elementType> &param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

    void in_vctDynamicMatrix(vctDynamicMatrix<_elementType> param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicMatrix_ref(const vctDynamicMatrix<_elementType> &param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

    vctDynamicMatrix<_elementType> out_vctDynamicMatrix(unsigned int rows, unsigned int cols) {
        copy.SetSize(rows, cols);
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return copy;
    }

    vctDynamicMatrix<_elementType> &out_vctDynamicMatrix_ref(unsigned int rows, unsigned int cols) {
        copy.SetSize(rows, cols);
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return copy;
    }

    const vctDynamicMatrix<_elementType> &out_const_vctDynamicMatrix_ref(unsigned int rows, unsigned int cols) {
        copy.SetSize(rows, cols);
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return copy;
    }

    vctDynamicMatrixRef<_elementType> out_vctDynamicMatrixRef(unsigned int rows, unsigned int cols) {
        copy.SetSize(rows, cols);
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return copy;
    }

    vctDynamicConstMatrixRef<_elementType> out_vctDynamicConstMatrixRef(unsigned int rows, unsigned int cols) {
        copy.SetSize(rows, cols);
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return copy;
    }

    inline _elementType GetItem(unsigned int rowIndex, unsigned int colIndex) const
    throw(std::out_of_range) {
        return copy.at(rowIndex, colIndex);
    }

    inline void SetItem(unsigned int rowIndex, unsigned int colIndex, _elementType value)
    throw(std::out_of_range) {
        copy.at(rowIndex, colIndex) = value;
    }

    inline unsigned int rows() const {
        return copy.rows();
    }

    inline unsigned int cols() const {
        return copy.cols();
    }
};