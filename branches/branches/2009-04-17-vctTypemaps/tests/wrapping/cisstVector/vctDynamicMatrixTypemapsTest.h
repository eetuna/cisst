/********************************
 PLACEHOLDER STRINGS TO LOOK FOR:

 TODO       todo
********************************/

#include <cisstVector.h>
#include <iostream>

class vctDynamicMatrixTypemapsTest
{

protected:

    vctDynamicMatrix<int> copy;

public:

    vctDynamicMatrixTypemapsTest()
    {}

    void in_argout_vctDynamicMatrix_ref(vctDynamicMatrix<int> &param, unsigned int sizeFactor) {
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

    void in_vctDynamicMatrixRef(vctDynamicMatrixRef<int> param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
        param += 1;
    }

    void in_vctDynamicConstMatrixRef(vctDynamicConstMatrixRef<int> param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicConstMatrixRef_ref(const vctDynamicConstMatrixRef<int> &param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicMatrixRef_ref(const vctDynamicMatrixRef<int> &param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

    void in_vctDynamicMatrix(vctDynamicMatrix<int> param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicMatrix_ref(const vctDynamicMatrix<int> &param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

#if 0
    vctDynamicMatrix<int> out_vctDynamicMatrix(unsigned int size) {
        copy.SetSize(size);
        vctRandom(copy, 0, 10);
        return copy;
    }

    /* We currently do not support the vctDynamicMatrixRef out typemap
    vctDynamicMatrixRef<int> out_vctDynamicMatrixRef(unsigned int size) throw(std::exception) {
        copy.SetSize(size);
        vctRandom(copy, 0, 10);     // TODO: this is actually not random!
        return vctDynamicMatrixRef<int>(copy);
    }*/
#endif

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
};