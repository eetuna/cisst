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

#if 0
    void in_argout_vctDynamicMatrix_ref(vctDynamicMatrix<int> &param, unsigned int sizefactor) {
        copy.SetSize(param.size());
        copy.Assign(param);
        param += 1;

        if (sizefactor != 0) {
            unsigned int size = param.size();
            unsigned int newsize = size * sizefactor;
            param.resize(newsize);

            // TODO: is there a better way to do this?
            for (unsigned int i = size; i < newsize; i++) {
                param[i] = param[i % size];
            }
        }
    }

    void in_vctDynamicMatrixRef(vctDynamicMatrixRef<int> param, unsigned int dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
        param += 1;
    }

    void in_vctDynamicConstMatrixRef(vctDynamicConstMatrixRef<int> param, unsigned int dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicConstMatrixRef_ref(const vctDynamicConstMatrixRef<int> &param, unsigned int dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicMatrixRef_ref(const vctDynamicMatrixRef<int> &param, unsigned int dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }
#endif

    void in_vctDynamicMatrix(vctDynamicMatrix<int> param, unsigned int dummy) {
        copy.SetSize(param.sizes());
        copy.Assign(param);
    }

#if 0
    void in_argout_const_vctDynamicMatrix_ref(const vctDynamicMatrix<int> &param, unsigned int dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }

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

    inline int getitem(unsigned int rowIndex, unsigned int colIndex) const
    throw(std::out_of_range) {
        return copy.at(rowIndex, colIndex);
    }

    inline int __getitem__(unsigned int index) const
    throw(std::out_of_range) {
        return copy.at(index);
    }

    inline void SetItem(unsigned int rowIndex, unsigned int colIndex, int value)
    throw(std::out_of_range) {
        copy.at(rowIndex, colIndex) = value;
    }

    inline void __setitem__(unsigned int index, int value)
    throw(std::out_of_range) {
        copy.at(index) = value;
    }

    inline unsigned int rows() const {
        return copy.rows();
    }

    inline unsigned int cols() const {
        return copy.cols();
    }
};