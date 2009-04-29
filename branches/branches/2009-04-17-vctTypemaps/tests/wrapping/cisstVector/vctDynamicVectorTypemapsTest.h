/********************************
 PLACEHOLDER STRINGS TO LOOK FOR:

 TODO       todo
********************************/

#include <cisstVector.h>
#include <iostream>

class vctDynamicVectorTypemapsTest
{
public:
    vctDynamicVector<int> copy;

    vctDynamicVectorTypemapsTest() {
    }

    void in_argout_vctDynamicVector_ref(vctDynamicVector<int> &param, unsigned int sizefactor) {
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

    void in_vctDynamicVectorRef(vctDynamicVectorRef<int> param, unsigned int dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
        param += 1;
    }

    void in_vctDynamicConstVectorRef(vctDynamicConstVectorRef<int> param, unsigned int dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicConstVectorRef_ref(const vctDynamicConstVectorRef<int> &param, unsigned int dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicVectorRef_ref(const vctDynamicVectorRef<int> &param, unsigned int dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }

    void in_vctDynamicVector(vctDynamicVector<int> param, unsigned int dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }

    void in_argout_const_vctDynamicVector_ref(const vctDynamicVector<int> &param, unsigned int dummy) {
        copy.SetSize(param.size());
        copy.Assign(param);
    }

    vctDynamicVector<int> out_vctDynamicVector(unsigned int size) {
        copy.SetSize(size);
        vctRandom(copy, 0, 10);
        return copy;
    }

    /* We currently do not support the vctDynamicVectorRef out typemap
    vctDynamicVectorRef<int> out_vctDynamicVectorRef(unsigned int size) throw(std::exception) {
        copy.SetSize(size);
        vctRandom(copy, 0, 10);
        return vctDynamicVectorRef<int>(copy);
    }*/

    inline int __getitem__(unsigned int index) const throw(std::out_of_range) {
        return copy.at(index);
    }

    inline void __setitem__(unsigned int index, int value) throw(std::out_of_range) {
        copy.at(index) = value;
    }

    unsigned int size() const {
        return copy.size();
    }
};