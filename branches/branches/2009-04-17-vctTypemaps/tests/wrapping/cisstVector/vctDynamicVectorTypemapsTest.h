#include <cisstVector.h>
#include <iostream>

class vctDynamicVectorTypemapsTest
{
public:
    vctDynamicVector<int> copy;

    void in_argout_vctDynamicVector_ref(vctDynamicVector<int> &param, unsigned int sizefactor) {
        copy.SetSize(param.size());
        copy.Assign(param);
        param += 1;
        
        if (sizefactor != 0) {
            unsigned int size = param.size() * sizefactor;
            param.resize(size);
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
};