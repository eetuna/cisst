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
};