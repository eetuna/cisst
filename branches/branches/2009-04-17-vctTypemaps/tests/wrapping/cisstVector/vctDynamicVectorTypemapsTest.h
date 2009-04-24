#include <cisstVector.h>
#include <iostream>

class vctDynamicVectorTypemapsTest
{
public:
    vctDynamicVector<int> copy;

    void in_argout_vctDynamicVector_ref(vctDynamicVector<int> &param, unsigned int size) {
        /*if (size == 0) {
            copy.SetSize(...);
            copy.Assign(param);
            param += 1.0;
        } else {
            param.Resize(size);    // [1, 2, 3, 4, 5] --> [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        }*/
        std::cout << "Wee" << std::endl;
    }
};