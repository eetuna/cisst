%module cisstVectorTypemapsTestPython

%include "cisstVector/cisstVector.i"


%header %{
    // Put header files here
    #include "vctDynamicVectorTypemapsTest.h"
    #include "vctDynamicMatrixTypemapsTest.h"
    #include "vctDynamicNArrayTypemapsTest.h"
    
    #include "vctFixedSizeVectorTypemapsTest.h"
    #include "vctFixedSizeMatrixTypemapsTest.h"
%}


%include "vctDynamicVectorTypemapsTest.h"
%template(vctDynamicVectorTypemapsTest_int) vctDynamicVectorTypemapsTest<int>;
%template(vctDynamicVectorTypemapsTest_double) vctDynamicVectorTypemapsTest<double>;

%include "vctDynamicMatrixTypemapsTest.h"
%template(vctDynamicMatrixTypemapsTest_int) vctDynamicMatrixTypemapsTest<int>;
%template(vctDynamicMatrixTypemapsTest_double) vctDynamicMatrixTypemapsTest<double>;

%include "vctDynamicNArrayTypemapsTest.h"
%template(vctDynamicNArrayTypemapsTest_int_4) vctDynamicNArrayTypemapsTest<int, 4>;
%template(vctDynamicNArrayTypemapsTest_double_4) vctDynamicNArrayTypemapsTest<double, 4>;

%include "vctFixedSizeVectorTypemapsTest.h"
%template(vctFixedSizeVectorTypemapsTest_int_4) vctFixedSizeVectorTypemapsTest<int, 4>;
%template(vctFixedSizeVectorTypemapsTest_uint_4) vctFixedSizeVectorTypemapsTest<unsigned int, 4>;
%template(vctFixedSizeVectorTypemapsTest_double_4) vctFixedSizeVectorTypemapsTest<double, 4>;

%include "vctFixedSizeMatrixTypemapsTest.h"
%template(vctFixedSizeMatrixTypemapsTest_int_4_4) vctFixedSizeMatrixTypemapsTest<int, 4, 4>;
%template(vctFixedSizeMatrixTypemapsTest_uint_4_4) vctFixedSizeMatrixTypemapsTest<unsigned int, 4, 4>;
%template(vctFixedSizeMatrixTypemapsTest_double_4_4) vctFixedSizeMatrixTypemapsTest<double, 4, 4>;
