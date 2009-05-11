%module vctDynamicMatrixTypemapsTestPython

%include "cisstVector/cisstVector.i"


%header %{
    // Put header files here
    #include "vctDynamicMatrixTypemapsTest.h"
%}

%include "vctDynamicMatrixTypemapsTest.h"

%template(vctDynamicMatrixTypemapsTest_int) vctDynamicMatrixTypemapsTest<int>;
%template(vctDynamicMatrixTypemapsTest_double) vctDynamicMatrixTypemapsTest<double>;
