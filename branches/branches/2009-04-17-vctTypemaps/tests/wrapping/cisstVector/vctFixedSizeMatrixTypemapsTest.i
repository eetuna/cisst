%module vctFixedSizeMatrixTypemapsTestPython

%include "cisstVector/cisstVector.i"


%header %{
    // Put header files here
    #include "vctFixedSizeMatrixTypemapsTest.h"
%}

%include "vctFixedSizeMatrixTypemapsTest.h"

%template(vctFixedSizeMatrixTypemapsTest_int_4_4) vctFixedSizeMatrixTypemapsTest<int, 4, 4>;
%template(vctFixedSizeMatrixTypemapsTest_uint_4_4) vctFixedSizeMatrixTypemapsTest<unsigned int, 4, 4>;
%template(vctFixedSizeMatrixTypemapsTest_double_4_4) vctFixedSizeMatrixTypemapsTest<double, 4, 4>;
