%module vctFixedSizeVectorTypemapsTestPython

%include "cisstVector/cisstVector.i"


%header %{
    // Put header files here
    #include "vctFixedSizeVectorTypemapsTest.h"
%}

%include "vctFixedSizeVectorTypemapsTest.h"

%template(vctFixedSizeVectorTypemapsTest_int_4) vctFixedSizeVectorTypemapsTest<int, 4>;
%template(vctFixedSizeVectorTypemapsTest_uint_4) vctFixedSizeVectorTypemapsTest<unsigned int, 4>;
%template(vctFixedSizeVectorTypemapsTest_double_4) vctFixedSizeVectorTypemapsTest<double, 4>;
