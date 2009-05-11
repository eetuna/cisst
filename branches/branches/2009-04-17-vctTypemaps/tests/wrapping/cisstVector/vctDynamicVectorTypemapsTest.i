%module vctDynamicVectorTypemapsTestPython

%include "cisstVector/cisstVector.i"


%header %{
    // Put header files here
    #include "vctDynamicVectorTypemapsTest.h"
%}

%include "vctDynamicVectorTypemapsTest.h"

%template(vctDynamicVectorTypemapsTest_int) vctDynamicVectorTypemapsTest<int>;
%template(vctDynamicVectorTypemapsTest_double) vctDynamicVectorTypemapsTest<double>;
