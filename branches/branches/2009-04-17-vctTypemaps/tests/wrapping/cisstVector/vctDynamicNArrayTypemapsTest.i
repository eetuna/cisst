%module vctDynamicNArrayTypemapsTestPython

%include "cisstVector/cisstVector.i"


%header %{
    // Put header files here
    #include "vctDynamicNArrayTypemapsTest.h"
%}

%include "vctDynamicNArrayTypemapsTest.h"

%template(vctDynamicNArrayTypemapsTest_int_4) vctDynamicNArrayTypemapsTest<int, 4>;
%template(vctDynamicNArrayTypemapsTest_double_4) vctDynamicNArrayTypemapsTest<double, 4>;
