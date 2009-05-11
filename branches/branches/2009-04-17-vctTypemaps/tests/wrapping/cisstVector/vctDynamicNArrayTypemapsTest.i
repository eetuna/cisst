%module vctDynamicNArrayTypemapsTestPython

%include "cisstVector/cisstVector.i"


%header %{
    // Put header files here
    #include "vctDynamicNArrayTypemapsTest.h"
%}

%include "vctDynamicNArrayTypemapsTest.h"

%template(vctDynamicNArrayTypemapsTest_int) vctDynamicNArrayTypemapsTest<int>;
%template(vctDynamicNArrayTypemapsTest_double) vctDynamicNArrayTypemapsTest<double>;
