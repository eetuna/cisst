###################################
# Authors: Daniel Li, Anton Deguet
###################################

#######################
# PLACEHOLDER STRINGS TO LOOK FOR:
#
# TODO          todo
#######################

# TODO: If I have time, Document why self.CObject[i] works and check which typemap(s) used
# TODO: Clean this code up

import copy
import numpy
import unittest

from vctDynamicVectorTypemapsTestPython import vctDynamicVectorTypemapsTest_double
import sys

class DynamicVectorTypemapsTest(unittest.TestCase):

    dtype = numpy.double

    ###########################################################################
    #   SET UP function                                                       #
    ###########################################################################

    def setUp(self):
        self.CObject = vctDynamicVectorTypemapsTest_double()


    ###########################################################################
    #   STANDARD TESTS - These are the library of tests that will be called   #
    #   by the test functions.                                                #
    ###########################################################################

    # Tests that the typemap throws an exception if the argument isn't an array
    def StdTestThrowUnlessIsArray(self, function):
        # Give a non-array; expect an exception
        exceptionOccurred = False
        try:
            badvar = 0.0
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an array; expect no exception
        goodvar = numpy.ones(10, dtype=self.dtype)
        #self.CObject.in_argout_vctDynamicVector_ref(goodvar, 0)      # DEBUG
        #print sys.getrefcount(goodvar)                               # DEBUG
        exec('self.CObject.' + function + '(goodvar, 0)')


    # Tests that the typemap throws an exception if the data type isn't int
    def StdTestThrowUnlessDataType(self, function):
        # Give an array of floats; expect an exception
        exceptionOccurred = False
        try:
            if (self.dtype != numpy.float64):
                badvar = numpy.ones(10, dtype=numpy.float64)
            else:
                badvar = numpy.ones(10, dtype=numpy.int32)
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an int; expect no exception
        goodvar = numpy.ones(10, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar, 0)')


    # Tests that the typemap throws an exception if the array isn't 1D
    def StdTestThrowUnlessDimension1(self, function):
        # Give a 2D array; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.array([[1, 2, 3], [4, 5, 6]])
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a 1D array; expect no exception
        goodvar = numpy.ones(10, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar, 0)')


    # Tests that the typemap throws an exception if the array isn't writable
    def StdTestThrowUnlessWritable(self, function):
        # Give a non-writable array; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones(10, dtype=self.dtype)
            badvar.setflags(write=False)
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a writable array; expect no exception
        goodvar = numpy.ones(10, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar, 0)')


    def StdTestThrowUnlessOwnsData(self, function):
        # Give a non-memory owning array; expect an exception
        exceptionOccurred = False
        try:
            temp = numpy.ones(10, dtype=self.dtype)
            badvar = temp[:]
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a memory-owning array; expect no exception
        goodvar = numpy.ones(10, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar, 0)')


    def StdTestThrowUnlessNotReferenced(self, function):
        # Give an array with a reference on it; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones(10, dtype=self.dtype)
            temp = badvar
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an array with no references; expect no exception
        goodvar = numpy.ones(10, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar, 0)')


    def StdTestThrowUnlessReturnedVectorIsWritable(self, function):
        # Expect the returned array to be writable
        SIZE = 10
        exec('v = self.CObject.' + function + '(SIZE)')
        assert(v.flags['WRITEABLE'] == True)


    def StdTestThrowUnlessReturnedVectorIsNonWritable(self, function):
        # Expect the returned array to be non-writable
        SIZE = 10
        exec('v = self.CObject.' + function + '(SIZE)')
        assert(v.flags['WRITEABLE'] == False)


    # Test if the C object reads the vector correctly
    def SpecTestThrowUnlessReadsCorrectly(self, function):
        vNew = numpy.random.rand(10)
        vNew = numpy.floor(vNew * 100)
        vNew = numpy.array(vNew, dtype=self.dtype)
        vOld = copy.deepcopy(vNew)
        size = vNew.size
        exec('self.CObject.' + function + '(vNew, 0)')

        assert(self.CObject.size() == size)
        assert(vNew.size == size)
        for i in xrange(size):
            # Test if the C object read the vector correctly
            assert(self.CObject[i] == vOld[i])
            # Test that the C object did not modify the vector
            assert(vNew[i] == vOld[i])


    # Test if the C object reads and modifies the vector correctly
    def SpecTestThrowUnlessReadsWritesCorrectly(self, function):
        vNew = numpy.random.rand(10)
        vNew = numpy.floor(vNew * 100)
        vNew = numpy.array(vNew, dtype=self.dtype)
        vOld = copy.deepcopy(vNew)
        size = vNew.size
        exec('self.CObject.' + function + '(vNew, 0)')

        assert(self.CObject.size() == size)
        assert(vNew.size == size)
        for i in xrange(size):
            # Test if the C object read the vector correctly
            assert(self.CObject[i] == vOld[i])
            # Test if the C object modified the vector correctly
            assert(vNew[i] == vOld[i] + 1)


    # Test if the C object reads, modifies, and resizes the vector correctly
    def SpecTestThrowUnlessReadsWritesResizesCorrectly(self, function):
        vNew = numpy.random.rand(10)
        vNew = numpy.floor(vNew * 100)
        vNew = numpy.array(vNew, dtype=self.dtype)
        vOld = copy.deepcopy(vNew)
        size = vNew.size
        SIZE_FACTOR = 3
        exec('self.CObject.' + function + '(vNew, SIZE_FACTOR)')

        assert(self.CObject.size() == size)
        assert(vNew.size == size * SIZE_FACTOR)
        for i in xrange(size):
            # Test if the C object read the vector correctly
            assert(self.CObject[i] == vOld[i])
            # Test if the C object modified the vector correctly
            assert(vNew[i] == vOld[i] + 1)
            # Test if the C object resized the vector correctly
            for j in xrange(SIZE_FACTOR):
                assert(vOld[i] + 1 == vNew[i + size*j])


    # Test if the C object returns a good vector
    def SpecTestThrowUnlessReceivesCorrectVector(self, function):
        SIZE = 10
        exec('v = self.CObject.' + function + '(SIZE)')

        assert(v.size == SIZE)
        for i in xrange(SIZE):
            assert(self.CObject[i] == v[i])


    ###########################################################################
    #   TEST FUNCTIONS - Put any new unit test here!  These are the unit      #
    #   tests that test the typemaps.  One test per typemap!                  #
    ###########################################################################

    def Test_in_argout_vctDynamicVector_ref(self):
        MY_NAME = 'in_argout_vctDynamicVector_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension1(MY_NAME)
        self.StdTestThrowUnlessWritable(MY_NAME)
        self.StdTestThrowUnlessOwnsData(MY_NAME)
        self.StdTestThrowUnlessNotReferenced(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsWritesCorrectly(MY_NAME)
        self.SpecTestThrowUnlessReadsWritesResizesCorrectly(MY_NAME)


    def Test_in_vctDynamicVectorRef(self):
        MY_NAME = 'in_vctDynamicVectorRef'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension1(MY_NAME)
        self.StdTestThrowUnlessWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsWritesCorrectly(MY_NAME)


    def Test_in_vctDynamicConstVectorRef(self):
        MY_NAME = 'in_vctDynamicConstVectorRef'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension1(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_in_argout_const_vctDynamicConstVectorRef_ref(self):
        MY_NAME = 'in_argout_const_vctDynamicConstVectorRef_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension1(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_in_argout_const_vctDynamicVectorRef_ref(self):
        MY_NAME = 'in_argout_const_vctDynamicVectorRef_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension1(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_in_vctDynamicVector(self):
        MY_NAME = 'in_vctDynamicVector'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension1(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_in_argout_const_vctDynamicVector_ref(self):
        MY_NAME = 'in_argout_const_vctDynamicVector_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension1(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_out_vctDynamicVector(self):
        MY_NAME = 'out_vctDynamicVector'

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedVectorIsWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectVector(MY_NAME)


    def Test_out_vctDynamicVector_ref(self):
        MY_NAME = 'out_vctDynamicVector_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedVectorIsWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectVector(MY_NAME)


    def Test_out_const_vctDynamicVector_ref(self):
        MY_NAME = 'out_const_vctDynamicVector_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedVectorIsNonWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectVector(MY_NAME)


    def Test_out_vctDynamicVectorRef(self):
        MY_NAME = 'out_vctDynamicVectorRef'

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedVectorIsWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectVector(MY_NAME)


    def Test_out_vctDynamicConstVectorRef(self):
        MY_NAME = 'out_vctDynamicConstVectorRef'

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedVectorIsNonWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectVector(MY_NAME)
