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

from vctFixedSizeVectorTypemapsTestPython import vctFixedSizeVectorTypemapsTest_double_4
import sys

class FixedSizeVectorTypemapsTest(unittest.TestCase):

    dtype = numpy.double

    ###########################################################################
    #   SET UP function                                                       #
    ###########################################################################

    def setUp(self):
        self.CObject = vctFixedSizeVectorTypemapsTest_double_4()


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
            exec('self.CObject.' + function + '(badvar)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an array; expect no exception
        size = self.CObject.size()
        goodvar = numpy.ones(size, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar)')


    # Tests that the typemap throws an exception if the data type isn't int
    def StdTestThrowUnlessDataType(self, function):
        size = self.CObject.size()

        # Give an array of floats; expect an exception
        exceptionOccurred = False
        try:
            if (self.dtype != numpy.float64):
                badvar = numpy.ones(size, dtype=numpy.float64)
            else:
                badvar = numpy.ones(size, dtype=numpy.int32)
            exec('self.CObject.' + function + '(badvar)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an int; expect no exception
        goodvar = numpy.ones(size, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar)')


    # Tests that the typemap throws an exception if the array isn't 1D
    def StdTestThrowUnlessDimension1(self, function):
        # Give a 2D array; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.array([[1, 2, 3], [4, 5, 6]])
            exec('self.CObject.' + function + '(badvar)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a 1D array; expect no exception
        size = self.CObject.size()
        goodvar = numpy.ones(size, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar)')


    # Tests that the typemap throws an exception if the array isn't writable
    def StdTestThrowUnlessWritable(self, function):
        size = self.CObject.size()

        # Give a non-writable array; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones(size, dtype=self.dtype)
            badvar.setflags(write=False)
            exec('self.CObject.' + function + '(badvar)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a writable array; expect no exception
        goodvar = numpy.ones(size, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar)')


    def StdTestThrowUnlessOwnsData(self, function):
        size = self.CObject.size()

        # Give a non-memory owning array; expect an exception
        exceptionOccurred = False
        try:
            temp = numpy.ones(size, dtype=self.dtype)
            badvar = temp[:]
            exec('self.CObject.' + function + '(badvar)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a memory-owning array; expect no exception
        goodvar = numpy.ones(size, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar)')


    def StdTestThrowUnlessNotReferenced(self, function):
        size = self.CObject.size()

        # Give an array with a reference on it; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones(size, dtype=self.dtype)
            temp = badvar
            exec('self.CObject.' + function + '(badvar)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an array with no references; expect no exception
        goodvar = numpy.ones(size, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar)')


    def StdTestThrowUnlessReturnedVectorIsWritable(self, function):
        # Expect the returned array to be writable
        exec('v = self.CObject.' + function + '()')
        assert(v.flags['WRITEABLE'] == True)


    def StdTestThrowUnlessReturnedVectorIsNonWritable(self, function):
        # Expect the returned array to be non-writable
        exec('v = self.CObject.' + function + '()')
        assert(v.flags['WRITEABLE'] == False)


    # Test if the C object reads the vector correctly
    def SpecTestThrowUnlessReadsCorrectly(self, function):
        size = self.CObject.size()
        vNew = numpy.random.rand(size)
        vNew = numpy.floor(vNew * 100)
        vNew = numpy.array(vNew, dtype=self.dtype)
        vOld = copy.deepcopy(vNew)
        size = vNew.size
        exec('self.CObject.' + function + '(vNew)')

        assert(self.CObject.size() == size)
        assert(vNew.size == size)
        for i in xrange(size):
            # Test if the C object read the vector correctly
            assert(self.CObject[i] == vOld[i])
            # Test that the C object did not modify the vector
            assert(vNew[i] == vOld[i])


    # Test if the C object reads and modifies the vector correctly
    def SpecTestThrowUnlessReadsWritesCorrectly(self, function):
        size = self.CObject.size()
        vNew = numpy.random.rand(size)
        vNew = numpy.floor(vNew * 100)
        vNew = numpy.array(vNew, dtype=self.dtype)
        vOld = copy.deepcopy(vNew)
        size = vNew.size
        exec('self.CObject.' + function + '(vNew)')

        assert(self.CObject.size() == size)
        assert(vNew.size == size)
        for i in xrange(size):
            # Test if the C object read the vector correctly
            assert(self.CObject[i] == vOld[i])
            # Test if the C object modified the vector correctly
            assert(vNew[i] == vOld[i] + 1)


    # Test if the C object returns a good vector
    def SpecTestThrowUnlessReceivesCorrectVector(self, function):
        exec('v = self.CObject.' + function + '()')

        size = self.CObject.size()
        for i in xrange(size):
            assert(self.CObject[i] == v[i])


    ###########################################################################
    #   TEST FUNCTIONS - Put any new unit test here!  These are the unit      #
    #   tests that test the typemaps.  One test per typemap!                  #
    ###########################################################################

    def Test_in_argout_vctFixedSizeVector_ref(self):
        MY_NAME = 'in_argout_vctFixedSizeVector_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension1(MY_NAME)
        self.StdTestThrowUnlessWritable(MY_NAME)
        self.StdTestThrowUnlessOwnsData(MY_NAME)
        self.StdTestThrowUnlessNotReferenced(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsWritesCorrectly(MY_NAME)


    def Test_out_vctFixedSizeVector_ref(self):
        MY_NAME = 'out_vctFixedSizeVector_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedVectorIsWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectVector(MY_NAME)


    def Test_in_vctFixedSizeVector(self):
        MY_NAME = 'in_vctFixedSizeVector'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension1(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_out_vctFixedSizeVector(self):
        MY_NAME = 'out_vctFixedSizeVector'

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedVectorIsWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectVector(MY_NAME)


    def Test_in_argout_const_vctFixedSizeVector_ref(self):
        MY_NAME = 'in_argout_const_vctFixedSizeVector_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension1(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_out_const_vctFixedSizeVector_ref(self):
        MY_NAME = 'out_const_vctFixedSizeVector_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedVectorIsNonWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectVector(MY_NAME)
