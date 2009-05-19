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

from cisstVectorTypemapsTestPython import vctFixedSizeVectorTypemapsTest_double_4
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
            function(badvar)
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an array; expect no exception
        size = self.CObject.size()
        goodvar = numpy.ones(size, dtype=self.dtype)
        function(goodvar)


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
            function(badvar)
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an int; expect no exception
        goodvar = numpy.ones(size, dtype=self.dtype)
        function(goodvar)


    # Tests that the typemap throws an exception if the array isn't 1D
    def StdTestThrowUnlessDimension1(self, function):
        # Give a 2D array; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.array([[1, 2, 3], [4, 5, 6]])
            function(badvar)
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a 1D array; expect no exception
        size = self.CObject.size()
        goodvar = numpy.ones(size, dtype=self.dtype)
        function(goodvar)


    # Tests that the typemap throws an exception if there is a size mismatch
    def StdTestThrowUnlessCorrectVectorSize(self, function):
        size = self.CObject.size()

        # Give a `size+1' array; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones(size + 1, dtype=self.dtype)
            function(badvar)
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a `size' array; expect no exception
        goodvar = numpy.ones(size, dtype=self.dtype)
        function(goodvar)


    # Tests that the typemap throws an exception if the array isn't writable
    def StdTestThrowUnlessWritable(self, function):
        size = self.CObject.size()

        # Give a non-writable array; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones(size, dtype=self.dtype)
            badvar.setflags(write=False)
            function(badvar)
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a writable array; expect no exception
        goodvar = numpy.ones(size, dtype=self.dtype)
        function(goodvar)


    def StdTestThrowUnlessReturnedVectorIsWritable(self, function):
        # Expect the returned array to be writable
        v = function()
        assert(v.flags['WRITEABLE'] == True)


    def StdTestThrowUnlessReturnedVectorIsNonWritable(self, function):
        # Expect the returned array to be non-writable
        v = function()
        assert(v.flags['WRITEABLE'] == False)


    # Test if the C object reads the vector correctly
    def SpecTestThrowUnlessReadsCorrectly(self, function):
        size = self.CObject.size()
        vNew = numpy.random.rand(size)
        vNew = numpy.floor(vNew * 100)
        vNew = numpy.array(vNew, dtype=self.dtype)
        vOld = copy.deepcopy(vNew)
        size = vNew.size
        function(vNew)

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
        function(vNew)

        assert(self.CObject.size() == size)
        assert(vNew.size == size)
        for i in xrange(size):
            # Test if the C object read the vector correctly
            assert(self.CObject[i] == vOld[i])
            # Test if the C object modified the vector correctly
            assert(vNew[i] == vOld[i] + 1)


    # Test if the C object returns a good vector
    def SpecTestThrowUnlessReceivesCorrectVector(self, function):
        v = function()

        size = self.CObject.size()
        for i in xrange(size):
            assert(self.CObject[i] == v[i])


    ###########################################################################
    #   TEST FUNCTIONS - Put any new unit test here!  These are the unit      #
    #   tests that test the typemaps.  One test per typemap!                  #
    ###########################################################################

    def Test_in_argout_vctFixedSizeVector_ref(self):
        function = self.CObject.in_argout_vctFixedSizeVector_ref

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(function)
        self.StdTestThrowUnlessDataType(function)
        self.StdTestThrowUnlessDimension1(function)
        self.StdTestThrowUnlessCorrectVectorSize(function)
        self.StdTestThrowUnlessWritable(function)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsWritesCorrectly(function)


    def Test_out_vctFixedSizeVector_ref(self):
        function = self.CObject.out_vctFixedSizeVector_ref

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedVectorIsWritable(function)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectVector(function)


    def Test_in_vctFixedSizeVector(self):
        function = self.CObject.in_vctFixedSizeVector

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(function)
        self.StdTestThrowUnlessDataType(function)
        self.StdTestThrowUnlessDimension1(function)
        self.StdTestThrowUnlessCorrectVectorSize(function)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(function)


    def Test_out_vctFixedSizeVector(self):
        function = self.CObject.out_vctFixedSizeVector

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedVectorIsWritable(function)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectVector(function)


    def Test_in_argout_const_vctFixedSizeVector_ref(self):
        function = self.CObject.in_argout_const_vctFixedSizeVector_ref

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(function)
        self.StdTestThrowUnlessDataType(function)
        self.StdTestThrowUnlessDimension1(function)
        self.StdTestThrowUnlessCorrectVectorSize(function)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(function)


    def Test_out_const_vctFixedSizeVector_ref(self):
        function = self.CObject.out_const_vctFixedSizeVector_ref

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedVectorIsNonWritable(function)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectVector(function)
