###################################
# Authors: Daniel Li, Anton Deguet
###################################

#######################
# PLACEHOLDER STRINGS TO LOOK FOR:
#
# TODO          todo
#######################

# TODO: If I have time, Document why self.CObject[i] works and check which typemap(s) used
# TODO: Check that StdTestThrowUnlessReads[Writes][Resizes]Correctly mirror each other
# TODO: Clean this code up

import copy
import numpy
import unittest

from vctDynamicVectorTypemapsTestPython import vctDynamicVectorTypemapsTest

class DynamicVectorTypemapsTest(unittest.TestCase):

    def setUp(self):
        self.CObject = vctDynamicVectorTypemapsTest()

    # Standard tests

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
        goodvar = numpy.ones(10, dtype=numpy.int32)
        exec('self.CObject.' + function + '(goodvar, 0)')

    # Tests that the typemap throws an exception if the data type isn't int
    def StdTestThrowUnlessDataType(self, function):
        # Give an array of floats; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones(10, dtype=numpy.float64)
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an int; expect no exception
        goodvar = numpy.ones(10, dtype=numpy.int32)
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
        goodvar = numpy.ones(10, dtype=numpy.int32)
        exec('self.CObject.' + function + '(goodvar, 0)')

    # Tests that the typemap throws an exception if the array isn't writable
    def StdTestThrowUnlessWritable(self, function):
        # Give a non-writable array; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones(10, dtype=numpy.int32)
            badvar.setflags(write=False)
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a writable array; expect no exception
        goodvar = numpy.ones(10, dtype=numpy.int32)
        exec('self.CObject.' + function + '(goodvar, 0)')

    def StdTestThrowUnlessOwnsData(self, function):
        # Give a non-memory owning array; expect an exception
        exceptionOccurred = False
        try:
            temp = numpy.ones(10, dtype=numpy.int32)
            badvar = temp[:]
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a memory-owning array; expect no exception
        goodvar = numpy.ones(10, dtype=numpy.int32)
        exec('self.CObject.' + function + '(goodvar, 0)')

    def StdTestThrowUnlessNotReferenced(self, function):
        # Give an array with a reference on it; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones(10, dtype=numpy.int32)
            temp = badvar
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an array with no references; expect no exception
        goodvar = numpy.ones(10, dtype=numpy.int32)
        exec('self.CObject.' + function + '(goodvar, 0)')

    # Test if the C object reads the vector correctly
    def StdTestThrowUnlessReadsCorrectly(self, function):
        v = numpy.ones(10, dtype=numpy.int32)   # TODO: randomize the vector
        size = v.size;
        exec('self.CObject.' + function + '(v, 0)')

        assert(self.CObject.size() == size)
        assert(v.size == size)
        for i in xrange(size):
            # Test if the C object read the vector correctly
            assert(self.CObject[i] == v[i])

    # Test if the C object reads and modifies the vector correctly
    def StdTestThrowUnlessReadsWritesCorrectly(self, function):
        v = numpy.ones(10, dtype=numpy.int32)   # TODO: randomize the vector
        vOrig = copy.deepcopy(v)
        exec('self.CObject.' + function + '(v, 0)')

        assert(self.CObject.size() == vOrig.size)
        assert(v.size == vOrig.size)
        for i in xrange(vOrig.size):
            # Test if the C object read the vector correctly
            assert(self.CObject[i] == vOrig[i])
            # Test if the C object modified the vector correctly
            assert(v[i] == vOrig[i] + 1)

    # Test if the C object reads, modifies, and resizes the vector correctly
    def StdTestThrowUnlessReadsWritesResizesCorrectly(self, function):
        v = numpy.ones(10, dtype=numpy.int32)   # TODO: randomize the vector
        vOrig = copy.deepcopy(v)
        SIZEFACTOR = 2
        exec('self.CObject.' + function + '(v, SIZEFACTOR)')

        assert(self.CObject.size() == vOrig.size)
        # TODO: Should I resize CObject.copy as well?  More generally, should I
        # look at making the C++ code for in_argout_vctDynamicVector_ref (see
        # the .h file) better?
        assert(v.size == vOrig.size * SIZEFACTOR)
        for i in xrange(vOrig.size):
            # Test if the C object read the vector correctly
            assert(self.CObject[i] == vOrig[i])
            # Test if the C object modified the vector correctly
            assert(v[i] == vOrig[i] + 1)
            # Test if the C object resized the vector correctly
            for j in xrange(SIZEFACTOR):
                assert(vOrig[i] + 1 == v[i + (vOrig.size * j)])

    # Tests

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
        self.StdTestThrowUnlessReadsWritesCorrectly(MY_NAME)
        self.StdTestThrowUnlessReadsWritesResizesCorrectly(MY_NAME)

    def Test_in_vctDynamicVectorRef(self):
        MY_NAME = 'in_vctDynamicVectorRef'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension1(MY_NAME)
        self.StdTestThrowUnlessWritable(MY_NAME)

        # Perform specialized tests
        self.StdTestThrowUnlessReadsWritesCorrectly(MY_NAME)

    def Test_in_vctDynamicConstVectorRef(self):
        MY_NAME = 'in_vctDynamicConstVectorRef'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension1(MY_NAME)

        # Perform specialized tests
        self.StdTestThrowUnlessReadsCorrectly(MY_NAME)

    def Test_in_argout_const_vctDynamicConstVectorRef_ref(self):
        MY_NAME = 'in_argout_const_vctDynamicConstVectorRef_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension1(MY_NAME)

        # Perform specialized tests
        self.StdTestThrowUnlessReadsCorrectly(MY_NAME)

    def Test_in_argout_const_vctDynamicVectorRef_ref(self):
        MY_NAME = 'in_argout_const_vctDynamicVectorRef_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension1(MY_NAME)

        # Perform specialized tests
        self.StdTestThrowUnlessReadsCorrectly(MY_NAME)

    def Test_in_vctDynamicVector(self):
        MY_NAME = 'in_vctDynamicVector'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension1(MY_NAME)

        # Perform specialized tests
        self.StdTestThrowUnlessReadsCorrectly(MY_NAME)

    def Test_in_argout_const_vctDynamicVector_ref(self):
        MY_NAME = 'in_argout_const_vctDynamicVector_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension1(MY_NAME)

        # Perform specialized tests
        self.StdTestThrowUnlessReadsCorrectly(MY_NAME)

    def Test_out_vctDynamicVector(self):
        MY_NAME = 'out_vctDynamicVector'

        # Perform specialized tests
        SIZE = 10
        v = self.CObject.out_vctDynamicVector(SIZE)
        assert(v.size == SIZE)  # to make sure v.size isn't zero
                                # TODO: do this for (almost) all tests
        for i in xrange(v.size):
            assert(self.CObject[i] == v[i])

# We currently do not support the vctDynamicVectorRef out typemap
#     def Test_out_vctDynamicVectorRef(self):
#         MY_NAME = 'out_vctDynamicVectorRef'

#         # Perform specialized tests
#         SIZE = 10
#         v = self.CObject.out_vctDynamicVectorRef(SIZE)
# #         assert(v.size == SIZE)  # to make sure v.size isn't zero
# #                                 # TODO: do this for (almost) all tests
# #         for i in xrange(v.size):
# #             assert(self.CObject[i] == v[i])
