#######################
# PLACEHOLDER STRINGS TO LOOK FOR:
#
# QUESTION      question to ask Anton
# TODO          todo
#######################

import copy
import numpy
import unittest

from vctDynamicVectorTypemapsTestPython import vctDynamicVectorTypemapsTest

class DynamicVectorTypemapsTest(unittest.TestCase):

    def setUp(self):
        self.CObject = vctDynamicVectorTypemapsTest()
        print '--new test--'

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
        # QUESTION: Should the test give any array (not just a 1D array of ints)?
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

    # Tests

    def Test_in_argout_vctDynamicVector_ref(self):
        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray('in_argout_vctDynamicVector_ref')
        self.StdTestThrowUnlessDataType('in_argout_vctDynamicVector_ref')
        self.StdTestThrowUnlessDimension1('in_argout_vctDynamicVector_ref')
        self.StdTestThrowUnlessWritable('in_argout_vctDynamicVector_ref')
        self.StdTestThrowUnlessOwnsData('in_argout_vctDynamicVector_ref')
        self.StdTestThrowUnlessNotReferenced('in_argout_vctDynamicVector_ref')

        # Perform specialized tests

        # Test if the C object reads and modifies the vector correctly
        # def StdTestThrowUnlessReadsWritesCorrectly(self, function):
        v = numpy.ones(10, dtype=numpy.int32)   # TODO: randomize the vector
        v_copy = copy.deepcopy(v)
        self.CObject.in_argout_vctDynamicVector_ref(v, 0)

        for i in xrange(v.size):
            # Test if the C object read the vector correctly
            assert(self.CObject.copy[i] == v_copy[i])
            # Test if the C object modified the vector correctly
            assert(v[i] == v_copy[i] + 1)



        # Test if the C object reads, modifies, and resizes the vector correctly
        # def StdTestThrowUnlessReadsWritesResizesCorrectly(self, function):
        v = numpy.ones(10, dtype=numpy.int32)   # TODO: randomize the vector
        v_copy = copy.deepcopy(v)
        SIZEFACTOR = 2
        self.CObject.in_argout_vctDynamicVector_ref(v, SIZEFACTOR)

        assert(v.size == v_copy.size * SIZEFACTOR)
        for i in xrange(v_copy.size):
            # Test if the C object read the vector correctly
            assert(self.CObject.copy[i] == v_copy[i])
            # Test if the C object modified the vector correctly
            assert(v[i] == v_copy[i] + 1)
        # QUESTION: [1, 2, 3, 4, 5] --> [1, 2, 3, 4, 5, 1, 2, 3, 4, 5] ?