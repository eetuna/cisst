###################################
# Authors: Daniel Li, Anton Deguet
###################################

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

    # Test if the C object reads the vector correctly
    def StdTestThrowUnlessReadsCorrectly(self, function):
        v = numpy.ones(10, dtype=numpy.int32)   # TODO: randomize the vector
        exec('self.CObject.' + function + '(v, 0)')

        for i in xrange(v.size):
            # Test if the C object read the vector correctly
            assert(self.CObject.copy[i] == v[i])

    # Test if the C object reads and modifies the vector correctly
    def StdTestThrowUnlessReadsWritesCorrectly(self, function):
        v = numpy.ones(10, dtype=numpy.int32)   # TODO: randomize the vector
        vOrig = copy.deepcopy(v)
        exec('self.CObject.' + function + '(v, 0)')

        for i in xrange(v.size):
            # Test if the C object read the vector correctly
            assert(self.CObject.copy[i] == vOrig[i])
            # Test if the C object modified the vector correctly
            assert(v[i] == vOrig[i] + 1)

    # Test if the C object reads, modifies, and resizes the vector correctly
    def StdTestThrowUnlessReadsWritesResizesCorrectly(self, function):
        v = numpy.ones(10, dtype=numpy.int32)   # TODO: randomize the vector
        vOrig = copy.deepcopy(v)
        SIZEFACTOR = 2
        exec('self.CObject.' + function + '(v, SIZEFACTOR)')

        assert(v.size == vOrig.size * SIZEFACTOR)
        for i in xrange(vOrig.size):
            # Test if the C object read the vector correctly
            assert(self.CObject.copy[i] == vOrig[i])
            # Test if the C object modified the vector correctly
            assert(v[i] == vOrig[i] + 1)
        # QUESTION: [1, 2, 3, 4, 5] --> [1, 2, 3, 4, 5, 1, 2, 3, 4, 5] ?

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