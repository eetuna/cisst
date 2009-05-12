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

from vctFixedSizeMatrixTypemapsTestPython import vctFixedSizeMatrixTypemapsTest
import sys

class FixedSizeMatrixTypemapsTest(unittest.TestCase):

    dtype = numpy.uint32

    ###########################################################################
    #   SET UP function                                                       #
    ###########################################################################

    def setUp(self):
        self.CObject = vctFixedSizeMatrixTypemapsTest()


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
        shape = tuple(self.CObject.sizes())
        goodvar = numpy.ones(shape, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar)')


    # Tests that the typemap throws an exception if the data type isn't int
    def StdTestThrowUnlessDataType(self, function):
        shape = tuple(self.CObject.sizes())

        # Give an array of floats; expect an exception
        exceptionOccurred = False
        try:
            if (self.dtype != numpy.float64):
                badvar = numpy.ones(shape, dtype=numpy.float64)
            else:
                badvar = numpy.ones(shape, dtype=numpy.int32)
            exec('self.CObject.' + function + '(badvar)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an int; expect no exception
        goodvar = numpy.ones(shape, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar)')


    # Tests that the typemap throws an exception if the array isn't 2D
    def StdTestThrowUnlessDimension2(self, function):
        # Give a 1D array; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones(10, dtype=self.dtype)
            exec('self.CObject.' + function + '(badvar)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a 2D array; expect no exception
        shape = tuple(self.CObject.sizes())
        goodvar = numpy.ones(shape, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar)')


    # Tests that the typemap throws an exception if the array isn't writable
    def StdTestThrowUnlessWritable(self, function):
        assert(False)
        shape = tuple(self.CObject.sizes())

        # Give a non-writable array; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones((5, 7), dtype=self.dtype)
            badvar.setflags(write=False)
            exec('self.CObject.' + function + '(badvar)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a writable array; expect no exception
        goodvar = numpy.ones((5, 7), dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar)')


    def StdTestThrowUnlessOwnsData(self, function):
        assert(False)
        shape = tuple(self.CObject.sizes())

        # Give a non-memory owning array; expect an exception
        exceptionOccurred = False
        try:
            temp = numpy.ones((5, 7), dtype=self.dtype)
            badvar = temp[:]
            exec('self.CObject.' + function + '(badvar)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a memory-owning array; expect no exception
        goodvar = numpy.ones((5, 7), dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar)')


    def StdTestThrowUnlessNotReferenced(self, function):
        assert(False)
        shape = tuple(self.CObject.sizes())

        # Give an array with a reference on it; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones((5, 7), dtype=self.dtype)
            temp = badvar
            exec('self.CObject.' + function + '(badvar)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an array with no references; expect no exception
        goodvar = numpy.ones((5, 7), dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar)')


    def StdTestThrowUnlessReturnedMatrixIsWritable(self, function):
        # Expect the returned array to be writable
        exec('v = self.CObject.' + function + '()')
        assert(v.flags['WRITEABLE'] == True)


    def StdTestThrowUnlessReturnedMatrixIsNonWritable(self, function):
        assert(False)

        # Expect the returned array to be non-writable
        ROWS = 5
        COLS = 7
        exec('v = self.CObject.' + function + '(ROWS, COLS)')
        assert(v.flags['WRITEABLE'] == False)


    # Test if the C object reads the vector correctly
    def SpecTestThrowUnlessReadsCorrectly(self, function):
        shape = tuple(self.CObject.sizes())

        vNew = numpy.random.random_sample(shape)
        vNew = numpy.floor(vNew * 100)
        vNew = numpy.array(vNew, dtype=self.dtype)
        vOld = copy.deepcopy(vNew)
        rows = vNew.shape[0]
        cols = vNew.shape[1]
        exec('self.CObject.' + function + '(vNew)')

        assert(self.CObject.rows() == rows and self.CObject.cols() == cols)
        assert(vNew.shape[0] == rows and vNew.shape[1] == cols)
        for r in xrange(rows):
            for c in xrange(cols):
                # Test if the C object read the vector correctly
                assert(self.CObject.GetItem(r,c) == vOld[r,c])
                # Test that the C object did not modify the vector
                assert(vNew[r,c] == vOld[r,c])


    # Test if the C object reads and modifies the vector correctly
    def SpecTestThrowUnlessReadsWritesCorrectly(self, function):
        assert(False)

        vNew = numpy.random.rand(5, 7)
        vNew = numpy.floor(vNew * 100)
        vNew = numpy.array(vNew, dtype=self.dtype)
        vOld = copy.deepcopy(vNew)
        rows = vNew.shape[0]
        cols = vNew.shape[1]
        exec('self.CObject.' + function + '(vNew)')

        assert(self.CObject.rows() == rows and self.CObject.cols() == cols)
        assert(vNew.shape[0] == rows and vNew.shape[1] == cols)
        for r in xrange(rows):
            for c in xrange(cols):
                # Test if the C object read the vector correctly
                assert(self.CObject.GetItem(r,c) == vOld[r,c])
                # Test if the C object modified the vector correctly
                assert(vNew[r,c] == vOld[r,c] + 1)


    # Test if the C object reads, modifies, and resizes the vector correctly
    def SpecTestThrowUnlessReadsWritesResizesCorrectly(self, function):
        assert(False)

        vNew = numpy.random.rand(5, 7)
        vNew = numpy.floor(vNew * 100)
        vNew = numpy.array(vNew, dtype=self.dtype)
        vOld = copy.deepcopy(vNew)
        rows = vNew.shape[0]
        cols = vNew.shape[1]
        SIZE_FACTOR = 3
        exec('self.CObject.' + function + '(vNew, SIZE_FACTOR)')

        assert(self.CObject.rows() == rows and self.CObject.cols() == cols)
        assert(vNew.shape[0] == rows * SIZE_FACTOR and vNew.shape[1] == cols * SIZE_FACTOR)
        for r in xrange(rows):
            for c in xrange(cols):
                # Test if the C object read the vector correctly
                assert(self.CObject.GetItem(r,c) == vOld[r,c])
                # Test if the C object modified the vector correctly
                assert(vNew[r,c] == vOld[r,c] + 1)
                # Test if the C object resized the vector correctly
                for r2 in xrange(SIZE_FACTOR):
                    for c2 in xrange(SIZE_FACTOR):
                        assert(vOld[r,c] + 1 == vNew[r + rows*r2, c + cols*c2])


    # Test if the C object returns a good vector
    def SpecTestThrowUnlessReceivesCorrectMatrix(self, function):
        exec('v = self.CObject.' + function + '()')

        rows = self.CObject.rows()
        cols = self.CObject.cols()
        assert(v.shape[0] == rows and v.shape[1] == cols)
        for r in xrange(rows):
            for c in xrange(cols):
                assert(self.CObject.GetItem(r,c) == v[r,c])


    ###########################################################################
    #   TEST FUNCTIONS - Put any new unit test here!  These are the unit      #
    #   tests that test the typemaps.  One test per typemap!                  #
    ###########################################################################

    def Test_in_vctFixedSizeMatrix(self):
        MY_NAME = 'in_vctFixedSizeMatrix'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension2(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_out_vctFixedSizeMatrix(self):
        MY_NAME = 'out_vctFixedSizeMatrix'

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedMatrixIsWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectMatrix(MY_NAME)
