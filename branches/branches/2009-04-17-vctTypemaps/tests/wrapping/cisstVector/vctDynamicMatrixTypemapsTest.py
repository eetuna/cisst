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

from vctDynamicMatrixTypemapsTestPython import vctDynamicMatrixTypemapsTest
import sys

class DynamicMatrixTypemapsTest(unittest.TestCase):

    ###########################################################################
    #   SET UP function                                                       #
    ###########################################################################

    def setUp(self):
        self.CObject = vctDynamicMatrixTypemapsTest()


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
        goodvar = numpy.ones((5, 7), dtype=numpy.int32)
        exec('self.CObject.' + function + '(goodvar, 0)')


    # Tests that the typemap throws an exception if the data type isn't int
    def StdTestThrowUnlessDataType(self, function):
        # Give an array of floats; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones((5, 7), dtype=numpy.float64)
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an int; expect no exception
        goodvar = numpy.ones((5, 7), dtype=numpy.int32)
        exec('self.CObject.' + function + '(goodvar, 0)')


    # Tests that the typemap throws an exception if the array isn't 2D
    def StdTestThrowUnlessDimension2(self, function):
        # Give a 1D array; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones(10, dtype=numpy.int32)
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a 2D array; expect no exception
        goodvar = numpy.ones((5, 7), dtype=numpy.int32)
        exec('self.CObject.' + function + '(goodvar, 0)')


    # Tests that the typemap throws an exception if the array isn't writable
    def StdTestThrowUnlessWritable(self, function):
        # Give a non-writable array; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones((5, 7), dtype=numpy.int32)
            badvar.setflags(write=False)
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a writable array; expect no exception
        goodvar = numpy.ones((5, 7), dtype=numpy.int32)
        exec('self.CObject.' + function + '(goodvar, 0)')


    def StdTestThrowUnlessOwnsData(self, function):
        # Give a non-memory owning array; expect an exception
        exceptionOccurred = False
        try:
            temp = numpy.ones((5, 7), dtype=numpy.int32)
            badvar = temp[:]
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a memory-owning array; expect no exception
        goodvar = numpy.ones((5, 7), dtype=numpy.int32)
        exec('self.CObject.' + function + '(goodvar, 0)')


    def StdTestThrowUnlessNotReferenced(self, function):
        # Give an array with a reference on it; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones((5, 7), dtype=numpy.int32)
            temp = badvar
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an array with no references; expect no exception
        goodvar = numpy.ones((5, 7), dtype=numpy.int32)
        exec('self.CObject.' + function + '(goodvar, 0)')


    # Test if the C object reads the vector correctly
    def StdTestThrowUnlessReadsCorrectly(self, function):
        vNew = numpy.random.rand(5, 7)
        vNew = numpy.floor(vNew * 100)
        vNew = numpy.array(vNew, dtype=numpy.int32)
        vOld = copy.deepcopy(vNew)
        rows = vNew.shape[0]
        cols = vNew.shape[1]
        exec('self.CObject.' + function + '(vNew, 0)')

        assert(self.CObject.rows() == rows and self.CObject.cols() == cols)
        assert(vNew.shape[0] == rows and vNew.shape[1] == cols)
        for r in xrange(rows):
            for c in xrange(cols):
                # Test if the C object read the vector correctly
                assert(self.CObject.GetItem(r,c) == vOld[r,c])
                # Test that the C object did not modify the vector
                assert(vNew[r,c] == vOld[r,c])


    # Test if the C object reads and modifies the vector correctly
    def StdTestThrowUnlessReadsWritesCorrectly(self, function):
        vNew = numpy.random.rand(5, 7)
        vNew = numpy.floor(vNew * 100)
        vNew = numpy.array(vNew, dtype=numpy.int32)
        vOld = copy.deepcopy(vNew)
        rows = vNew.shape[0]
        cols = vNew.shape[1]
        exec('self.CObject.' + function + '(vNew, 0)')

        assert(self.CObject.rows() == rows and self.CObject.cols() == cols)
        assert(vNew.shape[0] == rows and vNew.shape[1] == cols)
        for r in xrange(rows):
            for c in xrange(cols):
                # Test if the C object read the vector correctly
                assert(self.CObject.GetItem(r,c) == vOld[r,c])
                # Test if the C object modified the vector correctly
                assert(vNew[r,c] == vOld[r,c] + 1)


    # Test if the C object reads, modifies, and resizes the vector correctly
    def StdTestThrowUnlessReadsWritesResizesCorrectly(self, function):
        vNew = numpy.random.rand(5, 7)
        vNew = numpy.floor(vNew * 100)
        vNew = numpy.array(vNew, dtype=numpy.int32)
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


    ###########################################################################
    #   TEST FUNCTIONS - Put any new unit test here!  These are the unit      #
    #   tests that test the typemaps.  One test per typemap!                  #
    ###########################################################################

    def Test_in_argout_vctDynamicMatrix_ref(self):
        MY_NAME = 'in_argout_vctDynamicMatrix_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension2(MY_NAME)
        self.StdTestThrowUnlessWritable(MY_NAME)
        self.StdTestThrowUnlessOwnsData(MY_NAME)
        self.StdTestThrowUnlessNotReferenced(MY_NAME)

        # Perform specialized tests
        self.StdTestThrowUnlessReadsWritesCorrectly(MY_NAME)
        self.StdTestThrowUnlessReadsWritesResizesCorrectly(MY_NAME)


    def Test_in_vctDynamicMatrixRef(self):
        MY_NAME = 'in_vctDynamicMatrixRef'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension2(MY_NAME)
        self.StdTestThrowUnlessWritable(MY_NAME)

        # Perform specialized tests
        self.StdTestThrowUnlessReadsWritesCorrectly(MY_NAME)


    def Test_in_vctDynamicConstMatrixRef(self):
        MY_NAME = 'in_vctDynamicConstMatrixRef'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension2(MY_NAME)

        # Perform specialized tests
        self.StdTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_in_argout_const_vctDynamicConstMatrixRef_ref(self):
        MY_NAME = 'in_argout_const_vctDynamicConstMatrixRef_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension2(MY_NAME)

        # Perform specialized tests
        self.StdTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_in_argout_const_vctDynamicMatrixRef_ref(self):
        MY_NAME = 'in_argout_const_vctDynamicMatrixRef_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension2(MY_NAME)

        # Perform specialized tests
        self.StdTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_in_vctDynamicMatrix(self):
        MY_NAME = 'in_vctDynamicMatrix'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension2(MY_NAME)

        # Perform specialized tests
        self.StdTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_in_argout_const_vctDynamicMatrix_ref(self):
        MY_NAME = 'in_argout_const_vctDynamicMatrix_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension2(MY_NAME)

        # Perform specialized tests
        self.StdTestThrowUnlessReadsCorrectly(MY_NAME)


#     def Test_out_vctDynamicMatrix(self):
#         MY_NAME = 'out_vctDynamicMatrix'

#         # Perform specialized tests
#         SIZE = 10
#         v = self.CObject.out_vctDynamicMatrix(SIZE)
#         assert(v.size == SIZE)  # to make sure v.size isn't zero
#                                 # TODO: do this for (almost) all tests
#         for i in xrange(v.size):
#             assert(self.CObject[i] == v[i])


# We currently do not support the vctDynamicMatrixRef out typemap
#     def Test_out_vctDynamicMatrixRef(self):
#         MY_NAME = 'out_vctDynamicMatrixRef'

#         # Perform specialized tests
#         SIZE = 10
#         v = self.CObject.out_vctDynamicMatrixRef(SIZE)
# #         assert(v.size == SIZE)  # to make sure v.size isn't zero
# #                                 # TODO: do this for (almost) all tests
# #         for i in xrange(v.size):
# #             assert(self.CObject[i] == v[i])
