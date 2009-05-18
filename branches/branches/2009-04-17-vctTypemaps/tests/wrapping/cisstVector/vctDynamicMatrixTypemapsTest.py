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

from vctDynamicMatrixTypemapsTestPython import vctDynamicMatrixTypemapsTest_double
import sys

class DynamicMatrixTypemapsTest(unittest.TestCase):

    dtype = numpy.double

    ###########################################################################
    #   SET UP function                                                       #
    ###########################################################################

    def setUp(self):
        self.CObject = vctDynamicMatrixTypemapsTest_double()


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
        goodvar = numpy.ones((5, 7), dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar, 0)')


    # Tests that the typemap throws an exception if the data type isn't int
    def StdTestThrowUnlessDataType(self, function):
        # Give an array of floats; expect an exception
        exceptionOccurred = False
        try:
            if (self.dtype != numpy.float64):
                badvar = numpy.ones((5, 7), dtype=numpy.float64)
            else:
                badvar = numpy.ones((5, 7), dtype=numpy.int32)
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an int; expect no exception
        goodvar = numpy.ones((5, 7), dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar, 0)')


    # Tests that the typemap throws an exception if the array isn't 2D
    def StdTestThrowUnlessDimension2(self, function):
        # Give a 1D array; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones(10, dtype=self.dtype)
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a 2D array; expect no exception
        goodvar = numpy.ones((5, 7), dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar, 0)')


    # Tests that the typemap throws an exception if the array isn't writable
    def StdTestThrowUnlessWritable(self, function):
        # Give a non-writable array; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones((5, 7), dtype=self.dtype)
            badvar.setflags(write=False)
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a writable array; expect no exception
        goodvar = numpy.ones((5, 7), dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar, 0)')


    def StdTestThrowUnlessOwnsData(self, function):
        # Give a non-memory owning array; expect an exception
        exceptionOccurred = False
        try:
            temp = numpy.ones((5, 7), dtype=self.dtype)
            badvar = temp[:]
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a memory-owning array; expect no exception
        goodvar = numpy.ones((5, 7), dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar, 0)')


    def StdTestThrowUnlessNotReferenced(self, function):
        # Give an array with a reference on it; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones((5, 7), dtype=self.dtype)
            temp = badvar
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an array with no references; expect no exception
        goodvar = numpy.ones((5, 7), dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar, 0)')


    def StdTestThrowUnlessReturnedMatrixIsWritable(self, function):
        # Expect the returned array to be writable
        ROWS = 5
        COLS = 7
        exec('v = self.CObject.' + function + '(ROWS, COLS)')
        assert(v.flags['WRITEABLE'] == True)


    def StdTestThrowUnlessReturnedMatrixIsNonWritable(self, function):
        # Expect the returned array to be non-writable
        ROWS = 5
        COLS = 7
        exec('v = self.CObject.' + function + '(ROWS, COLS)')
        assert(v.flags['WRITEABLE'] == False)


    # Test if the C object reads the vector correctly
    def SpecTestThrowUnlessReadsCorrectly(self, function):
        STORAGE_ORDER = ['C', 'F']

        for stord in STORAGE_ORDER:
            vNew = numpy.random.rand(5, 7)
            vNew = numpy.floor(vNew * 100)
            vNew = numpy.array(vNew, dtype=self.dtype, order=stord)
            vOld = copy.deepcopy(vNew)
            vOld = numpy.array(vOld, order=stord)
            rows = vNew.shape[0]
            cols = vNew.shape[1]
            exec('self.CObject.' + function + '(vNew, 0)')

            assert(self.CObject.rows() == rows and self.CObject.cols() == cols)
            assert(vNew.shape[0] == rows and vNew.shape[1] == cols)
            try:
                assert(self.CObject.StorageOrder() == stord)
            except:
                print self.CObject.StorageOrder(), stord
                raise
            for r in xrange(rows):
                for c in xrange(cols):
                    # Test if the C object read the vector correctly
                    assert(self.CObject.GetItem(r,c) == vOld[r,c])
                    # Test that the C object did not modify the vector
                    assert(vNew[r,c] == vOld[r,c])


    # Test if the C object reads and modifies the vector correctly
    def SpecTestThrowUnlessReadsWritesCorrectly(self, function):
        STORAGE_ORDER = ['C', 'F']

        for stord in STORAGE_ORDER:
            vNew = numpy.random.rand(5, 7)
            vNew = numpy.floor(vNew * 100)
            vNew = numpy.array(vNew, dtype=self.dtype, order=stord)
            vOld = copy.deepcopy(vNew)
            vOld = numpy.array(vOld, order=stord)
            rows = vNew.shape[0]
            cols = vNew.shape[1]
            exec('self.CObject.' + function + '(vNew, 0)')

            assert(self.CObject.rows() == rows and self.CObject.cols() == cols)
            assert(vNew.shape[0] == rows and vNew.shape[1] == cols)
            try:
                assert(self.CObject.StorageOrder() == stord)
            except:
                print self.CObject.StorageOrder(), stord
                raise
            for r in xrange(rows):
                for c in xrange(cols):
                    # Test if the C object read the vector correctly
                    assert(self.CObject.GetItem(r,c) == vOld[r,c])
                    # Test if the C object modified the vector correctly
                    assert(vNew[r,c] == vOld[r,c] + 1)


    # Test if the C object reads, modifies, and resizes the vector correctly
    def SpecTestThrowUnlessReadsWritesResizesCorrectly(self, function):
        STORAGE_ORDER = ['C', 'F']

        for stord in STORAGE_ORDER:
            vNew = numpy.random.rand(5, 7)
            vNew = numpy.floor(vNew * 100)
            vNew = numpy.array(vNew, dtype=self.dtype, order=stord)
            vOld = copy.deepcopy(vNew)
            vOld = numpy.array(vOld, order=stord)
            rows = vNew.shape[0]
            cols = vNew.shape[1]
            SIZE_FACTOR = 3
            exec('self.CObject.' + function + '(vNew, SIZE_FACTOR)')

            assert(self.CObject.rows() == rows and self.CObject.cols() == cols)
            assert(vNew.shape[0] == rows * SIZE_FACTOR and vNew.shape[1] == cols * SIZE_FACTOR)
            try:
                assert(self.CObject.StorageOrder() == stord)
            except:
                print self.CObject.StorageOrder(), stord
                raise
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
        ROWS = 5
        COLS = 7
        exec('v = self.CObject.' + function + '(ROWS, COLS)')

        assert(v.shape[0] == ROWS and v.shape[1] == COLS)
        for r in xrange(ROWS):
            for c in xrange(COLS):
                assert(self.CObject.GetItem(r,c) == v[r,c])


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
        self.SpecTestThrowUnlessReadsWritesCorrectly(MY_NAME)
        self.SpecTestThrowUnlessReadsWritesResizesCorrectly(MY_NAME)


    def Test_in_vctDynamicMatrixRef(self):
        MY_NAME = 'in_vctDynamicMatrixRef'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension2(MY_NAME)
        self.StdTestThrowUnlessWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsWritesCorrectly(MY_NAME)


    def Test_in_vctDynamicConstMatrixRef(self):
        MY_NAME = 'in_vctDynamicConstMatrixRef'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension2(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_in_argout_const_vctDynamicConstMatrixRef_ref(self):
        MY_NAME = 'in_argout_const_vctDynamicConstMatrixRef_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension2(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_in_argout_const_vctDynamicMatrixRef_ref(self):
        MY_NAME = 'in_argout_const_vctDynamicMatrixRef_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension2(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_in_vctDynamicMatrix(self):
        MY_NAME = 'in_vctDynamicMatrix'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension2(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_in_argout_const_vctDynamicMatrix_ref(self):
        MY_NAME = 'in_argout_const_vctDynamicMatrix_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimension2(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_out_vctDynamicMatrix(self):
        MY_NAME = 'out_vctDynamicMatrix'

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedMatrixIsWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectMatrix(MY_NAME)


    def Test_out_vctDynamicMatrix_ref(self):
        MY_NAME = 'out_vctDynamicMatrix_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedMatrixIsWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectMatrix(MY_NAME)


    def Test_out_const_vctDynamicMatrix_ref(self):
        MY_NAME = 'out_const_vctDynamicMatrix_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedMatrixIsNonWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectMatrix(MY_NAME)


    def Test_out_vctDynamicMatrixRef(self):
        MY_NAME = 'out_vctDynamicMatrixRef'

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedMatrixIsWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectMatrix(MY_NAME)


    def Test_out_vctDynamicConstMatrixRef(self):
        MY_NAME = 'out_vctDynamicConstMatrixRef'

        exec('v = self.CObject.' + MY_NAME + '(5, 7)')
        #print v.flags

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedMatrixIsNonWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectMatrix(MY_NAME)
