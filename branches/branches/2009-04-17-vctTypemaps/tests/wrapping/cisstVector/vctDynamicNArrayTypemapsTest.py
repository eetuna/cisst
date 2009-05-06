###################################
# Authors: Daniel Li, Anton Deguet
###################################

#######################
# PLACEHOLDER STRINGS TO LOOK FOR:
#
# TODO          todo
#######################

# TODO: If I have time, Document why self.CObject[i] works and check which typemap(s) used
# TODO: Check that SpecTestThrowUnlessReads[Writes][Resizes]Correctly mirror each other
# TODO: Clean this code up

import copy
import numpy
import unittest

from vctDynamicNArrayTypemapsTestPython import vctDynamicNArrayTypemapsTest
import sys

class DynamicNArrayTypemapsTest(unittest.TestCase):

    ###########################################################################
    #   SET UP function                                                       #
    ###########################################################################

    def setUp(self):
        self.CObject = vctDynamicNArrayTypemapsTest()


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
        goodvar = numpy.ones((2, 3, 4), dtype=numpy.int32)
        exec('self.CObject.' + function + '(goodvar, 0)')


    # Tests that the typemap throws an exception if the data type isn't int
    def StdTestThrowUnlessDataType(self, function):
        # Give an array of floats; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones((2, 3, 4), dtype=numpy.float64)
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an int; expect no exception
        goodvar = numpy.ones((2, 3, 4), dtype=numpy.int32)
        exec('self.CObject.' + function + '(goodvar, 0)')


    # Tests that the typemap throws an exception if the array isn't of the
    # correct dimension
    def StdTestThrowUnlessDimensionN(self, function):
        # Give a (n-1)D array; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones((5, 7), dtype=numpy.int32)
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an n-D array; expect no exception
        goodvar = numpy.ones((2, 3, 4), dtype=numpy.int32)
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


    def StdTestThrowUnlessReturnedNArrayIsWritable(self, function):
        # Expect the returned array to be writable
        ROWS = 5
        COLS = 7
        exec('v = self.CObject.' + function + '(ROWS, COLS)')
        assert(v.flags['WRITEABLE'] == True)


    def StdTestThrowUnlessReturnedNArrayIsNonWritable(self, function):
        # Expect the returned array to be non-writable
        ROWS = 5
        COLS = 7
        exec('v = self.CObject.' + function + '(ROWS, COLS)')
        assert(v.flags['WRITEABLE'] == False)


    # Test if the C object reads the vector correctly
    def SpecTestThrowUnlessReadsCorrectly(self, function):
        vNew = numpy.random.rand(2, 3, 4)
        vNew = numpy.floor(vNew * 100)
        vNew = numpy.array(vNew, dtype=numpy.int32)
        vOld = copy.deepcopy(vNew)
        vSizes = numpy.array(vNew.shape)
        vSize = vNew.size
        exec('self.CObject.' + function + '(vNew, 0)')

        cSizes = numpy.ones(3, dtype=numpy.int32)
        self.CObject.sizes(cSizes)

        assert((cSizes == vSizes).all())
        assert((vNew.shape == vSizes).all())
        for i in xrange(vSize):
            # Convert metaindex i to tuple
            index = i
            indexList = []
            for j in vSizes[::-1]:
                r = index % j
                index /= j
                indexList.append(r)
            indexList.reverse()
            indexTuple = tuple(indexList)

            # Test if the C object read the vector correctly
            assert(self.CObject.GetItem(i) == vOld[indexTuple])
            # Test that the C object did not modify the vector
            assert(vNew[indexTuple] == vOld[indexTuple])


    # Test if the C object reads and modifies the vector correctly
    def SpecTestThrowUnlessReadsWritesCorrectly(self, function):
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
    def SpecTestThrowUnlessReadsWritesResizesCorrectly(self, function):
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


    # Test if the C object returns a good vector
    def SpecTestThrowUnlessReceivesCorrectNArray(self, function):
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

#     def Test_in_argout_vctDynamicNArray_ref(self):
#         MY_NAME = 'in_argout_vctDynamicNArray_ref'

#         # Perform battery of standard tests
#         self.StdTestThrowUnlessIsArray(MY_NAME)
#         self.StdTestThrowUnlessDataType(MY_NAME)
#         self.StdTestThrowUnlessDimensionN(MY_NAME)
#         self.StdTestThrowUnlessWritable(MY_NAME)
#         self.StdTestThrowUnlessOwnsData(MY_NAME)
#         self.StdTestThrowUnlessNotReferenced(MY_NAME)

#         # Perform specialized tests
#         self.SpecTestThrowUnlessReadsWritesCorrectly(MY_NAME)
#         self.SpecTestThrowUnlessReadsWritesResizesCorrectly(MY_NAME)


#     def Test_in_vctDynamicNArrayRef(self):
#         MY_NAME = 'in_vctDynamicNArrayRef'

#         # Perform battery of standard tests
#         self.StdTestThrowUnlessIsArray(MY_NAME)
#         self.StdTestThrowUnlessDataType(MY_NAME)
#         self.StdTestThrowUnlessDimensionN(MY_NAME)
#         self.StdTestThrowUnlessWritable(MY_NAME)

#         # Perform specialized tests
#         self.SpecTestThrowUnlessReadsWritesCorrectly(MY_NAME)


    def Test_in_vctDynamicConstNArrayRef(self):
        MY_NAME = 'in_vctDynamicConstNArrayRef'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimensionN(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


#     def Test_in_argout_const_vctDynamicConstNArrayRef_ref(self):
#         MY_NAME = 'in_argout_const_vctDynamicConstNArrayRef_ref'

#         # Perform battery of standard tests
#         self.StdTestThrowUnlessIsArray(MY_NAME)
#         self.StdTestThrowUnlessDataType(MY_NAME)
#         self.StdTestThrowUnlessDimensionN(MY_NAME)

#         # Perform specialized tests
#         self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


#     def Test_in_argout_const_vctDynamicNArrayRef_ref(self):
#         MY_NAME = 'in_argout_const_vctDynamicNArrayRef_ref'

#         # Perform battery of standard tests
#         self.StdTestThrowUnlessIsArray(MY_NAME)
#         self.StdTestThrowUnlessDataType(MY_NAME)
#         self.StdTestThrowUnlessDimensionN(MY_NAME)

#         # Perform specialized tests
#         self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


#     def Test_in_vctDynamicNArray(self):
#         MY_NAME = 'in_vctDynamicNArray'

#         # Perform battery of standard tests
#         self.StdTestThrowUnlessIsArray(MY_NAME)
#         self.StdTestThrowUnlessDataType(MY_NAME)
#         self.StdTestThrowUnlessDimensionN(MY_NAME)

#         # Perform specialized tests
#         self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


#     def Test_in_argout_const_vctDynamicNArray_ref(self):
#         MY_NAME = 'in_argout_const_vctDynamicNArray_ref'

#         # Perform battery of standard tests
#         self.StdTestThrowUnlessIsArray(MY_NAME)
#         self.StdTestThrowUnlessDataType(MY_NAME)
#         self.StdTestThrowUnlessDimensionN(MY_NAME)

#         # Perform specialized tests
#         self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


#     def Test_out_vctDynamicNArray(self):
#         MY_NAME = 'out_vctDynamicNArray'

#         # Perform battery of standard tests
#         self.StdTestThrowUnlessReturnedNArrayIsWritable(MY_NAME)

#         # Perform specialized tests
#         self.SpecTestThrowUnlessReceivesCorrectNArray(MY_NAME)


#     def Test_out_vctDynamicNArray_ref(self):
#         MY_NAME = 'out_vctDynamicNArray_ref'

#         # Perform battery of standard tests
#         self.StdTestThrowUnlessReturnedNArrayIsWritable(MY_NAME)

#         # Perform specialized tests
#         self.SpecTestThrowUnlessReceivesCorrectNArray(MY_NAME)


#     def Test_out_const_vctDynamicNArray_ref(self):
#         MY_NAME = 'out_const_vctDynamicNArray_ref'

#         # Perform battery of standard tests
#         self.StdTestThrowUnlessReturnedNArrayIsNonWritable(MY_NAME)

#         # Perform specialized tests
#         self.SpecTestThrowUnlessReceivesCorrectNArray(MY_NAME)


#     def Test_out_vctDynamicNArrayRef(self):
#         MY_NAME = 'out_vctDynamicNArrayRef'

#         # Perform battery of standard tests
#         self.StdTestThrowUnlessReturnedNArrayIsWritable(MY_NAME)

#         # Perform specialized tests
#         self.SpecTestThrowUnlessReceivesCorrectNArray(MY_NAME)


#     def Test_out_vctDynamicConstNArrayRef(self):
#         MY_NAME = 'out_vctDynamicConstNArrayRef'

#         exec('v = self.CObject.' + MY_NAME + '(5, 7)')
#         print v.flags

#         # Perform battery of standard tests
#         self.StdTestThrowUnlessReturnedNArrayIsNonWritable(MY_NAME)

#         # Perform specialized tests
#         self.SpecTestThrowUnlessReceivesCorrectNArray(MY_NAME)
