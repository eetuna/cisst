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

from cisstVectorTypemapsTestPython import vctDynamicNArrayTypemapsTest_double_4

class DynamicNArrayTypemapsTest(unittest.TestCase):

    dtype = numpy.double
    size_type = numpy.uint32

    ###########################################################################
    #   SET UP function                                                       #
    ###########################################################################

    def setUp(self):
        self.CObject = vctDynamicNArrayTypemapsTest_double_4()


    ###########################################################################
    #   HELPER FUNCTIONS - Used by the functions in this .py file             #
    ###########################################################################

    # Given a dimensionality, returns a vector of random sizes for an ndarray
    # of that dimension
    def HelperRandSizes(self, ndim):
        # TODO: Limit sizes to be something other than [1, 10]; possibly [3, 7]
        sizes = numpy.random.rand(ndim)
        sizes = numpy.floor(sizes * 10) + 1  # `+ 1' to avoid 0-sized dimension
        sizes = numpy.array(sizes, dtype=self.dtype)
        sizes = tuple(sizes)
        return sizes


    # Converts a given metaindex `index' on an array of shape `shape' to a
    # tuple index
    def MetaIndexToTuple(self, index, shape):
        indexList = []
        for j in shape[::-1]:
            r = index % j
            index /= j
            indexList.append(r)
        indexList.reverse()
        indexTuple = tuple(indexList)
        return indexTuple


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
        sizes = self.HelperRandSizes(self.CObject.Dim())
        goodvar = numpy.ones(sizes, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar, 0)')


    # Tests that the typemap throws an exception if the data type isn't int
    def StdTestThrowUnlessDataType(self, function):
        # Give an array of floats; expect an exception
        exceptionOccurred = False
        try:
            sizes = self.HelperRandSizes(self.CObject.Dim())
            if (self.dtype != numpy.float64):
                badvar = numpy.ones(sizes, dtype=numpy.float64)
            else:
                badvar = numpy.ones(sizes, dtype=numpy.int32)
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an int; expect no exception
        sizes = self.HelperRandSizes(self.CObject.Dim())
        goodvar = numpy.ones(sizes, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar, 0)')


    # Tests that the typemap throws an exception if the array isn't of the
    # correct dimension
    def StdTestThrowUnlessDimensionN(self, function):
        # Give a (n-1)D array; expect an exception
        exceptionOccurred = False
        try:
            sizes = self.HelperRandSizes(self.CObject.Dim() - 1)
            badvar = numpy.ones(sizes, dtype=self.dtype)
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an n-D array; expect no exception
        sizes = self.HelperRandSizes(self.CObject.Dim())
        goodvar = numpy.ones(sizes, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar, 0)')


    # Tests that the typemap throws an exception if the array isn't writable
    def StdTestThrowUnlessWritable(self, function):
        # Give a non-writable array; expect an exception
        exceptionOccurred = False
        try:
            sizes = self.HelperRandSizes(self.CObject.Dim())
            badvar = numpy.ones(sizes, dtype=self.dtype)
            badvar.setflags(write=False)
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a writable array; expect no exception
        sizes = self.HelperRandSizes(self.CObject.Dim())
        goodvar = numpy.ones(sizes, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar, 0)')


    def StdTestThrowUnlessOwnsData(self, function):
        # Give a non-memory owning array; expect an exception
        exceptionOccurred = False
        try:
            sizes = self.HelperRandSizes(self.CObject.Dim())
            temp = numpy.ones(sizes, dtype=self.dtype)
            badvar = temp[:]
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a memory-owning array; expect no exception
        sizes = self.HelperRandSizes(self.CObject.Dim())
        goodvar = numpy.ones(sizes, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar, 0)')


    def StdTestThrowUnlessNotReferenced(self, function):
        # Give an array with a reference on it; expect an exception
        exceptionOccurred = False
        try:
            sizes = self.HelperRandSizes(self.CObject.Dim())
            badvar = numpy.ones(sizes, dtype=self.dtype)
            temp = badvar
            exec('self.CObject.' + function + '(badvar, 0)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an array with no references; expect no exception
        sizes = self.HelperRandSizes(self.CObject.Dim())
        goodvar = numpy.ones(sizes, dtype=self.dtype)
        exec('self.CObject.' + function + '(goodvar, 0)')


    def StdTestThrowUnlessReturnedNArrayIsWritable(self, function):
        # Expect the returned array to be writable
        sizes = self.HelperRandSizes(self.CObject.Dim())
        sizes = numpy.array(sizes, dtype=numpy.uint32)
        exec('v = self.CObject.' + function + '(sizes)')
        assert(v.flags['WRITEABLE'] == True)


    def StdTestThrowUnlessReturnedNArrayIsNonWritable(self, function):
        # Expect the returned array to be non-writable
        sizes = self.HelperRandSizes(self.CObject.Dim())
        sizes = numpy.array(sizes, dtype=numpy.uint32)
        exec('v = self.CObject.' + function + '(sizes)')
        assert(v.flags['WRITEABLE'] == False)


    # Test if the C object reads the vector correctly
    def SpecTestThrowUnlessReadsCorrectly(self, function):
        ndim = self.CObject.Dim()

        sizes = self.HelperRandSizes(ndim)
        vNew = numpy.random.random_sample(sizes)
        vNew = numpy.floor(vNew * 100)
        vNew = numpy.array(vNew, dtype=self.dtype)
        vOld = copy.deepcopy(vNew)

        vShape = numpy.array(sizes)
        vSize = vNew.size

        exec('self.CObject.' + function + '(vNew, 0)')

        cShape = numpy.ones(ndim, dtype=self.size_type)
        self.CObject.sizes(cShape)

        assert((cShape == vShape).all())
        assert((vNew.shape == vShape).all())
        for i in xrange(vSize):
            indexTuple = self.MetaIndexToTuple(i, vShape)

            # Test if the C object read the vector correctly
            assert(self.CObject.GetItem(i) == vOld[indexTuple])
            # Test that the C object did not modify the vector
            assert(vNew[indexTuple] == vOld[indexTuple])


    # Test if the C object reads and modifies the vector correctly
    def SpecTestThrowUnlessReadsWritesCorrectly(self, function):
        ndim = self.CObject.Dim()

        sizes = self.HelperRandSizes(ndim)
        vNew = numpy.random.random_sample(sizes)
        vNew = numpy.floor(vNew * 100)
        vNew = numpy.array(vNew, dtype=self.dtype)
        vOld = copy.deepcopy(vNew)

        vShape = numpy.array(sizes)
        vSize = vNew.size

        exec('self.CObject.' + function + '(vNew, 0)')

        cShape = numpy.ones(ndim, dtype=self.size_type)
        self.CObject.sizes(cShape)

        assert((cShape == vShape).all())
        assert((vNew.shape == vShape).all())
        for i in xrange(vSize):
            indexTuple = self.MetaIndexToTuple(i, vShape)

            # Test if the C object read the vector correctly
            assert(self.CObject.GetItem(i) == vOld[indexTuple])
            # Test if the C object modified the vector correctly
            assert(vNew[indexTuple] == vOld[indexTuple] + 1)


    # Test if the C object resizes the vector correctly
    def SpecTestThrowUnlessResizesCorrectly(self, function):
        ndim = self.CObject.Dim()

        sizes = self.HelperRandSizes(ndim)
        vNew = numpy.random.random_sample(sizes)
        vNew = numpy.floor(vNew * 100)
        vNew = numpy.array(vNew, dtype=self.dtype)
        vOld = copy.deepcopy(vNew)

        vShape = numpy.array(sizes)
        vSize = vNew.size

        SIZE_FACTOR = 3

        exec('self.CObject.' + function + '(vNew, SIZE_FACTOR)')

        cShape = numpy.ones(ndim, dtype=self.size_type)
        self.CObject.sizes(cShape)

        assert((cShape == vShape).all())
        vShapeNew = numpy.array(vNew.shape)
        assert((vShapeNew == vShape * SIZE_FACTOR).all())
        for i in xrange(vNew.size):
            indexTuple = self.MetaIndexToTuple(i, vShape)

            # Test if the C object resized the vector correctly
            assert(vNew[indexTuple] == 17)


    # Test if the C object returns a good vector
    def SpecTestThrowUnlessReceivesCorrectNArray(self, function):
        ndim = self.CObject.Dim()
        sizes = self.HelperRandSizes(self.CObject.Dim())
        sizes = numpy.array(sizes, dtype=numpy.uint32)

        exec('v = self.CObject.' + function + '(sizes)')
        vShape = v.shape
        vSize = v.size

        for i in xrange(ndim):
            assert(vShape[i] == sizes[i]);

        for i in xrange(vSize):
            indexTuple = self.MetaIndexToTuple(i, vShape)
            assert(self.CObject.GetItem(i) == v[indexTuple])


    ###########################################################################
    #   TEST FUNCTIONS - Put any new unit test here!  These are the unit      #
    #   tests that test the typemaps.  One test per typemap!                  #
    ###########################################################################

    def Test_in_argout_vctDynamicNArray_ref(self):
        MY_NAME = 'in_argout_vctDynamicNArray_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimensionN(MY_NAME)
        self.StdTestThrowUnlessWritable(MY_NAME)
        self.StdTestThrowUnlessOwnsData(MY_NAME)
        self.StdTestThrowUnlessNotReferenced(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsWritesCorrectly(MY_NAME)
        self.SpecTestThrowUnlessResizesCorrectly(MY_NAME)


    def Test_in_vctDynamicNArrayRef(self):
        MY_NAME = 'in_vctDynamicNArrayRef'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimensionN(MY_NAME)
        self.StdTestThrowUnlessWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsWritesCorrectly(MY_NAME)


    def Test_in_vctDynamicConstNArrayRef(self):
        MY_NAME = 'in_vctDynamicConstNArrayRef'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimensionN(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_in_argout_const_vctDynamicConstNArrayRef_ref(self):
        MY_NAME = 'in_argout_const_vctDynamicConstNArrayRef_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimensionN(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_in_argout_const_vctDynamicNArrayRef_ref(self):
        MY_NAME = 'in_argout_const_vctDynamicNArrayRef_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimensionN(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_in_vctDynamicNArray(self):
        MY_NAME = 'in_vctDynamicNArray'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimensionN(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_in_argout_const_vctDynamicNArray_ref(self):
        MY_NAME = 'in_argout_const_vctDynamicNArray_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessIsArray(MY_NAME)
        self.StdTestThrowUnlessDataType(MY_NAME)
        self.StdTestThrowUnlessDimensionN(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReadsCorrectly(MY_NAME)


    def Test_out_vctDynamicNArray(self):
        MY_NAME = 'out_vctDynamicNArray'

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedNArrayIsWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectNArray(MY_NAME)


    def Test_out_vctDynamicNArray_ref(self):
        MY_NAME = 'out_vctDynamicNArray_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedNArrayIsWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectNArray(MY_NAME)


    def Test_out_const_vctDynamicNArray_ref(self):
        MY_NAME = 'out_const_vctDynamicNArray_ref'

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedNArrayIsNonWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectNArray(MY_NAME)


    def Test_out_vctDynamicNArrayRef(self):
        MY_NAME = 'out_vctDynamicNArrayRef'

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedNArrayIsWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectNArray(MY_NAME)


    def Test_out_vctDynamicConstNArrayRef(self):
        MY_NAME = 'out_vctDynamicConstNArrayRef'

        # Perform battery of standard tests
        self.StdTestThrowUnlessReturnedNArrayIsNonWritable(MY_NAME)

        # Perform specialized tests
        self.SpecTestThrowUnlessReceivesCorrectNArray(MY_NAME)
