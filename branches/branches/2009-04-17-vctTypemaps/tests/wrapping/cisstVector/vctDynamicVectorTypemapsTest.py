import numpy
import unittest

from vctDynamicVectorTypemapsTestPython import vctDynamicVectorTypemapsTest

class DynamicVectorTypemapsTest(unittest.TestCase):

    def setUp(self):
        self.CObject = vctDynamicVectorTypemapsTest()
        print '--new test--'

    # Standard tests

    def StdTest(self, function):
        var = numpy.ones(10)
        self.CObject.in_argout_vctDynamicVector_ref(var, 0)

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
        goodvar = numpy.ones(10)
        exec('self.CObject.' + function + '(goodvar)')

    # Tests that the typemap throws an exception if the data type isn't int
    def StdTestThrowUnlessDataType(self, function):
        # Give an array of floats; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.array([1., 1., 1.])
            exec('self.CObject.' + function + '(badvar)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give an int; expect no exception
        goodvar = numpy.array([1, 1, 1])
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
        goodvar = numpy.array([1, 1, 1])
        exec('self.CObject.' + function + '(goodvar)')

    # Tests that the typemap throws an exception if the array isn't writable
    def StdTestThrowUnlessWritable(self, function):
        # Give a non-writable array; expect an exception
        exceptionOccurred = False
        try:
            badvar = numpy.ones(10)
            badvar.setflags(write=False)
            exec('self.CObject.' + function + '(badvar)')
        except:
            exceptionOccurred = True
        assert(exceptionOccurred)

        # Give a writable array; expect no exception
        goodvar = numpy.ones(10)
        exec('self.CObject.' + function + '(goodvar)')

    def StdTestThrowUnlessOwnsData(self, function):
        pass

    def StdTestThrowUnlessNotReferenced(self, function):
        pass

    def StdTestThrowUnlessReadable(self, function):
        pass

    def StdTestThrowUnlessReadableWritable(self, function):
#       PyParam = numpy.ones(10)
#       PyParamOrig = deepcopy(PyParam)
#       self.CObject.in_argout_vctDynamicVector_ref(PyParam, 20)

#       for i:
#           self.CObject[i] == PyParamOrig[i]
#           PyParam[i] == PyParamOrig[i] + 1
        pass

    def StdTestThrowUnlessResizable(self, function):
        pass

    def Test_in_argout_vctDynamicVector_ref(self):
        # Perform battery of standard tests
        self.StdTest('in_argout_vctDynamicVector_ref')
#       self.StdTestThrowUnlessIsArray('in_argout_vctDynamicVector_ref')
#       self.StdTestThrowUnlessDataType('in_argout_vctDynamicVector_ref')
#       self.StdTestThrowUnlessDimension1('in_argout_vctDynamicVector_ref')
#       self.StdTestThrowUnlessWritable('in_argout_vctDynamicVector_ref')

        # Perform specialized tests
#       StdTestThrowUnlessReadableWritable('in_argout_vctDynamicVector_ref')
#       StdTestThrowUnlessResizable('in_argout_vctDynamicVector_ref')