import unittest
import numpy
import copy

from vctDynamicVectorTypemapsTestPython import vctDynamicVectorTypemapsTest
from sys import getrefcount


def Equal(a, b):
        if (a.size != b.size()):
                return False

        for i in range(0, a.size):
                if ( a[i] != b[i]):
                        return False

        return True



class DynamicVectorTypemapsTest(unittest.TestCase):

        # This function will be run prior to each test
        def setUp(self):
                self.numpyVector = numpy.ones(6, dtype=numpy.int32)
                #print "numpyVector ref count: " + str(getrefcount(self.numpyVector))
                #print self.numpyVector.dtype
                #print self.numpyVector.flags
                self.cisstVector = vctDynamicVectorTypemapsTest()
                #print "cisstVector ref count: " + str(getrefcount(self.cisstVector))

        #################################################################################
        #                   NumPy Vector to CISST Vector                                #
        #################################################################################

        # Tests if the data is the same using SetByCopyDVRef
        def TestSetByCopyDVRef(self):
                print "----- testSetByCopyDVRef -----"
                self.cisstVector.SetByCopyDVRef(self.numpyVector)
                print "numpyVector: " + str(self.numpyVector)
                print "cisstVector: " + str(self.cisstVector)
                self.assertEqual(Equal(self.numpyVector, self.cisstVector), True)

        # Check independence of data used in SetByCopyDVRef
        def TestSetByCopyDVRefIndependence(self):
                print "----- testSetByCopyDVRefIndependence -----"
                self.cisstVector.SetByCopyDVRef(self.numpyVector)
                print "numpyVector (before): " + str(self.numpyVector)
                print "cisstVector (before): " + str(self.cisstVector)
                self.numpyVector[4] = 2 #change the original
                print "numpyVector (after): " + str(self.numpyVector)
                print "cisstVector (after): " + str(self.cisstVector)
                self.assertEqual(Equal(self.numpyVector, self.cisstVector), False)

        # Tests that exception is raised by typemap if numpy array is not writable
        def TestSetByCopyDVRefWriteException(self):
                print "----- testSetByCopyDVRefWriteException -----"
                self.numpyVector.setflags(write=False)
                self.assertRaises(ValueError, self.cisstVector.SetByCopyDVRef, self.numpyVector)

        # Tests if the data is the same using SetByCopyConstDVRef
        def TestSetByCopyConstDVRef(self):
                print "----- testSetByCopyConstDVRef -----"
                self.cisstVector.SetByCopyConstDVRef(self.numpyVector)
                print "numpyVector: " + str(self.numpyVector)
                print "cisstVector: " + str(self.cisstVector)
                self.assertEqual(Equal(self.numpyVector, self.cisstVector), True)

        # Add test to check that Const vector is not affected by changing data

        # Tests if the data is the same using SetByRef
        def TestSetByRef(self):
                print "----- testSetByRef -----"
                self.cisstVector.SetByRef(self.numpyVector)
                #print "numpyVector ref count after reference created: " + str(getrefcount(self.numpyVector))
                print "numpyVector: " + str(self.numpyVector)
                print "cisstVector: " + str(self.cisstVector)
                self.assertEqual(Equal(self.numpyVector, self.cisstVector), True)

        # Tests if the data is the same using SetByConstRefAmp
        def TestSetByConstRefAmp(self):
            print "----- testSetByConstRefAmp -----"
            self.cisstVector.SetByConstRefAmp(self.numpyVector)
            print "numpyVector: " + str(self.numpyVector)
            print "cisstVector: " + str(self.cisstVector)
            self.assertEqual(Equal(self.numpyVector, self.cisstVector), True)

        # Tests if the data is the same using SetByConstConstRefAmp
        def TestSetByConstConstRefAmp(self):
            print "----- testSetByConstConstRefAmp -----"
            self.cisstVector.SetByConstRefAmp(self.numpyVector)
            print "numpyVector: " + str(self.numpyVector)
            print "cisstVector: " + str(self.cisstVector)
            self.assertEqual(Equal(self.numpyVector, self.cisstVector), True)

        # Tests that exception is raised by typemap if numpy array is not writable
        def TestSetByRefWriteException(self):
                print "----- testSetByRefWriteException -----"
                self.numpyVector.setflags(write=False)
                self.assertRaises(ValueError, self.cisstVector.SetByRef, self.numpyVector)

##        # Tests that exception is raised by typemap if numpy array doesn't own its memory
##        def TestSetByRefOwnException(self):
##                print "----- testSetByRefOwnException -----"
##                numpySlice = self.numpyVector[0:1] # get a slice containing the first element of numpyVector
##                print "numpyVector OWNDATA flag: " + str(self.numpyVector.flags.owndata)
##                print "numpySlice OWNDATA flag:  " + str(numpySlice.flags.owndata)
##                self.assertRaises(ValueError, self.cisstVector.SetByRef, numpySlice)

        # Test modifyVectorData function - modifies data in vector passed by ref, but dosn't change size
        def TestModifyVectorData(self):
                print "----- testModifyVectorData -----"
                print "numpyVector: " + str(self.numpyVector)
                self.cisstVector.ModifyVectorData(self.numpyVector, 0, 55)
                print "numpyVector: " + str(self.numpyVector)

        # Test modifyVectorData function on a slice - modifies data in vector passed by ref, but dosn't change size
        def TestModifyVectorDataSlice(self):
                print "----- testModifyVectorDataSlice -----"
                numpySlice = self.numpyVector[0:2] # get a slice containing the first 2 elements of numpyVector
                print "numpyVector: " + str(self.numpyVector)
                print "numpySlice: " + str(numpySlice)
                self.cisstVector.ModifyVectorData(numpySlice, 0, 55)
                print "numpyVector after modifying the slice: " + str(self.numpyVector)
                print "numpySlice after modifying the slice: " + str(numpySlice)
                self.cisstVector.ModifyVectorData(self.numpyVector, 1, 99)
                print "numpyVector after modifying the vector: " + str(self.numpyVector)
                print "numpySlice after modifying the vector: " + str(numpySlice)

        # Test the resizeVector function - changes size of vector passed by ref
        def TestResizeVector(self):
                print "----- testResizeVector -----"
                print "numpyVector before resize: " + str(self.numpyVector)
                #[newSize, newVector] = self.cisstVector.resizeVector(self.numpyVector, 0.5)
                newSize = self.cisstVector.ResizeVector(self.numpyVector, 0.5)
                print "new size: " + str(newSize)
                print "numpyVector after resize: " + str(self.numpyVector)

        # Tests that exception is raised by typemap when attempting to resize numpy array that has refs on it
        def TestResizeVectorRefException(self):
                print "----- testResizeVectorRefException -----"
                numpySlice = self.numpyVector[0:1] # get a slice containing the first element of numpyVector
                print "numpyVector ref count: " + str(getrefcount(self.numpyVector))
                #newSize = self.cisstVector.resizeVector(self.numpyVector, 0.5)
                self.assertRaises(ValueError, self.cisstVector.ResizeVector, self.numpyVector, 0.5)

        # Tests that exception is raised by typemap when attempting to resize a numpy array that doesn't own its data
        def TestResizeVectorOwnException(self):
                #call resize on numpySlice, should throw exception about not owning data
                print "----- testResizeVectorOwnException -----"
                numpySlice = self.numpyVector[0:4] # get a slice containing the first 4 elements of numpyVector
                self.assertRaises(ValueError, self.cisstVector.ResizeVector, self.numpyVector, 0.5)
                #make a deep copy of self.numpyVector and try resizing
                deepCopy = copy.deepcopy(numpySlice)
                print "numpySlice: " + str(numpySlice)
                print "deepCopy before resize: " + str(deepCopy)
                self.cisstVector.ResizeVector(deepCopy, 0.5)
                print "deepCopy after resize: " + str(deepCopy)

        # Tests if the data is the same using SetByCopy
        def TestSetByCopy(self):
                print "----- testSetByCopy -----"
                self.cisstVector.SetByCopy(self.numpyVector)
                print "numpyVector: " + str(self.numpyVector)
                print "cisstVector: " + str(self.cisstVector)
                self.assertEqual(Equal(self.numpyVector, self.cisstVector), True)

        # Check independence of data used in SetByCopy
        def TestSetByCopyIndependence(self):
                print "----- testSetByCopyIndependence -----"
                self.cisstVector.SetByCopy(self.numpyVector) #make a copy
                print "numpyVector (before): " + str(self.numpyVector)
                print "cisstVector (before): " + str(self.cisstVector)
                self.numpyVector[4] = 2 #change the original
                print "numpyVector (after): " + str(self.numpyVector)
                print "cisstVector (after): " + str(self.cisstVector)
                self.assertEqual(Equal(self.numpyVector, self.cisstVector), False)



        #################################################################################
        #                   CISST Vector to NumPy Vector                                #
        #################################################################################

        # Tests if the data is the same using GetByCopy
        def TestGetByCopy(self):
                print "----- testGetByCopy -----"
                self.cisstVector.SetByCopy(self.numpyVector)
                returnedCopy = self.cisstVector.GetByCopy()
                print "numpyVector:  " + str(self.numpyVector)
                print "cisstVector:  " + str(self.cisstVector)
                print "returnedCopy: " + str(returnedCopy)
                self.assertEqual(Equal(returnedCopy, self.cisstVector), True)

        # Check independence of data used in GetByCopy
        def TestGetByCopyIndependence(self):
                print "----- testGetByCopyIndependence -----"
                self.cisstVector.SetByCopy(self.numpyVector)
                returnedCopy = self.cisstVector.GetByCopy()
                print "cisstVector  (before): " + str(self.cisstVector)
                print "returnedCopy (before): " + str(returnedCopy)
                returnedCopy[4] = 2
                print "cisstVector  (after): " + str(self.cisstVector)
                print "returnedCopy (after): " + str(returnedCopy)
                self.assertEqual(Equal(returnedCopy, self.cisstVector), False)

        # Tests if the data is the same using GetByRef
        def TestGetByRef(self):
                print "----- testGetByRef -----"
                self.cisstVector.SetByCopy(self.numpyVector)
                returnedRef = self.cisstVector.GetByRef()
                print "numpyVector:  " + str(self.numpyVector)
                print "cisstVector:  " + str(self.cisstVector)
                print "returnedRef: " + str(returnedRef)
                print "numpyVector OWNDATA flag: " + str(self.numpyVector.flags.owndata)
                print "returnedRef OWNDATA flag: " + str(returnedRef.flags.owndata)
                self.assertEqual(Equal(returnedRef, self.cisstVector), True)

        # Check dependence of data used in GetByRef
        def TestGetByRefDependence(self):
                print "----- testGetByRefDependence -----"
                self.cisstVector.SetByCopy(self.numpyVector)
                returnedRef = self.cisstVector.GetByRef()
                print "cisstVector (before): " + str(self.cisstVector)
                print "returnedRef (before): " + str(returnedRef)
                self.cisstVector[4] = 2
                print "cisstVector (after): " + str(self.cisstVector)
                print "returnedRef (after): " + str(returnedRef)
                self.assertEqual(Equal(returnedRef, self.cisstVector), True)

        # Tests if the data is the same using GetByConstRef
        def TestGetByConstRef(self):
                print "----- testGetByConstRef -----"
                self.cisstVector.SetByCopy(self.numpyVector)
                returnedConstRef = self.cisstVector.GetByConstRef()
                print "numpyVector:  " + str(self.numpyVector)
                print "cisstVector:  " + str(self.cisstVector)
                print "returnedRef: " + str(returnedConstRef)
                self.assertEqual(Equal(returnedConstRef, self.cisstVector), True)

        # Check that returned const ref can't be modified
        def TestGetByConstRefModified(self):
                print "----- testGetByConstRefModified -----"
                self.cisstVector.SetByCopy(self.numpyVector)
                returnedConstRef = self.cisstVector.GetByConstRef()
                print "returnedConstRef WRITEABLE flag: " + str(returnedConstRef.flags.writeable)
                self.assertEqual(returnedConstRef.flags.writeable, False)



#if __name__ == '__main__':
#    unittest.main()
