# -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
# ex: set softtabstop=4 shiftwidth=4 tabstop=4 expandtab:

#
# $Id: $
#

# Author: Anton Deguet
# Date: 2010-01-20
#
# (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
# Reserved.

# --- begin cisst license - do not edit ---
# 
# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.
# 
# --- end cisst license ---

import unittest
import numpy

import cisstCommonPython as cisstCommon
import cisstVectorPython as cisstVector
import cisstMultiTaskPython as cisstMultiTask
import cisstMultiTaskPythonTestPython as cisstMultiTaskPythonTest # contains test classes

class PeriodicTaskTest(unittest.TestCase):
    def setUp(self):
        """Call before every test case."""

    def tearDown(self):
        """Call after every test case."""
        
    def TestType(self):
        """Test constructor and types of mtsPeriodicTask"""
        variable = cisstMultiTaskPythonTest.mtsPeriodicTaskTest(0.05)
        # check type
        self.failUnless(isinstance(variable, cisstMultiTaskPythonTest.mtsPeriodicTaskTest))
        self.failUnless(isinstance(variable, cisstMultiTask.mtsTaskPeriodic))
        self.failUnless(isinstance(variable, cisstMultiTask.mtsTaskContinuous))
        self.failUnless(isinstance(variable, cisstMultiTask.mtsDevice))
        self.failUnless(isinstance(variable, cisstCommon.cmnGenericObject))

    def TestUpdateFromC(self):
        """Test UpdateFromC for mtsPeriodicTask"""
        variable = cisstMultiTaskPythonTest.mtsPeriodicTaskTest(0.05)
        variable.UpdateFromC()
        # verify that both interfaces have been created
        self.failUnless(variable.__dict__.has_key("MainInterface"))
        self.failUnless(variable.__dict__.has_key("EmptyInterface")) # space should have been removed
        # test that MainInterface has been populated properly
        # command AddDouble(mtsDouble)
        self.failUnless(variable.MainInterface.__dict__.has_key("AddDouble"))
        self.failUnless(isinstance(variable.MainInterface.AddDouble, cisstMultiTask.mtsCommandWriteBase))
        proto = variable.MainInterface.AddDouble.GetArgumentPrototype()
        self.failUnless(isinstance(proto, cisstMultiTask.mtsDouble))
        # command ZeroAll(void)
        self.failUnless(variable.MainInterface.__dict__.has_key("ZeroAll"))
        self.failUnless(isinstance(variable.MainInterface.ZeroAll, cisstMultiTask.mtsCommandVoidBase))
        # command GetDouble(mtsDouble)
        self.failUnless(variable.MainInterface.__dict__.has_key("GetDouble"))
        self.failUnless(isinstance(variable.MainInterface.GetDouble, cisstMultiTask.mtsCommandReadBase))
        proto = variable.MainInterface.GetDouble.GetArgumentPrototype()
        self.failUnless(isinstance(proto, cisstMultiTask.mtsDouble))
        # command GetVector(mtsDoubleVec)
        self.failUnless(variable.MainInterface.__dict__.has_key("GetVector"))
        self.failUnless(isinstance(variable.MainInterface.GetVector, cisstMultiTask.mtsCommandReadBase))
        proto = variable.MainInterface.GetVector.GetArgumentPrototype()
        self.failUnless(isinstance(proto, cisstMultiTask.mtsDoubleVec))
        

