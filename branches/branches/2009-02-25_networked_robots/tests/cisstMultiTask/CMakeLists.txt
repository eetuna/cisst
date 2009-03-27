#
# $Id: CMakeLists.txt mjung5 $
#
# CMakeLists for cisstMultiTask tests
#
# (C) Copyright 2008-2009 Johns Hopkins University (JHU), All Rights
# Reserved.
# 
# --- begin cisst license - do not edit ---
# 
# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.
# 
# --- end cisst license ---


# all source files
SET(SOURCE_FILES
    #mtsTaskProxyTest.cpp
    mtsTaskManagerTest.cpp
    )

# all header files
SET(HEADER_FILES
    #mtsTaskProxyTest.h
    mtsTaskManagerTest.h
    )

# Add executable for C++ tests
ADD_EXECUTABLE(cisstMultiTaskTests ${SOURCE_FILES} ${HEADER_FILES})
TARGET_LINK_LIBRARIES(cisstMultiTaskTests cisstTestsMain )
CISST_REQUIRES("cisstMultiTaskTests" "cisstCommon;cisstMultiTask")

# Threads
FIND_PACKAGE(Threads REQUIRED)
TARGET_LINK_LIBRARIES(cisstMultiTaskTests ${CMAKE_THREAD_LIBS_INIT})

# To generate a CTest list of tests
CISST_ADD_TESTS(cisstMultiTaskTests)
