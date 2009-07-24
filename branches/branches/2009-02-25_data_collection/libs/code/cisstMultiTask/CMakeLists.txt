#
# $Id$
#
# CMakeLists for cisstMultiTask
#
# (C) Copyright 2003-2008 Johns Hopkins University (JHU), All Rights
# Reserved.
# 
# --- begin cisst license - do not edit ---
# 
# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.
# 
# --- end cisst license ---

# set the library name
SET(LIBRARY cisstMultiTask)

# set dependencies
SET(DEPENDENCIES
    cisstCommon
    cisstVector
    cisstOSAbstraction
    )

# all source files
SET(SOURCE_FILES
    mtsVector.cpp
    mtsMatrix.cpp
    mtsGenericObjectProxy.cpp
    mtsTransformationTypes.cpp

    mtsMulticastCommandVoid.cpp
    mtsMulticastCommandWriteBase.cpp

    mtsFunctionVoid.cpp
    mtsFunctionReadOrWrite.cpp
    mtsFunctionQualifiedReadOrWrite.cpp

    mtsCommandQueuedVoidBase.cpp
    mtsCommandQueuedWriteBase.cpp

    mtsDeviceInterface.cpp
    mtsDevice.cpp

    mtsMailBox.cpp
    mtsStateIndex.cpp
    mtsStateTable.cpp

    mtsTaskInterface.cpp
    mtsRequiredInterface.cpp
    mtsTask.cpp
    mtsTaskContinuous.cpp
    mtsTaskPeriodic.cpp
    mtsTaskFromCallback.cpp

    mtsTaskManager.cpp

    mtsCollectorBase.cpp
    mtsCollectorState.cpp

    mtsHistory.cpp
    )

# all header files
SET(HEADER_FILES
    mtsVector.h
    mtsMatrix.h
    mtsGenericObject.h
    mtsGenericObjectProxy.h
    mtsTransformationTypes.h

    mtsCommandBase.h
    mtsFunctionBase.h

    mtsCommandVoid.h
    mtsCommandVoidBase.h
    mtsMulticastCommandVoid.h
    mtsCommandQueuedVoid.h
    mtsCommandQueuedVoidBase.h
    mtsFunctionVoid.h

    mtsCommandRead.h
    mtsCommandWrite.h
    mtsCommandReadOrWriteBase.h
    mtsMulticastCommandWrite.h
    mtsMulticastCommandWriteBase.h
    mtsCommandQueuedWrite.h
    mtsCommandQueuedWriteBase.h
    mtsFunctionReadOrWrite.h

    mtsCommandQualifiedRead.h
    mtsCommandQualifiedReadOrWriteBase.h
    mtsFunctionQualifiedReadOrWrite.h

    mtsExport.h
    mtsForwardDeclarations.h

    mtsDeviceInterface.h
    mtsDevice.h

    mtsQueue.h
    mtsMailBox.h
    mtsStateIndex.h

    mtsStateArray.h
    mtsStateArrayBase.h
    mtsStateTable.h
    mtsStateData.h

    mtsTaskInterface.h
    mtsRequiredInterface.h
    mtsTask.h
    mtsTaskContinuous.h
    mtsTaskPeriodic.h
    mtsTaskFromCallback.h
    
    mtsTaskManager.h

    mtsCollectorBase.h    
    mtsCollectorState.h
    
    mtsHistory.h
    )

    # make sure all options are set for all devices to OFF
option(CISST_COLLECTOR_GUI "Build state data visualizer" ON)

if(CISST_COLLECTOR_GUI)

    SET(SOURCE_FILES
        ${SOURCE_FILES}
        mtsCollectorGUI.cpp
        )
        
    SET(HEADER_FILES
        ${HEADER_FILES}
        mtsCollectorGUI.h
        )

    include_directories(${cisst_SOURCE_DIR}/libs/etc
                           ${FLTK_INCLUDE_DIR})

    TARGET_LINK_LIBRARIES(LIBRARY ${FLTK_LIBRARIES})                            
    
endif(CISST_COLLECTOR_GUI)


CISST_ADD_LIBRARY_TO_PROJECT(cisstLibs)

