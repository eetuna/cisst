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

#SET_SOURCE_FILES_PROPERTIES(
#    mtsDeviceInterfaceProxy.h PROPERTIES GENERATED TRUE
#    mtsDeviceInterfaceProxy.cpp PROPERTIES GENERATED TRUE
#    mtsTaskManagerProxy.h PROPERTIES GENERATED TRUE
#    mtsTaskManagerProxy.cpp PROPERTIES GENERATED TRUE
#    )

# all source files
SET(SOURCE_FILES
    mtsVector.cpp

    mtsMulticastCommandVoid.cpp
    mtsMulticastCommandWriteBase.cpp

    mtsFunctionVoid.cpp
    mtsFunctionReadOrWrite.cpp
    mtsFunctionQualifiedReadOrWrite.cpp

    mtsCommandQueuedVoidBase.cpp
    mtsCommandQueuedWriteBase.cpp

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
    # mtsTaskManagerProxy.cpp
    mtsTaskManagerProxyClient.cpp
    mtsTaskManagerProxyServer.cpp
    
    mtsDeviceInterface.cpp
    # mtsDeviceInterfaceProxy.cpp
    mtsDeviceInterfaceProxyClient.cpp
    mtsDeviceInterfaceProxyServer.cpp

    mtsTaskGlobal.cpp
    )

# all header files
SET(HEADER_FILES
    mtsVector.h
    mtsMap.h

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
    
    mtsProxyBaseCommon.h
    mtsProxyBaseClient.h
    mtsProxyBaseServer.h
    
    mtsTaskManager.h
    # mtsTaskManagerProxy.h
    mtsTaskManagerProxyClient.h
    mtsTaskManagerProxyServer.h
        
    mtsDeviceInterface.h
    # mtsDeviceInterfaceProxy.h
    mtsDeviceInterfaceProxyClient.h
    mtsDeviceInterfaceProxyServer.h
    
    mtsTaskGlobal.h
    mtsDeviceProxy.h
    mtsCommandVoidProxy.h
    mtsCommandWriteProxy.h
    mtsCommandReadProxy.h
    mtsCommandQualifiedReadProxy.h
    mtsEventWriteGeneratorProxy.h
    )
    
SET(ICE_RESOURCE_FILES
    ${cisst_SOURCE_DIR}/libs/etc/${LIBRARY}/Ice/config.client
    ${cisst_SOURCE_DIR}/libs/etc/${LIBRARY}/Ice/config.server
    ${cisst_SOURCE_DIR}/libs/etc/${LIBRARY}/Ice/mtsDeviceInterfaceProxy.ice
    ${cisst_SOURCE_DIR}/libs/etc/${LIBRARY}/Ice/mtsTaskManagerProxy.ice
    )    

#
# ICE
#
SOURCE_GROUP(Resources FILES ${ICE_RESOURCE_FILES})

SET(SLICE_FILES
  mtsDeviceInterfaceProxy
  mtsTaskManagerProxy
)

foreach(ice_include_dir ${ICE_INCLUDE_DIR})
  set(SLICE_OPTIONS ${SLICE_OPTIONS} -I${ice_include_dir})
endforeach(ice_include_dir)

foreach(slice_file ${SLICE_FILES})
  set(GENERATED_CPP_FILE "${CMAKE_CURRENT_BINARY_DIR}/${slice_file}.cpp")
  set(GENERATED_H_FILE "${CMAKE_CURRENT_BINARY_DIR}/${slice_file}.h")

  add_custom_command(
    OUTPUT ${GENERATED_CPP_FILE} ${GENERATED_H_FILE} ${cisst_BINARY_DIR}/libs/include/cisstMultiTask/${slice_file}.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${ICE_HOME}/bin/slice2cpp
    ARGS    -I${ICE_SLICE_DIR} ${SLICE_OPTIONS} ${cisst_SOURCE_DIR}/libs/etc/cisstMultiTask/Ice/${slice_file}.ice
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different ${GENERATED_H_FILE} ${cisst_BINARY_DIR}/libs/include/cisstMultiTask
    COMMENT "Compiling ${slice_file}.ice"
  )

# ${CMAKE_COMMAND}
               # ARGS -E copy_if_different
               # \"${LIBRARY_MAIN_HEADER_TMP}\"
               # \"${LIBRARY_MAIN_HEADER}\") 
               
  ADD_CUSTOM_COMMAND(
    TARGET ${GENERATED_H_FILE}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${GENERATED_H_FILE} ${cisst_BINARY_DIR}/libs/include/cisstMultiTask)


  include_directories(${CMAKE_CURRENT_BINARY_DIR})
    
  set(EXTRA_SOURCE_FILES ${EXTRA_SOURCE_FILES} ${GENERATED_CPP_FILE})

endforeach(slice_file ${SLICE_FILES})

CISST_ADD_LIBRARY_TO_PROJECT(cisstLibs)

 TARGET_LINK_LIBRARIES( cisstMultiTask 
    optimized ${ICE_LIBRARY_DIR}/Ice.lib        debug ${ICE_LIBRARY_DIR}/Iced.lib
    optimized ${ICE_LIBRARY_DIR}/IceUtil.lib    debug ${ICE_LIBRARY_DIR}/IceUtild.lib )

