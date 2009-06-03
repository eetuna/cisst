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
#INCLUDE_DIRECTORIES(${${PROJECT_NAME}_SOURCE_DIR}/include/${LIBRARY})
SOURCE_GROUP(Resources FILES ${ICE_RESOURCE_FILES})

SET(SLICE_FILES
  mtsDeviceInterfaceProxy
  mtsTaskManagerProxy
)

FOREACH(slice_file ${SLICE_FILES})
  message ("slice ${slice_file}")
  set(GENERATED_CPP_FILE "${CMAKE_CURRENT_BINARY_DIR}/${slice_file}.cpp")
  set(GENERATED_H_FILE "${CMAKE_CURRENT_BINARY_DIR}/${slice_file}.h")
  #GET_SOURCE_FILE_PROPERTY(FOO ${GENERATED_CPP_FILE} GENERATED)
  #message ("foo ${FOO}")
  #SET_SOURCE_FILES_PROPERTIES(${GENERATED_CPP_FILE} GENERATED)
  #GET_SOURCE_FILE_PROPERTY(FOO ${GENERATED_CPP_FILE} GENERATED)
  #message ("foo ${FOO}")

#  add_custom_target("${CMAKE_CURRENT_BINARY_DIR}/${GENERATED_CPP_FILE}"
# "${ICE_HOME}/bin/slice2cpp -I${ICE_SLICE_DIR} -I${ICE_INCLUDE_DIR} ${cisst_SOURCE_DIR}/libs/etc/cisstMultiTask/Ice/${slice_file}.ice"
# DEPENDS   "${ICE_SLICE_DIR}/Ice/Identity.ice"
#)

  ADD_CUSTOM_COMMAND(
  OUTPUT ${GENERATED_CPP_FILE} ${GENERATED_H_FILE}
 #           ${cisst_SOURCE_DIR}/libs/include/${slice_file}.h
   DEPENDS ${ICE_SLICE_DIR}/Ice/Identity.ice 
   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
   COMMAND ${ICE_HOME}/bin/slice2cpp
   ARGS    -I${ICE_SLICE_DIR} -I${ICE_INCLUDE_DIR} ${cisst_SOURCE_DIR}/libs/etc/cisstMultiTask/Ice/${slice_file}.ice
   COMMENT "Compiling ${slice_file}.ice"
   )
    
    SET(ICE_SOURCE_FILES ${ICE_SOURCE_FILES} ${GENERATED_CPP_FILE})
    message("------------${CMAKE_CURRENT_BINARY_DIR}")
    message("dir ${GENERATED_CPP_FILE}")
     
#   ADD_DEPENDENCIES(${LIBRARY} ${slice_file}.cpp)

  #SET_SOURCE_FILES_PROPERTIES(${GENERATED_CPP_FILE} PROPERTIES GENERATED 1)
ENDFOREACH(slice_file ${SLICE_FILES})

   INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
   ADD_LIBRARY(cisstMultiTaskIce ${ICE_SOURCE_FILES})
# ADD_CUSTOM_TARGET(BUILD_ICE ALL 
  # DEPENDS ${cisst_SOURCE_DIR}/libs/code/mtsDeviceInterfaceProxy.cpp ${cisst_SOURCE_DIR}/libs/include/mtsDeviceInterfaceProxy.h
  # DEPENDS ${cisst_SOURCE_DIR}/libs/code/mtsTaskManagerProxy.cpp     ${cisst_SOURCE_DIR}/libs/include/mtsTaskManagerProxy.h
# )

# FOREACH(slice_file ${SLICE_FILES})
  # SET(UNIX_STYLE_PATH_BINARY "${cisst_BINARY_DIR}/libs/code/cisstMultiTask/")
  # SET(UNIX_STYLE_PATH_SOURCE_CODE "${cisst_SOURCE_DIR}/libs/code/cisstMultiTask/")
  # SET(UNIX_STYLE_PATH_SOURCE_INCLUDE "${cisst_SOURCE_DIR}/libs/include/cisstMultiTask/")
  # IF(WIN32)
    # SET(WINDOW_STYLE_PATH_BINARY ${UNIX_STYLE_PATH_BINARY})
    # SET(WINDOW_STYLE_PATH_SOURCE_CODE ${UNIX_STYLE_PATH_SOURCE_CODE})
    # SET(WINDOW_STYLE_PATH_SOURCE_INCLUDE ${UNIX_STYLE_PATH_SOURCE_INCLUDE})
    # FILE(TO_NATIVE_PATH ${UNIX_STYLE_PATH_BINARY} WINDOW_STYLE_PATH_BINARY)
# FILE(TO_NATIVE_PATH ${UNIX_STYLE_PATH_SOURCE_CODE} WINDOW_STYLE_PATH_SOURCE_CODE)
# FILE(TO_NATIVE_PATH ${UNIX_STYLE_PATH_SOURCE_INCLUDE} WINDOW_STYLE_PATH_SOURCE_INCLUDE)
    # ADD_CUSTOM_COMMAND(TARGET BUILD_ICE
      # POST_BUILD
      # COMMAND if exist \"${WINDOW_STYLE_PATH_BINARY}${slice_file}.cpp\" copy \"${WINDOW_STYLE_PATH_BINARY}${slice_file}.cpp\" \"${WINDOW_STYLE_PATH_SOURCE_CODE}\"
      # COMMAND if exist \"${WINDOW_STYLE_PATH_BINARY}${slice_file}.h\" copy \"${WINDOW_STYLE_PATH_BINARY}${slice_file}.h\" \"${WINDOW_STYLE_PATH_SOURCE_INCLUDE}\"
      # COMMAND if exist \"${WINDOW_STYLE_PATH_BINARY}${slice_file}.cpp\" del \"${WINDOW_STYLE_PATH_BINARY}${slice_file}.cpp\"
      # COMMAND if exist \"${WINDOW_STYLE_PATH_BINARY}${slice_file}.h\" del \"${WINDOW_STYLE_PATH_BINARY}${slice_file}.h\"
    # )
  # ELSE(WIN32)
    # #### TODO: ADD FILE_EXIST CHECK ROUTINE FOR UNIX, MAC, ETC. ####
    # ADD_CUSTOM_COMMAND(TARGET BUILD_ICE
      # POST_BUILD
      # COMMAND copy ${UNIX_STYLE_PATH_BINARY}${slice_file}.cpp ${UNIX_STYLE_PATH_SOURCE_CODE}
      # COMMAND copy ${UNIX_STYLE_PATH_BINARY}${slice_file}.h ${UNIX_STYLE_PATH_SOURCE_INCLUDE}
      # COMMAND del ${UNIX_STYLE_PATH_BINARY}${slice_file}.cpp
      # COMMAND del ${UNIX_STYLE_PATH_BINARY}${slice_file}.h
    # )
  # ENDIF(WIN32)
# ENDFOREACH(slice_file ${SLICE_FILES})

# ADD_DEPENDENCIES(${LIBRARY} BUILD_ICE)



# LINK_LIBRARIES( optimized ${ICE_LIBRARY_DIR}/Ice.lib        debug ${ICE_LIBRARY_DIR}/Iced.lib
                # optimized ${ICE_LIBRARY_DIR}/IceUtil.lib    debug ${ICE_LIBRARY_DIR}/IceUtild.lib )

CISST_ADD_LIBRARY_TO_PROJECT(cisstLibs)



# FOREACH(slice_file ${SLICE_FILES})
     
# ADD_DEPENDENCIES(${LIBRARY} cisstMultiTaskIce)

# ENDFOREACH(slice_file ${SLICE_FILES})