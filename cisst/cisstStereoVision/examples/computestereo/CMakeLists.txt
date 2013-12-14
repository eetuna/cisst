#
# $Id$
#
# (C) Copyright 2006-2011 Johns Hopkins University (JHU), All Rights
# Reserved.
#
# --- begin cisst license - do not edit ---
#
# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.
#
# --- end cisst license ---

cmake_minimum_required (VERSION 2.6)

# create a list of libraries needed for this project
set (REQUIRED_CISST_LIBRARIES cisstCommon cisstVector cisstOSAbstraction cisstMultiTask cisstStereoVision)

# find cisst and make sure the required libraries have been compiled
find_package (cisst REQUIRED ${REQUIRED_CISST_LIBRARIES})

if (cisst_FOUND_AS_REQUIRED)

  # load cisst configuration
  include (${CISST_USE_FILE})

  if (CISST_SVL_HAS_X11 OR WIN32)

    add_executable (svlExComputeStereo computestereo.cpp)
    set_property (TARGET svlExComputeStereo PROPERTY FOLDER "cisstStereoVision/examples")
	if (CISST_SVL_HAS_ISSI) 
	include_directories( ${ISSILIB_INCLUDE_DIR} )
	link_directories(${ISSILIB_LIB_DIR})
	add_definitions(-DISSI_DATA_DIR=${ISSILIB_DATA_DIR})
	add_definitions( -DISSI_PREDEFINED_DaVinciSSC -DISSI_PREDEFINED_DaVinciSi -DISSI_PREDEFINED_EndoscopeImage -DISSI_PREDEFINED_Zeego -DISSI_PREDEFINED_Trumpf -DISSI_PREDEFINED_ULI -DISSI_PREDEFINED_Zpp -DISSI_PREDEFINED_FluoroImage -DISSI_PREDEFINED_RenderedImage)
	                      
	target_link_libraries (svlExComputeStereo
						   ${ISSILIB_LIB})
	endif (CISST_SVL_HAS_ISSI)      
    cisst_target_link_libraries (svlExComputeStereo ${REQUIRED_CISST_LIBRARIES})

    # Copy example images to the build tree
    if (WIN32)
      configure_file (${CMAKE_CURRENT_SOURCE_DIR}/tsukuba3.bmp
                      ${CMAKE_CURRENT_BINARY_DIR}/tsukuba3.bmp COPYONLY)
      configure_file (${CMAKE_CURRENT_SOURCE_DIR}/tsukuba4.bmp
                      ${CMAKE_CURRENT_BINARY_DIR}/tsukuba4.bmp COPYONLY)
      configure_file (${CMAKE_CURRENT_SOURCE_DIR}/venus_l.bmp
                      ${CMAKE_CURRENT_BINARY_DIR}/venus_l.bmp COPYONLY)
      configure_file (${CMAKE_CURRENT_SOURCE_DIR}/venus_r.bmp
                      ${CMAKE_CURRENT_BINARY_DIR}/venus_r.bmp COPYONLY)
    else (WIN32)
      configure_file (${CMAKE_CURRENT_SOURCE_DIR}/tsukuba3.bmp
                      ${EXECUTABLE_OUTPUT_PATH}/tsukuba3.bmp COPYONLY)
      configure_file (${CMAKE_CURRENT_SOURCE_DIR}/tsukuba4.bmp
                      ${EXECUTABLE_OUTPUT_PATH}/tsukuba4.bmp COPYONLY)
      configure_file (${CMAKE_CURRENT_SOURCE_DIR}/venus_l.bmp
                      ${EXECUTABLE_OUTPUT_PATH}/venus_l.bmp COPYONLY)
      configure_file (${CMAKE_CURRENT_SOURCE_DIR}/venus_r.bmp
                      ${EXECUTABLE_OUTPUT_PATH}/venus_r.bmp COPYONLY)
    endif (WIN32)

  else (CISST_SVL_HAS_X11 OR WIN32)
    message ("Information: code in ${CMAKE_CURRENT_SOURCE_DIR} will not be compiled, it requires CISST_SVL_HAS_X11 or WIN32")
  endif (CISST_SVL_HAS_X11 OR WIN32)

else (cisst_FOUND_AS_REQUIRED)
  message ("Information: code in ${CMAKE_CURRENT_SOURCE_DIR} will not be compiled, it requires ${REQUIRED_CISST_LIBRARIES}")
endif (cisst_FOUND_AS_REQUIRED)
