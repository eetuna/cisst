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
set (REQUIRED_CISST_LIBRARIES cisstCommon cisstVector cisstOSAbstraction cisstMultiTask cisstStereoVision cisstStereoVisionQt)

# find cisst and make sure the required libraries have been compiled
find_package (cisst REQUIRED ${REQUIRED_CISST_LIBRARIES})

if (cisst_FOUND)

  # load cisst configuration
  include (${CISST_USE_FILE})

  if (CISST_SVL_HAS_X11 OR WIN32)

    if (CISST_HAS_QT)
      set (USED_CISST_LIBRARIES cisstCommon cisstVector cisstOSAbstraction cisstMultiTask cisstStereoVision cisstStereoVisionQt)
    else (CISST_HAS_QT)
      set (USED_CISST_LIBRARIES cisstCommon cisstVector cisstOSAbstraction cisstMultiTask cisstStereoVision)
    endif (CISST_HAS_QT)
    add_executable (svlExCameraViewer cameraviewer.cpp)
    set_property (TARGET svlExCameraViewer PROPERTY FOLDER "cisstStereoVision/examples")
    if (CISST_SVL_HAS_ISSI) 
		include_directories( ${ISSILIB_INCLUDE_DIR} )
		link_directories(${ISSILIB_LIB_DIR})
		add_definitions(-DISSI_DATA_DIR=${ISSILIB_DATA_DIR})
		add_definitions( -DISSI_PREDEFINED_DaVinciSSC -DISSI_PREDEFINED_DaVinciSi -DISSI_PREDEFINED_EndoscopeImage -DISSI_PREDEFINED_Zeego -DISSI_PREDEFINED_Trumpf -DISSI_PREDEFINED_ULI -DISSI_PREDEFINED_Zpp -DISSI_PREDEFINED_FluoroImage -DISSI_PREDEFINED_RenderedImage)     
		target_link_libraries (svlExCameraViewer
								${ISSILIB_LIB})
    endif (CISST_SVL_HAS_ISSI)     
    cisst_target_link_libraries (svlExCameraViewer ${USED_CISST_LIBRARIES})

  else (CISST_SVL_HAS_X11 OR WIN32)
    message ("Information: code in ${CMAKE_CURRENT_SOURCE_DIR} will not be compiled, it requires CISST_SVL_HAS_X11 or WIN32")
  endif (CISST_SVL_HAS_X11 OR WIN32)

else (cisst_FOUND)
  message ("Information: code in ${CMAKE_CURRENT_SOURCE_DIR} will not be compiled, it requires ${REQUIRED_CISST_LIBRARIES}")
endif (cisst_FOUND)


