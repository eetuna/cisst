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

project (ui3ExDaVinciStereoVideo)

# sawOpenIGTLink component
cisst_find_saw_component (sawOpenIGTLink REQUIRED)
include_directories (${sawOpenIGTLink_INCLUDE_DIR})
link_directories (${sawOpenIGTLink_LIBRARY_DIR})

# ISSI
find_package(ISSILIB REQUIRED)
include_directories( ${ISSILIB_INCLUDE_DIR} )
link_directories(${ISSILIB_LIB_DIR})
add_definitions(-DISSI_DATA_DIR=${ISSILIB_DATA_DIR})
add_definitions( -DISSI_PREDEFINED_DaVinciSSC -DISSI_PREDEFINED_DaVinciSi -DISSI_PREDEFINED_Ducati -DISSI_PREDEFINED_EndoscopeImage -DISSI_PREDEFINED_Zeego -DISSI_PREDEFINED_Trumpf -DISSI_PREDEFINED_ULI -DISSI_PREDEFINED_Zpp -DISSI_PREDEFINED_FluoroImage -DISSI_PREDEFINED_RenderedImage)

# sawDaVinci component
#include_directories (${sawIntuitiveDaVinci_INCLUDE_DIR})
#link_directories (${sawIntuitiveDaVinci_LIBRARY_DIR})
include_directories (${${CISST_UI3_ISI_WRAPPER_USED}_INCLUDE_DIR})
link_directories (${${CISST_UI3_ISI_WRAPPER_USED}_LIBRARY_DIR})
add_definitions (-D${CISST_UI3_ISI_WRAPPER_DEFINITION})

add_executable (ui3ExDaVinciStereoVideo
				ISSIEventHandler.h
				svlWindow.h
                main.cpp)

set_property (TARGET ui3ExDaVinciStereoVideo PROPERTY FOLDER "cisst3DUserInterface/examples")

target_link_libraries (ui3ExDaVinciStereoVideo
                       #${sawIntuitiveDaVinci_LIBRARIES}
                       ${${CISST_UI3_ISI_WRAPPER_USED}_LIBRARIES}
                       ui3ExBehaviors
                       ${sawOpenIGTLink_LIBRARIES}
                       ${ISSILIB_LIB})

cisst_target_link_libraries (ui3ExDaVinciStereoVideo
                             cisstCommon cisstVector cisstOSAbstraction cisstMultiTask
                             cisstStereoVision cisstParameterTypes cisst3DUserInterface)

# Copy VTK models to the build tree
if (WIN32)
    configure_file (${CMAKE_CURRENT_SOURCE_DIR}/probeEnd.STL
                    ${CMAKE_CURRENT_BINARY_DIR}/probeEnd.STL COPYONLY)
else (WIN32)
    configure_file (${CMAKE_CURRENT_SOURCE_DIR}/probeEnd.STL
                    ${EXECUTABLE_OUTPUT_PATH}/probeEnd.STL COPYONLY)
endif (WIN32)