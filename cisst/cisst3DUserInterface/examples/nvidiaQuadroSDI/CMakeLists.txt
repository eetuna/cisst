#
# $Id: CMakeLists.txt 3331 2012-01-10 22:34:09Z adeguet1 $
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

project (ui3ExNVIDIAQuadroSDI)

# sawSensablePhantom configuration
include_directories (${sawIntuitiveDaVinci_INCLUDE_DIR})
link_directories (${sawIntuitiveDaVinci_LIBRARY_DIR})

add_executable (ui3ExNVIDIAQuadroSDI
                svlSDI.cpp)

set_property (TARGET ui3ExNVIDIAQuadroSDI PROPERTY FOLDER "cisst3DUserInterface/examples")

target_link_libraries (ui3ExNVIDIAQuadroSDI
                       ${sawIntuitiveDaVinci_LIBRARIES}
                       )
if (CISST_SVL_HAS_ISSI)    
include_directories( ${ISSILIB_INCLUDE_DIR} )
link_directories(${ISSILIB_LIB_DIR})
add_definitions(-DISSI_DATA_DIR=${ISSILIB_DATA_DIR})
add_definitions( -DISSI_PREDEFINED_DaVinciSSC -DISSI_PREDEFINED_DaVinciSi -DISSI_PREDEFINED_EndoscopeImage -DISSI_PREDEFINED_Zeego -DISSI_PREDEFINED_Trumpf -DISSI_PREDEFINED_ULI -DISSI_PREDEFINED_Zpp -DISSI_PREDEFINED_FluoroImage -DISSI_PREDEFINED_RenderedImage)
                   
target_link_libraries (ui3ExNVIDIAQuadroSDI
                       ${sawIntuitiveDaVinci_LIBRARIES}
                       ${ISSILIB_LIB})

else (CISST_SVL_HAS_ISSI)
target_link_libraries (ui3ExNVIDIAQuadroSDI
                       ${sawIntuitiveDaVinci_LIBRARIES}
                       )
endif (CISST_SVL_HAS_ISSI)                         

cisst_target_link_libraries (ui3ExNVIDIAQuadroSDI
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
