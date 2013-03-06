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
                       ${sawIntuitiveDaVinci_LIBRARIES})

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
