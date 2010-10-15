#
# $Id: $
#
# (C) Copyright 2005-2010 Johns Hopkins University (JHU), All Rights
# Reserved.
#
# --- begin cisst license - do not edit ---
#
# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.
#
# --- end cisst license ---

project (svlExMultitask2)

include_directories (${CMAKE_CURRENT_SOURCE_DIR})

add_library (svlExMultitask2Lib
             # mtsManagerComponent.h
	     # mtsManagerComponent.cpp
             exampleFilter.h
             exampleFilter.cpp
             exampleComponent.h
             exampleComponent.cpp)

add_executable (svlExMultitask2Video videoMain.cpp)
target_link_libraries (svlExMultitask2Video svlExMultitask2Lib)

add_executable (svlExMultitask2Application applicationMain.cpp)
target_link_libraries (svlExMultitask2Application svlExMultitask2Lib)

# link with the cisst libraries
cisst_target_link_libraries (svlExMultitask2Lib cisstCommon cisstOSAbstraction cisstMultiTask cisstStereoVision)
cisst_target_link_libraries (svlExMultitask2Application cisstCommon cisstOSAbstraction cisstMultiTask cisstDevices cisstStereoVision)
cisst_target_link_libraries (svlExMultitask2Video cisstCommon cisstOSAbstraction cisstMultiTask cisstStereoVision)
