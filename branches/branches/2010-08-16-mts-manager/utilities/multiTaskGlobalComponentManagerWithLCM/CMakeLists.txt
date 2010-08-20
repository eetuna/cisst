#
# $Id: CMakeLists.txt 1236 2010-02-26 20:38:21Z adeguet1 $
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

project (multiTaskGlobalComponentManagerWithLCM)

# Create GCM with LCM that manages sineTask

# name the main executable and specifies with source files to use
ADD_EXECUTABLE(multiTaskGlobalComponentManagerWithLCM
               main.cpp
               sineTask.h
               sineTask.cpp)

# link with the cisst libraries
CISST_REQUIRES("multiTaskGlobalComponentManagerWithLCM" "cisstCommon;cisstOSAbstraction;cisstMultiTask")



# Create LCM that manages displayTask
INCLUDE_DIRECTORIES(${multiTaskGlobalComponentManagerWithLCM_SOURCE_DIR})

FLTK_WRAP_UI(multiTaskGlobalComponentManagerWithLCMDisp displayUI.fl)

ADD_EXECUTABLE(multiTaskGlobalComponentManagerWithLCMDisp
               mainDisp.cpp
               displayTask.h
               displayTask.cpp
               ${multiTaskGlobalComponentManagerWithLCMDisp_FLTK_UI_SRCS})

CISST_REQUIRES("multiTaskGlobalComponentManagerWithLCMDisp" "cisstCommon;cisstOSAbstraction;cisstMultiTask")

