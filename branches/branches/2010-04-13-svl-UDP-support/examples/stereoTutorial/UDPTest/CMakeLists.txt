#
# $Id: CMakeLists.txt 1236 2010-02-26 20:38:21Z adeguet1 $
#
# (C) Copyright 2006-2008 Johns Hopkins University (JHU), All Rights
# Reserved.
#
# --- begin cisst license - do not edit ---
# 
# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.
# 
# --- end cisst license ---

# UDP Sender
ADD_EXECUTABLE(UDPSenderTest UDPSender.cpp)
CISST_REQUIRES("UDPSenderTest" "cisstCommon;cisstVector;cisstOSAbstraction;cisstStereoVision")

# UDP Receiver
IF(CISST_SVL_HAS_X11 OR WIN32)
    ADD_EXECUTABLE(UDPReceiverTest UDPReceiver.cpp)
    CISST_REQUIRES("UDPReceiverTest" "cisstCommon;cisstVector;cisstOSAbstraction;cisstStereoVision")
ENDIF(CISST_SVL_HAS_X11 OR WIN32)
