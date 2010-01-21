/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: main.cpp 682 2009-08-14 03:18:33Z adeguet1 $
  
  Author(s):  Min Yang Jung
  Created on: 2010-01-20

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#include <cisstCommon/cmnPortability.h>
#include <cisstCommon/cmnLogger.h>
#include <cisstOSAbstraction/osaSleep.h>
#include <cisstOSAbstraction/osaThreadedLogFile.h>
#include <cisstMultiTask/mtsManagerGlobal.h>

int main(int CMN_UNUSED(argc), char ** CMN_UNUSED(argv))
{
    // log configuration, add a log per thread
    osaThreadedLogFile threadedLog("GlobalComponentManager");
    cmnLogger::GetMultiplexer()->AddChannel(threadedLog, CMN_LOG_LOD_RUN_WARNING);
    // specify a higher, more verbose log level for these classes
    cmnClassRegister::SetLoD("mtsManagerGlobal", CMN_LOG_LOD_RUN_WARNING);

    // Create and start global component manager
    mtsManagerGlobal globalComponentManager;
    if (!globalComponentManager.StartServer()) {
        CMN_LOG_INIT_ERROR << "Failed to start global component manager." << std::endl;
        return 1;
    }
    CMN_LOG_INIT_VERBOSE << "Global component manager started..." << std::endl;

    //
    // TODO: instead of 1, mtsManagerGlobal::StartServer() can be blocking
    // such that while loop ends if mtsManagerGlobal finishes??
    //
    while (1) {
        osaSleep(10 * cmn_ms);
    }

    return 0;
}
