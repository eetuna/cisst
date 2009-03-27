/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsTaskManagerProxyCommon.cpp 145 2009-03-18 23:32:40Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-03-17

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstMultiTask/mtsTaskManagerProxyCommon.h>

Ice::CommunicatorPtr mtsTaskManagerProxyCommon::communicator;

static void onCtrlC(int)
{
    if(mtsTaskManagerProxyCommon::communicator) {
        try {
            mtsTaskManagerProxyCommon::communicator->shutdown();
        } catch(const Ice::CommunicatorDestroyedException&) {
            //
            // This might occur if we receive more than one signal.
            //
        }
    }
}

CMN_IMPLEMENT_SERVICES(mtsTaskManagerProxyCommon);


mtsTaskManagerProxyCommon::mtsTaskManagerProxyCommon() 
    : RunningFlag(false), InitSuccessFlag(false),
      IceCommunicator(NULL)
{
    IceUtil::CtrlCHandler ctrCHandler(onCtrlC);
}

mtsTaskManagerProxyCommon::~mtsTaskManagerProxyCommon()
{
}