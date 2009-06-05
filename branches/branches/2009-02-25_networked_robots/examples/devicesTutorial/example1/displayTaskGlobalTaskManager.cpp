/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */
/* $Id: displayTaskGlobalTaskManager.cpp 332 2009-05-11 00:57:59Z mjung5 $ */


#include "displayTaskGlobalTaskManager.h"
#include "displayUIGlobalTaskManager.h"

CMN_IMPLEMENT_SERVICES(displayTaskGlobalTaskManager);

displayTaskGlobalTaskManager::displayTaskGlobalTaskManager(const std::string & taskName, double period):
    mtsTaskPeriodic(taskName, period, false, 500),
    ExitFlag(false)
{
}

void displayTaskGlobalTaskManager::Configure(const std::string & CMN_UNUSED(filename))
{
}

void displayTaskGlobalTaskManager::Startup(void)
{
    // make the UI visible
    UI.show(0, NULL);
}

void displayTaskGlobalTaskManager::Run(void)
{
    // process events
    this->ProcessQueuedEvents();
    // update the UI, process UI events 
    if (Fl::check() == 0) {
        ExitFlag = true;
    }
}

/*
  Author(s):  Min Yang Jung
  Created on: 2009-06-03

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/
