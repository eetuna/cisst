/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */
/* $Id: UITask.cpp 75 2009-02-24 16:47:20Z adeguet1 $ */

#include <math.h>
#include "UITask.h"
#include "displayUI.h"

UITask::UITask(const std::string & taskName, double period):
    mtsTaskPeriodic(taskName, period, false, 5000),
    ExitFlag(false)
{
}

void UITask::Configure(const std::string & CMN_UNUSED(filename))
{
    // define some values, ideally these come from a configuration
    // file and then configure the user interface
    double maxValue = 0.5; double minValue = 5.0;
    StartValue =  1.0;
    UI.Amplitude->bounds(minValue, maxValue);
    UI.Amplitude->value(StartValue);
    AmplitudeData.Data = StartValue;
}

void UITask::Startup(void) 
{
    // set the initial amplitude based on the configuration
    AmplitudeData.Data = StartValue;

    // make the UI visible
    UI.show(0, NULL);
}

void UITask::Run(void)
{
    // update the UI, process UI events 
    if (Fl::check() == 0) {
        ExitFlag = true;
    }
}

/*
  Author(s):  Ankur Kapoor, Peter Kazanzides, Anton Deguet
  Created on: 2004-04-30

  (C) Copyright 2004-2008 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/
