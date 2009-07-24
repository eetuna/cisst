/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCollectorGUI.h 2009-03-20 mjung5

  Author(s):  Min Yang Jung
  Created on: 2009-07-23

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#ifndef _mtsCollectorGUI_h
#define _mtsCollectorGUI_h

#include <cisstMultiTask/mtsStateTable.h>
#include <cisstMultiTask/multiplot.h>

#include <cisstMultiTask/mtsExport.h>

#include <string>

/*!
  \ingroup cisstMultiTask

*/
class CISST_EXPORT mtsCollectorGUI
{
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

protected:
    MULTIPLOT * GraphPane;

    /*! Initialize the graph pane registered. */
    void Initialize();

public:
    mtsCollectorGUI(MULTIPLOT * graphPane);
    ~mtsCollectorGUI() {}

    void UpdateUI(const double newValue);
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsCollectorGUI)

#endif // _mtsCollectorGUI_h
