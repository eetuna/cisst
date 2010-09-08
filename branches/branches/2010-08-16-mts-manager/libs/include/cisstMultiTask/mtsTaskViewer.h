/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Peter Kazanzides
  Created on: 2010-09-07

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

/*!
  \file
  \brief Definition of the task viewer
  \ingroup cisstMultiTask
*/

#ifndef _mtsTaskViewer_h
#define _mtsTaskViewer_h


#include <cisstOSAbstraction/osaSocket.h>
#include <cisstMultiTask/mtsTaskPeriodic.h>
#include <cisstMultiTask/mtsParameterTypes.h>

class mtsTaskViewer : public mtsTaskPeriodic
{
   CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

protected:

    osaSocket JGraphSocket;
    bool JGraphSocketConnected;

    osaSocket UDrawSocket;
    bool UDrawSocketConnected;

    bool ConnectToJGraph(const std::string &ipAddress = "localhost", unsigned short port = 4444);
    bool ConnectToUDrawGraph(const std::string &ipAddress = "localhost", unsigned short port = 2554);

    bool IsProxyComponent(const std::string & componentName) const;

    void SendAllInfo(void);

    std::string GetComponentInGraphFormat(const std::string & processName, const std::string & componentName) const;
    std::string GetComponentInUDrawGraphFormat(const std::string & processName, const std::string & componentName) const;

    // Event Handlers
    void AddComponent(const mtsDescriptionComponent &componentInfo);
    void AddConnection(const mtsDescriptionConnection &connectionInfo);

public:

    mtsTaskViewer(const std::string & name, double periodicityInSeconds);

    virtual ~mtsTaskViewer();

    void Configure(const std::string & CMN_UNUSED(filename)) {}

    void Startup(void);

    void Run(void);

    void Cleanup(void);

};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsTaskViewer)

#endif // _mts_TaskViewer.h
