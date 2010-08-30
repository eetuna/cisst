/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: 

  Author(s):  Min Yang Jung
  Created on: 2010-08-29

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

/*!
  \brief Declaration of Manager Component Server
  \ingroup cisstMultiTask

  This class defines the manager component server which is managed by the local 
  component manager (LCM) that runs with the global component manager (GCM).
  Only one manager component server exists in the whole system and all the other
  manager components should be of type manager component client
  (mtsManagerComponentClient) which internally gets connected to the manager 
  component server when they start.

  This component provides services for other manager component clients to allow
  dynamic component creation and connection request (disconnection and 
  reconnection will be handled later).

  \note Related classes: mtsManagerComponentBase, mtsManagerComponentClient 
*/

#ifndef _mtsManagerComponentServer_h
#define _mtsManagerComponentServer_h

#include <cisstMultiTask/mtsManagerComponentBase.h>

class mtsManagerComponentServer : public mtsManagerComponentBase
{
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    mtsManagerComponentServer();
    ~mtsManagerComponentServer();

    void Startup(void);
    void Run(void);
    void Cleanup(void);
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsManagerComponentServer);

#endif // _mtsManagerComponentServer_h
