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
  \brief Declaration of Manager Component Client
  \ingroup cisstMultiTask

  This class defines the manager component client which is managed by all local 
  component managers (LCMs).  An instance of this class is automatically created 
  and gets connected to the manager component server which runs on LCM that runs
  with the global component manager (GCM).
  
  This component has two sets of interfaces, one for communication with the 
  manager component server and the other one for command exchange between other
  manager component clients.
  
  \note Related classes: mtsManagerComponentBase, mtsManagerComponentServer
*/

#ifndef _mtsManagerComponentClient_h
#define _mtsManagerComponentClient_h

#include <cisstMultiTask/mtsManagerComponentBase.h>

class mtsManagerComponentClient : public mtsManagerComponentBase
{
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    mtsManagerComponentClient(const std::string & processName);
    ~mtsManagerComponentClient();

    void Startup(void);
    void Run(void);
    void Cleanup(void);
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsManagerComponentClient);

#endif // _mtsManagerComponentClient_h
