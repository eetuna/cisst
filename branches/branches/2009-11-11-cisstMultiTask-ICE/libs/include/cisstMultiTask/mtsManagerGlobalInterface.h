/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerGlobalInterface.h 794 2009-09-01 21:43:56Z pkazanz1 $

  Author(s):  Min Yang Jung
  Created on: 2009-11-15

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

/*!
  \file
  \brief Definition of mtsManagerGlobalInterface
  \ingroup cisstMultiTask

  This class defines an interface used by local task manager to communicate 
  with the global manager.  The interface is defined as a pure abstract 
  class because there are two different cases that the interface covers:

  Standalone Mode: Inter-thread communication, no ICE.  Local task manager 
    directly connects to the global manager that runs in the same process.
    In this case, mtsTaskManager::ManagerGlobal is of type mtsManagerGlobal.

  Network mode: Inter-process communication, ICE enabled.  Local task manager
    connects to the global manager via a proxy for the global manager.
    In this case, mtsTaskManager::ManagerGlobal is of type mtsManagerGlobalProxyClient.

  \note Please refer to mtsManagerGlobal and mtsManagerGlobalProxyClient  as well.
*/

#ifndef _mtsManagerGlobalInterface_h
#define _mtsManagerGlobalInterface_h

#include <cisstCommon/cmnGenericObject.h>

class CISST_EXPORT mtsManagerGlobalInterface : public cmnGenericObject {

public:
    /*! Register a component to the global manager. */
    virtual bool AddComponent(
        const std::string & processName, const std::string & componentName) = 0;

    /*! Connect two components. */
    virtual bool Connect(
        const std::string & clientProcessName,
        const std::string & clientComponentName,
        const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName,
        const std::string & serverComponentName,
        const std::string & serverProvidedInterfaceName) = 0;

    /*! Remove a component from the global manager. */
    virtual bool RemoveComponent(
        const std::string & processName, const std::string & componentName) = 0;
};

#endif // _mtsManagerGlobalInterface_h

