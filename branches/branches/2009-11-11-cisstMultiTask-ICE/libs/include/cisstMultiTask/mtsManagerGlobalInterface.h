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

  This class defines an interface used by local component manager to communicate 
  with the global component manager.  The interface is defined as a pure abstract 
  class because there are two different cases that the interface are used for:

  Standalone Mode: Inter-thread communication, no ICE.  A local component manager 
    directly connects to the global component manager that runs in the same process.
    In this case, mtsManagerLocal::ManagerGlobal is of type mtsManagerGlobal.

  Network mode: Inter-process communication, ICE enabled.  A local component 
    manager connects to the global component manager via a proxy for it.
    In this case, mtsManagerLocal::ManagerGlobal is of type mtsManagerGlobalProxyClient.

  \note Please refer to mtsManagerGlobal and mtsManagerGlobalProxyClient for details.
*/

#ifndef _mtsManagerGlobalInterface_h
#define _mtsManagerGlobalInterface_h

#include <cisstCommon/cmnGenericObject.h>

class CISST_EXPORT mtsManagerGlobalInterface : public cmnGenericObject {

public:
    //-------------------------------------------------------------------------
    //  Process Management
    //-------------------------------------------------------------------------
    /*! Register a process. */
    virtual bool AddProcess(const std::string & processName) = 0;

    /*! Find a process. */
    virtual bool FindProcess(const std::string & processName) const = 0;

    /*! Remove a process. */
    virtual bool RemoveProcess(const std::string & processName) = 0;

    //-------------------------------------------------------------------------
    //  Component Management
    //-------------------------------------------------------------------------
    /*! Register a component. */
    virtual bool AddComponent(const std::string & processName, const std::string & componentName) = 0;

    /*! Find a component using process name and component name */
    virtual bool FindComponent(const std::string & processName, const std::string & componentName) const = 0;

    /*! Remove a component. */
    virtual bool RemoveComponent(const std::string & processName, const std::string & componentName) = 0;

    //-------------------------------------------------------------------------
    //  Interface Management
    //-------------------------------------------------------------------------
    /*! Register an interface. Note that adding/removing an interface can be run-time. */
    virtual bool AddProvidedInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName) = 0;

    virtual bool AddRequiredInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName) = 0;

    /*! Find an interface using process name, component name, and interface name */
    virtual bool FindProvidedInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName) const = 0;

    virtual bool FindRequiredInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName) const = 0;

    /*! Remove an interface. Note that adding/removing an interface can be run-time. */
    virtual bool RemoveProvidedInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName) = 0;

    virtual bool RemoveRequiredInterface(
        const std::string & processName, const std::string & componentName, const std::string & interfaceName) = 0;

    //-------------------------------------------------------------------------
    //  Connection Management
    //-------------------------------------------------------------------------
    /*! Connect two interfaces */
    virtual bool Connect(
        const std::string & clientProcessName,
        const std::string & clientComponentName,
        const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName,
        const std::string & serverComponentName,
        const std::string & serverProvidedInterfaceName) = 0;

    /*! Disconnect two interfaces */
    virtual void Disconnect(
        const std::string & clientProcessName,
        const std::string & clientComponentName,
        const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName,
        const std::string & serverComponentName,
        const std::string & serverProvidedInterfaceName) = 0;
};

#endif // _mtsManagerGlobalInterface_h

