/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsGlobalManager.h 794 2009-09-01 21:43:56Z pkazanz1 $

  Author(s):  Min Yang Jung
  Created on: 2009-11-12

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
  \brief Definition of the global manager
  \ingroup cisstMultiTask

  TODO: add description
*/

#ifndef _mtsGlobalManager_h
#define _mtsGlobalManager_h

#include <cisstCommon/cmnGenericObject.h>
#include <cisstCommon/cmnClassRegister.h>
#include <cisstMultiTask/mtsGlobalManagerInterface.h>
#include <cisstMultiTask/mtsForwardDeclarations.h>

#include <cisstMultiTask/mtsExport.h>

#include <map>

class CISST_EXPORT mtsGlobalManager : public mtsGlobalManagerInterface {

    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

protected:
    //-------------------------------------------------------------------------
    //  Data Structure Definition
    //-------------------------------------------------------------------------
    /*! Connection map: (interface id, connected interface info) */
    typedef struct {
        std::string ProcessName;
        std::string ComponentName;
        std::string InterfaceName;
    } ConnectedInterfaceInfo;

    typedef std::map<std::string, ConnectedInterfaceInfo *> ConnectionMapType;

    /*! Component map: (component id, connection map)*/
    typedef std::map<std::string, ConnectionMapType *> ComponentMapType;

    /*! Process map: (connected process id, component map) */
    typedef std::map<std::string, ComponentMapType *> ProcessMapType;

    /*! Process proxy map: (connected process id, proxy for the process) */
    typedef std::map<std::string, mtsGlobalManagerProxyServer *> ProcessProxyMapType;

    ProcessProxyMapType ProcessProxyMap;

    /* MJUNG: 11/16/09
       The following definition generates C4053 warning. This can be resolved by
       introducing auxiliary structure definitions.
       (see http://msdn.microsoft.com/en-us/library/074af4b6(VS.80).aspx)
    */
    ProcessMapType ProcessMap;

public:
    /*! Constructor and destructor */
    mtsGlobalManager();

    ~mtsGlobalManager();

    /*! Register a component to the global manager. */
    bool AddComponent(
        const std::string & processName, const std::string & componentName);

    /*! Connect two components. */
    bool Connect(
        const std::string & clientProcessName,
        const std::string & clientComponentName,
        const std::string & clientRequiredInterfaceName,
        const std::string & serverProcessName,
        const std::string & serverComponentName,
        const std::string & serverProvidedInterfaceName);

    /*! Remove a component from the global manager. */
    bool RemoveComponent(
        const std::string & processName, const std::string & componentName);
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsGlobalManager)

#endif // _mtsGlobalManager_h

