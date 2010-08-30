/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: 

  Author(s):  Anton Deguet, Min Yang Jung
  Created on: 2010-08-29

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#ifndef _mtsManagerComponentBase_h
#define _mtsManagerComponentBase_h

#include <cisstMultiTask/mtsTaskFromSignal.h>

//-----------------------------------------------------------------------------
//  Component Description
//
class mtsDescriptionComponent: public mtsGenericObject
{
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    std::string ProcessName;
    std::string ComponentName;
    std::string ClassName;

    void ToStream(std::ostream & outputStream) const;
    void SerializeRaw(std::ostream & outputStream) const;
    void DeSerializeRaw(std::istream & inputStream);
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsDescriptionComponent);


//-----------------------------------------------------------------------------
//  Connection Description
//
class mtsDescriptionConnection: public mtsGenericObject
{
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    struct FullInterface {
        std::string ProcessName;
        std::string ComponentName;
        std::string InterfaceName;
    };

    FullInterface Client;
    FullInterface Server;

    void ToStream(std::ostream & outputStream) const;
    void SerializeRaw(std::ostream & outputStream) const;
    void DeSerializeRaw(std::istream & inputStream);
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsDescriptionConnection);

//-----------------------------------------------------------------------------
//  Manager Component Base Class
//
class mtsManagerComponentBase : public mtsTaskFromSignal
{
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    mtsManagerComponentBase(const std::string & componentName);
    virtual ~mtsManagerComponentBase();

    virtual void Startup(void) = 0;
    virtual void Run(void);
    virtual void Cleanup(void);
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsManagerComponentBase);

#endif // _mtsManagerComponentBase_h

/*
class mtsManagerComponent: public mtsTaskFromSignal
{
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    mtsManagerComponent(const std::string & componentName);
    ~mtsManagerComponent() {};
    void Startup(void);
    void Run(void);
    void Cleanup(void) {};

    // method to add standardized interface to component to talk to this
    static void AddInterfaceToManager(mtsComponent * component);

    void ConnectToRemoteManager(const std::string & processName);

protected:

    // added to interface for components to create local or remote
    void CreateComponent(const mtsDescriptionNewComponent & component);
    void Connect(const mtsDescriptionConnection & connection);

    // added to interface for managers to create local
    void CreateComponentLocally(const mtsDescriptionNewComponent & component);
    void ConnectLocally(const mtsDescriptionConnection & connection);
    
    struct OtherManager {
        mtsFunctionWrite CreateComponent;
        mtsFunctionWrite Connect;
        mtsInterfaceRequired * RequiredInterface; // might be useful?
    };

    cmnNamedMap<OtherManager> OtherManagers;
};
*/
