/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
 $Id: $

 Author(s):  Anton Deguet
 Created on: 2010

 (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
 Reserved.

 --- begin cisst license - do not edit ---

 This software is provided "as is" under an open source license, with
 no warranty.  The complete license can be found in license.txt and
 http://www.cisst.org/cisst/license.txt.

 --- end cisst license ---

 */

#ifndef _mtsManagerComponent_h
#define _mtsManagerComponent_h

#include <cisstMultiTask/mtsTaskFromSignal.h>

class mtsDescriptionNewComponent: public mtsGenericObject
{
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);
 public:
    std::string ProcessName;
    std::string ClassName;
    std::string ComponentName;

    inline void ToStream(std::ostream & outputStream) const {
        mtsGenericObject::ToStream(outputStream);
        outputStream << std::endl
		     << "Process: " << this->ProcessName
		     << " Class: " << this->ClassName
		     << " Name: " << this->ComponentName << std::endl;
    }

    inline void SerializeRaw(std::ostream & outputStream) const
    {
        mtsGenericObject::SerializeRaw(outputStream);
	cmnSerializeRaw(outputStream, this->ProcessName);
	cmnSerializeRaw(outputStream, this->ClassName);
	cmnSerializeRaw(outputStream, this->ComponentName);
    }

    inline void DeSerializeRaw(std::istream & inputStream)
    {
        mtsGenericObject::DeSerializeRaw(inputStream);
	cmnDeSerializeRaw(inputStream, this->ProcessName);
	cmnDeSerializeRaw(inputStream, this->ClassName);
	cmnDeSerializeRaw(inputStream, this->ComponentName);
    }

};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsDescriptionNewComponent);

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

    // added to interface for managers to create local
    void CreateComponentLocally(const mtsDescriptionNewComponent & component);
    
    struct OtherManager {
	mtsFunctionWrite CreateComponent;
	mtsInterfaceRequired * RequiredInterface; // might be useful?
    };

    cmnNamedMap<OtherManager> OtherManagers;
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsManagerComponent);

#endif // _mtsManagerComponent_h
