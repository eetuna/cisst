/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsParameterTypes.h 1726 2010-08-30 05:07:54Z mjung5 $

  Author(s):  Anton Deguet, Min Yang Jung
  Created on: 2010-09-01

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#ifndef _mtsParameterTypes_h
#define _mtsParameterTypes_h

#include <cisstMultiTask/mtsGenericObject.h>

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
    int ConnectionID;

    void ToStream(std::ostream & outputStream) const;
    void SerializeRaw(std::ostream & outputStream) const;
    void DeSerializeRaw(std::istream & inputStream);
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsDescriptionConnection);

//-----------------------------------------------------------------------------
//  Component Status Control
//
class mtsComponentStatusControl : public mtsGenericObject
{
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    typedef enum { 
        COMPONENT_START = 0, 
        COMPONENT_STOP, 
        COMPONENT_RESUME 
    } ComponentStatusCommand;

    std::string ProcessName;
    std::string ComponentName;
    double      DelayInSecond;
    ComponentStatusCommand Command;

    void ToStream(std::ostream & outputStream) const;
    void SerializeRaw(std::ostream & outputStream) const;
    void DeSerializeRaw(std::istream & inputStream);
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsComponentStatusControl);

#endif // _mtsParameterTypes_h
