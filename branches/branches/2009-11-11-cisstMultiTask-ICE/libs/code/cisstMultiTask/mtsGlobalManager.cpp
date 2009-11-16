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

#include <cisstMultiTask/mtsGlobalManager.h>

CMN_IMPLEMENT_SERVICES(mtsGlobalManager);

mtsGlobalManager::mtsGlobalManager()
{
}

mtsGlobalManager::~mtsGlobalManager()
{
}

bool mtsGlobalManager::AddComponent(const std::string & processName, const std::string & componentName)
{
    return true;
}

bool mtsGlobalManager::Connect(
    const std::string & clientProcessName,
    const std::string & clientComponentName,
    const std::string & clientRequiredInterfaceName,
    const std::string & serverProcessName,
    const std::string & serverComponentName,
    const std::string & serverProvidedInterfaceName)
{
    return true;
}

    /*! Remove a component from the global manager. */
bool mtsGlobalManager::RemoveComponent(
    const std::string & processName, const std::string & componentName)
{
    return true;
}