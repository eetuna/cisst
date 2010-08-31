/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerComponentClient.cpp 1726 2010-08-30 05:07:54Z mjung5 $

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

#include <cisstMultiTask/mtsManagerComponentClient.h>
#include <cisstMultiTask/mtsInterfaceProvided.h>
#include <cisstMultiTask/mtsInterfaceRequired.h>

CMN_IMPLEMENT_SERVICES(mtsManagerComponentClient);

std::string mtsManagerComponentClient::NameOfInterfaceRequired = "GCMServiceInterfaceRequired";

mtsManagerComponentClient::mtsManagerComponentClient(const std::string & componentName)
    : mtsManagerComponentBase(componentName)
{
    mtsInterfaceRequired * required = 
        AddInterfaceRequired(mtsManagerComponentClient::NameOfInterfaceRequired);
    if (required) {
        required->AddFunction("GetNamesOfProcesses", GetNamesOfProcesses);
    } else {
        std::string err("Failed to add required interface: ");
        err += mtsManagerComponentClient::NameOfInterfaceRequired;
        cmnThrow(std::runtime_error(err));
    }
}

mtsManagerComponentClient::~mtsManagerComponentClient()
{
}

void mtsManagerComponentClient::Startup(void)
{
   CMN_LOG_CLASS_INIT_VERBOSE << "Manager component CLIENT starts" << std::endl;
}

void mtsManagerComponentClient::Run(void)
{
    mtsManagerComponentBase::Run();
}

void mtsManagerComponentClient::Cleanup(void)
{
}

void mtsManagerComponentClient::Test(void)
{
    mtsStdStringVec vec;
    GetNamesOfProcesses(vec);

    static int cnt = 0;
    std::cout << ++cnt << ") size: " << vec.size() << std::endl;

    for (unsigned int i = 0; i < vec.size(); ++i) {
        std::cout << "\t" << vec(i) << std::endl;
    }
}