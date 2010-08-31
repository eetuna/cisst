/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerComponentServer.cpp 1726 2010-08-30 05:07:54Z mjung5 $

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

#include <cisstMultiTask/mtsManagerComponentServer.h>
#include <cisstMultiTask/mtsManagerGlobal.h>

CMN_IMPLEMENT_SERVICES(mtsManagerComponentServer);

std::string mtsManagerComponentServer::NameOfManagerComponentServer = "MNGR-COMP-SERVER";
std::string mtsManagerComponentServer::NameOfInterfaceProvided = "GCMServiceInterfaceProvided";

mtsManagerComponentServer::mtsManagerComponentServer(mtsManagerGlobal * gcm)
    : mtsManagerComponentBase(mtsManagerComponentServer::NameOfManagerComponentServer),
      GCM(gcm)
{
    // Prevent this component from being created more than once
    static int instanceCount = 0;
    if (instanceCount != 0) {
        cmnThrow(std::runtime_error("Error in creating manager component server: it's already created"));
    }

    mtsInterfaceProvided * provided = 
        AddInterfaceProvided(mtsManagerComponentServer::NameOfInterfaceProvided);
    if (provided) {
        provided->AddCommandRead(&mtsManagerComponentServer::GetNamesOfProcesses, this, "GetNamesOfProcesses");
    } else {
        std::string err("Failed to add provided interface: ");
        err += mtsManagerComponentServer::NameOfInterfaceProvided;
        cmnThrow(std::runtime_error(err));
    }
}

mtsManagerComponentServer::~mtsManagerComponentServer()
{
}

void mtsManagerComponentServer::Startup(void)
{
    CMN_LOG_CLASS_INIT_VERBOSE << "Manager component SERVER starts" << std::endl;
}

void mtsManagerComponentServer::Run(void)
{
    mtsManagerComponentBase::Run();
}

void mtsManagerComponentServer::Cleanup(void)
{
}

void mtsManagerComponentServer::GetNamesOfProcesses(mtsStdStringVec & stdStringVec) const
{
    std::vector<std::string> namesOfProcesses;
    GCM->GetNamesOfProcesses(namesOfProcesses);

    const size_t n = namesOfProcesses.size();
    stdStringVec.SetSize(n);
    for (unsigned int i = 0; i < n; ++i) {
        stdStringVec(i) = namesOfProcesses[i];
    }
}