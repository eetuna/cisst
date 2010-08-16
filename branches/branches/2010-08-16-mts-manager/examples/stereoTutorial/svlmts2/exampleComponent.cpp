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

#include "exampleComponent.h"
#include "mtsManagerComponent.h"

CMN_IMPLEMENT_SERVICES(exampleComponent);

exampleComponent::exampleComponent(const std::string & componentName):
    mtsTaskFromSignal(componentName, 50)
{
    UseSeparateLogFileDefault();
    mtsInterfaceProvided * interfaceProvided;
    interfaceProvided = this->AddInterfaceProvided("UI");
    interfaceProvided->AddCommandVoid(&exampleComponent::CreateVideo, this, "CreateVideo");
    interfaceProvided->AddCommandVoid(&exampleComponent::StartVideo, this, "StartVideo");

    mtsInterfaceRequired * interfaceRequired;
    interfaceRequired = this->AddInterfaceRequired("ToManager");
    interfaceRequired->AddFunction("CreateComponent", this->Manager.CreateComponent);
}

void exampleComponent::Startup(void)
{
}

void exampleComponent::Run(void)
{
    ProcessQueuedCommands();
    ProcessQueuedEvents();
}

void exampleComponent::CreateVideo(void)
{
    CMN_LOG_CLASS_RUN_VERBOSE << "CreateVideo" << std::endl;
    mtsDescriptionNewComponent newComponent;
    newComponent.ProcessName = "svlExMultitask2Video";

    // stream manager
    newComponent.ClassName = "svlStreamManager";
    newComponent.ComponentName = "Stream";
    Manager.CreateComponent(newComponent);
    // source video file
    newComponent.ClassName = "svlFilterSourceVideoFile";
    newComponent.ComponentName = "Source";
    Manager.CreateComponent(newComponent);
    // image window
    newComponent.ClassName = "svlFilterImageWindow";
    newComponent.ComponentName = "Window";
    Manager.CreateComponent(newComponent);
}

void exampleComponent::StartVideo(void)
{
    CMN_LOG_CLASS_RUN_VERBOSE << "StartVideo" << std::endl;
    
}
