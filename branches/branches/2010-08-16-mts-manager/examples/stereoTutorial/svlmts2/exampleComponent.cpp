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

#include <cisstOSAbstraction/osaSleep.h>

CMN_IMPLEMENT_SERVICES(exampleComponent);

exampleComponent::exampleComponent(const std::string & componentName):
    mtsTaskFromSignal(componentName, 50),
    StreamControl("StreamControl", this),
    SourceSettings("SourceSettings", this),
    WindowSettings("WindowSettings", this)
{
    UseSeparateLogFileDefault();

    mtsInterfaceProvided * interfaceProvided;
    interfaceProvided = this->AddInterfaceProvided("UI");
    interfaceProvided->AddCommandVoid(&exampleComponent::CreateVideo, this, "CreateVideo");
    interfaceProvided->AddCommandVoid(&exampleComponent::StartVideo, this, "StartVideo");

    mtsInterfaceRequired * interfaceRequired;
    interfaceRequired = this->AddInterfaceRequired("ToManager");
    interfaceRequired->AddFunction("CreateComponent", this->Manager.CreateComponent);
    interfaceRequired->AddFunction("Connect", this->Manager.Connect);
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

    // connect to the remote stream manager to connect it to the
    // source.  Later revisions should allow to connect the stream
    // manager to the source unsing mtsComponent::Interface::Connect
    mtsDescriptionConnection connection;
    connection.Server.ProcessName = "svlExMultitask2Video";
    connection.Server.ComponentName = "Stream";
    connection.Server.InterfaceName = "Control";
    connection.Client.ProcessName = "svlExMultitask2Application";
    connection.Client.ComponentName = this->GetName();
    connection.Client.InterfaceName =  "StreamControl";
    Manager.Connect(connection);
    osaSleep(1.0 * cmn_s);

    // connect to source and window control
    connection.Server.ProcessName = "svlExMultitask2Video";
    connection.Server.ComponentName = "Source";
    connection.Server.InterfaceName = "Settings";
    connection.Client.ProcessName = "svlExMultitask2Application";
    connection.Client.ComponentName = this->GetName();
    connection.Client.InterfaceName =  "SourceSettings";
    Manager.Connect(connection);

    connection.Server.ProcessName = "svlExMultitask2Video";
    connection.Server.ComponentName = "Window";
    connection.Server.InterfaceName = "Settings";
    connection.Client.ProcessName = "svlExMultitask2Application";
    connection.Client.ComponentName = this->GetName();
    connection.Client.InterfaceName =  "WindowSettings";
    Manager.Connect(connection);

    // connect filters together
    connection.Server.ProcessName = "svlExMultitask2Video";
    connection.Server.ComponentName = "Source";
    connection.Server.InterfaceName = "output";
    connection.Client.ProcessName = "svlExMultitask2Video";
    connection.Client.ComponentName = "Window";
    connection.Client.InterfaceName = "input";
    Manager.Connect(connection);

    // configure remote stream manager
    StreamControl.SetSourceFilter(mtsStdString("Source"));
}


void exampleComponent::StartVideo(void)
{
    CMN_LOG_CLASS_RUN_VERBOSE << "StartVideo" << std::endl;

    SourceSettings.SetLoop(true);
    SourceSettings.SetChannels(mtsInt(1));
    SourceSettings.SetFilename(mtsStdString("crop2.avi"));

    WindowSettings.SetTitle(mtsStdString("Image window"));
    WindowSettings.SetPosition(vctInt2(20, 20));

    StreamControl.Initialize();
    StreamControl.Play();
}
