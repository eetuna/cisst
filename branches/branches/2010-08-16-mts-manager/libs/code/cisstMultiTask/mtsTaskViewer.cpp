/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Peter Kazanzides
  Created on: 2010-09-07

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstCommon/cmnUnits.h>
#include <cisstOSAbstraction/osaSleep.h>
#include <cisstMultiTask/mtsTaskViewer.h>
#include <cisstMultiTask/mtsInterfaceRequired.h>

CMN_IMPLEMENT_SERVICES(mtsTaskViewer)

mtsTaskViewer::mtsTaskViewer(const std::string & name, double periodicityInSeconds) :
    mtsTaskPeriodic(name, periodicityInSeconds),
    JGraphSocket(osaSocket::TCP),
    JGraphSocketConnected(false),
    UDrawSocket(osaSocket::TCP),
    UDrawSocketConnected(false)
{
    // Extend internal required interface (to Manager Component) to include event handlers
    mtsInterfaceRequired *required = GetInterfaceRequired(mtsComponent::NameOfInterfaceInternalRequired);
    if (required) {
        required->AddEventHandlerWrite(&mtsTaskViewer::AddComponent, this, mtsComponent::EventNames::AddComponent);
        required->AddEventHandlerWrite(&mtsTaskViewer::AddConnection, this, mtsComponent::EventNames::AddConnection);
    }
}

mtsTaskViewer::~mtsTaskViewer()
{
    this->Cleanup();
}

void mtsTaskViewer::Startup(void)
{
    CMN_LOG_CLASS_INIT_VERBOSE << "Startup called" << std::endl;
    // Try to connect to JGraph or UDrawGraph viewer
    if (!JGraphSocketConnected)
        ConnectToJGraph();
    if (!UDrawSocketConnected)
        ConnectToUDrawGraph();
    if (JGraphSocketConnected || UDrawSocketConnected)
        SendAllInfo();
}

void mtsTaskViewer::Run(void)
{
    if (!UDrawSocketConnected) {
       ConnectToUDrawGraph();
       if (UDrawSocketConnected) {
           CMN_LOG_CLASS_INIT_VERBOSE << "Run: Sending all info" << std::endl;
           SendAllInfo();
       }
    }
#if 0
    // Could also periodically check for connection to JGraph-based program.
    if (!JGrahSocketConnected) {
       ConnectToJGraph();
       if (JGraphSocketConnected) {
           CMN_LOG_CLASS_INIT_VERBOSE << "Run: Sending all info" << std::endl;
           SendAllInfo();
       }
    }
#endif
    ProcessQueuedCommands();
    ProcessQueuedEvents();
}

void mtsTaskViewer::Cleanup(void)
{
    if (JGraphSocketConnected) {
        JGraphSocket.Close();
        JGraphSocketConnected = false;
    }

    if (UDrawSocketConnected) {
        UDrawSocket.Close();
        UDrawSocketConnected = false;
    }
}

//*************************************** Event Handlers ******************************************************

void mtsTaskViewer::AddComponent(const mtsDescriptionComponent &componentInfo)
{
    if (JGraphSocketConnected) {
        std::string buffer = GetComponentInGraphFormat(componentInfo.ProcessName, componentInfo.ComponentName);
        if (buffer != "") {
            CMN_LOG_CLASS_INIT_VERBOSE << "Sending " << buffer << std::endl;
            JGraphSocket.Send(buffer);
        }
    }
    if (UDrawSocketConnected) {
        std::string buffer = GetComponentInUDrawGraphFormat(componentInfo.ProcessName, componentInfo.ComponentName);
        if (buffer != "") {
            CMN_LOG_CLASS_INIT_VERBOSE << "Sending " << buffer << std::endl;
            UDrawSocket.Send(buffer);
            char response[256];
            if (UDrawSocket.Receive(response, sizeof(response), 1.0))
                CMN_LOG_CLASS_INIT_VERBOSE << "Received response from UDraw(Graph): " << response << std::endl;
        }
    }
}

void mtsTaskViewer::AddConnection(const mtsDescriptionConnection &connection)
{
    // Send to TaskViewer if present
    if (JGraphSocketConnected) {
        std::string message = "add edge [" + connection.Client.ProcessName + ":" + connection.Client.ComponentName + ", "
                                           + connection.Server.ProcessName + ":" + connection.Server.ComponentName + ", "
                                           + connection.Client.InterfaceName + ", "
                                           + connection.Server.InterfaceName + "]\n";
        CMN_LOG_CLASS_INIT_VERBOSE << "Sending " << message << std::endl;
        JGraphSocket.Send(message);
    }
    if (UDrawSocketConnected) {
        char response[256];
        std::string message("graph(update([],[new_edge(\"");
        sprintf(response, "%d", connection.ConnectionID);
        message.append(response);
        message.append("\", \"C\", [a(\"OBJECT\", \"");
        message.append(response);
        message.append("\"), a(\"INFO\", \"");
        message.append(connection.Client.InterfaceName + "<->" + connection.Server.InterfaceName);
        message.append("\")], \"");
        message.append(connection.Client.ProcessName + ":" + connection.Client.ComponentName);
        message.append("\", \"");
        message.append(connection.Server.ProcessName + ":" + connection.Server.ComponentName);
        message.append("\")]))\n");
        CMN_LOG_CLASS_INIT_VERBOSE << "Sending " << message << std::endl;
        UDrawSocket.Send(message);
        if (UDrawSocket.Receive(response, sizeof(response), 1.0))
            CMN_LOG_CLASS_INIT_VERBOSE << "Received response from UDraw(Graph): " << response << std::endl;
    }
}

//********************************* Local (protected) methods ***********************************************

bool mtsTaskViewer::ConnectToJGraph(const std::string &ipAddress, unsigned short port)
{
    // Try to connect to the JGraph application software (Java program).
    // Note that the JGraph application also sends event messages back via the socket,
    // though we don't currently read them.
    CMN_LOG_CLASS_INIT_WARNING << "Attempting to connect to JGraph TaskViewer" << std::endl;
    JGraphSocketConnected = JGraphSocket.Connect(ipAddress, port);
    if (JGraphSocketConnected) {
        osaSleep(1.0 * cmn_s);  // need to wait or JGraph server will not start properly
    }
    return JGraphSocketConnected;
}

bool mtsTaskViewer::ConnectToUDrawGraph(const std::string &ipAddress, unsigned short port)
{
    // Try to connect to UDrawGraph on port 2554
    // (Note: default UDrawGraph port is 2542, but this may be a target for hackers).
    UDrawSocketConnected = UDrawSocket.Connect(ipAddress, port);
    // wait for initial OK
    if (UDrawSocketConnected) {
        char response[256];
        if (UDrawSocket.Receive(response, sizeof(response), 3.0)) {
            CMN_LOG_CLASS_INIT_VERBOSE << "Received response from UDraw(Graph): " << response << std::endl;
            UDrawSocket.Send("graph(new([]))\n");
            if (UDrawSocket.Receive(response, sizeof(response), 1.0))
               CMN_LOG_CLASS_INIT_VERBOSE << "Received response from UDraw(Graph), new: " << response << std::endl;
        }
    }
    return UDrawSocketConnected;
}

bool mtsTaskViewer::IsProxyComponent(const std::string & componentName) const
{
    // PK: Need to fix this to be more robust
    return (componentName.find("Proxy", componentName.length()-5) != std::string::npos);
}

void mtsTaskViewer::SendAllInfo(void)
{
    // Now, send all existing components and connections
    std::vector<std::string> processList;
    std::vector<std::string> componentList;
    size_t i, j;  // could use iterators instead
    RequestGetNamesOfProcesses(processList);
    for (i = 0; i < processList.size(); i++) {
        componentList.clear();
        RequestGetNamesOfComponents(processList[i], componentList);
        for (j = 0; j < componentList.size(); j++) {
            // Ignore proxy components
            if (!IsProxyComponent(componentList[j])) {
                mtsDescriptionComponent arg;
                arg.ProcessName = processList[i];
                arg.ComponentName = componentList[j];
                this->AddComponent(arg);
            }
        }
        std::vector<mtsDescriptionConnection> connectionList;
        RequestGetListOfConnections(connectionList);
        for (i = 0; i < connectionList.size(); i++)
            this->AddConnection(connectionList[i]);
    }
}

std::string mtsTaskViewer::GetComponentInGraphFormat(const std::string &processName,
                                                     const std::string &componentName) const
{
// PK TEMP
#if 0
    size_t i;
    std::vector<std::string> requiredList;
    std::vector<std::string> providedList;
    GetNamesOfInterfacesRequiredOrInput(processName, componentName, requiredList);
    GetNamesOfInterfacesProvidedOrOutput(processName, componentName, providedList);
    // For now, ignore components that don't have any interfaces
    if ((requiredList.size() == 0) && (providedList.size() == 0))
        return "";
    std::string buffer;
    buffer = "add taska [[" + processName + ":" + componentName + "],[";
    for (i = 0; i < requiredList.size(); i++) {
        buffer += requiredList[i];
        if (i < requiredList.size()-1)
            buffer += ",";
    }
    buffer += "],[";
    for (i = 0; i < providedList.size(); i++) {
        buffer += providedList[i];
        if (i < providedList.size()-1)
            buffer += ",";
    }
    buffer += "]]\n";
#else
    std::string buffer;
    buffer = "add taska [[" + processName + ":" + componentName + "],[],[]]\n";
#endif
    return buffer;
}

std::string mtsTaskViewer::GetComponentInUDrawGraphFormat(const std::string &processName,
                                                          const std::string &componentName) const
{
// PK TEMP
#if 0
    std::vector<std::string> requiredList;
    std::vector<std::string> providedList;
    GetNamesOfInterfacesRequiredOrInput(processName, componentName, requiredList);
    GetNamesOfInterfacesProvidedOrOutput(processName, componentName, providedList);
    // For now, ignore components that don't have any interfaces
    if ((requiredList.size() == 0) && (providedList.size() == 0))
        return "";
#endif
    std::string buffer("graph(update([new_node(\"");
    buffer.append(processName + ":" + componentName);
    buffer.append("\",\"B\",[a(\"OBJECT\",\""); 
    buffer.append(componentName);
    buffer.append("\"), a(\"INFO\", \"");
    buffer.append(processName + ":" + componentName);
    buffer.append("\")])],[]))\n");
    return buffer;
}
