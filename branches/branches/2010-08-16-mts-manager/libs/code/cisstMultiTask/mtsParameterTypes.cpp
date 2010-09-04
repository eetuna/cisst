/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsParameterTypes.cpp 1726 2010-08-30 05:07:54Z mjung5 $

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

#include <cisstMultiTask/mtsParameterTypes.h>

//-----------------------------------------------------------------------------
//  Component Description
//
CMN_IMPLEMENT_SERVICES(mtsDescriptionComponent);

void mtsDescriptionComponent::ToStream(std::ostream & outputStream) const 
{
    mtsGenericObject::ToStream(outputStream);
    outputStream << std::endl
                 << "Process: " << this->ProcessName
                 << ", Class: " << this->ClassName
                 << ", Name: " << this->ComponentName << std::endl;
}

void mtsDescriptionComponent::SerializeRaw(std::ostream & outputStream) const
{
    mtsGenericObject::SerializeRaw(outputStream);
    cmnSerializeRaw(outputStream, this->ProcessName);
    cmnSerializeRaw(outputStream, this->ClassName);
    cmnSerializeRaw(outputStream, this->ComponentName);
}

void mtsDescriptionComponent::DeSerializeRaw(std::istream & inputStream)
{
    mtsGenericObject::DeSerializeRaw(inputStream);
    cmnDeSerializeRaw(inputStream, this->ProcessName);
    cmnDeSerializeRaw(inputStream, this->ClassName);
    cmnDeSerializeRaw(inputStream, this->ComponentName);
}


//-----------------------------------------------------------------------------
//  Connection Description
//
CMN_IMPLEMENT_SERVICES(mtsDescriptionConnection);

void mtsDescriptionConnection::ToStream(std::ostream & outputStream) const
{
    mtsGenericObject::ToStream(outputStream);
    outputStream << std::endl
                 << "Client process: " << this->Client.ProcessName
                 << ", component: " << this->Client.ComponentName
                 << ", interface: " << this->Client.InterfaceName << std::endl
                 << "Server process: " << this->Server.ProcessName
                 << ", component: " << this->Server.ComponentName
                 << ", interface: " << this->Server.InterfaceName << std::endl;
}

void mtsDescriptionConnection::SerializeRaw(std::ostream & outputStream) const
{
    mtsGenericObject::SerializeRaw(outputStream);
    cmnSerializeRaw(outputStream, this->Client.ProcessName);
    cmnSerializeRaw(outputStream, this->Client.ComponentName);
    cmnSerializeRaw(outputStream, this->Client.InterfaceName);
    cmnSerializeRaw(outputStream, this->Server.ProcessName);
    cmnSerializeRaw(outputStream, this->Server.ComponentName);
    cmnSerializeRaw(outputStream, this->Server.InterfaceName);
}

void mtsDescriptionConnection::DeSerializeRaw(std::istream & inputStream)
{
    mtsGenericObject::DeSerializeRaw(inputStream);
    cmnDeSerializeRaw(inputStream, this->Client.ProcessName);
    cmnDeSerializeRaw(inputStream, this->Client.ComponentName);
    cmnDeSerializeRaw(inputStream, this->Client.InterfaceName);
    cmnDeSerializeRaw(inputStream, this->Server.ProcessName);
    cmnDeSerializeRaw(inputStream, this->Server.ComponentName);
    cmnDeSerializeRaw(inputStream, this->Server.InterfaceName);
}
