/*
  $Id: mtsCollectorSample.cpp 2009-03-02 mjung5

  Author(s):  Min Yang Jung
  Created on: 2009-03-20

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#include <cisstMultiTask/mtsCollectorSample.h>

CMN_IMPLEMENT_SERVICES(mtsCollectorSample)

//-------------------------------------------------------
//	Constructor, Destructor, and Initializer
//-------------------------------------------------------
mtsCollectorSample::mtsCollectorSample(const std::string & collectorName, double period)
    : mtsCollectorBase(collectorName, period), 
      NextReadIndex(0)
{
}

mtsCollectorSample::~mtsCollectorSample()
{
}

//-------------------------------------------------------
//	Signal Management
//-------------------------------------------------------
bool mtsCollectorSample::AddSignal(const std::string & taskName, 
                                 const std::string & signalName, 
                                 const std::string & format)
{	
    return true;
}

//-------------------------------------------------------
//	Collecting Data
//-------------------------------------------------------
void mtsCollectorSample::Collect(void)
{
}
