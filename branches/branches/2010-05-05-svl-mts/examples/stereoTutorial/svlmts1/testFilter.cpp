/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
 $Id: $
 
 Author(s):  Balazs Vagvolgyi
 Created on: 2010
 
 (C) Copyright 2006-2010 Johns Hopkins University (JHU), All Rights
 Reserved.
 
 --- begin cisst license - do not edit ---
 
 This software is provided "as is" under an open source license, with
 no warranty.  The complete license can be found in license.txt and
 http://www.cisst.org/cisst/license.txt.
 
 --- end cisst license ---
 
 */

#include "testFilter.h"

#include <cisstMultiTask/mtsInterfaceProvided.h>
#include <cisstStereoVision/svlDraw.h>


/***************************/
/*** svlFilterTest class ***/
/***************************/

CMN_IMPLEMENT_SERVICES(svlFilterTest)
CMN_IMPLEMENT_SERVICES_TEMPLATED(svlFilterTestConfigProxy)

svlFilterTest::svlFilterTest() :
    svlFilterBase(),
    StateTable(3, "StateTable")
{
    StateTable.AddData(Settings, "Settings");
    mtsInterfaceProvided* provided= AddInterfaceProvided("Settings");
    if (provided) {
        provided->AddCommandReadState(StateTable, Settings, "Get");
        provided->AddCommandWrite(&svlFilterTest::SetConfig, this, "Set");
    }

    AddInput("input", true);
    AddInputType("input", svlTypeImageRGB);

    AddOutput("output", true);
    SetAutomaticOutputType(true);
}

svlFilterTest::~svlFilterTest()
{
}

void svlFilterTest::SetConfig(const mtsGenericObjectProxy<Config>& data)
{
    SetParam1(data.Data.IntValue1, data.Data.IntValue2);
    SetParam2(data.Data.DblValue1, data.Data.DblValue2);
    SetParam3(data.Data.BoolValue);
}

void svlFilterTest::SetParam1(unsigned int value1, unsigned int value2)
{
    Settings.Data.IntValue1 = value1;
    Settings.Data.IntValue2 = value2;
}

void svlFilterTest::SetParam2(double value1, double value2)
{
    Settings.Data.DblValue1 = value1;
    Settings.Data.DblValue2 = value2;
}

void svlFilterTest::SetParam3(bool value)
{
    Settings.Data.BoolValue = value;
}

int svlFilterTest::Initialize(svlSample* syncInput, svlSample* &syncOutput)
{
    syncOutput = syncInput;
    return SVL_OK;
}

int svlFilterTest::Process(svlProcInfo* procInfo, svlSample* syncInput, svlSample* &syncOutput)
{
    syncOutput = syncInput;

    _OnSingleThread(procInfo)
    {
        std::stringstream strstr;
        strstr << Settings.Data;
        svlDraw::Text(dynamic_cast<svlSampleImageRGB*>(syncInput),
                      0,
                      svlPoint2D(4, 20),
                      strstr.str(),
                      14,
                      svlRGB(255, 255, 255));

        Settings.Data.IntValue1 = FrameCounter;
        StateTable.Advance();
    }

    return SVL_OK;
}

std::ostream & operator << (std::ostream & stream, const svlFilterTest::Config & objref)
{
    stream << objref.DblValue1 << ", "
           << objref.DblValue2 << ", "
           << objref.IntValue1 << ", "
           << objref.IntValue2 << ", "
           << objref.BoolValue;
    return stream;
}

