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

#ifndef _testFilter_h
#define _testFilter_h

#include <cisstStereoVision/svlFilterBase.h>
#include <cisstMultiTask/mtsStateTable.h>

// Always include last!
#include <cisstStereoVision/svlExport.h>


class svlFilterTest : public svlFilterBase
{
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    struct Config
    {
        double DblValue1;
        double DblValue2;
        int IntValue1;
        int IntValue2;
        bool BoolValue;

        friend std::ostream & operator << (std::ostream & stream, const Config & objref);
    };
    void SetConfig(const mtsGenericObjectProxy<Config>& data);

public:
    svlFilterTest();
    virtual ~svlFilterTest();

    void SetParam1(unsigned int value1, unsigned int value2);
    void SetParam2(double value1, double value2);
    void SetParam3(bool value);

protected:
    virtual int Initialize(svlSample* syncInput, svlSample* &syncOutput);
    virtual int Process(svlProcInfo* procInfo, svlSample* syncInput, svlSample* &syncOutput);

protected:
    mtsStateTable StateTable;
    mtsGenericObjectProxy<Config> Settings;
};

typedef mtsGenericObjectProxy<svlFilterTest::Config> svlFilterTestConfigProxy;
CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlFilterTestConfigProxy);
CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlFilterTest)

#endif // _testFilter_h

