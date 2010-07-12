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


#ifndef _svlFilterIO_h
#define _svlFilterIO_h

#include <cisstVector/vctDynamicVectorTypes.h>
#include <cisstStereoVision/svlTypes.h>
#include <cisstStereoVision/svlBufferSample.h>

// Always include last!
#include <cisstStereoVision/svlExport.h>


// Forward declarations
class svlFilterBase;
class svlFilterOutput;
class svlStreamManager;
class svlStreamProc;
class svlStreamBranchSource;


class CISST_EXPORT svlFilterInput
{
friend class svlStreamManager;
friend class svlStreamProc;
friend class svlFilterBase;
friend class svlFilterOutput;

public:
    svlFilterInput(svlFilterBase* owner, bool trunk, const std::string &name);
    ~svlFilterInput();

    bool IsTrunk() const;
    svlStreamType GetType() const;
    const std::string& GetName() const;
    svlFilterBase* GetFilter();
    svlFilterBase* GetConnectedFilter();
    int AddType(svlStreamType type);
    bool IsTypeSupported(svlStreamType type);
    bool IsConnected() const;
    svlFilterOutput* GetConnection();

    int Disconnect();

    int PushSample(const svlSample* sample);
    svlSample* PullSample(bool waitfornew, double timeout = 5.0);

    double GetTimestamp();

private:
    svlFilterBase* Filter;
    const bool Trunk;
    const std::string Name;
    bool Connected;
    svlFilterOutput* Connection;
    svlFilterBase* ConnectedFilter;
    vctDynamicVector<svlStreamType> SupportedTypes;
    svlStreamType Type;

    svlBufferSample* Buffer;

    double Timestamp;
};


class CISST_EXPORT svlFilterOutput
{
friend class svlStreamManager;
friend class svlStreamProc;
friend class svlFilterBase;

public:
    svlFilterOutput(svlFilterBase* owner, bool trunk, const std::string &name);
    ~svlFilterOutput();

    bool IsTrunk() const;
    svlStreamType GetType() const;
    const std::string& GetName() const;
    svlFilterBase* GetFilter();
    svlFilterBase* GetConnectedFilter();
    int SetType(svlStreamType type);
    bool IsConnected() const;
    svlFilterInput* GetConnection();
    int GetDroppedSampleCount();
    int GetBufferUsage();
    double GetBufferUsageRatio();

    int SetThreadCount(unsigned int threadcount);
    int SetBufferSize(unsigned int buffersize);
    int SetBlock(bool block);
    int Connect(svlFilterInput *input);
    int Disconnect();

    void SetupSample(svlSample* sample);
    void PushSample(const svlSample* sample);

    double GetTimestamp();

private:
    svlFilterBase* Filter;
    const bool Trunk;
    const std::string Name;
    bool Connected;
    svlFilterInput* Connection;
    svlFilterBase* ConnectedFilter;
    svlStreamType Type;

    unsigned int ThreadCount;
    unsigned int BufferSize;
    bool Blocked;
    svlStreamManager* Stream;
    svlStreamBranchSource* BranchSource;

    double Timestamp;
};

#endif // _svlFilterIO_h

