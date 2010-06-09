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


#ifndef _svlFilterSourceBase_h
#define _svlFilterSourceBase_h

#include <cisstOSAbstraction/osaStopwatch.h>
#include <cisstStereoVision/svlFilterBase.h>

// Always include last!
#include <cisstStereoVision/svlExport.h>


// Forward declarations
class svlStreamManager;
class svlStreamProc;


class CISST_EXPORT svlFilterSourceBase : public svlFilterBase
{
friend class svlFilterOutput;
friend class svlStreamManager;
friend class svlStreamProc;

public:
    svlFilterSourceBase();
    svlFilterSourceBase(bool autotimestamps);
    virtual ~svlFilterSourceBase();

    virtual double GetTargetFrequency();
    virtual int SetTargetFrequency(double hertz);
    virtual void SetLoop(bool loop = true);
    virtual bool GetLoop();
/*
    virtual int SetFramePos(int position);
    virtual int GetFramePos();
    virtual int SetFrameRange(int from, int to);
    virtual int GetFrameRange(int& from, int& to);

    virtual int SetTimePos(double position);
    virtual double GetTimePos();
    virtual int SetTimeRange(double from, double to);
    virtual int GetTimeRange(double& from, double& to);
*/
protected:
    virtual int Initialize(svlSample* &syncOutput);
    virtual int OnStart(unsigned int procCount);
    virtual int Process(svlProcInfo* procInfo, svlSample* &syncOutput) = 0;
    virtual void OnStop();
    virtual int Release();

    int RestartTargetTimer();
    int StopTargetTimer();
    int WaitForTargetTimer();
    bool IsTargetTimerRunning();

    double TargetFrequency;
    bool LoopFlag;

private:
    // Dispatched to source-specific methods declared above
    int Initialize(svlSample* syncInput, svlSample* &syncOutput);
    int Process(svlProcInfo* procInfo, svlSample* syncInput, svlSample* &syncOutput);

    // Hide input setup methods from derived classes
    int AddInput(const std::string &inputname, bool trunk = true);
    int AddInputType(const std::string &inputname, svlStreamType type);
    int UpdateTypes(svlFilterInput &input, svlStreamType type);

    bool AutoTimestamp;
    osaStopwatch TargetTimer;
    double TargetStartTime;
    double TargetFrameTime;
};


class CISST_EXPORT svlFilterSourceImageBase : public svlFilterSourceBase
{
public:
    svlFilterSourceImageBase();
    svlFilterSourceImageBase(bool autotimestamps);

    virtual int GetWidth(unsigned int videoch = SVL_LEFT) = 0;
    virtual int GetHeight(unsigned int videoch = SVL_LEFT) = 0;
};

#endif // _svlFilterSourceBase_h

