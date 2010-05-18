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


#ifndef _svlFilterBase_h
#define _svlFilterBase_h

#include <map>

#include <cisstStereoVision/svlTypes.h>
#include <cisstStereoVision/svlFilterIO.h>
#include <cisstStereoVision/svlSyncPoint.h>

// Always include last!
#include <cisstStereoVision/svlExport.h>


// Forward declarations
class svlFilterSourceBase;
class svlStreamManager;
class svlStreamProc;


class CISST_EXPORT svlFilterBase
{
friend class svlFilterInput;
friend class svlFilterOutput;
friend class svlStreamManager;
friend class svlStreamProc;

    typedef std::map<std::string, svlFilterInput*> _Inputs;
    typedef std::map<std::string, svlFilterOutput*> _Outputs;

public:
    svlFilterBase();
    virtual ~svlFilterBase();

    bool IsInitialized();
    bool IsRunning();
    unsigned int GetFrameCounter();

    svlFilterInput* GetInput();
    svlFilterOutput* GetOutput();
    svlFilterInput* GetInput(const std::string &inputname);
    svlFilterOutput* GetOutput(const std::string &outputname);

protected:
    unsigned int FrameCounter;

    svlFilterInput* AddInput(const std::string &inputname, bool trunk = true);
    svlFilterOutput* AddOutput(const std::string &outputname, bool trunk = true);
    int AddInputType(const std::string &inputname, svlStreamType type);
    int SetOutputType(const std::string &outputname, svlStreamType type);
    void SetAutomaticOutputType(bool autotype);

    virtual int UpdateTypes(svlFilterInput &input, svlStreamType type);
    virtual int Initialize(svlSample* syncInput, svlSample* &syncOutput) = 0;
    virtual int OnStart(unsigned int procCount);
    virtual int Process(svlProcInfo* procInfo, svlSample* syncInput, svlSample* &syncOutput) = 0;
    virtual void OnStop();
    virtual int Release();

    int IsDataValid(svlStreamType type, svlSample* data);
    bool IsNewSample(svlSample* sample);

private:
    bool Initialized;
    bool Running;
    double PrevInputTimestamp;
    bool AutoType;
    _Inputs Inputs;
    _Outputs Outputs;
};

#endif // _svlFilterBase_h

