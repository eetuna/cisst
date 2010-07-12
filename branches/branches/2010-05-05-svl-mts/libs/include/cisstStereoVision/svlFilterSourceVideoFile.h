/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$
  
  Author(s):  Balazs Vagvolgyi
  Created on: 2006 

  (C) Copyright 2006-2007 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#ifndef _svlFilterSourceVideoFile_h
#define _svlFilterSourceVideoFile_h

#include <cisstStereoVision/svlFilterSourceBase.h>
#include <cisstStereoVision/svlVideoIO.h>

#include <cisstMultiTask/mtsFixedSizeVectorTypes.h>
#include <cisstMultiTask/mtsStateTable.h>

// Always include last!
#include <cisstStereoVision/svlExport.h>


class CISST_EXPORT svlFilterSourceVideoFile : public svlFilterSourceBase
{
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    class Config
    {
    public:
        Config();
        Config(const Config& config);

        int                           Channels;
        vctDynamicVector<std::string> FilePath;
        vctDynamicVector<int>         Length;
        vctDynamicVector<int>         Position;
        vctDynamicVector<vctInt2>     Range;
        double                        Framerate;
        bool                          Loop;

        void SetChannels(const int channels);
        friend std::ostream & operator << (std::ostream & stream, const Config & objref);
    };
    void confSet(const mtsGenericObjectProxy<Config>& data);
    void confSetChannels(const mtsInt& channels);
    void confSetPathL(const mtsStdString& filepath);
    void confSetPathR(const mtsStdString& filepath);
    void confSetPosL(const mtsInt& position);
    void confSetPosR(const mtsInt& position);
    void confSetRangeL(const mtsInt2& position);
    void confSetRangeR(const mtsInt2& position);
    void confSetFramerate(const mtsDouble& framerate);
    void confSetLoop(const mtsBool& loop);

public:
    svlFilterSourceVideoFile();
    svlFilterSourceVideoFile(unsigned int channelcount);
    ~svlFilterSourceVideoFile();

    int SetChannelCount(unsigned int channelcount);
    int DialogFilePath(unsigned int videoch = SVL_LEFT);

    int SetFilePath(const std::string &filepath, unsigned int videoch = SVL_LEFT);
    int GetFilePath(std::string &filepath, unsigned int videoch = SVL_LEFT) const;
    int SetPosition(const int position, unsigned int videoch = SVL_LEFT);
    int GetPosition(unsigned int videoch = SVL_LEFT) const;
    int SetRange(const vctInt2 range, unsigned int videoch = SVL_LEFT);
    int GetRange(vctInt2& range, unsigned int videoch = SVL_LEFT) const;
    int GetLength(unsigned int videoch = SVL_LEFT) const;

protected:
    virtual int Initialize(svlSample* &syncOutput);
    virtual int OnStart(unsigned int procCount);
    virtual int Process(svlProcInfo* procInfo, svlSample* &syncOutput);
    virtual int Release();

private:
    svlSampleImage* OutputImage;
    vctDynamicVector<svlVideoCodecBase*> Codec;
    double FirstTimestamp;
    bool ResetTimer;
    osaStopwatch Timer;

protected:
    mtsStateTable StateTable;
    mtsGenericObjectProxy<Config> Settings;
};

typedef mtsGenericObjectProxy<svlFilterSourceVideoFile::Config> svlFilterSourceVideoFileConfigProxy;
CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlFilterSourceVideoFileConfigProxy);
CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlFilterSourceVideoFile)

#endif // _svlFilterSourceVideoFile_h

