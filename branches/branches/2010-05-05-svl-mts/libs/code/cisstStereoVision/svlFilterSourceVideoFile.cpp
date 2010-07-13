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

#include <cisstStereoVision/svlFilterSourceVideoFile.h>
#include <cisstMultiTask/mtsInterfaceProvided.h>
#include <cisstOSAbstraction/osaSleep.h>


#ifdef _MSC_VER
    // Quick fix for Visual Studio Intellisense:
    // The Intellisense parser can't handle the CMN_UNUSED macro
    // correctly if defined in cmnPortability.h, thus
    // we should redefine it here for it.
    // Removing this part of the code will not effect compilation
    // in any way, on any platforms.
    #undef CMN_UNUSED
    #define CMN_UNUSED(argument) argument
#endif


/***************************************/
/*** svlFilterSourceVideoFile class ****/
/***************************************/

CMN_IMPLEMENT_SERVICES(svlFilterSourceVideoFile)
CMN_IMPLEMENT_SERVICES_TEMPLATED(svlFilterSourceVideoFileConfigProxy)

svlFilterSourceVideoFile::svlFilterSourceVideoFile() :
    svlFilterSourceBase(false),  // manual timestamp management
    OutputImage(0),
    FirstTimestamp(-1.0),
    ResetTimer(false),
    StateTable(3, "StateTable")
{
    // Add provided interface for settings management
    StateTable.AddData(Settings, "Settings");
    mtsInterfaceProvided* provided= AddInterfaceProvided("Settings");
    if (provided) {
        provided->AddCommandReadState(StateTable, Settings, "Get");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSet,          this, "Set");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetChannels,  this, "SetChannels");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetPathL,     this, "SetFilename");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetPathL,     this, "SetLeftFilename");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetPathR,     this, "SetRightFilename");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetPosL,      this, "SetPosition");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetPosL,      this, "SetLeftPosition");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetPosR,      this, "SetRightPosition");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetRangeL,    this, "SetRange");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetRangeL,    this, "SetLeftRange");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetRangeR,    this, "SetRightRange");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetFramerate, this, "SetFramerate");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetLoop,      this, "SetLoop");
    }

    AddOutput("output", true);
    SetAutomaticOutputType(false);
}

svlFilterSourceVideoFile::svlFilterSourceVideoFile(unsigned int channelcount) :
    svlFilterSourceBase(false),  // manual timestamp management
    OutputImage(0),
    FirstTimestamp(-1.0),
    ResetTimer(false),
    StateTable(3, "StateTable")
{
    // Add provided interface for settings management
    StateTable.AddData(Settings, "Settings");
    mtsInterfaceProvided* provided= AddInterfaceProvided("Settings");
    if (provided) {
        provided->AddCommandReadState(StateTable, Settings, "Get");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSet,          this, "Set");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetChannels,  this, "SetChannels");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetPathL,     this, "SetFilename");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetPathL,     this, "SetLeftFilename");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetPathR,     this, "SetRightFilename");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetPosL,      this, "SetPosition");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetPosL,      this, "SetLeftPosition");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetPosR,      this, "SetRightPosition");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetRangeL,    this, "SetRange");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetRangeL,    this, "SetLeftRange");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetRangeR,    this, "SetRightRange");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetFramerate, this, "SetFramerate");
        provided->AddCommandWrite(&svlFilterSourceVideoFile::confSetLoop,      this, "SetLoop");
    }

    AddOutput("output", true);
    SetAutomaticOutputType(false);
    SetChannelCount(channelcount);
}

svlFilterSourceVideoFile::~svlFilterSourceVideoFile()
{
    Release();

    if (OutputImage) delete OutputImage;
}

svlFilterSourceVideoFile::Config::Config() :
    Channels(0),
    Framerate(-1.0),
    Loop(true)
{
}

svlFilterSourceVideoFile::Config::Config(const svlFilterSourceVideoFile::Config& config)
{
    SetChannels(config.Channels);
    FilePath  = config.FilePath;
    Length    = config.Length;
    Position  = config.Position;
    Range     = config.Range;
    Framerate = config.Framerate;
    Loop      = config.Loop;
}

void svlFilterSourceVideoFile::Config::SetChannels(const int channels)
{
    if (channels < 0) return;

    FilePath.SetSize(channels);
    Length.SetSize(channels);
    Position.SetSize(channels);
    Range.SetSize(channels);

    Channels = channels;
    Length.SetAll(-1);
    Position.SetAll(-1); 
    Range.SetAll(vctInt2(-1, -1));
    Framerate = -1.0;
    Loop = true;
}

std::ostream & operator << (std::ostream & stream, const svlFilterSourceVideoFile::Config & objref)
{
    for (int i = 0; i < objref.Channels; i ++) {
        if (i > 0) stream << ", ";
        else {
            stream << objref.Framerate << "Hz, ";
            if (objref.Loop) stream << "loop=ON, ";
            else stream << "loop=OFF, ";
        }
        stream << "("
               << objref.FilePath[i] << ", "
               << objref.Length[i]   << ", "
               << objref.Position[i] << ", "
               << objref.Range[i][0] << ", "
               << objref.Range[i][1] << ")";
    }
    return stream;
}

void svlFilterSourceVideoFile::confSet(const mtsGenericObjectProxy<Config>& data)
{
    if (data.Data.Channels < 0) return;

    SetChannelCount(static_cast<unsigned int>(data.Data.Channels));
    for (unsigned int i = 0; i < data.Data.Channels; i ++) {
        SetFilePath(data.Data.FilePath[i], i);
        SetPosition(data.Data.Position[i], i);
        SetRange(data.Data.Range[i], i);
    }
    SetTargetFrequency(data.Data.Framerate);
    SetLoop(data.Data.Loop);
}

void svlFilterSourceVideoFile::confSetChannels(const mtsInt& channels)
{
    SetChannelCount(static_cast<unsigned int>(channels));
}

void svlFilterSourceVideoFile::confSetPathL(const mtsStdString& filepath)
{
    SetFilePath(filepath, SVL_LEFT);
}

void svlFilterSourceVideoFile::confSetPathR(const mtsStdString& filepath)
{
    SetFilePath(filepath, SVL_RIGHT);
}

void svlFilterSourceVideoFile::confSetPosL(const mtsInt& position)
{
    SetPosition(position, SVL_LEFT);
}

void svlFilterSourceVideoFile::confSetPosR(const mtsInt& position)
{
    SetPosition(position, SVL_RIGHT);
}

void svlFilterSourceVideoFile::confSetRangeL(const mtsInt2& range)
{
    SetRange(range, SVL_LEFT);
}

void svlFilterSourceVideoFile::confSetRangeR(const mtsInt2& range)
{
    SetRange(range, SVL_RIGHT);
}

void svlFilterSourceVideoFile::confSetFramerate(const mtsDouble& framerate)
{
    SetTargetFrequency(framerate);
}

void svlFilterSourceVideoFile::confSetLoop(const mtsBool& loop)
{
    SetLoop(loop);
}

int svlFilterSourceVideoFile::SetChannelCount(unsigned int channelcount)
{
    if (OutputImage) return SVL_FAIL;

    if (channelcount == 1) {
        GetOutput()->SetType(svlTypeImageRGB);
        OutputImage = new svlSampleImageRGB;
    }
    else if (channelcount == 2) {
        GetOutput()->SetType(svlTypeImageRGBStereo);
        OutputImage = new svlSampleImageRGBStereo;
    }
    else return SVL_FAIL;

    Codec.SetSize(channelcount);
    Codec.SetAll(0);

    Settings.Data.SetChannels(channelcount);

    return SVL_OK;
}

int svlFilterSourceVideoFile::Initialize(svlSample* &syncOutput)
{
    if (OutputImage == 0) return SVL_FAIL;
    syncOutput = OutputImage;

    Release();

    unsigned int width, height;
    double framerate;
    int ret = SVL_OK;

    for (unsigned int i = 0; i < OutputImage->GetVideoChannels(); i ++) {

        // Get video codec for file extension
        Codec[i] = svlVideoIO::GetCodec(Settings.Data.FilePath[i]);
        // Open video file
        if (!Codec[i] || Codec[i]->Open(Settings.Data.FilePath[i], width, height, framerate) != SVL_OK) {
            ret = SVL_FAIL;
            break;
        }

        if (i == 0) {
            // The first video channel defines the video
            // framerate of all channels in the stream
            Settings.Data.Framerate = framerate;
        }

        Settings.Data.Length[i] = Codec[i]->GetEndPos() + 1;
        Settings.Data.Position[i] = Codec[i]->GetPos();

        // Create image sample of matching dimensions
        OutputImage->SetSize(i, width, height);
    }

    // Initialize timestamp for case of timestamp errors
    OutputImage->SetTimestamp(0.0);

    if (ret != SVL_OK) Release();
    return ret;
}

int svlFilterSourceVideoFile::OnStart(unsigned int CMN_UNUSED(procCount))
{
    if (TargetFrequency < 0.1) TargetFrequency = Settings.Data.Framerate;
    RestartTargetTimer();

    return SVL_OK;
}

int svlFilterSourceVideoFile::Process(svlProcInfo* procInfo, svlSample* &syncOutput)
{
    syncOutput = OutputImage;

    // Try to keep TargetFrequency
    _OnSingleThread(procInfo) WaitForTargetTimer();

    unsigned int idx, videochannels = OutputImage->GetVideoChannels();
    double timestamp, timespan;
    int pos, ret = SVL_OK;

    _ParallelLoop(procInfo, idx, videochannels)
    {
        if (Codec[idx]) {

            pos = Codec[idx]->GetPos();
            Settings.Data.Position[idx] = pos;

            if (Settings.Data.Range[idx][0] >= 0 &&
                Settings.Data.Range[idx][0] <= Settings.Data.Range[idx][1]) {
                // Check if position is outside of the playback segment
                if (pos < Settings.Data.Range[idx][0] ||
                    pos > Settings.Data.Range[idx][1]) {
                    Codec[idx]->SetPos(Settings.Data.Range[idx][0]);
                    ResetTimer = true;
                }
            }

            ret = Codec[idx]->Read(0, *OutputImage, idx, true);
            if (ret == SVL_VID_END_REACHED) {
                if (!LoopFlag) ret = SVL_STOP_REQUEST;
                else {
                    // Loop around
                    ret = Codec[idx]->Read(0, *OutputImage, idx, true);
                }
            }
            if (ret != SVL_OK) break;

            if (idx == 0) {
                if (!IsTargetTimerRunning()) {

                    // Synchronizing all channels to channel #0
                    timestamp = Codec[idx]->GetTimestamp();
                    if (timestamp > 0.0) {

                        // Try to keep orignal frame intervals
                        if (ResetTimer || Codec[idx]->GetPos() == 1) {
                            FirstTimestamp = timestamp;
                            Timer.Reset();
                            Timer.Start();
                        }
                        else {
                            timespan = (timestamp - FirstTimestamp) - Timer.GetElapsedTime();
                            if (timespan > 0.0) osaSleep(timespan);
                        }

                        // Set timestamp to the one stored in the video file
                        OutputImage->SetTimestamp(timestamp);

                        continue;
                    }
                }

                // Ask Stream Manager for current timestamp
                OutputImage->SetTimestamp(-1.0);
            }
        }
    }

    return ret;
}

int svlFilterSourceVideoFile::Release()
{
    for (unsigned int i = 0; i < Codec.size(); i ++) {
        svlVideoIO::ReleaseCodec(Codec[i]);
        Codec[i] = 0;
    }

    if (Timer.IsRunning()) Timer.Stop();

    Settings.Data.Framerate = -1.0;

    return SVL_OK;
}

int svlFilterSourceVideoFile::DialogFilePath(unsigned int videoch)
{
    if (OutputImage == 0) return SVL_FAIL;
    if (IsInitialized() == true)
        return SVL_ALREADY_INITIALIZED;

    if (videoch >= OutputImage->GetVideoChannels()) return SVL_WRONG_CHANNEL;

    std::ostringstream out;
    out << "Open video file [channel #" << videoch << "]";
    std::string title(out.str());

    return svlVideoIO::DialogFilePath(false, title, Settings.Data.FilePath[videoch]);
}

int svlFilterSourceVideoFile::SetFilePath(const std::string &filepath, unsigned int videoch)
{
    if (OutputImage == 0) return SVL_FAIL;
    if (IsInitialized() == true)
        return SVL_ALREADY_INITIALIZED;

    if (videoch >= OutputImage->GetVideoChannels()) return SVL_WRONG_CHANNEL;

    Settings.Data.FilePath[videoch] = filepath;

    return SVL_OK;
}

int svlFilterSourceVideoFile::GetFilePath(std::string &filepath, unsigned int videoch) const
{
    if (Settings.Data.FilePath.size() <= videoch) return SVL_FAIL;
    filepath = Settings.Data.FilePath[videoch];
    return SVL_OK;
}

int svlFilterSourceVideoFile::SetPosition(const int position, unsigned int videoch)
{
    if (Codec.size() <= videoch || !Codec[videoch]) return SVL_FAIL;
    Codec[videoch]->SetPos(position);
    ResetTimer = true;
    return SVL_OK;
}

int svlFilterSourceVideoFile::GetPosition(unsigned int videoch) const
{
    if (Codec.size() <= videoch || !Codec[videoch]) return SVL_FAIL;
    return Codec[videoch]->GetPos();
}

int svlFilterSourceVideoFile::SetRange(const vctInt2 range, unsigned int videoch)
{
    if (Codec.size() <= videoch) return SVL_FAIL;
    Settings.Data.Range[videoch] = range;
    return SVL_OK;
}

int svlFilterSourceVideoFile::GetRange(vctInt2& range, unsigned int videoch) const
{
    if (Codec.size() <= videoch) return SVL_FAIL;
    range = Settings.Data.Range[videoch];
    return SVL_OK;
}

int svlFilterSourceVideoFile::GetLength(unsigned int videoch) const
{
    if (Codec.size() <= videoch || !Codec[videoch]) return SVL_FAIL;
    return (Codec[videoch]->GetEndPos() + 1);
}

