/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: StereoPlayerTest.cpp 1352 2010-03-27 17:38:21Z dmirota1 $
  
  Author(s):  Min Yang Jung
  Created on: 2010

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/


#ifdef _WIN32
#include <conio.h>
#endif // _WIN32

#include <iostream>
#include <string>
#include <cisstCommon.h>
#include <cisstOSAbstraction.h>
#include <cisstStereoVision.h>


using namespace std;

//
// FPS Filter
//
class CFPSFilter : public svlFilterBase
{
public:
    CFPSFilter() :
        svlFilterBase(),
        Manager(0),
        ShowFramerate(true)

    {
        AddSupportedType(svlTypeImageRGB, svlTypeImageRGB);
    }

protected:
    int Initialize(svlSample* inputdata)
    {
        OutputData = inputdata;
        return SVL_OK;
    }

    int ProcessFrame(svlProcInfo* procInfo, svlSample* CMN_UNUSED(inputdata) = 0)
    {
        if (!ShowFramerate) return SVL_OK;

        _OnSingleThread(procInfo) {
            unsigned int framecount = GetFrameCounter();
            if ((framecount % 30) == 0) {
#ifdef _WIN32
                DWORD now;
                now = ::GetTickCount();

                if (framecount > 0) {
                    DWORD msec = now - StartMSec;
                    std::cerr << "\rFrame #: " << framecount << "; "
                              << std::setprecision(1) << std::fixed << (double)30000 / msec << " fps";
                    if (Manager) {
                        std::cerr << " (Buffer: " << Manager->Branch("Recorder").GetBufferUsageRatio() * 100.0
                                  << "%, Dropped: " << Manager->Branch("Recorder").GetDroppedSampleCount() << ")";
                    }
                    std::cerr << "     \r";
                }

                StartMSec = now;
#endif // _WIN32

#ifdef __GNUC__
                timeval now;
                gettimeofday(&now, 0);

                if (framecount > 0) {
                    int sec = now.tv_sec - StartSec;
                    int usec = now.tv_usec - StartUSec;
                    usec += 1000000 * sec;
                    std::cerr << "\rFrame #: " << framecount << "; "
                              << std::setprecision(1) << std::fixed << (double)30000000 / usec << " fps";
                    if (Manager) {
                        std::cerr << " (Buffer: " << Manager->Branch("Recorder").GetBufferUsageRatio() * 100.0
                                  << "%, Dropped: " << Manager->Branch("Recorder").GetDroppedSampleCount() << ")";
                    }
                    std::cerr << "     \r";
                }

                StartSec = now.tv_sec;
                StartUSec = now.tv_usec;
#endif // __GNUC__
            }
        }

        return SVL_OK;
    }

public:
    svlStreamManager* Manager;
    bool ShowFramerate;
    bool Recording;

private:
#ifdef _WIN32
    DWORD StartMSec;
#endif // _WIN32
#ifdef __GNUC__
    unsigned int StartSec;
    unsigned int StartUSec;
#endif // __GNUC__
};

////////////////////
//  Video Player  //
////////////////////

int MonoVideoPlayer(const std::string pathname)
{
    svlInitialize();

    // instantiating SVL stream and filters
    svlStreamManager viewer_stream(4);
    svlFilterSourceVideoFile viewer_source(1);
    svlFilterImageWindow viewer_window;

    // FPS filter setup
    CFPSFilter viewer_fps;
    viewer_fps.ShowFramerate = true;
    //viewer_fps.Manager = &viewer_stream;

    // setup source
    if (pathname.empty()) {
        viewer_source.DialogFilePath();
    }
    else {
        if (viewer_source.SetFilePath(pathname) != SVL_OK) {
            cerr << endl << "Wrong file name... " << endl;
            exit(1);
        }
    }

    // setup image window
    viewer_window.SetTitleText("Video Player");
    viewer_window.EnableTimestampInTitle();

    // chain filters to pipeline
    if (viewer_stream.Trunk().Append(&viewer_source) != SVL_OK) exit(1);
    if (viewer_stream.Trunk().Append(&viewer_window) != SVL_OK) exit(1);
    if (viewer_stream.Trunk().Append(&viewer_fps) != SVL_OK) exit(1);

    cerr << endl << "Starting stream... ";

    // initialize and start stream
    if (viewer_stream.Start() != SVL_OK) exit(1);

    cerr << "Done" << endl;

    // wait for keyboard input in command window
    int ch;

    do {
        cerr << endl << "Keyboard commands:" << endl << endl;
        cerr << "  In command window:" << endl;
        cerr << "    'q'   - Quit" << endl << endl;

        ch = cmnGetChar();
        osaSleep(1.0 * cmn_ms);
    } while (ch != 'q');

    cerr << endl;

    // stop stream
    viewer_stream.Stop();

    // destroy pipeline
    viewer_stream.RemoveAll();

    return 0;
}

int StereoVideoPlayer(int argc, char** argv)
{
    std::string sourceleft, sourceright;

    svlInitialize();

    // instantiating SVL stream and filters
    svlStreamManager viewer_stream(4);
    svlFilterSourceVideoFile viewer_source(2);
    svlFilterStereoImageJoiner viewer_joiner;
    svlFilterImageWindow viewer_window;

    // FPS filter setup
    CFPSFilter viewer_fps;
    viewer_fps.ShowFramerate = true;
    //viewer_fps.Manager = &viewer_stream;

    // Input files
    if (argc == 3) {
        if (viewer_source.SetFilePath(argv[1], SVL_LEFT) != SVL_OK) {
            cerr << endl << "Invalid file name: " << argv[1] << endl;
            exit(1);
        }
        viewer_source.GetFilePath(sourceleft);

        if (viewer_source.SetFilePath(argv[2], SVL_RIGHT) != SVL_OK) {
            cerr << endl << "Invalid file name: " << argv[1] << endl;
            exit(1);
        }
        viewer_source.GetFilePath(sourceright);
    } else {
        if (viewer_source.DialogFilePath(SVL_LEFT) != SVL_OK) {
            cerr << " -!- No source file has been selected." << endl;
            exit(1);
        }
        viewer_source.GetFilePath(sourceleft, SVL_LEFT);

        if (viewer_source.DialogFilePath(SVL_RIGHT) != SVL_OK) {
            cerr << " -!- No source file has been selected." << endl;
            return -1;
        }
        viewer_source.GetFilePath(sourceright, SVL_RIGHT);
    }

    cout << "Left source : " << sourceleft << endl;
    cout << "Right source: " << sourceright << endl;

    // Set property of source and writer
    viewer_source.SetTargetFrequency(30.0); // as fast as possible
    viewer_source.SetLoop(true);

    // setup image window
    viewer_window.SetTitleText("Video Player");
    viewer_window.EnableTimestampInTitle();

    // chain filters to pipeline
    if (viewer_stream.Trunk().Append(&viewer_source) != SVL_OK) exit(1);
    if (viewer_stream.Trunk().Append(&viewer_joiner) != SVL_OK) exit(1);
    if (viewer_stream.Trunk().Append(&viewer_window) != SVL_OK) exit(1);
    if (viewer_stream.Trunk().Append(&viewer_fps) != SVL_OK) exit(1);

    cerr << endl << "Starting stream... ";

    // initialize and start stream
    if (viewer_stream.Start() != SVL_OK) exit(1);

    cerr << "Done" << endl;

    // wait for keyboard input in command window
    int ch;

    do {
        cerr << endl << "Keyboard commands:" << endl << endl;
        cerr << "  In command window:" << endl;
        cerr << "    'q'   - Quit" << endl << endl;

        ch = cmnGetChar();
        osaSleep(1.0 * cmn_ms);
    } while (ch != 'q');

    cerr << endl;

    // stop stream
    viewer_stream.Stop();

    // destroy pipeline
    viewer_stream.RemoveAll();

    return 0;
}

int StereoVideoPlayerUDP(int argc, char** argv, unsigned int numThread, bool enableNetwork)
{
    cout << "Stereo video player test: " << endl;
    cout << "Number of thread(s) : " << numThread << endl;
    cout << "Serialization       : " << "enabled" << endl;
    cout << "Networking          : " << (enableNetwork ? "enabled" : "disabled") << endl;

    std::string sourceleft, sourceright, destination;

    svlInitialize();

    svlStreamManager converter_stream(numThread);
    svlFilterSourceVideoFile converter_source(2); // # of source channels
    svlFilterStereoImageJoiner converter_joiner;
    svlFilterVideoFileWriter converter_writer;

    // FPS filter setup
    CFPSFilter viewer_fps;
    viewer_fps.ShowFramerate = true;

    // Input files
    destination = (enableNetwork ? "d:\\1.udp" : "d:\\0.udp");
    if (argc == 4) {
        if (converter_source.SetFilePath(argv[2], SVL_LEFT) != SVL_OK) {
            cerr << endl << "Invalid file name: " << argv[2] << endl;
            exit(1);
        }
        converter_source.GetFilePath(sourceleft);

        if (converter_source.SetFilePath(argv[3], SVL_RIGHT) != SVL_OK) {
            cerr << endl << "Invalid file name: " << argv[3] << endl;
            exit(1);
        }
        converter_source.GetFilePath(sourceright);

        if (converter_writer.SetFilePath(destination) != SVL_OK) {
            cerr << endl << "Invalid file name: " << destination << endl;
            exit(1);
        }
        converter_writer.GetFilePath(destination);
    } else {
        if (converter_source.DialogFilePath(SVL_LEFT) != SVL_OK) {
            cerr << " -!- No source file has been selected." << endl;
            exit(1);
        }
        converter_source.GetFilePath(sourceleft, SVL_LEFT);

        if (converter_source.DialogFilePath(SVL_RIGHT) != SVL_OK) {
            cerr << " -!- No source file has been selected." << endl;
            return -1;
        }
        converter_source.GetFilePath(sourceright, SVL_RIGHT);

        if (converter_writer.DialogFilePath() != SVL_OK) {
            cerr << endl << "Invalid file name: " << destination << endl;
            exit(1);
        }
        converter_writer.GetFilePath(destination);
    }

    cout << "Left source : " << sourceleft << endl;
    cout << "Right source: " << sourceright << endl;
    cout << "Target file : " << destination << endl;

    // Set property of source and writer
    converter_source.SetTargetFrequency(30.0); // as fast as possible
    converter_source.SetLoop(true);
    
    std::string encoderleft, encoderright;
    converter_writer.GetCodecName(encoderleft, SVL_LEFT);
    converter_writer.GetCodecName(encoderright, SVL_RIGHT);

    // chain filters to pipeline
    converter_stream.Trunk().Append(&converter_source);
    converter_stream.Trunk().Append(&converter_joiner);
    converter_stream.Trunk().Append(&converter_writer);
    converter_stream.Trunk().Append(&viewer_fps);

    cerr << "Converting: '" << sourceleft << "' and '" << sourceright << "' to '" << destination <<"' using codec: '" << encoderleft << "'" << endl;

    // initialize and start stream
    if (converter_stream.Start() != SVL_OK) goto labError;

    do {
        //cerr << " > Frames processed: " << converter_source.GetFrameCounter() << "     \r";
    } while (converter_stream.IsRunning() && converter_stream.WaitForStop(0.5) == SVL_WAIT_TIMEOUT);
    //cerr << " > Frames processed: " << converter_source.GetFrameCounter() << "           " << endl;

    if (converter_stream.GetStreamStatus() < 0) {
        // Some error
        cerr << " -!- Error occured during conversion." << endl;
    }
    else {
        // Success
        cerr << " > Conversion done." << endl;
    }

    // destroy pipeline
    converter_stream.RemoveAll();

labError:
    return 0;
}

//////////////////////////////////
//             main             //
//////////////////////////////////

int main(int argc, char** argv)
{
    cout << "StereoPlayerTest operation_mode [video_src_1] [video_src_2]" << endl;
    cout << "  operation_mode:" << endl;
    cout << "    1    Monovideo   without serialization" << endl;
    cout << "    2    Stereovideo without serialization" << endl;
    cout << "    3    Stereovideo with serialization, single thread, networking disabled" << endl;
    cout << "    4    Stereovideo with serialization, 4 threads,     networking disabled" << endl;
    cout << "    5    Stereovideo with serialization, 16 threads,    networking disabled" << endl;
    cout << "    6    Stereovideo with serialization, single thread, networking enabled" << endl;
    cout << "    7    Stereovideo with serialization, 4 threads,     networking enabled" << endl;
    cout << "    8    Stereovideo with serialization, 16 threads,    networking enabled" << endl;

    int mode;
    if (argc == 1) {
        mode = 1;
    } else {
        mode = atoi(argv[1]);
    }
    cout << endl << "Operation mode: " << mode << endl;

    string src, dest;
    switch (mode) {
            // Mono video without serialization
        case 1:
            if (argc == 3) {
                MonoVideoPlayer(argv[2]);
            } else {
                MonoVideoPlayer("");
            }
            break;

            // Stereo video without serialization
        case 2:
            StereoVideoPlayer(argc, argv);
            break;

            // Stereovideo with serialization, single thread, networking disabled
        case 3:
            StereoVideoPlayerUDP(argc, argv, 1, false);
            break;

            // Stereovideo with serialization, 4 threads,     networking disabled
        case 4:
            StereoVideoPlayerUDP(argc, argv, 4, false);
            break;

            // Stereovideo with serialization, 16 threads,    networking disabled
        case 5:
            StereoVideoPlayerUDP(argc, argv, 16, false);
            break;

            // Stereovideo with serialization, single thread, networking enabled
        case 6:
            StereoVideoPlayerUDP(argc, argv, 1, true);
            break;

            // Stereovideo with serialization, 4 threads,     networking enabled
        case 7:
            StereoVideoPlayerUDP(argc, argv, 4, true);
            break;

            // Stereovideo with serialization, 16 threads,    networking enabled
        case 8:
            StereoVideoPlayerUDP(argc, argv, 16, true);
            break;

        default:
            ;
    }

    return 1;
}

