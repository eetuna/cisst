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

// Choose input source: file or live stream(camera)
#define STREAM_FROM_FILE
//#define STREAM_FROM_CAMERA

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

int StereoVideoPlayerUDP(int argc, char** argv, unsigned int numThread, bool enableNetwork, bool useLiveStream)
{
    cout << endl;
    cout << "Number of thread(s) : " << numThread << endl;
    cout << "Networking          : " << (enableNetwork ? "enabled" : "disabled") << endl;
    cout << "Stream Source       : " << (useLiveStream ? "camera" : "file") << endl;

    svlInitialize();

    svlStreamManager converter_stream(numThread);

    // Read stream from files
    svlFilterSourceVideoFile converter_file_source(2); // # of source channels
    std::string sourceleft, sourceright, destination;
    
    // Read stream from cameras
    svlFilterSourceVideoCapture converter_camera_source(2);

    svlFilterStereoImageJoiner converter_joiner;
    svlFilterVideoFileWriter converter_writer;

    // FPS filter setup
    CFPSFilter viewer_fps;
    viewer_fps.ShowFramerate = true;

    // Input files
    if (!useLiveStream) {
        destination = (enableNetwork ? "1.udp" : "0.udp");
        if (argc == 6) {
            if (converter_file_source.SetFilePath(argv[4], SVL_LEFT) != SVL_OK) {
                cerr << endl << "Invalid file name: " << argv[2] << endl;
                exit(1);
            }
            converter_file_source.GetFilePath(sourceleft);

            if (converter_file_source.SetFilePath(argv[5], SVL_RIGHT) != SVL_OK) {
                cerr << endl << "Invalid file name: " << argv[3] << endl;
                exit(1);
            }
            converter_file_source.GetFilePath(sourceright);

            if (converter_writer.SetFilePath(destination) != SVL_OK) {
                cerr << endl << "Invalid file name: " << destination << endl;
                exit(1);
            }
            converter_writer.GetFilePath(destination);
        } else {
            if (converter_file_source.DialogFilePath(SVL_LEFT) != SVL_OK) {
                cerr << " -!- No source file has been selected." << endl;
                exit(1);
            }
            converter_file_source.GetFilePath(sourceleft, SVL_LEFT);

            if (converter_file_source.DialogFilePath(SVL_RIGHT) != SVL_OK) {
                cerr << " -!- No source file has been selected." << endl;
                return -1;
            }
            converter_file_source.GetFilePath(sourceright, SVL_RIGHT);

            if (converter_writer.DialogFilePath() != SVL_OK) {
                cerr << endl << "Invalid file name: " << destination << endl;
                exit(1);
            }
            converter_writer.GetFilePath(destination);
        }

        converter_file_source.SetTargetFrequency(30.0); // as fast as possible
        converter_file_source.SetLoop(true);

        cout << "Left source : " << sourceleft << endl;
        cout << "Right source: " << sourceright << endl;
        cout << "Target file : " << destination << endl;
    } 
    else {
        if (converter_writer.SetFilePath(enableNetwork ? "1.udp" : "0.udp") != SVL_OK) {
            cout << "error: set file path " << endl;
            exit(1);
        }
        if (converter_camera_source.LoadSettings("stereodevice.dat") != SVL_OK) {
            cout << endl;
            converter_camera_source.DialogSetup(SVL_LEFT);
            converter_camera_source.DialogSetup(SVL_RIGHT);
        }
    }
    
    std::string encoderleft, encoderright;
    converter_writer.GetCodecName(encoderleft, SVL_LEFT);
    converter_writer.GetCodecName(encoderright, SVL_RIGHT);

    // chain filters to pipeline
    if (useLiveStream) {
        if (converter_stream.Trunk().Append(&converter_camera_source) != SVL_OK) {
            cerr << "Error append camera source" << endl;
        }
    } else {
        if (converter_stream.Trunk().Append(&converter_file_source) != SVL_OK) {
            cerr << "Error append file source" << endl;
        }
    }
    if (converter_stream.Trunk().Append(&converter_joiner) != SVL_OK) {
        cerr << "Error append joiner" << endl;
    }
    if (converter_stream.Trunk().Append(&converter_writer) != SVL_OK) {
        cerr << "Error append writer" << endl;
    }
    if (converter_stream.Trunk().Append(&viewer_fps) != SVL_OK) {
        cerr << "Error append fps" << endl;
    }
    //cerr << "Converting: '" << sourceleft << "' and '" << sourceright << "' to '" << destination <<"' using codec: '" << encoderleft << "'" << endl;

    // initialize and start stream
    if (converter_stream.Start() != SVL_OK) {
        cerr << "Failed to start stream" << endl;
        exit(1);
    }

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

    return 0;
}

//////////////////////////////////
//             main             //
//////////////////////////////////

int main(int argc, char** argv)
{
    cout << "StereoPlayerTest live_stream thread_count enable_network [video_src_1] [video_src_2]" << endl;
    cout << endl;
    cout << "   live_stream   : 0 to read video stream from files" << endl;
    cout << "                   1 to use live stream from cameras" << endl;
    cout << "   thread_count  : Number of worker threads (available CPU core)" << endl;
    cout << "   enable_networK: 0 to skip sending UDP packets" << endl;
    cout << "                   1 to generate and send UDP packets" << endl;
    cout << "   [video_src_1] : path to video source (left eye) (optional)" << endl;
    cout << "   [video_src_2] : path to video source (right eye) (optional)" << endl;

    bool useLiveStream = false;
    unsigned int numThread = 0;
    bool enableNetwork = false;

    if (argc == 4) {
        useLiveStream = (atoi(argv[1]) == 0 ? false : true);
        numThread = atoi(argv[2]);
        enableNetwork = (atoi(argv[3]) == 0 ? false : true);

        StereoVideoPlayerUDP(argc, argv, numThread, enableNetwork, useLiveStream);
    }
    
    return 1;
}

