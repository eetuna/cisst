/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: videoconverter.cpp 1330 2010-03-22 19:15:48Z mjung5 $
  
  Author(s):  Balazs Vagvolgyi
  Created on: 2009

  (C) Copyright 2006-2009 Johns Hopkins University (JHU), All Rights
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


///////////////////////
//  Video Converter  //
///////////////////////

int VideoConverter()
{
    std::string sourceleft, sourceright, destination;

    svlInitialize();

    svlStreamManager converter_stream(4);
    svlFilterSourceVideoFile converter_source(2); // # of source channels
    svlFilterStereoImageJoiner converter_joiner;
    svlFilterVideoFileWriter converter_writer;

    // converter_joiner.SetLayout(svlFilterStereoImageJoiner::SideBySide);

    if (converter_source.DialogFilePath(SVL_LEFT) != SVL_OK) {
        cerr << " -!- No source file has been selected." << endl;
        return -1;
    }
    converter_source.GetFilePath(sourceleft, SVL_LEFT);

    if (converter_source.DialogFilePath(SVL_RIGHT) != SVL_OK) {
        cerr << " -!- No source file has been selected." << endl;
        return -1;
    }
    converter_source.GetFilePath(sourceright, SVL_RIGHT);

    converter_source.SetTargetFrequency(30.0); // as fast as possible
    //converter_source.SetLoop(false);
    converter_source.SetLoop(true);

    if (converter_writer.DialogFilePath() != SVL_OK) {
        cerr << " -!- No destination file has been selected." << endl;
        return -1;
    }
    converter_writer.GetFilePath(destination);
//    if (converter_writer.DialogFilePath(SVL_RIGHT) != SVL_OK) {
//        cerr << " -!- No destination file has been selected." << endl;
//        return -1;
//    }
//    converter_writer.GetFilePath(destinationright, SVL_RIGHT);
/*
    if (converter_writer.LoadCodec("codec.dat") != SVL_OK) {
        if (converter_writer.DialogCodec() != SVL_OK) {
            cerr << " -!- Unable to set up compression." << endl;
            return -1;
        }
        converter_writer.SaveCodec("codec.dat");
    }
*/
//    if (converter_writer.LoadCodec("codecright.dat", SVL_RIGHT) != SVL_OK) {
//        if (converter_writer.DialogCodec(SVL_RIGHT) != SVL_OK) {
//            cerr << " -!- Unable to set up compression." << endl;
//            return -1;
//        }
//        converter_writer.SaveCodec("codecright.dat", SVL_RIGHT);
//    }

    std::string encoderleft, encoderright;
    converter_writer.GetCodecName(encoderleft, SVL_LEFT);
    converter_writer.GetCodecName(encoderright, SVL_RIGHT);

    // chain filters to pipeline
    converter_stream.Trunk().Append(&converter_source);
    converter_stream.Trunk().Append(&converter_joiner);
    converter_stream.Trunk().Append(&converter_writer);

    cerr << "Converting: '" << sourceleft << "' and '" << sourceright << "' to '" << destination <<"' using codec: '" << encoderleft << "'" << endl;
//    cerr << "Converting: '" << sourceright << "' to '" << destinationright <<"' using codec: '" << encoderright << "'" << endl;

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
    // log configuration
    cmnLogger::SetLoD(CMN_LOG_LOD_VERY_VERBOSE);

    cerr << endl << "stereoTutorialVideoConverter - cisstStereoVision example by Balazs Vagvolgyi" << endl;
    cerr << "See http://www.cisst.org/cisst for details." << endl;

    VideoConverter();

    return 1;
}

