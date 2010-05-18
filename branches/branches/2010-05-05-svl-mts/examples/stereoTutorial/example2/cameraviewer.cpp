/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$
  
  Author(s):  Balazs Vagvolgyi
  Created on: 2008

  (C) Copyright 2006-2008 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/


#include <iostream>
#include <string>
#include <cisstCommon.h>
#include <cisstOSAbstraction.h>
#include <cisstStereoVision.h>

using namespace std;


class svlOverlayAsyncOutputProperties : public svlOverlayStaticText
{
public:
    svlOverlayAsyncOutputProperties();
    svlOverlayAsyncOutputProperties(unsigned int videoch,
                                    bool visible,
                                    svlFilterOutput* output,
                                    svlRect rect,
                                    double fontsize,
                                    svlRGB txtcolor);
    svlOverlayAsyncOutputProperties(unsigned int videoch,
                                    bool visible,
                                    svlFilterOutput* output,
                                    svlRect rect,
                                    double fontsize,
                                    svlRGB txtcolor,
                                    svlRGB bgcolor);
    virtual ~svlOverlayAsyncOutputProperties();

protected:
    virtual void DrawInternal(svlSampleImage* bgimage, svlSample* input);

private:
    svlFilterOutput* Output;
};


/*********************************************/
/*** svlOverlayAsyncOutputProperties class ***/
/*********************************************/

svlOverlayAsyncOutputProperties::svlOverlayAsyncOutputProperties() :
    svlOverlayStaticText(),
    Output(0)
{
}

svlOverlayAsyncOutputProperties::svlOverlayAsyncOutputProperties(unsigned int videoch,
                                                                 bool visible,
                                                                 svlFilterOutput* output,
                                                                 svlRect rect,
                                                                 double fontsize,
                                                                 svlRGB txtcolor) :
    svlOverlayStaticText(videoch, visible, "", rect, fontsize, txtcolor),
    Output(output)
{
}

svlOverlayAsyncOutputProperties::svlOverlayAsyncOutputProperties(unsigned int videoch,
                                                                 bool visible,
                                                                 svlFilterOutput* output,
                                                                 svlRect rect,
                                                                 double fontsize,
                                                                 svlRGB txtcolor,
                                                                 svlRGB bgcolor) :
    svlOverlayStaticText(videoch, visible, "", rect, fontsize, txtcolor, bgcolor),
    Output(output)
{
}

svlOverlayAsyncOutputProperties::~svlOverlayAsyncOutputProperties()
{
}

void svlOverlayAsyncOutputProperties::DrawInternal(svlSampleImage* bgimage, svlSample* CMN_UNUSED(input))
{
    if (Output) {
        double usageratio = Output->GetBufferUsageRatio();
        int dropped = Output->GetDroppedSampleCount();

        std::stringstream strstr;
        strstr << "Buffer: " << std::fixed << std::setprecision(1) << usageratio * 100.0 << "%, Dropped: " << dropped;
        SetText(strstr.str());
    }

    svlOverlayStaticText::DrawInternal(bgimage, 0);
}



///////////////////////////////////
//     Window callback class     //
///////////////////////////////////

class CViewerWindowCallback : public svlImageWindowCallbackBase
{
public:
    CViewerWindowCallback() :
        svlImageWindowCallbackBase()
        ,ImageWriterFilter(0)
        ,RecorderFilter(0)
        ,SplitterOutput(0)
        ,Recording(false)
    {
    }

    void OnUserEvent(unsigned int CMN_UNUSED(winid), bool ascii, unsigned int eventid)
    {
        // handling user inputs
        if (ascii) {
            switch (eventid) {
                case 's':
                {
                    if (ImageWriterFilter) {
                        ImageWriterFilter->Record(1);
                        cout << endl << " >>> Snapshot saved <<<" << endl;
                    }
                }
                break;

                case ' ':
                    if (RecorderFilter) {
                        if (Recording) {
                            RecorderFilter->Pause();
                            SplitterOutput->SetBlock(true);
                            Recording = false;
                            cout << endl << " >>> Recording paused <<<" << endl;
                        }
                        else {
                            SplitterOutput->SetBlock(false);
                            RecorderFilter->Record(-1);
                            Recording = true;
                            cout << endl << " >>> Recording started <<<" << endl;
                        }
                    }
                break;

                default:
                    return;
            }
        }
    }

    svlFilterImageFileWriter* ImageWriterFilter;
    svlFilterVideoFileWriter* RecorderFilter;
    svlFilterOutput* SplitterOutput;
    bool Recording;
};


////////////////////
//  CameraViewer  //
////////////////////

int CameraViewer(bool interpolation, bool save, int width, int height)
{
    svlInitialize();

    // instantiating SVL stream and filters
    svlStreamManager viewer_stream(1);
    svlFilterSourceVideoCapture viewer_source(1);
    svlFilterSplitter viewer_splitter;
    svlFilterImageResizer viewer_resizer;
    svlFilterImageWindow viewer_window;
    svlFilterImageOverlay viewer_overlay;
    CViewerWindowCallback viewer_window_cb;
    svlFilterVideoFileWriter viewer_videowriter;
    svlFilterImageFileWriter viewer_imagewriter;
    svlFilterImageWindow viewer_window2;

    // setup source
    // Delete "device.dat" to reinitialize input device
    if (viewer_source.LoadSettings("device.dat") != SVL_OK) {
        cout << endl;
        viewer_source.DialogSetup();
    }

    // setup splitter
    viewer_splitter.AddOutput("output2", 8);
    svlFilterOutput* output = viewer_splitter.GetOutput("output2");

    // setup writer
    if (save == true) {
        viewer_videowriter.DialogFilePath();
        viewer_videowriter.DialogCodec();
        viewer_videowriter.Pause();
    }

    // setup image writer
    viewer_imagewriter.SetFilePath("image_", "bmp");
    viewer_imagewriter.EnableTimestamps();
    viewer_imagewriter.Pause();

    // setup resizer
    if (width > 0 && height > 0) {
        viewer_resizer.SetInterpolation(interpolation);
        viewer_resizer.SetOutputSize(width, height);
    }

    // setup image window
    if (save == true) {
        viewer_window_cb.RecorderFilter = &viewer_videowriter;
        viewer_window_cb.SplitterOutput = output;
    }
    viewer_window_cb.ImageWriterFilter = &viewer_imagewriter;
    viewer_window.SetCallback(&viewer_window_cb);
    viewer_window.SetTitleText("Camera Viewer");
    viewer_window.EnableTimestampInTitle();


    // Add buffer status overlay
    svlOverlayAsyncOutputProperties buffer_overlay(SVL_LEFT,
                                                   true,
                                                   output,
                                                   svlRect(4, 4, 225, 20),
                                                   14.0,
                                                   svlRGB(255, 255, 255),
                                                   svlRGB(0, 128, 0));
    viewer_overlay.AddOverlay(buffer_overlay);

    // Add framerate overlay
    svlOverlayFramerate fps_overlay(SVL_LEFT,
                                    true,
                                    &viewer_window,
                                    svlRect(4, 24, 47, 40),
                                    14.0,
                                    svlRGB(255, 255, 255),
                                    svlRGB(128, 0, 0));
    viewer_overlay.AddOverlay(fps_overlay);


    // chain filters to pipeline
    viewer_stream.SetSourceFilter(&viewer_source);
    viewer_source.GetOutput()->Connect(viewer_imagewriter.GetInput());
    if (width > 0 && height > 0) {
        viewer_imagewriter.GetOutput()->Connect(viewer_resizer.GetInput());
        viewer_resizer.GetOutput()->Connect(viewer_splitter.GetInput());
    }
    else {
        viewer_imagewriter.GetOutput()->Connect(viewer_splitter.GetInput());
    }
    viewer_splitter.GetOutput()->Connect(viewer_overlay.GetInput());
    viewer_overlay.GetOutput()->Connect(viewer_window.GetInput());

    if (save == true) {
        // put the recorder on a branch in order to enable buffering
        output->SetBlock(true);
//        output->Connect(viewer_window2.GetInput());
        output->Connect(viewer_videowriter.GetInput());
    }

    cerr << endl << "Starting stream... ";

    // initialize and start stream
    if (viewer_stream.Start() != SVL_OK) goto labError;

    cerr << "Done" << endl;

    // wait for keyboard input in command window
    int ch;

    do {
        cerr << endl << "Keyboard commands:" << endl << endl;
        cerr << "  In image window:" << endl;
        if (save == true) {
            cerr << "    SPACE - Video recorder control: Record/Pause" << endl;
        }
        cerr << "    's'   - Take image snapshot" << endl;
        cerr << "  In command window:" << endl;
        cerr << "    'i'   - Adjust image properties" << endl;
        cerr << "    'q'   - Quit" << endl << endl;

        ch = cmnGetChar();

        switch (ch) {
            case 'i':
                // Adjust image properties
                cerr << endl << endl;
                viewer_source.DialogImageProperties();
                cerr << endl;
            break;

            default:
            break;
        }
        osaSleep(1.0 * cmn_ms);
    } while (ch != 'q');

    cerr << endl;

    // stop stream
    viewer_stream.Stop();

    // save settings
    viewer_source.SaveSettings("device.dat");

labError:
    return 0;
}


//////////////////////////////////
//             main             //
//////////////////////////////////

int ParseNumber(char* string, unsigned int maxlen)
{
    if (string == 0 || maxlen == 0) return -1;

    int ivalue, j;
    char ch;

    // parse number
    j = 0;
    ivalue = 0;
    ch = string[j];
    // 4 digits max
    while (ch != 0 && j < (int)maxlen) {
        // check if number
        ch -= '0';
        if (ch > 9 || ch < 0) {
            ivalue = -1;
            break;
        }
        ivalue = ivalue * 10 + ch;
        // step to next digit
        j ++;
        ch = string[j];
    }
    if (j == 0) ivalue = -1;

    return ivalue;
}

int main(int argc, char** argv)
{
    cerr << endl << "stereoTutorialCameraViewer - cisstStereoVision example by Balazs Vagvolgyi" << endl;
    cerr << "See http://www.cisst.org/cisst for details." << endl;
    cerr << "Enter 'stereoTutorialCameraViewer -?' for help." << endl;

    //////////////////////////////
    // parsing arguments
    int i, options, ivalue, width, height;
    bool interpolation, save;

    options = argc - 1;
    interpolation = false;
    width = -1;
    height = -1;
    save = true;

    for (i = 1; i <= options; i ++) {
        if (argv[i][0] != '-') continue;

        switch (argv[i][1]) {
            case '?':
                cerr << "Command line format:" << endl;
                cerr << "     stereoTutorialCameraViewer [options]" << endl;
                cerr << "Options:" << endl;
                cerr << "     -v        Save video file" << endl;
                cerr << "     -i        Interpolation ON [default: OFF]" << endl;
                cerr << "     -w#       Displayed image width" << endl;
                cerr << "     -h#       Displayed image height" << endl;
                cerr << "Examples:" << endl;
                cerr << "     stereoTutorialCameraViewer" << endl;
                cerr << "     stereoTutorialCameraViewer -v -i -w1024 -h768" << endl;
                return 1;
            break;

            case 'i':
                interpolation = true;
            break;

            case 'v':
                save = true;
            break;

            case 'w':
                ivalue = ParseNumber(argv[i] + 2, 4);
                if (ivalue > 0) width = ivalue;
            break;

            case 'h':
                ivalue = ParseNumber(argv[i] + 2, 4);
                if (ivalue > 0) height = ivalue;
            break;

            default:
                // NOP
            break;
        }
    }

    //////////////////////////////
    // starting viewer

    CameraViewer(interpolation, save, width, height);

    cerr << "Quit" << endl;
    return 1;
}

