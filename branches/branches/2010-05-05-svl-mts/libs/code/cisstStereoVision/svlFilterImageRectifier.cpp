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

#include <cisstStereoVision/svlFilterImageRectifier.h>


/******************************************/
/*** svlFilterImageRectifier class ********/
/******************************************/

CMN_IMPLEMENT_SERVICES(svlFilterImageRectifier)

svlFilterImageRectifier::svlFilterImageRectifier() :
    svlFilterBase(),
    cmnGenericObject(),
    OutputImage(0),
    InterpolationEnabled(true)
{
    AddInput("input", true);
    AddInputType("input", svlTypeImageRGB);
    AddInputType("input", svlTypeImageRGBStereo);

    AddOutput("output", true);
    SetAutomaticOutputType(true);

    RectifLUT.SetAll(0);
}

svlFilterImageRectifier::~svlFilterImageRectifier()
{
    for (unsigned int i = 0; i < SVL_MAX_CHANNELS; i ++) ReleaseLUT(RectifLUT[i]);
    if (OutputImage) delete OutputImage;
}

int svlFilterImageRectifier::Initialize(svlSample* syncInput, svlSample* &syncOutput)
{
    svlSampleImage* input = dynamic_cast<svlSampleImage*>(syncInput);

    unsigned int i, channels;

    channels = std::min(SVL_MAX_CHANNELS, input->GetVideoChannels());
    for (i = 0; i < channels; i ++) {
        if (RectifLUT[i]) {
            if (RectifLUT[i]->Width != input->GetWidth(i) ||
                RectifLUT[i]->Height != input->GetHeight(i))
                return SVL_FAIL;
        }
    }

    // Preparing output sample
    if (OutputImage) delete OutputImage;
    OutputImage = dynamic_cast<svlSampleImage*>(input->GetNewInstance());
    OutputImage->SetSize(*input);

    channels = OutputImage->GetVideoChannels();
    for (i = 0; i < channels; i ++) {
        memset(OutputImage->GetUCharPointer(i), 0, OutputImage->GetDataSize(i));
    }

    syncOutput = OutputImage;

    return SVL_OK;
}

int svlFilterImageRectifier::Process(svlProcInfo* procInfo, svlSample* syncInput, svlSample* &syncOutput)
{
    syncOutput = OutputImage;
    _SkipIfAlreadyProcessed(syncInput, syncOutput);

    svlSampleImage* id = dynamic_cast<svlSampleImage*>(syncInput);
    unsigned int videochannels = id->GetVideoChannels();
    unsigned int idx;

    _ParallelLoop(procInfo, idx, videochannels)
    {
        // Processing
        if (RectifLUT[idx]) {
            Rectify(RectifLUT[idx],
                    id->GetUCharPointer(idx),
                    OutputImage->GetUCharPointer(idx),
                    InterpolationEnabled);
        }
        else {
            memcpy(OutputImage->GetUCharPointer(idx), id->GetUCharPointer(idx), id->GetDataSize(idx));
        }
    }

    return SVL_OK;
}

int svlFilterImageRectifier::LoadTable(const std::string &filepath, unsigned int videoch, int exponentlen)
{
    if (IsInitialized() == true) return SVL_ALREADY_INITIALIZED;
    if (videoch >= SVL_MAX_CHANNELS) return SVL_FAIL;

    ReleaseLUT(RectifLUT[videoch]);
    RectifLUT[videoch] = new RectificationLUT;
    memset(RectifLUT[videoch], 0, sizeof(RectificationLUT));

    int ret = LoadRectificationData(RectifLUT[videoch], filepath, exponentlen);
    if (ret != SVL_OK) {
        ReleaseLUT(RectifLUT[videoch]);
        RectifLUT[videoch] = 0;
    }
    return ret;
}

void svlFilterImageRectifier::EnableInterpolation(bool enable)
{
    InterpolationEnabled = enable;
}

int svlFilterImageRectifier::LoadRectificationData(RectificationLUT* rectdata, const std::string &filepath, int explen)
{
    if (rectdata == 0) return SVL_FAIL;

    std::ifstream file(filepath.c_str(), std::ios_base::in | std::ios_base::binary);
    if (!file.is_open()) return SVL_FAIL;

    ResetLUT(rectdata);

    double dbl;
    const unsigned int maxwidth = 1920;
    const unsigned int maxheight = 1200;
    const unsigned int size = maxwidth * maxheight;
    double* dblbuf = new double[size];
    char* chbuf    = new char[(16 * size) + 1];
    int valcnt, i;

    // lutpos:
    //          1 - width, height
    //          2 - destination index lut
    //          3 - source index lut 1
    //          4 - source index lut 2
    //          5 - source index lut 3
    //          6 - source index lut 4
    //          7 - source blending lut 1
    //          8 - source blending lut 2
    //          9 - source blending lut 3
    //         10 - source blending lut 4
    int lutpos = 0;

    while (lutpos < 10) {
        switch (lutpos) {
            case 0:
                if (LoadLine(file, &dbl, chbuf, 1, explen) < 1) goto labError;
                rectdata->Height = static_cast<int>(dbl);
                if (LoadLine(file, &dbl, chbuf, 1, explen) < 1) goto labError;
                rectdata->Width = static_cast<int>(dbl);

                if (rectdata->Width > maxwidth || rectdata->Height > maxheight) goto labError;

                lutpos ++;
            break;

            case 1:
                valcnt = LoadLine(file, dblbuf, chbuf, size, explen);
                if (valcnt < 1) goto labError;
                rectdata->idxDestSize = valcnt;
                rectdata->idxDest = new unsigned int[valcnt];
                for (i = 0; i < valcnt; i ++) {
                    rectdata->idxDest[i] = static_cast<unsigned int>(dblbuf[i] + 0.5);
                }

                lutpos ++;
            break;

            case 2:
                valcnt = LoadLine(file, dblbuf, chbuf, size, explen);
                if (valcnt < 1) goto labError;
                rectdata->idxSrc1Size = valcnt;
                rectdata->idxSrc1 = new unsigned int[valcnt];
                for (i = 0; i < valcnt; i ++) {
                    rectdata->idxSrc1[i] = static_cast<unsigned int>(dblbuf[i] + 0.5);
                }

                lutpos ++;
            break;

            case 3:
                valcnt = LoadLine(file, dblbuf, chbuf, size, explen);
                if (valcnt < 1) goto labError;
                rectdata->idxSrc2Size = valcnt;
                rectdata->idxSrc2 = new unsigned int[valcnt];
                for (i = 0; i < valcnt; i ++) {
                    rectdata->idxSrc2[i] = static_cast<unsigned int>(dblbuf[i] + 0.5);
                }

                lutpos ++;
            break;

            case 4:
                valcnt = LoadLine(file, dblbuf, chbuf, size, explen);
                if (valcnt < 1) goto labError;
                rectdata->idxSrc3Size = valcnt;
                rectdata->idxSrc3 = new unsigned int[valcnt];
                for (i = 0; i < valcnt; i ++) {
                    rectdata->idxSrc3[i] = static_cast<unsigned int>(dblbuf[i] + 0.5);
                }

                lutpos ++;
            break;

            case 5:
                valcnt = LoadLine(file, dblbuf, chbuf, size, explen);
                if (valcnt < 1) goto labError;
                rectdata->idxSrc4Size = valcnt;
                rectdata->idxSrc4 = new unsigned int[valcnt];
                for (i = 0; i < valcnt; i ++) {
                    rectdata->idxSrc4[i] = static_cast<unsigned int>(dblbuf[i] + 0.5);
                }

                lutpos ++;
            break;

            case 6:
                valcnt = LoadLine(file, dblbuf, chbuf, size, explen);
                if (valcnt < 1) goto labError;
                rectdata->blendSrc1Size = valcnt;
                rectdata->blendSrc1 = new unsigned char[valcnt];
                for (i = 0; i < valcnt; i ++) {
                    rectdata->blendSrc1[i] = static_cast<unsigned char>(dblbuf[i] * 256);
                }

                lutpos ++;
            break;

            case 7:
                valcnt = LoadLine(file, dblbuf, chbuf, size, explen);
                if (valcnt < 1) goto labError;
                rectdata->blendSrc2Size = valcnt;
                rectdata->blendSrc2 = new unsigned char[valcnt];
                for (i = 0; i < valcnt; i ++) {
                    rectdata->blendSrc2[i] = static_cast<unsigned char>(dblbuf[i] * 256);
                }

                lutpos ++;
            break;

            case 8:
                valcnt = LoadLine(file, dblbuf, chbuf, size, explen);
                if (valcnt < 1) goto labError;
                rectdata->blendSrc3Size = valcnt;
                rectdata->blendSrc3 = new unsigned char[valcnt];
                for (i = 0; i < valcnt; i ++) {
                    rectdata->blendSrc3[i] = static_cast<unsigned char>(dblbuf[i] * 256);
                }

                lutpos ++;
            break;

            case 9:
                valcnt = LoadLine(file, dblbuf, chbuf, size, explen);
                if (valcnt < 1) goto labError;
                rectdata->blendSrc4Size = valcnt;
                rectdata->blendSrc4 = new unsigned char[valcnt];
                for (i = 0; i < valcnt; i ++) {
                    rectdata->blendSrc4[i] = static_cast<unsigned char>(dblbuf[i] * 256);
                }

                lutpos ++;
            break;
        }
    }

    file.close();

    valcnt = rectdata->idxDestSize;
    if (rectdata->idxSrc1Size != valcnt ||
        rectdata->idxSrc2Size != valcnt ||
        rectdata->idxSrc3Size != valcnt ||
        rectdata->idxSrc4Size != valcnt ||
        rectdata->blendSrc1Size != valcnt ||
        rectdata->blendSrc2Size != valcnt ||
        rectdata->blendSrc3Size != valcnt ||
        rectdata->blendSrc4Size != valcnt) goto labError;

    TransposeLUTArray(rectdata->idxDest, valcnt, rectdata->Width, rectdata->Height);
    TransposeLUTArray(rectdata->idxSrc1, valcnt, rectdata->Width, rectdata->Height);
    TransposeLUTArray(rectdata->idxSrc2, valcnt, rectdata->Width, rectdata->Height);
    TransposeLUTArray(rectdata->idxSrc3, valcnt, rectdata->Width, rectdata->Height);
    TransposeLUTArray(rectdata->idxSrc4, valcnt, rectdata->Width, rectdata->Height);

    for (i = 0; i < valcnt; i ++) {
        rectdata->idxDest[i] *= 3;
        rectdata->idxSrc1[i] *= 3;
        rectdata->idxSrc2[i] *= 3;
        rectdata->idxSrc3[i] *= 3;
        rectdata->idxSrc4[i] *= 3;
    }

    if (dblbuf) delete [] dblbuf;
    if (chbuf) delete [] chbuf;

    return SVL_OK;

labError:
    if (file.is_open()) file.close();

    ResetLUT(rectdata);

    if (dblbuf) delete [] dblbuf;
    if (chbuf) delete [] chbuf;

    return SVL_FAIL;
}

int svlFilterImageRectifier::LoadLine(std::ifstream &file, double* dblbuf, char* chbuf, unsigned int size, int explen)
{
    unsigned int bufsize = (16 * size) + 1; // max text line length
    unsigned int datalen; // actual data size
    unsigned int bufpos = 0;
    unsigned int filepos;
    char* tbuf;
    char ch;
    int ival;
    int counter = 0;
    int linebreak = 0;
    double dbl;
    double negexp[] = {1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001};
	double negexpTens[] = {1.0, 0.0000000001, 0.00000000001, 0.000000000001, 0.000000000001, 0.000000000001, 0.000000000001, 0.000000000001, 0.000000000001, 0.0000000000001, 0.0000000000001};
    double posexp[] = {1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0, 1000000000.0};

    filepos = file.tellg();
    file.read(chbuf, bufsize);
    datalen = file.gcount();
    file.clear();

    if (explen == 3) {
        while (datalen > 16 ) {
            bufpos += 2;

            tbuf = chbuf + bufpos;
            ch = *tbuf - 48; if (ch < 0 || ch > 9) break;

        // ?.xxxxxxxes00x
            dbl = ch;

            tbuf += 2;
            ch = *tbuf - 48; if (ch < 0 || ch > 9) break;

        // x.?xxxxxxes00x
            dbl += 0.1 * ch;

            tbuf ++;
            ch = *tbuf - 48; if (ch < 0 || ch > 9) break;

        // x.x?xxxxxes00x
            dbl += 0.01 * ch;

            tbuf ++;
            ch = *tbuf - 48; if (ch < 0 || ch > 9) break;

        // x.xx?xxxxes00x
            dbl += 0.001 * ch;

            tbuf ++;
            ch = *tbuf - 48; if (ch < 0 || ch > 9) break;

        // x.xxx?xxxes00x
            dbl += 0.0001 * ch;

            tbuf ++;
            ch = *tbuf - 48; if (ch < 0 || ch > 9) break;

        // x.xxxx?xxes00x
            dbl += 0.00001 * ch;

            tbuf ++;
            ch = *tbuf - 48; if (ch < 0 || ch > 9) break;

        // x.xxxxx?xes00x
            dbl += 0.000001 * ch;

            tbuf ++;
            ch = *tbuf - 48; if (ch < 0 || ch > 9) break;

        // x.xxxxxx?es00x
            dbl += 0.0000001 * ch;

            tbuf += 2;

        // x.xxxxxxxe?00x
            if (*tbuf == '-') {
                tbuf += 3;
                ival = *tbuf - 48; if (ival < 0 || ival > 9) break;

            // x.xxxxxxxe-00?
                dbl *= negexp[ival];
            }
            else {
                tbuf += 3;
                ival = *tbuf - 48; if (ival < 0 || ival > 9) break;

            // x.xxxxxxxe+00?
                dbl *= posexp[ival];
            }

            dblbuf[counter] = dbl;
            bufpos += 14;
            counter ++;
            datalen -= 16;

            // if end of line, seek back
            ch = chbuf[bufpos];
            if (ch == 0x0a) { // LF
                file.seekg(filepos + bufpos + 1);
                linebreak = 1;
                break;
            }
            else if (ch == 0x0d) { // CRLF
                file.seekg(filepos + bufpos + 2);
                linebreak = 1;
                break;
            }
        }
    }
    else if (explen == 2) {
        while (datalen > 16 ) {
            bufpos += 3;

            tbuf = chbuf + bufpos;
            ch = *tbuf - 48; if (ch < 0 || ch > 9) break;

        // ?.xxxxxxxes0x
            dbl = ch;

            tbuf += 2;
            ch = *tbuf - 48; if (ch < 0 || ch > 9) break;

        // x.?xxxxxxes0x
            dbl += 0.1 * ch;

            tbuf ++;
            ch = *tbuf - 48; if (ch < 0 || ch > 9) break;

        // x.x?xxxxxes0x
            dbl += 0.01 * ch;

            tbuf ++;
            ch = *tbuf - 48; if (ch < 0 || ch > 9) break;

        // x.xx?xxxxes0x
            dbl += 0.001 * ch;

            tbuf ++;
            ch = *tbuf - 48; if (ch < 0 || ch > 9) break;

        // x.xxx?xxxes0x
            dbl += 0.0001 * ch;

            tbuf ++;
            ch = *tbuf - 48; if (ch < 0 || ch > 9) break;

        // x.xxxx?xxes0x
            dbl += 0.00001 * ch;

            tbuf ++;
            ch = *tbuf - 48; if (ch < 0 || ch > 9) break;

        // x.xxxxx?xes0x
            dbl += 0.000001 * ch;

            tbuf ++;
            ch = *tbuf - 48; if (ch < 0 || ch > 9) break;

        // x.xxxxxx?es0x
            dbl += 0.0000001 * ch;

            tbuf += 2;

        // x.xxxxxxxe?0x
            if (*tbuf == '-') {
                tbuf ++;
                ival = *tbuf - 48; if (ival < 0 || ival > 9) break;

            // x.xxxxxxxe-?x
                dbl *= negexpTens[ival];

				tbuf ++;
                ival = *tbuf - 48; if (ival < 0 || ival > 9) break;

            // x.xxxxxxxe-x?
                dbl *= negexp[ival];
            }
            else {
                tbuf += 2;
                ival = *tbuf - 48; if (ival < 0 || ival > 9) break;

            // x.xxxxxxxe+0?
                dbl *= posexp[ival];
            }

            dblbuf[counter] = dbl;
            bufpos += 13;
            counter ++;
            datalen -= 16;

            // if end of line, seek back
            ch = chbuf[bufpos];
            if (ch == 0x0a) { // LF
                file.seekg(filepos + bufpos + 1);
                linebreak = 1;
                break;
            }
            else if (ch == 0x0d) { // CRLF
                file.seekg(filepos + bufpos + 2);
                linebreak = 1;
                break;
            }
        }
    }

    if (linebreak == 1) return counter;
    return 0;
}

void svlFilterImageRectifier::TransposeLUTArray(unsigned int* index, unsigned int size, unsigned int width, unsigned int height)
{
    unsigned int i, x, y, val;

    for (i = 0; i < size; i ++) {
        val = index[i] - 1;
        x = val / height;
        y = val % height;
        index[i] = y * width + x;
    }
}

void svlFilterImageRectifier::Rectify(RectificationLUT* rectdata, unsigned char* srcimg, unsigned char* destimg, bool interpolation)
{
    unsigned char *srcbld1, *srcbld2, *srcbld3, *srcbld4;
    unsigned int *destidx, *srcidx1, *srcidx2, *srcidx3, *srcidx4;
    unsigned char *destr, *destg, *destb;
    unsigned char *srcr, *srcg, *srcb;
    unsigned int destofs, srcofs;
    unsigned int resr, resg, resb;
    unsigned int blnd;

    const unsigned int destlen = rectdata->idxDestSize;

    if (interpolation) {
        destidx = rectdata->idxDest;
        srcidx1 = rectdata->idxSrc1;
        srcidx2 = rectdata->idxSrc2;
        srcidx3 = rectdata->idxSrc3;
        srcidx4 = rectdata->idxSrc4;
        srcbld1 = rectdata->blendSrc1;
        srcbld2 = rectdata->blendSrc2;
        srcbld3 = rectdata->blendSrc3;
        srcbld4 = rectdata->blendSrc4;

        for (unsigned int i = 0; i < destlen; i ++) {

            // interpolation - 1st source pixel and weight
            srcofs = *srcidx1;
            srcr = srcimg + srcofs;
            srcg = srcr + 1;
            srcb = srcg + 1;

            blnd = *srcbld1;
            resr = blnd * (*srcr);
            resg = blnd * (*srcg);
            resb = blnd * (*srcb);

            // interpolation - 2nd source pixel and weight
            srcofs = *srcidx2;
            srcr = srcimg + srcofs;
            srcg = srcr + 1;
            srcb = srcg + 1;

            blnd = *srcbld2;
            resr += blnd * (*srcr);
            resg += blnd * (*srcg);
            resb += blnd * (*srcb);

            // interpolation - 3rd source pixel and weight
            srcofs = *srcidx3;
            srcr = srcimg + srcofs;
            srcg = srcr + 1;
            srcb = srcg + 1;

            blnd = *srcbld3;
            resr += blnd * (*srcr);
            resg += blnd * (*srcg);
            resb += blnd * (*srcb);

            // interpolation - 4th source pixel and weight
            srcofs = *srcidx4;
            srcr = srcimg + srcofs;
            srcg = srcr + 1;
            srcb = srcg + 1;

            blnd = *srcbld4;
            resr += blnd * (*srcr);
            resg += blnd * (*srcg);
            resb += blnd * (*srcb);

            // destination pixel
            destofs = *destidx;
            destr = destimg + destofs;
            destg = destr + 1;
            destb = destg + 1;

            *destr = static_cast<unsigned char>(resr >> 8);
            *destg = static_cast<unsigned char>(resg >> 8);
            *destb = static_cast<unsigned char>(resb >> 8);

            destidx ++;
            srcidx1 ++;
            srcidx2 ++;
            srcidx3 ++;
            srcidx4 ++;
            srcbld1 ++;
            srcbld2 ++;
            srcbld3 ++;
            srcbld4 ++;
        }
    }
    else {
        destidx = rectdata->idxDest;
        srcidx1 = rectdata->idxSrc1;
        svlRGB *prgb1, *prgb2;

        for (unsigned int i = 0; i < destlen; i ++) {

            // sampling - 1st source pixel
            srcofs = *srcidx1;
            prgb1 = reinterpret_cast<svlRGB*>(srcimg + srcofs);

            // copying value
            destofs = *destidx;
            prgb2 = reinterpret_cast<svlRGB*>(destimg + destofs);

            *prgb2 = *prgb1;

            destidx ++;
            srcidx1 ++;
        }
    }
}

void svlFilterImageRectifier::ResetLUT(RectificationLUT* lut)
{
    if (lut) {
        if (lut->idxDest) delete [] lut->idxDest;
        if (lut->idxSrc1) delete [] lut->idxSrc1;
        if (lut->idxSrc2) delete [] lut->idxSrc2;
        if (lut->idxSrc3) delete [] lut->idxSrc3;
        if (lut->idxSrc4) delete [] lut->idxSrc4;
        if (lut->blendSrc1) delete [] lut->blendSrc1;
        if (lut->blendSrc2) delete [] lut->blendSrc2;
        if (lut->blendSrc3) delete [] lut->blendSrc3;
        if (lut->blendSrc4) delete [] lut->blendSrc4;
        memset(lut, 0, sizeof(RectificationLUT));
    }
}

void svlFilterImageRectifier::ReleaseLUT(RectificationLUT* lut)
{
    if (lut) {
        ResetLUT(lut);
        delete lut;
    }
}

