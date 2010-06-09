/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: $
  
  Author(s):  Balazs Vagvolgyi
  Created on: 2009

  (C) Copyright 2006-2007 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#include <cisstStereoVision/svlTypes.h>


/******************************/
/*** svlSample class **********/
/******************************/

svlSample::svlSample() :
    cmnGenericObject(),
    EncoderParameter(-1)
{
}

svlSample::~svlSample()
{
}

bool svlSample::IsImage() const
{
    return false;
}

bool svlSample::IsInitialized() const
{
    return false;
}

void svlSample::SetTimestamp(double ts)
{
    Timestamp = ts;
}

double svlSample::GetTimestamp() const
{
    return Timestamp;
}

svlSample* svlSample::GetNewFromType(svlStreamType type)
{
    switch (type) {
        case svlTypeInvalid:           return 0;                              break;
        case svlTypeStreamSource:      return 0;                              break;
        case svlTypeStreamSink:        return 0;                              break;
        case svlTypeImageCustom:       return 0;                              break;
        case svlTypeImageRGB:          return new svlSampleImageRGB;          break;
        case svlTypeImageRGBA:         return new svlSampleImageRGBA;         break;
        case svlTypeImageRGBStereo:    return new svlSampleImageRGBStereo;    break;
        case svlTypeImageRGBAStereo:   return new svlSampleImageRGBAStereo;   break;
        case svlTypeImageMono8:        return new svlSampleImageMono8;        break;
        case svlTypeImageMono8Stereo:  return new svlSampleImageMono8Stereo;  break;
        case svlTypeImageMono16:       return new svlSampleImageMono16;       break;
        case svlTypeImageMono16Stereo: return new svlSampleImageMono16Stereo; break;
        case svlTypeImageMonoFloat:    return new svlSampleImageMonoFloat;    break;
        case svlTypeImage3DMap:        return new svlSampleImage3DMap;        break;
        case svlTypeTransform3D:       return new svlSampleTransform3D;       break;
        case svlTypeTargets:           return new svlSampleTargets;           break;
        case svlTypeText:              return new svlSampleText;              break;
    }
    return 0;
}

void svlSample::SetEncoder(const std::string & codec, const int parameter)
{
    Encoder = codec;
    EncoderParameter = parameter;
}

void svlSample::GetEncoder(std::string & codec, int & parameter) const
{
    codec = Encoder;
    parameter = EncoderParameter;
}


/*****************************/
/*** svlSampleImage class ****/
/*****************************/

svlSampleImage::svlSampleImage() :
    svlSample()
{
    SetEncoder("bmp", 0);
}

svlSampleImage::~svlSampleImage()
{
}

bool svlSampleImage::IsImage() const
{
    return true;
}


/***********************************/
/*** svlSampleImageCustom class ****/
/***********************************/

CMN_IMPLEMENT_SERVICES_TEMPLATED(svlSampleImageMono8)
CMN_IMPLEMENT_SERVICES_TEMPLATED(svlSampleImageMono8Stereo)
CMN_IMPLEMENT_SERVICES_TEMPLATED(svlSampleImageMono16)
CMN_IMPLEMENT_SERVICES_TEMPLATED(svlSampleImageMono16Stereo)
CMN_IMPLEMENT_SERVICES_TEMPLATED(svlSampleImageRGB)
CMN_IMPLEMENT_SERVICES_TEMPLATED(svlSampleImageRGBA)
CMN_IMPLEMENT_SERVICES_TEMPLATED(svlSampleImageRGBStereo)
CMN_IMPLEMENT_SERVICES_TEMPLATED(svlSampleImageRGBAStereo)
CMN_IMPLEMENT_SERVICES_TEMPLATED(svlSampleImageMonoFloat)
CMN_IMPLEMENT_SERVICES_TEMPLATED(svlSampleImage3DMap)


/**********************************/
/*** svlSampleTransform3D class ***/
/**********************************/

CMN_IMPLEMENT_SERVICES(svlSampleTransform3D)

svlSampleTransform3D::svlSampleTransform3D() :
    svlSample()
{
}

svlSample* svlSampleTransform3D::GetNewInstance() const
{
    return new svlSampleTransform3D;
}

svlStreamType svlSampleTransform3D::GetType() const
{
    return svlTypeTransform3D;
}

int svlSampleTransform3D::SetSize(const svlSample* sample)
{
    if (!sample || sample->GetType() != svlTypeTransform3D) return SVL_FAIL;
    return SVL_OK;
}

int svlSampleTransform3D::SetSize(const svlSample& sample)
{
    if (sample.GetType() != svlTypeTransform3D) return SVL_FAIL;
    return SVL_OK;
}

int svlSampleTransform3D::CopyOf(const svlSample* sample)
{
    if (!sample || sample->GetType() != svlTypeTransform3D) return SVL_FAIL;
    
    const svlSampleTransform3D* samplexform = dynamic_cast<const svlSampleTransform3D*>(sample);
    memcpy(GetUCharPointer(), samplexform->GetUCharPointer(), GetDataSize());
    SetTimestamp(sample->GetTimestamp());
    
    return SVL_OK;
}

int svlSampleTransform3D::CopyOf(const svlSample& sample)
{
    if (sample.GetType() != svlTypeTransform3D) return SVL_FAIL;

    const svlSampleTransform3D* samplexform = dynamic_cast<const svlSampleTransform3D*>(&sample);
    memcpy(GetUCharPointer(), samplexform->GetUCharPointer(), GetDataSize());
    SetTimestamp(sample.GetTimestamp());

    return SVL_OK;
}

bool svlSampleTransform3D::IsInitialized() const
{
    return true;
}

unsigned char* svlSampleTransform3D::GetUCharPointer()
{
    return reinterpret_cast<unsigned char*>(Matrix.Pointer());
}

const unsigned char* svlSampleTransform3D::GetUCharPointer() const
{
    return reinterpret_cast<const unsigned char*>(Matrix.Pointer());
}

unsigned int svlSampleTransform3D::GetDataSize() const
{
    return (Matrix.size() * sizeof(double));
}

void svlSampleTransform3D::SerializeRaw(std::ostream & outputStream) const
{
    cmnSerializeRaw(outputStream, Matrix);
}

void svlSampleTransform3D::DeSerializeRaw(std::istream & inputStream)
{
    cmnDeSerializeRaw(inputStream, Matrix);
}

svlSampleTransform3D::svlSampleTransform3D(const vct4x4 & matrix) :
    svlSample()
{
    Matrix.Assign(matrix);
}

svlSampleTransform3D::svlSampleTransform3D(const vctFrm4x4 & frame) :
    svlSample()
{
    Matrix.Assign(frame);
}

svlSampleTransform3D::svlSampleTransform3D(const vctRot3 & rotation, const vct3 & translation) :
    svlSample()
{
    Matrix.Element(0, 0) = rotation.Element(0, 0);
        Matrix.Element(0, 1) = rotation.Element(0, 1);
            Matrix.Element(0, 2) = rotation.Element(0, 2);
    Matrix.Element(1, 0) = rotation.Element(1, 0);
        Matrix.Element(1, 1) = rotation.Element(1, 1);
            Matrix.Element(1, 2) = rotation.Element(1, 2);
    Matrix.Element(2, 0) = rotation.Element(2, 0);
        Matrix.Element(2, 1) = rotation.Element(2, 1);
            Matrix.Element(2, 2) = rotation.Element(2, 2);
    Matrix.Element(0, 3) = translation.X();
        Matrix.Element(1, 3) = translation.Y();
            Matrix.Element(2, 3) = translation.Z();
    Matrix.Element(3, 0) = 0.0;
        Matrix.Element(3, 1) = 0.0;
            Matrix.Element(3, 2) = 0.0;
                Matrix.Element(3, 3) = 1.0;
}

vct4x4 & svlSampleTransform3D::GetMatrixRef()
{
    return Matrix;
}

const vct4x4 & svlSampleTransform3D::GetMatrixRef() const
{
    return Matrix;
}

vctFrm4x4 svlSampleTransform3D::GetFrame() const
{
    return vctFrm4x4(Matrix);
}

vctRot3 svlSampleTransform3D::GetRotation() const
{
    vctRot3 rotation;

    rotation.Element(0, 0) = Matrix.Element(0, 0);
        rotation.Element(0, 1) = Matrix.Element(0, 1);
            rotation.Element(0, 2) = Matrix.Element(0, 2);
    rotation.Element(1, 0) = Matrix.Element(1, 0);
        rotation.Element(1, 1) = Matrix.Element(1, 1);
            rotation.Element(1, 2) = Matrix.Element(1, 2);
    rotation.Element(2, 0) = Matrix.Element(2, 0);
        rotation.Element(2, 1) = Matrix.Element(2, 1);
            rotation.Element(2, 2) = Matrix.Element(2, 2);

    return rotation;
}

vct3 svlSampleTransform3D::GetTranslation() const
{
    vct3 translation;

    translation.X() = Matrix.Element(0, 3);
        translation.Y() = Matrix.Element(1, 3);
            translation.Z() = Matrix.Element(2, 3);

    return translation;
}

double* svlSampleTransform3D::GetDoublePointer()
{
    return Matrix.Pointer();
}

const double* svlSampleTransform3D::GetDoublePointer() const
{
    return Matrix.Pointer();
}

void svlSampleTransform3D::Identity()
{
    Matrix = vct4x4::Eye();
}


/******************************/
/*** svlSampleTargets class ***/
/******************************/

CMN_IMPLEMENT_SERVICES(svlSampleTargets)

svlSampleTargets::svlSampleTargets() :
    svlSample(),
    Channels(0),
    Dimensions(0)
{
}

svlSampleTargets::svlSampleTargets(const svlSampleTargets & targets) :
    svlSample()
{
    CopyOf(targets);
}

svlSample* svlSampleTargets::GetNewInstance() const
{
    return new svlSampleTargets;
}

svlStreamType svlSampleTargets::GetType() const
{
    return svlTypeTargets;
}

int svlSampleTargets::SetSize(const svlSample* sample)
{
    const svlSampleTargets* targets = dynamic_cast<const svlSampleTargets*>(sample);
    if (targets == 0) return SVL_FAIL;

    Channels = targets->GetChannels();
    Dimensions = targets->GetDimensions();
    Matrix.SetSize(1 + Channels * (1 + Dimensions), targets->GetMaxTargets());

    return SVL_OK;
}

int svlSampleTargets::SetSize(const svlSample& sample)
{
    const svlSampleTargets* targets = dynamic_cast<const svlSampleTargets*>(&sample);
    if (targets == 0) return SVL_FAIL;
    
    Channels = targets->GetChannels();
    Dimensions = targets->GetDimensions();
    Matrix.SetSize(1 + Channels * (1 + Dimensions), targets->GetMaxTargets());
    
    return SVL_OK;
}

int svlSampleTargets::CopyOf(const svlSample* sample)
{
    if (SetSize(sample) != SVL_OK) return SVL_FAIL;
    memcpy(GetUCharPointer(), sample->GetUCharPointer(), GetDataSize());
    SetTimestamp(sample->GetTimestamp());
    return SVL_OK;
}

int svlSampleTargets::CopyOf(const svlSample& sample)
{
    if (SetSize(sample) != SVL_OK) return SVL_FAIL;
    memcpy(GetUCharPointer(), sample.GetUCharPointer(), GetDataSize());
    SetTimestamp(sample.GetTimestamp());
    return SVL_OK;
}

bool svlSampleTargets::IsInitialized() const
{
    return true;
}

unsigned char* svlSampleTargets::GetUCharPointer()
{
    return reinterpret_cast<unsigned char*>(Matrix.Pointer());
}

const unsigned char* svlSampleTargets::GetUCharPointer() const
{
    return reinterpret_cast<const unsigned char*>(Matrix.Pointer());
}

unsigned int svlSampleTargets::GetDataSize() const
{
    return (Matrix.size() * sizeof(int));
}

void svlSampleTargets::SerializeRaw(std::ostream & outputStream) const
{
    cmnSerializeRaw(outputStream, Channels);
    cmnSerializeRaw(outputStream, Dimensions);
    cmnSerializeRaw(outputStream, Matrix);
}

void svlSampleTargets::DeSerializeRaw(std::istream & inputStream)
{
    cmnDeSerializeRaw(inputStream, Channels);
    cmnDeSerializeRaw(inputStream, Dimensions);
    cmnDeSerializeRaw(inputStream, Matrix);
    if (Matrix.rows() != (1 + Channels * (1 + Dimensions))) {
        // Error
        Channels = 0;
        Dimensions = 0;
        Matrix.SetSize(0, 0);
    }
}

svlSampleTargets::svlSampleTargets(unsigned int dimensions, unsigned int maxtargets, unsigned int channels) :
    svlSample(),
    Channels(channels),
    Dimensions(dimensions)
{
    SetSize(Dimensions, maxtargets, channels);
    ResetTargets();
}

void svlSampleTargets::SetSize(unsigned int dimensions, unsigned int maxtargets, unsigned int channels)
{
    Dimensions = dimensions;
    Channels = channels;
    Matrix.SetSize(1 + channels * (1 + dimensions), maxtargets);
}

void svlSampleTargets::SetDimensions(unsigned int dimensions)
{
    Dimensions = dimensions;
    SetSize(dimensions, Matrix.cols(), Channels);
}

unsigned int svlSampleTargets::GetDimensions() const
{
    return Dimensions;
}

void svlSampleTargets::SetMaxTargets(unsigned int maxtargets)
{
    SetSize(Dimensions, maxtargets, Channels);
}

unsigned int svlSampleTargets::GetMaxTargets() const
{
    return Matrix.cols();
}

void svlSampleTargets::SetChannels(unsigned int channels)
{
    Channels = channels;
    SetSize(Dimensions, Matrix.cols(), channels);
}

unsigned int svlSampleTargets::GetChannels() const
{
    return Channels;
}

vctDynamicVectorRef<int> svlSampleTargets::GetFlagVectorRef()
{
    return Matrix.Row(0);
}

const vctDynamicConstVectorRef<int> svlSampleTargets::GetFlagVectorRef() const
{
    return Matrix.Row(0);
}

vctDynamicVectorRef<int> svlSampleTargets::GetConfidenceVectorRef(unsigned int channel)
{
    return Matrix.Row(1 + channel * (1 + Dimensions));
}

const vctDynamicConstVectorRef<int> svlSampleTargets::GetConfidenceVectorRef(unsigned int channel) const
{
    return Matrix.Row(1 + channel * (1 + Dimensions));
}

vctDynamicMatrixRef<int> svlSampleTargets::GetPositionMatrixRef(unsigned int channel)
{
    return vctDynamicMatrixRef<int>(Matrix, 1 + channel * (1 + Dimensions) + 1, 0, Dimensions, Matrix.cols());
}

const vctDynamicConstMatrixRef<int> svlSampleTargets::GetPositionMatrixRef(unsigned int channel) const
{
    return vctDynamicConstMatrixRef<int>(Matrix, 1 + channel * (1 + Dimensions) + 1, 0, Dimensions, Matrix.cols());
}

int* svlSampleTargets::GetFlagPointer()
{
    return GetFlagVectorRef().Pointer();
}

const int* svlSampleTargets::GetFlagPointer() const
{
    return GetFlagVectorRef().Pointer();
}

int* svlSampleTargets::GetConfidencePointer(unsigned int channel)
{
    return GetConfidenceVectorRef(channel).Pointer();
}

const int* svlSampleTargets::GetConfidencePointer(unsigned int channel) const
{
    return GetConfidenceVectorRef(channel).Pointer();
}

int* svlSampleTargets::GetPositionPointer(unsigned int channel)
{
    return GetPositionMatrixRef(channel).Pointer();
}

const int* svlSampleTargets::GetPositionPointer(unsigned int channel) const
{
    return GetPositionMatrixRef(channel).Pointer();
}

void svlSampleTargets::ResetTargets()
{
    memset(Matrix.Pointer(), 0, GetDataSize());
}

void svlSampleTargets::SetFlag(unsigned int targetid, int value)
{
    if (targetid < Matrix.cols() && Dimensions > 0) GetFlagVectorRef().Element(targetid) = value;
}

int svlSampleTargets::GetFlag(unsigned int targetid) const
{
    if (targetid >= Matrix.cols() || Dimensions < 1) return SVL_FAIL;
    return GetFlagVectorRef().Element(targetid);
}

void svlSampleTargets::SetConfidence(unsigned int targetid, int value, unsigned int channel)
{
    if (targetid >= Matrix.cols() || Dimensions < 1 || channel >= Channels) return;
    GetConfidenceVectorRef(channel).Element(targetid) = value;
}

int svlSampleTargets::GetConfidence(unsigned int targetid, unsigned int channel) const
{
    if (targetid >= Matrix.cols() || Dimensions < 1 || channel >= Channels) return SVL_FAIL;
    return GetConfidenceVectorRef(channel).Element(targetid);
}

void svlSampleTargets::SetPosition(unsigned int targetid, const vctInt2& value, unsigned int channel)
{
    if (targetid >= Matrix.cols() || Dimensions != 2 || channel >= Channels) return;
    
    int* ptr = Matrix.Pointer(2 + channel * 3, targetid);
    
    *ptr = value[0]; ptr += Matrix.cols();
    *ptr = value[1];
}

void svlSampleTargets::SetPosition(unsigned int targetid, const vctInt3& value, unsigned int channel)
{
    if (targetid >= Matrix.cols() || Dimensions != 3 || channel >= Channels) return;
    
    int* ptr = Matrix.Pointer(2 + channel * 4, targetid);
    const unsigned int stride = Matrix.cols();
    
    *ptr = value[0]; ptr += stride;
    *ptr = value[1]; ptr += stride;
    *ptr = value[2];
}

int svlSampleTargets::GetPosition(unsigned int targetid, vctInt2& value, unsigned int channel) const
{
    if (targetid >= Matrix.cols() || Dimensions != 2 || channel >= Channels) return SVL_FAIL;
    
    const int* ptr = Matrix.Pointer(2 + channel * 3, targetid);
    
    value[0] = ptr[0];
    value[1] = ptr[Matrix.cols()];
    
    return SVL_OK;
}

int svlSampleTargets::GetPosition(unsigned int targetid, vctInt3& value, unsigned int channel) const
{
    if (targetid >= Matrix.cols() || Dimensions != 3 || channel >= Channels) return SVL_FAIL;
    
    const int* ptr = Matrix.Pointer(2 + channel * 4, targetid);
    unsigned int stride = Matrix.cols();
    
    value[0] = ptr[0];
    value[1] = ptr[stride]; stride <<= 1;
    value[2] = ptr[stride];
    
    return SVL_OK;
}


/***************************/
/*** svlSampleText class ***/
/***************************/

CMN_IMPLEMENT_SERVICES(svlSampleText)

svlSampleText::svlSampleText() : svlSample()
{
}

svlSample* svlSampleText::GetNewInstance() const
{
    return new svlSampleText;
}

svlStreamType svlSampleText::GetType() const
{
    return svlTypeText;
}

int svlSampleText::SetSize(const svlSample* sample)
{
    const svlSampleText* text = dynamic_cast<const svlSampleText*>(sample);
    if (text == 0) return SVL_FAIL;

    String.resize(text->GetSize());

    return SVL_OK;
}

int svlSampleText::SetSize(const svlSample& sample)
{
    const svlSampleText* text = dynamic_cast<const svlSampleText*>(&sample);
    if (text == 0) return SVL_FAIL;
    
    String.resize(text->GetSize());
    
    return SVL_OK;
}

int svlSampleText::CopyOf(const svlSample* sample)
{
    const svlSampleText* text = dynamic_cast<const svlSampleText*>(sample);
    if (text == 0) return SVL_FAIL;
    
    String.assign(text->GetStringRef());
    
    return SVL_OK;
}

int svlSampleText::CopyOf(const svlSample& sample)
{
    const svlSampleText* text = dynamic_cast<const svlSampleText*>(&sample);
    if (text == 0) return SVL_FAIL;

    String.assign(text->GetStringRef());

    return SVL_OK;
}

bool svlSampleText::IsInitialized() const
{
    return true;
}

unsigned char* svlSampleText::GetUCharPointer()
{
    return reinterpret_cast<unsigned char*>(const_cast<char*>(String.c_str()));
}

const unsigned char* svlSampleText::GetUCharPointer() const
{
    return reinterpret_cast<const unsigned char*>(String.c_str());
}

unsigned int svlSampleText::GetDataSize() const
{
    return String.size();
}

void svlSampleText::SerializeRaw(std::ostream & outputStream) const
{
    cmnSerializeRaw(outputStream, String);
}

void svlSampleText::DeSerializeRaw(std::istream & inputStream)
{
    cmnDeSerializeRaw(inputStream, String);
}

svlSampleText::svlSampleText(const std::string & text) :
    svlSample(),
    String(text)
{
}

void svlSampleText::SetText(const std::string & text)
{
    String = text;
}

std::string & svlSampleText::GetStringRef()
{
    return String;
}

const std::string & svlSampleText::GetStringRef() const
{
    return String;
}

char* svlSampleText::GetCharPointer()
{
    return const_cast<char*>(String.c_str());
}

const char* svlSampleText::GetCharPointer() const
{
    return String.c_str();
}

unsigned int svlSampleText::GetSize() const
{
    return String.size();
}

unsigned int svlSampleText::GetLength() const
{
    return String.size();
}


/*********************/
/*** svlRect class ***/
/*********************/

svlRect::svlRect() :
    left(0),
    top(0),
    right(0),
    bottom(0)
{
}

svlRect::svlRect(int left, int top, int right, int bottom) :
    left(left),
    top(top),
    right(right),
    bottom(bottom)
{
}

void svlRect::Assign(const svlRect & rect)
{
    left = rect.left;
    top = rect.top;
    right = rect.right;
    bottom = rect.bottom;
}

void svlRect::Assign(int left, int top, int right, int bottom)
{
    this->left = left;
    this->top = top;
    this->right = right;
    this->bottom = bottom;
}

void svlRect::Normalize()
{
    int temp;
    if (left > right) {
        temp = right;
        right = left;
        left = temp;
    }
    if (top > bottom) {
        temp = bottom;
        bottom = top;
        top = temp;
    }
}

void svlRect::Trim(int minx, int maxx, int miny, int maxy)
{
    if (left < minx) left = minx;
    if (left > maxx) left = maxx;
    if (right < minx) right = minx;
    if (right > maxx) right = maxx;
    if (top < miny) top = miny;
    if (top > maxy) top = maxy;
    if (bottom < miny) bottom = miny;
    if (bottom > maxy) bottom = maxy;
}


/************************/
/*** svlPoint2D class ***/
/************************/

svlPoint2D::svlPoint2D() :
    x(0),
    y(0)
{
}

svlPoint2D::svlPoint2D(int x, int y)
{
    svlPoint2D::x = x;
    svlPoint2D::y = y;
}

void svlPoint2D::Assign(const svlPoint2D & point)
{
    x = point.x;
    y = point.y;
}

void svlPoint2D::Assign(int x, int y)
{
    svlPoint2D::x = x;
    svlPoint2D::y = y;
}


/*************************/
/*** svlTarget2D class ***/
/*************************/

svlTarget2D::svlTarget2D() :
    used(false),
    visible(false),
    conf(0),
    pos(0, 0)
{
}

svlTarget2D::svlTarget2D(bool used, bool visible, unsigned char conf, int x, int y)
{
    svlTarget2D::used    = used;
    svlTarget2D::visible = visible;
    svlTarget2D::conf    = conf;
    pos.Assign(x, y);
}

svlTarget2D::svlTarget2D(bool used, bool visible, unsigned char conf, svlPoint2D & pos)
{
    svlTarget2D::used    = used;
    svlTarget2D::visible = visible;
    svlTarget2D::conf    = conf;
    svlTarget2D::pos     = pos;
}

svlTarget2D::svlTarget2D(int x, int y)
{
    used    = true;
    visible = true;
    conf    = 255;
    pos.Assign(x, y);
}

svlTarget2D::svlTarget2D(svlPoint2D & pos)
{
    used             = true;
    visible          = true;
    conf             = 255;
    svlTarget2D::pos = pos;
}

void svlTarget2D::Assign(const svlTarget2D & target)
{
    used    = target.used;
    visible = target.visible;
    conf    = target.conf;
    pos     = target.pos;
}

void svlTarget2D::Assign(bool used, bool visible, unsigned char conf, int x, int y)
{
    svlTarget2D::used    = used;
    svlTarget2D::visible = visible;
    svlTarget2D::conf    = conf;
    pos.Assign(x, y);
}

void svlTarget2D::Assign(bool used, bool visible, unsigned char conf, svlPoint2D & pos)
{
    svlTarget2D::used    = used;
    svlTarget2D::visible = visible;
    svlTarget2D::conf    = conf;
    svlTarget2D::pos     = pos;
}

void svlTarget2D::Assign(int x, int y)
{
    used    = true;
    visible = true;
    conf    = 255;
    pos.Assign(x, y);
}

void svlTarget2D::Assign(svlPoint2D & pos)
{
    used             = true;
    visible          = true;
    conf             = 255;
    svlTarget2D::pos = pos;
}


/*************************/
/*** svlRGB class ********/
/*************************/

svlRGB::svlRGB() :
    b(0), g(0), r(0)
{
}

svlRGB::svlRGB(unsigned char r_, unsigned char g_, unsigned char b_) :
    b(b_), g(g_), r(r_)
{
}

void svlRGB::Assign(const svlRGB & color)
{
    r = color.r;
    g = color.g;
    b = color.b;
}

void svlRGB::Assign(unsigned char r_, unsigned char g_, unsigned char b_)
{
    r = r_;
    g = g_;
    b = b_;
}


/*************************/
/*** svlRGBA class *******/
/*************************/

svlRGBA::svlRGBA() :
    b(0), g(0), r(0), a(0)
{
}

svlRGBA::svlRGBA(const svlRGB & rgb, unsigned char a_) :
    a(a_)
{
    r = rgb.r;
    g = rgb.g;
    b = rgb.b;
}

svlRGBA::svlRGBA(unsigned char r_, unsigned char g_, unsigned char b_, unsigned char a_) :
    b(b_), g(g_), r(r_), a(a_)
{
}

void svlRGBA::Assign(const svlRGBA & color)
{
    r = color.r;
    g = color.g;
    b = color.b;
    a = color.a;
}

void svlRGBA::Assign(const svlRGB & rgb, unsigned char a_)
{
    r = rgb.r;
    g = rgb.g;
    b = rgb.b;
    a = a_;
}

void svlRGBA::Assign(unsigned char r_, unsigned char g_, unsigned char b_, unsigned char a_)
{
    r = r_;
    g = g_;
    b = b_;
    a = a_;
}

