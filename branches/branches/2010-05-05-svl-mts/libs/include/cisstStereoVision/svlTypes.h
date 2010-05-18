/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: svlTypes.h 1212 2010-02-19 01:47:33Z bvagvol1 $
  
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

#ifndef _svlTypes_h
#define _svlTypes_h

#include <cisstCommon/cmnLogger.h>
#include <cisstVector/vctDynamicMatrixTypes.h>
#include <cisstVector/vctFixedSizeMatrixTypes.h>
#include <cisstVector/vctFixedSizeVectorTypes.h>
#include <cisstVector/vctTransformationTypes.h>
#include <cisstStereoVision/svlDefinitions.h>
#include <cisstStereoVision/svlImageIO.h>
#include <cisstStereoVision/svlConfig.h>

// Always include last!
#include <cisstStereoVision/svlExport.h>


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


#if (CISST_SVL_HAS_OPENCV == ON)
    #if (CISST_OS == CISST_WINDOWS) || (CISST_OS == CISST_DARWIN)
        #include <cv.h>
        #include <highgui.h>
    #else
        #include <opencv/cv.h>
        #include <opencv/highgui.h>
    #endif
#else // CISST_SVL_HAS_OPENCV
    typedef void IplImage;
#endif // CISST_SVL_HAS_OPENCV


//////////////////////////
// Forward declarations //
//////////////////////////

class osaCriticalSection;
class svlSyncPoint;
struct svlPoint2D;
struct svlTarget2D;


//////////////////////////////
// Stream type enumerations //
//////////////////////////////

enum svlStreamType
{
     svlTypeInvalid           // Default in base class
    ,svlTypeStreamSource      // Capture sources have an input connector of this type
    ,svlTypeStreamSink        // Render filters may have an output connector of this type
    ,svlTypeImageRGB          // Single RGB image
    ,svlTypeImageRGBA         // Single RGBA image
    ,svlTypeImageRGBStereo    // Dual RGB image
    ,svlTypeImageRGBAStereo   // Dual RGBA image
    ,svlTypeImageMono8        // Single Grayscale image (8bpp)
    ,svlTypeImageMono8Stereo  // Dual Grayscale image (8bpp)
    ,svlTypeImageMono16       // Single Grayscale image (16bpp)
    ,svlTypeImageMono16Stereo // Dual Grayscale image (16bpp)
    ,svlTypeImageMonoFloat    // Single float image (32bpp)
    ,svlTypeImage3DMap        // Three floats per pixel for storing 3D coordinates
    ,svlTypeImageCustom       // Custom, un-enumerated image format
    ,svlTypeTransform3D       // 3D transformation
    ,svlTypeTargets           // Vector of N dimensional points
    ,svlTypeText              // Textual data
};


////////////////////////////////
// Stereo layout enumerations //
////////////////////////////////

enum svlStereoLayout
{
     svlLayoutInterlaced
    ,svlLayoutInterlacedRL
    ,svlLayoutSideBySide
    ,svlLayoutSideBySideRL
};


/////////////////////////////////////////
// Kernel matching metric enumerations //
/////////////////////////////////////////

enum svlErrorMetric
{
     svlSAD
    ,svlSSD
    ,svlWSSD
    ,svlNCC
};


////////////////////////////
// Image type definitions //
////////////////////////////

typedef vctDynamicMatrix<unsigned char> svlImageMono8;
typedef vctDynamicMatrix<unsigned short> svlImageMono16;
typedef vctDynamicMatrix<float> svlImageMonoFloat;
typedef vctDynamicMatrix<unsigned char> svlImageRGB;
typedef vctDynamicMatrix<unsigned char> svlImageRGBA;


////////////////////////////////////
// Type checking helper functions //
////////////////////////////////////

template <class __ValueType>
static bool IsTypeFloat(__ValueType CMN_UNUSED(val)) { return false; }
template <>
inline bool IsTypeFloat<float>(float CMN_UNUSED(val)) { return true; }

template <class __ValueType>
static bool IsTypeUChar(__ValueType CMN_UNUSED(val)) { return false; }
template <>
inline bool IsTypeUChar<unsigned char>(unsigned char CMN_UNUSED(val)) { return true; }

template <class __ValueType>
static bool IsTypeUWord(__ValueType CMN_UNUSED(val)) { return false; }
template <>
inline bool IsTypeUWord<unsigned short>(unsigned short CMN_UNUSED(val)) { return true; }


///////////////////////////////////////
// Stream data structure definitions //
///////////////////////////////////////

class CISST_EXPORT svlSample : public cmnGenericObject
{
public:
    svlSample();
    virtual ~svlSample();
    virtual svlSample* GetNewInstance() const = 0;
    virtual svlStreamType GetType() const = 0;
    virtual int SetSize(const svlSample* sample) = 0;
    virtual int SetSize(const svlSample& sample) = 0;
    virtual int CopyOf(const svlSample* sample) = 0;
    virtual int CopyOf(const svlSample& sample) = 0;
    virtual bool IsImage() const;
    virtual bool IsInitialized() const;
    virtual unsigned char* GetUCharPointer() = 0;
    virtual const unsigned char* GetUCharPointer() const = 0;
    virtual unsigned int GetDataSize() const = 0;
    virtual void SerializeRaw(std::ostream & outputStream) const = 0;
    virtual void DeSerializeRaw(std::istream & inputStream) = 0;

public:
    void SetTimestamp(double ts);
    double GetTimestamp() const;
    static svlSample* GetNewFromType(svlStreamType type);
    void SetEncoder(const std::string & codec, const int parameter);
    void GetEncoder(std::string & codec, int & parameter) const;

private:
    double Timestamp; // [seconds]
    std::string Encoder;
    int EncoderParameter;
};


class CISST_EXPORT svlSampleImage : public svlSample
{
public:
    svlSampleImage();
    virtual ~svlSampleImage();

    virtual svlSample* GetNewInstance() const = 0;
    virtual svlStreamType GetType() const = 0;
    virtual int SetSize(const svlSample* sample) = 0;
    virtual int SetSize(const svlSample& sample) = 0;
    virtual int CopyOf(const svlSample* sample) = 0;
    virtual int CopyOf(const svlSample& sample) = 0;
    virtual bool IsImage() const;
    virtual bool IsInitialized() const = 0;
    virtual unsigned char* GetUCharPointer() = 0;
    virtual const unsigned char* GetUCharPointer() const = 0;
    virtual unsigned int GetDataSize() const = 0;
    virtual void SerializeRaw(std::ostream & outputStream) const = 0;
    virtual void DeSerializeRaw(std::istream & inputStream) = 0;

    virtual IplImage* IplImageRef(const unsigned int videochannel = 0) const = 0;
    virtual unsigned char* GetUCharPointer(const unsigned int videochannel) = 0;
    virtual const unsigned char* GetUCharPointer(const unsigned int videochannel) const = 0;
    virtual unsigned char* GetUCharPointer(const unsigned int videochannel, const unsigned int x, const unsigned int y) = 0;
    virtual const unsigned char* GetUCharPointer(const unsigned int videochannel, const unsigned int x, const unsigned int y) const = 0;
    virtual void SetSize(const unsigned int width, const unsigned int height) = 0;
    virtual void SetSize(const unsigned int videochannel, const unsigned int width, const unsigned int height) = 0;
    virtual unsigned int GetVideoChannels() const = 0;
    virtual unsigned int GetDataChannels() const = 0;
    virtual unsigned int GetBPP() const = 0;
    virtual unsigned int GetWidth(const unsigned int videochannel = 0) const = 0;
    virtual unsigned int GetHeight(const unsigned int videochannel = 0) const = 0;
    virtual unsigned int GetDataSize(const unsigned int videochannel) const = 0;
};


template <class _ValueType, unsigned int _DataChannels, unsigned int _VideoChannels>
class CISST_EXPORT svlSampleImageCustom : public svlSampleImage
{
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:

    //////////////////
    // Constructors //
    //////////////////

    svlSampleImageCustom() :
        svlSampleImage(),
        OwnData(true)
    {
        for (unsigned int vch = 0; vch < _VideoChannels; vch ++) {
            Image[vch] = new vctDynamicMatrix<_ValueType>;
#if (CISST_SVL_HAS_OPENCV == ON)
            int ocvdepth = GetOCVDepth();
            if (ocvdepth >= 0) OCVImageHeader[vch] = cvCreateImageHeader(cvSize(0, 0), ocvdepth, _DataChannels);
            else OCVImageHeader[vch] = 0;
#endif // CISST_SVL_HAS_OPENCV
        }
    }

    svlSampleImageCustom(bool owndata) :
        svlSampleImage(),
        OwnData(owndata)
    {
        for (unsigned int vch = 0; vch < _VideoChannels; vch ++) {
            if (OwnData) {
                Image[vch] = new vctDynamicMatrix<_ValueType>;
            }
            else Image[vch] = 0;
#if (CISST_SVL_HAS_OPENCV == ON)
            int ocvdepth = GetOCVDepth();
            if (ocvdepth >= 0) OCVImageHeader[vch] = cvCreateImageHeader(cvSize(0, 0), ocvdepth, _DataChannels);
            else OCVImageHeader[vch] = 0;
#endif // CISST_SVL_HAS_OPENCV
        }
    }


    ////////////////
    // Destructor //
    ////////////////

    ~svlSampleImageCustom()
    {
        for (unsigned int vch = 0; vch < _VideoChannels; vch ++) {
            if (OwnData) delete Image[vch];
#if (CISST_SVL_HAS_OPENCV == ON)
            if (OCVImageHeader[vch]) cvReleaseImageHeader(&(OCVImageHeader[vch]));
#endif // CISST_SVL_HAS_OPENCV
        }
    }


    //////////////////////////////
    // Inherited from svlSample //
    //////////////////////////////

    svlSample* GetNewInstance() const
    {
        return new svlSampleImageCustom<_ValueType, _DataChannels, _VideoChannels>;
    }

    svlStreamType GetType() const
    {
        if (IsTypeUChar<_ValueType>(static_cast<_ValueType>(0))) {
            if (_DataChannels == 1) {
                if (_VideoChannels == 1) return svlTypeImageMono8;
                if (_VideoChannels == 2) return svlTypeImageMono8Stereo;
            }
            if (_DataChannels == 3) {
                if (_VideoChannels == 1) return svlTypeImageRGB;
                if (_VideoChannels == 2) return svlTypeImageRGBStereo;
            }
            if (_DataChannels == 4) {
                if (_VideoChannels == 1) return svlTypeImageRGBA;
                if (_VideoChannels == 2) return svlTypeImageRGBAStereo;
            }
        }
        if (IsTypeUWord<_ValueType>(static_cast<_ValueType>(0))) {
            if (_DataChannels == 1) {
                if (_VideoChannels == 1) return svlTypeImageMono16;
                if (_VideoChannels == 2) return svlTypeImageMono16Stereo;
            }
        }
        if (IsTypeFloat<_ValueType>(static_cast<_ValueType>(0))) {
            if (_DataChannels == 1 && _VideoChannels == 1) return svlTypeImageMonoFloat;
            if (_DataChannels == 3 && _VideoChannels == 1) return svlTypeImage3DMap;
        }
        return svlTypeImageCustom;
    }

    int SetSize(const svlSample* sample)
    {
        const svlSampleImage* sampleimage = dynamic_cast<const svlSampleImage*>(sample);
        if (sampleimage == 0) return SVL_FAIL;
        unsigned int samplevideochannels = sampleimage->GetVideoChannels();
        for (unsigned int vch = 0; vch < _VideoChannels && vch < samplevideochannels; vch ++) {
            SetSize(vch, sampleimage->GetWidth(vch), sampleimage->GetHeight(vch));
        }
        return SVL_OK;
    }
    
    int SetSize(const svlSample& sample)
    {
        const svlSampleImage* sampleimage = dynamic_cast<const svlSampleImage*>(&sample);
        if (sampleimage == 0) return SVL_FAIL;
        unsigned int samplevideochannels = sampleimage->GetVideoChannels();
        for (unsigned int vch = 0; vch < _VideoChannels && vch < samplevideochannels; vch ++) {
            SetSize(vch, sampleimage->GetWidth(vch), sampleimage->GetHeight(vch));
        }
        return SVL_OK;
    }

    int CopyOf(const svlSample* sample)
    {
        if (!sample) return SVL_FAIL;
        if (sample->GetType() != GetType() || SetSize(sample) != SVL_OK) return SVL_FAIL;

        const svlSampleImage* sampleimage = dynamic_cast<const svlSampleImage*>(sample);
        for (unsigned int vch = 0; vch < _VideoChannels; vch ++) {
            memcpy(GetUCharPointer(vch), sampleimage->GetUCharPointer(vch), GetDataSize(vch));
        }
        SetTimestamp(sample->GetTimestamp());

        return SVL_OK;
    }

    int CopyOf(const svlSample& sample)
    {
        if (sample.GetType() != GetType() || SetSize(sample) != SVL_OK) return SVL_FAIL;

        const svlSampleImage* sampleimage = dynamic_cast<const svlSampleImage*>(&sample);
        for (unsigned int vch = 0; vch < _VideoChannels; vch ++) {
            memcpy(GetUCharPointer(vch), sampleimage->GetUCharPointer(vch), GetDataSize(vch));
        }
        SetTimestamp(sample.GetTimestamp());

        return SVL_OK;
    }

    bool IsImage() const
    {
        return true;
    }

    bool IsInitialized() const
    {
        for (unsigned int vch = 0; vch < _VideoChannels; vch ++) {
            if (Image[vch] == 0 ||
                Image[vch]->width() < _DataChannels ||
                Image[vch]->height() < 1) return false;
        }
        return true;
    }

    unsigned char* GetUCharPointer()
    {
        return reinterpret_cast<unsigned char*>(GetPointer());
    }

    const unsigned char* GetUCharPointer() const
    {
        return reinterpret_cast<const unsigned char*>(GetPointer());
    }

    unsigned int GetDataSize() const
    {
        return GetDataSize(0);
    }

    virtual void SerializeRaw(std::ostream & outputStream) const
    {
        std::string codec;
        int compression;
        GetEncoder(codec, compression);
        cmnSerializeRaw(outputStream, GetType());
        cmnSerializeRaw(outputStream, GetTimestamp());
        cmnSerializeRaw(outputStream, codec);
        for (unsigned int vch = 0; vch < _VideoChannels; vch ++) {
            if (svlImageIO::Write(*this, vch, codec, outputStream, compression) != SVL_OK) {
                cmnThrow("svlSampleImageCustom::SerializeRaw(): Error occured with svlImageIO::Write");
            }
        }
    }

    virtual void DeSerializeRaw(std::istream & inputStream)
    {
        int type = -1;
        double timestamp;
        std::string codec;
        cmnDeSerializeRaw(inputStream, type);
        if (type != GetType()) {
            CMN_LOG_CLASS_RUN_ERROR << "Deserialized sample type mismatch " << std::endl;
            return;
        }
        cmnDeSerializeRaw(inputStream, timestamp);
        SetTimestamp(timestamp);
        cmnDeSerializeRaw(inputStream, codec);
        for (unsigned int vch = 0; vch < _VideoChannels; vch ++) {
            if (svlImageIO::Read(*this, vch, codec, inputStream, false) != SVL_OK) {
                cmnThrow("svlSampleImageCustom::DeSerializeRaw(): Error occured with svlImageIO::Read");
            }
        }
    }

    ///////////////////////////////////
    // Inherited from svlSampleImage //
    ///////////////////////////////////

#if (CISST_SVL_HAS_OPENCV == ON)
    IplImage* IplImageRef(const unsigned int videochannel = 0) const
#else // CISST_SVL_HAS_OPENCV
    IplImage* IplImageRef(const unsigned int CMN_UNUSED(videochannel) = 0) const
#endif // CISST_SVL_HAS_OPENCV
    {
#if (CISST_SVL_HAS_OPENCV == ON)
        if (videochannel < _VideoChannels) return OCVImageHeader[videochannel];
        else return 0;
#else // CISST_SVL_HAS_OPENCV
        CMN_LOG_INIT_WARNING << "Class svlSampleImageCustom: IplImageRef() called while OpenCV is disabled" << std::endl;
        return 0;
#endif // CISST_SVL_HAS_OPENCV
    }

    unsigned char* GetUCharPointer(const unsigned int videochannel)
    {
        return reinterpret_cast<unsigned char*>(GetPointer(videochannel));
    }

    const unsigned char* GetUCharPointer(const unsigned int videochannel) const
    {
        return reinterpret_cast<const unsigned char*>(GetPointer(videochannel));
    }

    unsigned char* GetUCharPointer(const unsigned int videochannel, const unsigned int x, const unsigned int y)
    {
        return reinterpret_cast<unsigned char*>(GetPointer(videochannel, x, y));
    }

    const unsigned char* GetUCharPointer(const unsigned int videochannel, const unsigned int x, const unsigned int y) const
    {
        return reinterpret_cast<const unsigned char*>(GetPointer(videochannel, x, y));
    }

    void SetSize(const unsigned int width, const unsigned int height)
    {
        for (unsigned int vch = 0; vch < _VideoChannels; vch ++) {
            SetSize(vch, width, height);
        }
    }

    void SetSize(const unsigned int videochannel, const unsigned int width, const unsigned int height)
    {
        if (videochannel < _VideoChannels && Image[videochannel]) {
            if (GetWidth (videochannel) == width &&
                GetHeight(videochannel) == height) return;
            Image[videochannel]->SetSize(height,  width * _DataChannels);
#if (CISST_SVL_HAS_OPENCV == ON)
            if (OCVImageHeader[videochannel]) {
                cvInitImageHeader(OCVImageHeader[videochannel],
                                  cvSize(width, height),
                                  GetOCVDepth(),
                                  _DataChannels);
                cvSetData(OCVImageHeader[videochannel],
                          GetPointer(videochannel),
                          width * GetBPP());
            }
#endif // CISST_SVL_HAS_OPENCV
        }
    }

    unsigned int GetVideoChannels() const
    {
        return _VideoChannels;
    }

    unsigned int GetDataChannels() const
    {
        return _DataChannels;
    }

    unsigned int GetBPP() const
    {
        return (sizeof(_ValueType) * _DataChannels);
    }

    unsigned int GetWidth(const unsigned int videochannel = 0) const
    {
        if (videochannel < _VideoChannels && Image[videochannel]) return (Image[videochannel]->width() / _DataChannels);
        return 0;
    }

    unsigned int GetHeight(const unsigned int videochannel = 0) const
    {
        if (videochannel < _VideoChannels && Image[videochannel]) return Image[videochannel]->height();
        return 0;
    }

    unsigned int GetDataSize(const unsigned int videochannel) const
    {
        if (videochannel < _VideoChannels && Image[videochannel]) {
            return (GetBPP() * GetWidth(videochannel) * GetHeight(videochannel));
        }
        return 0;
    }


    ///////////////////////////////////////////
    // svlSampleImageCustom specific methods //
    ///////////////////////////////////////////

    int SetMatrix(vctDynamicMatrix<_ValueType>* matrix, unsigned int videochannel = 0)
    {
        if (!OwnData && videochannel < _VideoChannels) {
            Image[videochannel] = matrix;
#if (CISST_SVL_HAS_OPENCV == ON)
            if (OCVImageHeader[videochannel]) {
                cvInitImageHeader(OCVImageHeader[videochannel],
                                  cvSize(GetWidth(videochannel), GetHeight(videochannel)),
                                  GetOCVDepth(),
                                  _DataChannels);
                cvSetData(OCVImageHeader[videochannel],
                          GetPointer(videochannel),
                          GetWidth(videochannel) * _DataChannels);
            }
#endif // CISST_SVL_HAS_OPENCV
            return SVL_OK;
        }
        return SVL_FAIL;
    }

    vctDynamicMatrix<_ValueType> & GetMatrixRef(const unsigned int videochannel = 0)
    {
        if (videochannel < _VideoChannels && Image[videochannel]) return *(Image[videochannel]);
        else return InvalidMatrix;
    }

    const vctDynamicMatrix<_ValueType> & GetMatrixRef(const unsigned int videochannel = 0) const
    {
        if (videochannel < _VideoChannels && Image[videochannel]) return *(Image[videochannel]);
        else return InvalidMatrix;
    }

    _ValueType* GetPointer(const unsigned int videochannel = 0)
    {
        if (videochannel < _VideoChannels && Image[videochannel]) return Image[videochannel]->Pointer();
        return 0;
    }

    const _ValueType* GetPointer(const unsigned int videochannel = 0) const
    {
        if (videochannel < _VideoChannels && Image[videochannel]) return Image[videochannel]->Pointer();
        return 0;
    }

    _ValueType* GetPointer(const unsigned int videochannel, const unsigned int x, const unsigned int y)
    {
        if (videochannel < _VideoChannels && Image[videochannel]) {
            return Image[videochannel]->Pointer(y, x * _DataChannels);
        }
        return 0;
    }

    const _ValueType* GetPointer(const unsigned int videochannel, const unsigned int x, const unsigned int y) const
    {
        if (videochannel < _VideoChannels && Image[videochannel]) {
            return Image[videochannel]->Pointer(y, x * _DataChannels);
        }
        return 0;
    }

private:
    bool OwnData;
    vctDynamicMatrix<_ValueType>* Image[_VideoChannels];
    vctDynamicMatrix<_ValueType> InvalidMatrix;

#if (CISST_SVL_HAS_OPENCV == ON)
    IplImage* OCVImageHeader[_VideoChannels];

    int GetOCVDepth()
    {
        if (IsTypeUChar<_ValueType>(static_cast<_ValueType>(0))) return IPL_DEPTH_8U;
        if (IsTypeUWord<_ValueType>(static_cast<_ValueType>(0))) return IPL_DEPTH_16U;
        if (IsTypeFloat<_ValueType>(static_cast<_ValueType>(0))) return IPL_DEPTH_32F;
        return -1;
    }
#endif // CISST_SVL_HAS_OPENCV
};

typedef svlSampleImageCustom<unsigned char,  1, 1>   svlSampleImageMono8;
typedef svlSampleImageCustom<unsigned char,  1, 2>   svlSampleImageMono8Stereo;
typedef svlSampleImageCustom<unsigned short, 1, 1>   svlSampleImageMono16;
typedef svlSampleImageCustom<unsigned short, 1, 2>   svlSampleImageMono16Stereo;
typedef svlSampleImageCustom<unsigned char,  3, 1>   svlSampleImageRGB;
typedef svlSampleImageCustom<unsigned char,  4, 1>   svlSampleImageRGBA;
typedef svlSampleImageCustom<unsigned char,  3, 2>   svlSampleImageRGBStereo;
typedef svlSampleImageCustom<unsigned char,  4, 2>   svlSampleImageRGBAStereo;
typedef svlSampleImageCustom<float,          1, 1>   svlSampleImageMonoFloat;
typedef svlSampleImageCustom<float,          3, 1>   svlSampleImage3DMap;

CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlSampleImageMono8)
CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlSampleImageMono8Stereo)
CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlSampleImageMono16)
CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlSampleImageMono16Stereo)
CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlSampleImageRGB)
CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlSampleImageRGBA)
CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlSampleImageRGBStereo)
CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlSampleImageRGBAStereo)
CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlSampleImageMonoFloat)
CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlSampleImage3DMap)


class CISST_EXPORT svlSampleTransform3D : public svlSample
{
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    svlSampleTransform3D();

    svlSample* GetNewInstance() const;
    svlStreamType GetType() const;
    int SetSize(const svlSample* sample);
    int SetSize(const svlSample& sample);
    int CopyOf(const svlSample* sample);
    int CopyOf(const svlSample& sample);
    bool IsInitialized() const;
    unsigned char* GetUCharPointer();
    const unsigned char* GetUCharPointer() const;
    unsigned int GetDataSize() const;
    void SerializeRaw(std::ostream & outputStream) const;
    void DeSerializeRaw(std::istream & inputStream);

    svlSampleTransform3D(const vct4x4 & matrix);
    svlSampleTransform3D(const vctFrm4x4 & frame);
    svlSampleTransform3D(const vctRot3 & rotation, const vct3 & translation);
    vct4x4 & GetMatrixRef();
    const vct4x4 & GetMatrixRef() const;
    vctFrm4x4 GetFrame() const;
    vctRot3 GetRotation() const;
    vct3 GetTranslation() const;
    double* GetDoublePointer();
    const double* GetDoublePointer() const;
    void Identity();

protected:
    vct4x4 Matrix;
};

CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlSampleTransform3D)


class CISST_EXPORT svlSampleTargets : public svlSample
{
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    svlSampleTargets();
    svlSampleTargets(const svlSampleTargets & targets);

    svlSample* GetNewInstance() const;
    svlStreamType GetType() const;
    int SetSize(const svlSample* sample);
    int SetSize(const svlSample& sample);
    int CopyOf(const svlSample* sample);
    int CopyOf(const svlSample& sample);
    bool IsInitialized() const;
    unsigned char* GetUCharPointer();
    const unsigned char* GetUCharPointer() const;
    unsigned int GetDataSize() const;
    void SerializeRaw(std::ostream & outputStream) const;
    void DeSerializeRaw(std::istream & inputStream);

    svlSampleTargets(unsigned int dimensions, unsigned int maxtargets, unsigned int channels);
    void SetSize(unsigned int dimensions, unsigned int maxtargets, unsigned int channels);
    void SetDimensions(unsigned int dimensions);
    unsigned int GetDimensions() const;
    void SetMaxTargets(unsigned int maxtargets);
    unsigned int GetMaxTargets() const;
    void SetChannels(unsigned int channels);
    unsigned int GetChannels() const;

    vctDynamicVectorRef<int> GetFlagVectorRef();
    const vctDynamicConstVectorRef<int> GetFlagVectorRef() const;
    vctDynamicVectorRef<int> GetConfidenceVectorRef(unsigned int channel = 0);
    const vctDynamicConstVectorRef<int> GetConfidenceVectorRef(unsigned int channel = 0) const;
    vctDynamicMatrixRef<int> GetPositionMatrixRef(unsigned int channel = 0);
    const vctDynamicConstMatrixRef<int> GetPositionMatrixRef(unsigned int channel = 0) const;
    int* GetFlagPointer();
    const int* GetFlagPointer() const;
    int* GetConfidencePointer(unsigned int channel = 0);
    const int* GetConfidencePointer(unsigned int channel = 0) const;
    int* GetPositionPointer(unsigned int channel = 0);
    const int* GetPositionPointer(unsigned int channel = 0) const;
    void ResetTargets();

    void SetFlag(unsigned int targetid, int value);
    int GetFlag(unsigned int targetid) const;
    void SetConfidence(unsigned int targetid, int value, unsigned int channel = 0);
    int GetConfidence(unsigned int targetid, unsigned int channel = 0) const;
    void SetPosition(unsigned int targetid, const vctDynamicVector<int> & value, unsigned int channel = 0);
    int GetPosition(unsigned int targetid, vctDynamicVector<int> & value, unsigned int channel = 0) const;

protected:
    unsigned int Channels;
    unsigned int Dimensions;
    vctDynamicMatrix<int> Matrix;
};

CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlSampleTargets)


class CISST_EXPORT svlSampleText : public svlSample
{
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    svlSampleText();
    svlSample* GetNewInstance() const;
    svlStreamType GetType() const;
    int SetSize(const svlSample* sample);
    int SetSize(const svlSample& sample);
    int CopyOf(const svlSample* sample);
    int CopyOf(const svlSample& sample);
    bool IsInitialized() const;
    unsigned char* GetUCharPointer();
    const unsigned char* GetUCharPointer() const;
    unsigned int GetDataSize() const;
    void SerializeRaw(std::ostream & outputStream) const;
    void DeSerializeRaw(std::istream & inputStream);

    svlSampleText(const std::string & text);
    void SetText(const std::string & text);
    std::string & GetStringRef();
    const std::string & GetStringRef() const;
    char* GetCharPointer();
    const char* GetCharPointer() const;
    unsigned int GetSize() const;
    unsigned int GetLength() const;

protected:
    std::string String;
};

CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlSampleText)


#pragma pack(1)

//////////////////////////////////////////////
// Miscellaneous structure type definitions //
//////////////////////////////////////////////

struct CISST_EXPORT svlProcInfo
{
    unsigned int        count;
    unsigned int        id;
    svlSyncPoint*       sync;
    osaCriticalSection* cs;
};

struct CISST_EXPORT svlRect
{
    svlRect();
    svlRect(int left, int top, int right, int bottom);
    void Assign(const svlRect & rect);
    void Assign(int left, int top, int right, int bottom);
    void Normalize();
    void Trim(const int minx, const int maxx, const int miny, const int maxy);

    int left;
    int top;
    int right;
    int bottom;
};

struct CISST_EXPORT svlPoint2D
{
    svlPoint2D();
    svlPoint2D(int x, int y);
    void Assign(const svlPoint2D & point);
    void Assign(int x, int y);
    
    int x;
    int y;
};

struct CISST_EXPORT svlTarget2D
{
    svlTarget2D();
    svlTarget2D(bool used, bool visible, unsigned char conf, int x, int y);
    svlTarget2D(bool used, bool visible, unsigned char conf, svlPoint2D & pos);
    svlTarget2D(int x, int y);
    svlTarget2D(svlPoint2D & pos);
    void Assign(const svlTarget2D & target);
    void Assign(bool used, bool visible, unsigned char conf, int x, int y);
    void Assign(bool used, bool visible, unsigned char conf, svlPoint2D & pos);
    void Assign(int x, int y);
    void Assign(svlPoint2D & pos);
    
    bool          used;
    bool          visible;
    unsigned char conf;
    svlPoint2D    pos;
};


/////////////////////////////////
// Image structure definitions //
/////////////////////////////////

struct CISST_EXPORT svlBMPFileHeader
{
    unsigned short bfType;
    unsigned int bfSize;
    unsigned short bfReserved1;
    unsigned short bfReserved2;
    unsigned int bfOffBits;
};

struct CISST_EXPORT svlDIBHeader
{
    unsigned int biSize;
    int biWidth;
    int biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int biCompression;
    unsigned int biSizeImage;
    int biXPelsPerMeter;
    int biYPelsPerMeter;
    unsigned int biClrUsed;
    unsigned int biClrImportant;
};

struct CISST_EXPORT svlRGB
{
    svlRGB();
    svlRGB(unsigned char r, unsigned char g, unsigned char b);
    void Assign(const svlRGB & color);
    void Assign(unsigned char r, unsigned char g, unsigned char b);

    unsigned char b;
    unsigned char g;
    unsigned char r;
};

struct CISST_EXPORT svlRGBA
{
    svlRGBA();
    svlRGBA(const svlRGB & rgb, unsigned char a);
    svlRGBA(unsigned char r, unsigned char g, unsigned char b, unsigned char a);
    void Assign(const svlRGBA & color);
    void Assign(const svlRGB & rgb, unsigned char a);
    void Assign(unsigned char r, unsigned char g, unsigned char b, unsigned char a);

    unsigned char b;
    unsigned char g;
    unsigned char r;
    unsigned char a;
};

#pragma pack()


#endif // _svlTypes_h

