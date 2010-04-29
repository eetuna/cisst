/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: svlVideoCodecUDP.h 1236 2010-02-26 20:38:21Z adeguet1 $
  
  Author(s):  Min Yang Jung
  Created on: 2010-03-06

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#ifndef _svlVideoCodecUDP_h
#define _svlVideoCodecUDP_h

#include <cisstCommon/cmnSerializer.h>
#include <cisstCommon/cmnDeSerializer.h>
#include <cisstOSAbstraction/osaTimeServer.h>
#include <cisstStereoVision/svlVideoIO.h>
#include <cisstStereoVision/svlStreamDefs.h>

#include <limits> // for statistics

//-------------------------------------------------------------------------
//  Constant Definitions
//-------------------------------------------------------------------------
/*! String to identify header correctly */
#define DELIMITER_STRING      "JHU_TELESURGERY_RESEARCH"
/*! Maximum length of the string above */
#define DELIMITER_STRING_SIZE 28
/*! Size of unit UDP message */
#define UNIT_MESSAGE_SIZE     1300
/*! Maximum number of subimages.  Equal to the number of worker thread(s) and 
    set by the number of processors available. */
#define MAX_SUBIMAGE_COUNT    32
/*! Total length of serialized cisst class service.  Assume that it does not 
    exceed 100 bytes. */
#define MAX_SERIALIZED_CISST_CLASS_SERVICE_SIZE 100

class svlVideoCodecUDP : public svlVideoCodecBase, public cmnGenericObject
{
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    svlVideoCodecUDP();
    virtual ~svlVideoCodecUDP();

    // Methods required by the base class (svlVideoIO.h)
    int GetBegPos() const { return 0; }
    int GetEndPos() const { return 0; }
    int GetPos() const    { return 0; }
    int SetPos(const int CMN_UNUSED(pos)) { return SVL_OK; }
    int SetCompression(const svlVideoIO::Compression * CMN_UNUSED(compression)) { return SVL_OK; }
    int DialogCompression() { return SVL_OK; }

    int Close();

    //-------------------------------------------------------------------------
    //  UDP Sender
    //-------------------------------------------------------------------------
    /*! Initialize UDP sender (e.g. create send socket) */
    int Create(const std::string &filename, const unsigned int width, const unsigned int height, const double framerate);

    /*! Serialize image object and send it to receiver across networks */
    int Write(svlProcInfo* procInfo, const svlSampleImageBase &image, const unsigned int videoch);

    //-------------------------------------------------------------------------
    //  UDP Receiver
    //-------------------------------------------------------------------------
    /*! Initialize UDP receiver (e.g. create client socket, determine image size) */
    int Open(const std::string &filename, unsigned int &width, unsigned int &height, double & framerate);

    /*! Deserialize byte stream image object to rebuild original image object */
    int Read(svlProcInfo* procInfo, svlSampleImageBase &image, const unsigned int videoch, const bool noresize = false);

protected:
    /*! Typedef for udp codec type */
    typedef enum { UDP_SENDER, UDP_RECEIVER } UDPCodecType;
    UDPCodecType CodecType;

    /*! Variables for image properties */
    unsigned int Width;
    unsigned int Height;

    /*! Timeserver. Used for timestamping packets */
    osaTimeServer TimeServer;

    /*! Flag to turn on or off udp message generation.  Used for testing 
        purposes (see StereoPlayerTest). */
    bool NetworkEnabled;

    //-------------------------------------------------------------------------
    //  Auxiliary class for statistics
    //-------------------------------------------------------------------------
    class Stat {
    protected:
        std::list<double> History;
        double Max;
        double Min;
        double Avg;
        double Sum;
        std::string Name;
        size_t MaxSampleSize;

    public:
        Stat(const std::string & name, const unsigned int maxSampleSize) 
            : Max(DBL_MIN), Min(DBL_MAX), Avg(0.0), Sum(0.0), Name(name), MaxSampleSize(maxSampleSize)
        {}

        void AddSample(const double sample) {
            if (History.size() == MaxSampleSize) {
                Clear();
            }

            History.push_back(sample);
            
            if (Max < sample) Max = sample;
            if (Min > sample) Min = sample;
            Sum += sample;
        }

        void Clear(void) {
            History.clear();
            Max = DBL_MIN;
            Min = DBL_MAX;
            Avg = 0.0;
            Sum = 0.0;
        }

        inline size_t GetSize(void) const {
            return History.size();
        }
        inline double GetAverage(void) const {
            return (Sum / (double) History.size());
        }
        inline double GetStd(void) const {
            double temp = 0.0;
            std::list<double>::const_iterator it = History.begin();
            const std::list<double>::const_iterator itEnd = History.end();
            for (; it != itEnd; ++it) {
                temp += (*it) * (*it);
            }
            temp /= (double) History.size();

            double avg = GetAverage();
            temp -= avg * avg;

            return sqrt(temp);
        }
        inline double GetMax(void) const {
            return Max;
        }
        inline double GetMin(void) const {
            return Min;
        }

        void Print(void) const {
            std::cout.setf(std::ios::fixed, std::ios::floatfield);
            std::cout.setf(std::ios::showpoint);
            std::cout.precision(3);
            std::cout << Name << " (" << (int) GetSize() << ") : avg " << GetAverage() << " (" << GetStd() << "), min " 
                << GetMin() << ", max " << GetMax() << std::endl;
        }
    };

    /*! Structure to contain the experiment results */
    typedef struct {
        unsigned int FrameNo;
        unsigned int FrameSize;
        unsigned int FPS;
        double TimeSerialization;
        double TimeDeSerialization;
        double TimeProcessing;
        double Timestamp;
    } ExperimentResultElement;

    typedef std::vector<ExperimentResultElement> ExperimentResultElementsType;
    ExperimentResultElementsType ExperimentResultElements;

    /*! FPS (Frame-per-second) */
    unsigned int FrameCountPerSecond;
    /*! Last time when FPS was calculated */
    double LastFPSTick;
    /*! Last time when delay was calculated */
    double LastDelayTick;
    /*! Instances of Stat class above */
    Stat * StatFPS;
    Stat * StatOverhead;
    Stat * StatDelay;

    /*! Write experiment results to log file */
    void ReportResults(void);

    //-------------------------------------------------------------------------
    //  Network (UDP) Support
    //-------------------------------------------------------------------------
    /*! Definition of MSG_HEADER message */
    class MSG_HEADER {
    public:
        /*! Frame sequence number */
        unsigned int FrameSeq;

        /*! Serialized cisst class services */
        char CisstClassService[MAX_SERIALIZED_CISST_CLASS_SERVICE_SIZE];
        char CisstClassServiceSize;

        /*! Image size */
        unsigned short Width;
        unsigned short Height;

        /*! Total number of subimages */
        char SubImageCount;
        /*! Subimage sizes */
        unsigned int SubImageSize[MAX_SUBIMAGE_COUNT];

        /*! Delimiter to differentiate MSG_HEADER and MSG_BODY (MJ: don't need 
            to edit this field) */
        char Delimiter[DELIMITER_STRING_SIZE];

        /*! Timestamp right before this message is sent to network */
        //double Timestamp;

        /*! Constructor */
        MSG_HEADER() : FrameSeq(0), CisstClassServiceSize(0), SubImageCount(0)
        {
            memset(CisstClassService, 0, MAX_SERIALIZED_CISST_CLASS_SERVICE_SIZE);
            memset(SubImageSize, 0, sizeof(unsigned int) * MAX_SUBIMAGE_COUNT);
            memset(Delimiter, 0, DELIMITER_STRING_SIZE);
            strncpy(Delimiter, DELIMITER_STRING, DELIMITER_STRING_SIZE);
        }

        void Print(void) {
            std::cout << "HEADER.FrameSeq         = " << FrameSeq << std::endl;
            std::cout << "HEADER.ClassServiceSize = " << (int) CisstClassServiceSize << std::endl;
            std::cout << "HEADER.SubImageCount    = " << (int) SubImageCount << std::endl;
            std::cout << "HEADER.SubImageSize     = ";
            for (int i = 0; i < SubImageCount; ++i) {
                std::cout << SubImageSize[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "HEADER.Delimiter        = " << Delimiter << std::endl;
        }
    };

    /*! Definition of MSG_PAYLOAD message */
    class MSG_PAYLOAD {
    public:
        /*! Frame sequence of an image that this payload belongs to */
        unsigned int FrameSeq;
        /*! Payload size */
        unsigned short PayloadSize;
        /*! Fragmented image data (with serialization) */
        char Payload[UNIT_MESSAGE_SIZE];
        /*! Timestamp right before this message is sent to network */
        //double Timestamp;
    };

    /*! UDP sockets */
    int SocketSend, SocketRecv;
    /*! Frame sequence number this is being transmitted now */
    unsigned int CurrentSeq;

    /*! Create send/recv sockets and send/receive threads */
    bool CreateSocket(const UDPCodecType type);
    /*! Get one image frame from network */
    unsigned int GetOneImage(double & senderTick);
    /*! Send one image frame to network */
    int SendUDP(void);
    /*! Cleanup resouces */
    void SocketCleanup(void);

    //-------------------------------------------------------------------------
    //  Image Compression and CISST (De)Serialization
    //-------------------------------------------------------------------------
    /*! CISST Serializer and Deserializer */
    cmnSerializer * Serializer;
    cmnDeSerializer * DeSerializer;

    /*! Parallel serialization using multiple CPU cores */
    unsigned int ProcessCount;
    char * SerializedClassService;
    unsigned int SerializedClassServiceSize;

    /*! Temporary buffer and its size */
    unsigned char* BufferYUV;
    unsigned char* BufferCompression;
    unsigned int BufferYUVSize;
    unsigned int BufferCompressionSize;

    /*! Offset and size of subimage */
    vctDynamicVector<unsigned int> SubImageOffset;
    vctDynamicVector<unsigned int> SubImageSize;

    /*! Deserialize byte stream image and rebuild original object */
    void DeSerialize(const std::string & serializedObject, cmnGenericObject & originalObject);
    cmnGenericObject * DeSerialize(const std::string & serializedObject);
};

CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlVideoCodecUDP)

#endif // _svlVideoCodecUDP_h

