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

#define DELIMITER_STRING      "JHU_TELESURGERY_RESEARCH"
#define DELIMITER_STRING_SIZE 28
#define UNIT_MESSAGE_SIZE     1300

class svlVideoCodecUDP : public svlVideoCodecBase, public cmnGenericObject
{
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

public:
    svlVideoCodecUDP();
    virtual ~svlVideoCodecUDP();

    // Methods definitions required by the base class (svlVideoIO.h)
    //-------------------------------------------------------------------------
    //  UDP Sender
    //-------------------------------------------------------------------------
    /*! Extract receiver ip and port information from the selected .udp file
        A udp file consists of a pair of ip and port deliminated by a space. */
    int Create(const std::string &filename, const unsigned int width, const unsigned int height, const double framerate);

    /*! Serialize SVL image object and send it to receiver(s) over networks */
    int Write(svlProcInfo* procInfo, const svlSampleImageBase &image, const unsigned int videoch);

    //-------------------------------------------------------------------------
    //  UDP Receiver
    //-------------------------------------------------------------------------
    /*! Open client socket to receive stream data and initialize image width and
        height */
    int Open(const std::string &filename, unsigned int &width, unsigned int &height, double &framerate);

    /*! Deserialize byte stream image object to generate SVL image object */
    int Read(svlProcInfo* procInfo, svlSampleImageBase &image, const unsigned int videoch, const bool noresize = false);

    //-------------------------------------------------------------------------
    //  Common Methods
    //-------------------------------------------------------------------------
    int Close();

    int GetBegPos() const { return 0; }
    int GetEndPos() const { return 0; }
    int GetPos() const    { return Pos; }
    int SetPos(const int CMN_UNUSED(pos));
    /*
    int SetCompression(const svlVideoIO::Compression *compression);
    svlVideoIO::Compression* GetCompression() const;
    int DialogCompression();
    */
    int SetCompression(const svlVideoIO::Compression * CMN_UNUSED(compression)) { return SVL_OK; }
    int DialogCompression() { return SVL_OK; }

protected:
    /*! Typedef for this codec: sender or receiver */
    typedef enum { UDP_SENDER, UDP_RECEIVER } UDPCodecType;
    UDPCodecType CodecType;

    /*! Variables for image properties */
    unsigned int Width;
    unsigned int Height;
    int BegPos;
    int EndPos;
    int Pos;
    bool Writing;
    bool Opened;

    /*! Timeserver */
    osaTimeServer TimeServer;

    /*! For testing purposes (see StereoPlayerTest) */
    bool NetworkEnabled;

    /* Auxiliary class for statistics */
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

    // FPS
    unsigned int FrameCountPerSecond;
    // Last time when FPS was calculated
    double LastFPSTick;
    // Last time when delay was calculated
    double LastDelayTick;
    // Statistic variables
    Stat * StatFPS;
    Stat * StatOverhead;
    Stat * StatDelay;

    /*! \brief Generate log file and write experiment results into the file */
    void ReportResults(void);

    //-------------------------------------------------------------------------
    //  Network (UDP) Support
    //-------------------------------------------------------------------------
    class MSG_HEADER {
    public:
        // Frame sequence number
        unsigned int FrameSeq;
        // Total size (in bytes) of serialized image data
        unsigned int SerializedSize;
        // Delimiter to differentiate MSG_HEADER and MSG_BODY (don't edit this field)
        char Delimiter[DELIMITER_STRING_SIZE];
        // Timestamp right before this message is sent to network
        double Timestamp;

        MSG_HEADER() {
            memset(Delimiter, 0, DELIMITER_STRING_SIZE);
            strncpy(Delimiter, DELIMITER_STRING, DELIMITER_STRING_SIZE);
            SerializedSize = 0;
        }

        void Print(void) {
            std::cout << "HEADER.FrameSeq       = " << FrameSeq << std::endl;
            std::cout << "HEADER.SerializedSize = " << SerializedSize << std::endl;
            std::cout << "HEADER.Delimiter      = " << Delimiter << std::endl;
        }
    };

    class MSG_PAYLOAD {
    public:
        // Frame sequence of an image that this payload belongs to
        unsigned int FrameSeq;
        // Payload size
        unsigned int PayloadSize;
        // Timestamp right before this message is sent to network
        double Timestamp;
        // Fragmented image data (with serialization)
        char Payload[UNIT_MESSAGE_SIZE];
    };

    /*! UDP socket support */
    int SocketSend, SocketRecv;
    /*! Frame sequence number this is being transmitted now */
    unsigned int CurrentSeq;

    /*! Create send/recv sockets and send/receive threads */
    bool CreateSocket(const UDPCodecType type);
    /*! Get one image frame from network */
    unsigned int GetOneImage(double & senderTick);
    /*! Send one image frame to network */
    int SendUDP(const unsigned char * serializedImage, const size_t serializedImageSize);
    /*! Cleanup resouces */
    void SocketCleanup(void);

    //-------------------------------------------------------------------------
    //  CISST Serialization and Deserialization
    //-------------------------------------------------------------------------
    /*! Serializer and Deserializer. */
    cmnSerializer * Serializer;
    cmnDeSerializer * DeSerializer;

    /*! Support for serialization of subimages using multiple CPU */
    unsigned int ProcessCount;
    unsigned char * SerializationBuffer;
    unsigned int SerializationBufferSize;

    unsigned char* yuvBuffer;
    unsigned int yuvBufferSize;
    unsigned char* comprBuffer;
    unsigned int comprBufferSize;
    vctDynamicVector<unsigned int> ComprPartOffset;
    vctDynamicVector<unsigned int> ComprPartSize;

    vctDynamicVector<unsigned int> saveBuffer;
    unsigned int saveBufferSize;
    unsigned int SaveBufferUsedSize;
    unsigned int SaveBufferUsedID;

    vctDynamicVector<unsigned int> SubImageTimeForSerialization;
    vctDynamicVector<unsigned int> SubImageSerializedSize;
    //vctDynamicVector<unsigned int> SubImageOffset;
    //vctDynamicVector<unsigned int> SubImageSize;

    //void Serialize(const cmnGenericObject & originalObject, std::string & serializedObject);
    //void Serialize(svlProcInfo * procInfo, const svlSampleImageBase &image, const unsigned int videoch, const unsigned int procId);
    void DeSerialize(const std::string & serializedObject, cmnGenericObject & originalObject);
    cmnGenericObject * DeSerialize(const std::string & serializedObject);
};

CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlVideoCodecUDP)

#endif // _svlVideoCodecUDP_h

