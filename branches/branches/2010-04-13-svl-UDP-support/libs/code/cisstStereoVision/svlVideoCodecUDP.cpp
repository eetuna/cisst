/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: svlVideoCodecUDP.cpp 1236 2010-02-26 20:38:21Z adeguet1 $
  
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

#include "svlVideoCodecUDP.h"
#include <cisstOSAbstraction/osaSleep.h>
#include <cisstOSAbstraction/osaGetTime.h>
#include <cisstOSAbstraction/osaTimeServer.h>
#include <cisstStereoVision/svlConverters.h>
#include <cisstStereoVision/svlStreamManager.h>
#include <iostream>
#include <fstream>
#include "zlib.h" // for zlib compression

//#define _DEBUG_

#define USE_COMPRESSION
#ifdef USE_COMPRESSION
//#define COMPRESSION_ARG 10
//#define COMPRESSION_ARG 50
//#define COMPRESSION_ARG 75
#define COMPRESSION_ARG 95
#endif

#define IMAGE_WIDTH  (1920 * 2)
#define IMAGE_HEIGHT 1080
//#define IMAGE_WIDTH  (256*2)
//#define IMAGE_HEIGHT 240

// Socket support
#if (CISST_OS == CISST_WINDOWS)
#include <winsock2.h>
#include <ws2tcpip.h>
#define WINSOCKVERSION MAKEWORD(2,2)
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <errno.h>
#include <string.h>  // for memset
#endif

/*! Network support */
//#define UDP_RECEIVER_IP   "127.0.0.1"
//#define UDP_RECEIVER_IP   "thin3.compsci.jhu.edu"
#define UDP_RECEIVER_IP   "lcsr-minyang.compscidhcp.jhu.edu"
#define UDP_RECV_PORT 20705

struct sockaddr_in SendToAddr;

// Receive buffer (TODO: improve this)
#define RECEIVE_BUFFER_SIZE (10 * 1024 * 1024) // 20 MB
char ReceiveBuffer[RECEIVE_BUFFER_SIZE];

/*! Internal buffer for serialization and deserialization. */
std::stringstream SerializationStreamBuffer;
std::stringstream DeSerializationStreamBuffer;

CMN_IMPLEMENT_SERVICES(svlVideoCodecUDP)

svlVideoCodecUDP::svlVideoCodecUDP() :
    svlVideoCodecBase(),
    cmnGenericObject(),
    Width(0), Height(0), BegPos(-1), EndPos(0), Pos(-1),
    CurrentSeq(0),
    Writing(false),
    Opened(false),
    StatFPS(0), StatOverhead(0), StatDelay(0),
    SerializationBuffer(0), SerializationBufferSize(0), ProcessCount(0),
    yuvBuffer(0), yuvBufferSize(0), comprBuffer(0), comprBufferSize(0),
    NetworkEnabled(true)
{
    SetName("UDP Stream");
    SetExtensionList(".udp;");
    SetMultithreaded(true);

#if (CISST_OS == CISST_WINDOWS)
    WSADATA wsaData;
    int retval = WSAStartup(WINSOCKVERSION, &wsaData);
    if (retval != 0) {
        std::cerr << "svlVideoCodecUDP: WSAStartup() failed with error code " << retval << std::endl;
        return;
    }
#endif

    SocketSend = SocketRecv = 0;

    // Initialize CISST serializer and deserializer
    Serializer = new cmnSerializer(SerializationStreamBuffer);
    DeSerializer = new cmnDeSerializer(DeSerializationStreamBuffer);

    // For statistics
    FrameCountPerSecond = 0;
    LastFPSTick = 0.0;
    LastDelayTick = 0.0;

    // Allocate compression buffer if not done yet
    unsigned int size = IMAGE_WIDTH * IMAGE_HEIGHT * 3;
    size += size / 100 + 4096;
    comprBuffer = new unsigned char[size];
    comprBufferSize = size;

    // Allocate YUV buffer if not done yet
    size = IMAGE_WIDTH * IMAGE_HEIGHT * 2;
    if (!yuvBuffer) {
        yuvBuffer = new unsigned char[size];
        yuvBufferSize = size;
    }
    else if (yuvBuffer && yuvBufferSize < size) {
        delete [] yuvBuffer;
        yuvBuffer = new unsigned char[size];
        yuvBufferSize = size;
    }
}

svlVideoCodecUDP::~svlVideoCodecUDP()
{
    Close();

    if (Serializer) delete Serializer;
    if (DeSerializer) delete DeSerializer;
    if (SerializationBuffer) delete [] SerializationBuffer;

    if (yuvBuffer) delete [] yuvBuffer;
    if (comprBuffer) delete [] comprBuffer;

    if (StatFPS) delete StatFPS;
    if (StatDelay) delete StatDelay;
    if (StatOverhead) delete StatOverhead;
}

// Create a pair of UDP sockets (client/server)
bool svlVideoCodecUDP::CreateSocket(const UDPCodecType type)
{
    CodecType = type;
    std::cout << "svlVideoCodecUDP: Codec type: ";

    if (type == UDP_RECEIVER) {
        std::cout << "UDP receiver" << std::endl;
        // Create recv socket
        SocketRecv = socket(AF_INET, SOCK_DGRAM, 0);
        if (SocketRecv < 0) {
            std::cerr << "svlVideoCodecUDP: Failed to create UDP send socket" << std::endl;
            return false;
        }

        struct sockaddr_in name;
        name.sin_family = AF_INET; 
        name.sin_addr.s_addr = INADDR_ANY; 
        name.sin_port = htons(UDP_RECV_PORT);

        if (bind(SocketRecv, (struct sockaddr *)&name, sizeof(name)) < 0) {
            std::cerr << "svlVideoCodecUDP: Failed to bind recv socket" << std::endl;
            return false;
        }

        std::cout << "svlVideoCodecUDP: receive socket is created." << std::endl;

        // For statistics
        StatFPS = new Stat("FPS", 10);
        StatDelay = new Stat("Network Delay", 1000);
        StatOverhead = new Stat("Overhead", 1000);

        return true;
    }

    if (type == UDP_SENDER) {
        std::cout << "UDP sender" << std::endl;
        // Create send socket
        SocketSend = socket(AF_INET, SOCK_DGRAM, 0);
        if (SocketSend < 0) {
            std::cerr << "svlVideoCodecUDP: Failed to create UDP recv socket" << std::endl;
            return false;
        }

        struct hostent hostEnt;
        long hostNum;
        struct hostent * pHostEnt = gethostbyname(UDP_RECEIVER_IP);
        if (!pHostEnt) {
            std::cerr << "svlVideoCodecUDP: Cannot get host name: " << UDP_RECEIVER_IP << std::endl;
            return false;
        }

        memcpy(&hostEnt, pHostEnt, sizeof(hostEnt));
        memcpy(&hostNum, hostEnt.h_addr_list[0], sizeof(hostNum));

        SendToAddr.sin_family = AF_INET;
        SendToAddr.sin_addr.s_addr = hostNum;
        SendToAddr.sin_port = htons(UDP_RECV_PORT);

        std::cout << "svlVideoCodecUDP: send socket is created (dest: " 
            << UDP_RECEIVER_IP << ":" << UDP_RECV_PORT << std::endl;

        // For statistics
        StatFPS = new Stat("FPS-Sender", 10);
        StatOverhead = new Stat("Processing Delay", 1000);

        return true;
    }

    return false;
}

void svlVideoCodecUDP::SocketCleanup(void)
{
    if (SocketSend) {
#if (CISST_OS == CISST_WINDOWS)
        closesocket(SocketSend);
#else
        close(SocketSend);
#endif
    }

    if (SocketRecv) {
#if (CISST_OS == CISST_WINDOWS)
        closesocket(SocketRecv);
#else
        close(SocketRecv);
#endif
    }

#if (CISST_OS == CISST_WINDOWS)
    WSACleanup();
#endif
}

int svlVideoCodecUDP::Create(const std::string &filename, const unsigned int width, const unsigned int height, const double CMN_UNUSED(framerate))
{
    if (Opened || width < 1 || height < 1) return SVL_FAIL;

    // Create send socket if it's not created yet
    if (SocketSend == 0) {
        if (!CreateSocket(UDP_SENDER)) {
            std::cout << "svlVideoCodecUDP: failed to create socket" << std::endl;
            return SVL_FAIL;
        }
    }

    // TODO: Extract receiver ip and port information from the selected .udp file
    std::string ip(filename);
    std::cout << "svlVideoCodecUDP: selected file: " << filename << std::endl;

    // Allocate serialization buffer
    if (!SerializationBuffer) {
        const unsigned int size = IMAGE_WIDTH * IMAGE_HEIGHT * 2;
        SerializationBuffer = new unsigned char[size];
        SerializationBufferSize = size;
        std::cout << "svlVideoCodecUDP: size of buffer for serialization: " << SerializationBufferSize << std::endl;
    }

    Width = IMAGE_WIDTH;
    Height = IMAGE_HEIGHT;
    Pos = 0;

    // For testing purposes
    std::string str("0.udp");
    size_t found = filename.find(str);
    if (found != std::string::npos) {
        NetworkEnabled = false;
        std::cout << "svlVideoCodecUDP: networking feature is disabled" << std::endl;
    } else {
        NetworkEnabled = true;
        std::cout << "svlVideoCodecUDP: networking feature is enabled" << std::endl;
    }

    return SVL_OK;
}

int svlVideoCodecUDP::Write(svlProcInfo* procInfo, const svlSampleImageBase &image, const unsigned int videoch)
{
    static int frameNo = 0;
    if (procInfo->id == 0) {
        // For testing
        /*
        if (frameNo++ >= 3) {
            ReportResults();
            //exit(1);
            return SVL_FAIL;
        }
        */

        // The first frame takes longer to process(?)
        if (++frameNo == 1) {
            osaSleep(1.0);
        }
    }

    //
    // Initialize multi-threaded image serialization
    //
    // Remember total number of subimages
    ProcessCount = procInfo->count;
    const unsigned int procId = procInfo->id;
    bool err = false;

    if (frameNo == 1) {
        if (procInfo->id == 0) {
            std::cout << "svlVideoCodecUDP: processor count: " << ProcessCount << std::endl;
            ComprPartOffset.SetSize(procInfo->count);
            ComprPartSize.SetSize(procInfo->count);

            SubImageTimeForSerialization.SetSize(ProcessCount);
            SubImageSerializedSize.SetSize(ProcessCount);
        }
    }
    
    _SynchronizeThreads(procInfo);

    /*
    ExperimentResultElement result;
    result.FrameNo = frameNo;
    result.FPS = 0; // will be updated later
    result.Timestamp = osaGetTime();

    const double ticProcessing = osaGetTime();
    */

    // temporary compression. 
    // TODO: Need to fix this
//#ifdef USE_COMPRESSION
//    const_cast<svlSampleImageBase&>(image).SetEncoder("jpg", COMPRESSION_ARG);
//#endif

    // Serialize image data
    /*
    unsigned int serializedSize;
    const double tic = osaGetTime();
    std::string s;
    Serialize(image, s);
    const double toc = osaGetTime();

    serializedSize = s.size();
    result.FrameSize = serializedSize; //image.GetDataSize();
    const char * dest = s.c_str();
    //std::cout << serializedSize << std::endl;

    result.TimeSerialization = toc - tic;
    */

    const unsigned int procid = procInfo->id;
    const unsigned int proccount = procInfo->count;
    unsigned int start, end, size, offset;
    unsigned long comprsize;
    int compr = 9;//Codec->data[0];

    // Multithreaded compression phase
    while (1) {
        // Compute part size and offset
        size = Height / proccount + 1;
        comprsize = comprBufferSize / proccount;
        start = procid * size;
        if (start >= Height) break;
        end = start + size;
        if (end > Height) end = Height;
        offset = start * Width;
        size = Width * (end - start);
        ComprPartOffset[procid] = procid * comprsize;

        // Convert RGB to YUV422 planar format
        svlConverter::RGB24toYUV422P(const_cast<unsigned char*>(image.GetUCharPointer(videoch)) + offset * 3, yuvBuffer + offset * 2, size);

        // Compress part
        if (compress2(comprBuffer + ComprPartOffset[procid], &comprsize, yuvBuffer + offset * 2, size * 2, compr) != Z_OK) {
            err = true;
            break;
        }
        ComprPartSize[procid] = comprsize;

        break;
    }

    _SynchronizeThreads(procInfo);

    if (err) return SVL_FAIL;

    // Single threaded data serialization
    _OnSingleThread(procInfo)
    {
        // Only the first subimage contains serialized class service information.

        // Do cisst serialization (refer to cmnSerializer::Serialize())
        // without serializing image frame data (note that the sencond
        // argument of Serialize() is false).
        SerializationStreamBuffer.str("");
        Serializer->Serialize(image, false);

        // Serialize cisst class service information
        size_t pos = 0;
        const std::string str = SerializationStreamBuffer.str();
        // - size of serialized cisst class service
        size_t serializedSize = str.size();

        memcpy(&SerializationBuffer[pos], &serializedSize, sizeof(unsigned int));
        pos += sizeof(unsigned int);
        // - serialize cisst class service
        memcpy(&SerializationBuffer[pos], str.c_str(), str.size());
        pos += str.size();
#ifdef _DEBUG_
        std::cout << "Serialized class service size : " << serializedSize << std::endl;
#endif

        // Do svlStream serialization (refer to svlSampleImageCustom<>::SerializeRaw())
        /*
        std::string codec;
        int compression;
        image.GetEncoder(codec, compression);

        svlStreamType type = image.GetType();
        double timestamp = image.GetTimestamp();

        // Serialize stream type
        memcpy(&SerializationBuffer[pos], &type, sizeof(type));
        pos += sizeof(type);
        // Serialize timestamp
        memcpy(&SerializationBuffer[pos], &timestamp, sizeof(timestamp));
        pos += sizeof(timestamp);
        // Serialize codec information
        memcpy(&SerializationBuffer[pos], codec.c_str(), codec.size());
        pos += codec.size();
        */
        // Serialize processor count
        memcpy(&SerializationBuffer[pos], &ProcessCount, sizeof(ProcessCount));
        pos += sizeof(ProcessCount);
#ifdef _DEBUG_
        std::cout << "Processor count : " << ProcessCount << std::endl;
#endif

        for (size_t i = 0; i < ProcessCount; ++i) {
            // Serialize subimage size
            memcpy(&SerializationBuffer[pos], &ComprPartSize[i], sizeof(unsigned int));
            pos += sizeof(unsigned int);
            // Serialize subimage itself
            memcpy(&SerializationBuffer[pos], comprBuffer + ComprPartOffset[i], ComprPartSize[i]);
            pos += ComprPartSize[i];
#ifdef _DEBUG_
            std::cout << "[" << i << "] size: " << ComprPartSize[i] << ", offset: " << ComprPartOffset[i] << std::endl;
#endif
        }

        if (NetworkEnabled) {
            if (SendUDP(SerializationBuffer, pos) == SVL_FAIL) {
                std::cerr << "svlVideoCodecUDP: failed to send UDP messages" << std::endl;
                return SVL_FAIL;
            }
        }
    }

    /*
    serializedSize = s.size();
    result.FrameSize = serializedSize; //image.GetDataSize();
    const char * dest = s.c_str();
    //std::cout << serializedSize << std::endl;

    result.TimeSerialization = toc - tic;

    // SEND MSG TO NETWORK

    const double tocProcessing = osaGetTime();

    result.TimeProcessing = tocProcessing - ticProcessing;

    // Track fps and processing overhead
    ++FrameCountPerSecond;
    StatOverhead->AddSample(tocProcessing - ticProcessing);
    if (LastFPSTick == 0.0) {
        LastFPSTick = osaGetTime();
    } else {
        if (osaGetTime() - LastFPSTick > 1.0 * cmn_s) {
            StatFPS->AddSample(FrameCountPerSecond);

            StatFPS->Print();
            StatOverhead->Print();
            std::cout << std::endl;

            result.FPS = FrameCountPerSecond;

            FrameCountPerSecond = 0;
            LastFPSTick = osaGetTime();
        }
    }

    ExperimentResultElements.push_back(result);
    */

    return SVL_OK;
}

int svlVideoCodecUDP::Open(const std::string &filename, unsigned int &width, unsigned int &height, double &framerate)
{
    if (Opened) return SVL_FAIL;

    if (SocketRecv == 0) {
        // Create sockets
        if (!CreateSocket(UDP_RECEIVER)) {
            std::cerr << "svlVideoCodecUDP: failed to create socket" << std::endl;
            return SVL_FAIL;
        }
    }

    // TODO: Parse filename
    //std::cout << "Open called with file: " << filename << std::endl;

    std::cout << "Waiting for the first frame to get image information..." << std::endl;

    /*
    // Wait for one complete image to be transferred. If it arrives, extract 
    // image size information (width and height)
    double senderTick;
    unsigned int serializedSize;
    unsigned int _width = 0;
    //svlSampleImageBase * image;
    //for (int i = 0; i < 1; i++) { // 2 sec (30Hz)
    int cnt = 0;
    while (_width == 0) {
        serializedSize = GetOneImage(senderTick);
    //    if (serializedSize == 0) {
    //        continue;
    //    }
        std::cout << "serializedSize: " << serializedSize << std::endl;

        // Deserialize data
        std::string serializedData(ReceiveBuffer, serializedSize);
        cmnGenericObject * object = DeSerialize(serializedData);
        //object->Services()->TypeInfoPointer()->name
        svlSampleImageBase * image = dynamic_cast<svlSampleImageBase *>(object);//(object->Services()->Create());
        if (!image) {
            std::cout << "ERROR: NULL" << std::endl;
            exit(1);
        }
        
        width = image->GetWidth();
        height = image->GetHeight();
        std::cout << "Image size: " << width << " x " << height << std::endl;

        _width = width;
    }

    //if (serializedSize == 0) {
    //    return SVL_FAIL;
    //}

    //std::cout << "----------------------" << std::endl;

    //width = image->GetWidth();
    //height = image->GetHeight();
    std::cout << "========== Image size: " << width << " x " << height << std::endl;

    //exit(1);
    //*/

    /*  // drop.avi
    width = 256 * 2;
    height = 240;
    //*/
    /*  // capture.avi
    width = 640 * 2;
    height = 480;riali
    /*/
    width = IMAGE_WIDTH;
    height = IMAGE_HEIGHT;
    //*/

    Pos = 0;
    framerate = -1.0;
    Opened = true;

    return SVL_OK;
}

int svlVideoCodecUDP::Read(svlProcInfo* procInfo, svlSampleImageBase & image, const unsigned int videoch, const bool CMN_UNUSED(noresize))
{
    // for testing
    static int frameNo = 0;
    /*
    //if (frameNo++ == 100) {
    if (frameNo++ == 3) {
        ReportResults();
        exit(1);
    }
    */

    /*
    ExperimentResultElement result;
    result.FrameNo = frameNo;
    result.FPS = 0; // will be updated later
    result.Timestamp = osaGetTime();

    const double ticProcessing = osaGetTime();
    */

    std::cout << ++frameNo << std::endl;

    // Receive video stream data via UDP
    double senderTick;
    unsigned int serializedSize = GetOneImage(senderTick);
    if (serializedSize == 0) {
        return SVL_FAIL;
    }
    //result.FrameSize = serializedSize;

    // Uses only a single thread
    if (procInfo && procInfo->id != 0) return SVL_OK;

    // Allocate image buffer if not done yet
    if (IMAGE_WIDTH != image.GetWidth(videoch) || IMAGE_HEIGHT != image.GetHeight(videoch)) {
        image.SetSize(videoch, IMAGE_WIDTH, IMAGE_HEIGHT);
    }

    // Deserialize cisst class service information
    size_t pos = 0;
    unsigned int serializedCisstServiceSize = 0;
    // - Serialize size of serialized cisst class service
    memcpy(&serializedCisstServiceSize, ReceiveBuffer + pos, sizeof(unsigned int));
    pos += sizeof(unsigned int);
#ifdef _DEBUG_
    std::cout << "Serialized class service size : " << serializedCisstServiceSize << std::endl;
#endif
    // - Deserialize cisst class service
    DeSerializationStreamBuffer.str("");
    DeSerializationStreamBuffer.write(ReceiveBuffer + pos, serializedCisstServiceSize);
    DeSerializer->DeSerialize(image, false);
    pos += serializedCisstServiceSize;

    // Do svlStream deserialization (refer to svlSampleImageCustom<>::DeSerializeRaw())
    // - Deserialize processor count (i.e., total number of subimages)
    unsigned int processorCount = 0;
    memcpy(&processorCount, ReceiveBuffer + pos, sizeof(unsigned int));
    pos += sizeof(unsigned int);
#ifdef _DEBUG_
    std::cout << "Processor count : " << processorCount << std::endl;
#endif

    // - Image itself
    unsigned char* img = image.GetUCharPointer(videoch);
    unsigned int i, compressedpartsize, offset;
    unsigned long longsize;
    int ret = SVL_FAIL;

    offset = 0;
    for (i = 0; i < processorCount; ++i) {
        // Deserialize subimage size
        memcpy(&compressedpartsize, ReceiveBuffer + pos, sizeof(unsigned int));
        pos += sizeof(unsigned int);
        //if (compressedpartsize == 0 || compressedpartsize > comprBufferSize) return SVL_FAIL;
        // Deserialize subimage itself
        memcpy(comprBuffer, ReceiveBuffer + pos, compressedpartsize);
        pos += compressedpartsize;
#ifdef _DEBUG_
        std::cout << "[" << i << "] size: " << compressedpartsize << std::endl;
#endif

        // Decompress frame part
        longsize = yuvBufferSize - offset;
        if (uncompress(yuvBuffer + offset, &longsize, comprBuffer, compressedpartsize) != Z_OK) {
            std::cout << "ERROR: Uncompress failed" << std::endl;
            exit(1);
            return SVL_FAIL;
        }

        // Convert YUV422 planar to RGB format
        svlConverter::YUV422PtoRGB24(yuvBuffer + offset, img + offset * 3 / 2, longsize >> 1);

        offset += longsize;
    }

    return SVL_OK;

    /*
    // Deserialize data
    const double tic = osaGetTime();
    std::string serializedData(ReceiveBuffer, serializedSize);
    DeSerialize(serializedData, image);
    const double toc = osaGetTime();

    result.TimeDeSerialization = toc - tic;

    std::cout << serializedSize << std::endl;

#ifdef _DEBUG_
    std::cout << "Successfully deserialized: " << serializedSize << std::endl;
    std::cout << "Size: " << image.GetWidth() << " x " << image.GetHeight() << std::endl;
    std::cout << "Ch: " << image.GetDataChannels() << std::endl;
#endif

    const double tocProcessing = osaGetTime();

    result.TimeProcessing = tocProcessing - ticProcessing; 

    // Track fps and processing overhead
    ++FrameCountPerSecond;
    StatOverhead->AddSample(tocProcessing - ticProcessing);
    if (LastFPSTick == 0.0) {
        LastFPSTick = osaGetTime();
    } else {
        if (osaGetTime() - LastFPSTick > 1.0 * cmn_s) {
            StatFPS->AddSample(FrameCountPerSecond);

            StatFPS->Print();
            StatOverhead->Print();
            std::cout << std::endl;

            result.FPS = FrameCountPerSecond;

            FrameCountPerSecond = 0;
            LastFPSTick = osaGetTime();
        }
    }

    // Delay update
    double delay = osaGetTime() - senderTick;
    StatDelay->AddSample(delay);
    if (LastDelayTick == 0.0) {
        LastDelayTick = osaGetTime();
    } else {
        if (osaGetTime() - LastDelayTick > 1.0 * cmn_s) {
            StatDelay->Print();
            std::cout << std::endl;

            LastDelayTick = osaGetTime();
        }
    }

    ExperimentResultElements.push_back(result);
    */

    return SVL_OK;
}

int svlVideoCodecUDP::Close()
{
    SocketCleanup();

    //std::cout << "Close called" << std::endl;

    Pos = -1;

    return SVL_OK;
}

int svlVideoCodecUDP::SetPos(const int CMN_UNUSED(pos))
{
    std::cout << "SetPos called" << std::endl;

    return SVL_OK;
}

#if 0
void svlVideoCodecUDP::Serialize(const cmnGenericObject & originalObject, std::string & serializedObject) 
{
    try {
        SerializationStreamBuffer.str("");
        Serializer->Serialize(originalObject);
        serializedObject = SerializationStreamBuffer.str();
    } catch (std::runtime_error e) {
        std::cerr << "Serialization failed: " << originalObject.ToString() << std::endl;
        std::cerr << e.what() << std::endl;
        serializedObject = "";
    }
}

void svlVideoCodecUDP::Serialize(svlProcInfo * procInfo, const svlSampleImageBase &image, const unsigned int videoch, const unsigned int procId)
{
    // Get offset and size of subimage that this thread should serialize
    
#if 0
    // Size of original image
    const size_t totalSize = image.GetDataSize(videoch);
#ifdef _DEBUG_
    std::cout << "svlVideoCodecUDP: Serialize: [" << procId << "/" << ProcessCount << "] total size: " << totalSize << std::endl;
#endif
    // size of a subimage
    const size_t subImageSize = totalSize / ProcessCount;
    // buffer chunk size
    const size_t bufferChunkSize = SerializationBufferSize / ProcessCount;

    // src image start offset
    const size_t srcStartOffset = procId * subImageSize;
    // src image end offset
    const size_t srcEndOffset = srcStartOffset + subImageSize;
    // dest buffer start offset
    size_t destStartOffset = procId * bufferChunkSize;
    size_t destStartOffsetCurrent = destStartOffset;

    // Only the first subimage contains serialized class service information.
    _OnSingleThread(procInfo) {
        // Do cisst serialization (refer to cmnSerializer::Serialize())
        Serializer->Serialize(image, false);
        // Copy serialized stream (class service information) into buffer chunk
        std::string str = SerializationStreamBuffer.str();
        memcpy(&SerializationBuffer[destStartOffsetCurrent], str.c_str(), str.size());
        destStartOffsetCurrent += str.size();
        // Do svlStream serialization (refer to svlSampleImageCustom<>::SerializeRaw())
        std::string codec;
        int compression;
        image.GetEncoder(codec, compression);

        svlStreamType type = image.GetType();
        double timestamp = image.GetTimestamp();

        memcpy(&SerializationBuffer[destStartOffset], &type, sizeof(type));
        destStartOffsetCurrent += sizeof(type);
        memcpy(&SerializationBuffer[destStartOffset], &timestamp, sizeof(timestamp));
        destStartOffsetCurrent += sizeof(timestamp);
        memcpy(&SerializationBuffer[destStartOffset], codec.c_str(), codec.size());
        destStartOffsetCurrent += codec.size();
    }

    // Serialize subimage content based on procId
    const size_t usedSize = destStartOffsetCurrent - destStartOffset;
    /*
    if (SVL_OK != svlImageIO::Write(
        image, videoch, codec, &SerializationBuffer[destStartOffsetCurrent], bufferChunkSize - usedSize, compression))
    {
        cmnThrow("svlVideoCodecUDP: Serialize error occured with svlImageIO::Write");
    }
    */
#endif
}
#endif

void svlVideoCodecUDP::DeSerialize(const std::string & serializedObject, cmnGenericObject & originalObject) 
{
    try {
        DeSerializationStreamBuffer.str("");
        DeSerializationStreamBuffer << serializedObject;
        DeSerializer->DeSerialize(originalObject);
    }  catch (std::runtime_error e) {
        std::cerr << "DeSerialization failed: " << e.what() << std::endl;
    }
}

cmnGenericObject * svlVideoCodecUDP::DeSerialize(const std::string & serializedObject) 
{
    cmnGenericObject * deserializedObject = 0;
    try {
        DeSerializationStreamBuffer.str("");
        DeSerializationStreamBuffer << serializedObject;
        deserializedObject = DeSerializer->DeSerialize();
    }  catch (std::runtime_error e) {
        std::cerr << "DeSerialization failed: " << e.what() << std::endl;
        return 0;
    }

    return deserializedObject;
}

unsigned int svlVideoCodecUDP::GetOneImage(double & senderTick)
{
    // UDP receiver expects MSG_HEADER message first to get the total length of 
    // a serialized image frame and then reads up to that size bytes.
    fd_set mask;
    //struct timeval timeout;
    struct sockaddr_in from_addr;
    socklen_t from_len = sizeof(from_addr);    

    bool received = false;
    int byteRecv = 0;
    unsigned int totalByteRecv = 0;
    unsigned int serializedSize;
    
    MSG_PAYLOAD * payload = 0;

    //timeout.tv_sec = 0; // consider the worst case (1 sec transmission delay)
    //timeout.tv_usec = 0;

    FD_ZERO(&mask);
    FD_SET(SocketRecv, &mask);

    int ret;
    char buf[UNIT_MESSAGE_SIZE * 2];
    while (!received) {
        ret = select(FD_SETSIZE, &mask, 0, 0, 0); // blocking
        // Socket error
        if (ret < 0) {
            std::cerr << "svlVideoCodecUDP: select error: " << byteRecv << " bytes received so far, errono: ";
#if (CISST_OS == CISST_WINDOWS)
            std::cerr << WSAGetLastError() << std::endl;
#else
            std::cerr << strerror(errno) << std::endl;
#endif
            return 0;
        }

        if (FD_ISSET(SocketRecv, &mask)) {
            byteRecv = recvfrom(SocketRecv, buf, UNIT_MESSAGE_SIZE * 2, 0, (struct sockaddr *) &from_addr, &from_len);
            // Socket error
            if (byteRecv < 0) {
                std::cerr << "svlVideoCodecUDP: failed to receive UDP message: " << byteRecv << " bytes received so far, errono: ";
#if (CISST_OS == CISST_WINDOWS)
                std::cerr << WSAGetLastError() << std::endl;
#else
                std::cerr << strerror(errno) << std::endl;
#endif
                return 0;
            }

#ifdef _DEBUG_
            std::cout << "received: " << byteRecv << " bytes" << std::endl;
#endif

            //
            // Get header
            //
            // Check message size
            if (byteRecv == sizeof(MSG_HEADER)) {
                // Check content to make sure this is MSG_HEADER
                MSG_HEADER * header = (MSG_HEADER*) buf;
#ifdef _DEBUG_
                header->Print();
#endif
                if (strncmp(DELIMITER_STRING, header->Delimiter, DELIMITER_STRING_SIZE) == 0) {
                    CurrentSeq = header->FrameSeq;
#ifdef _DEBUG_
                    std::cout << "sequence number: " << CurrentSeq << std::endl;
#endif
                    serializedSize = header->SerializedSize;

#ifdef _DEBUG_
                    std::cout << "serialized size: " << serializedSize << " bytes" << std::endl;
#endif

                    if (serializedSize == 0) {
                        return 0;
                    }

                    continue;
                }
            }

            //
            // Get payload
            //
            payload = reinterpret_cast<MSG_PAYLOAD *>(buf);
            if (CurrentSeq != 0) {
                if (payload->FrameSeq != CurrentSeq) {
                    // TODO: Check if this is OK
                    CurrentSeq = 0;
                    totalByteRecv = 0;
                    continue;
                    /*
                    // drop all data received until now
                    totalByteRecv = 0;
                    // start new image frame
                    // waiting for header
                    */
                } else {
                    memcpy(ReceiveBuffer + totalByteRecv, payload->Payload, payload->PayloadSize);
                    totalByteRecv += payload->PayloadSize;
                    
                    //std::cout << "READ: " << byteRecv << ", " << payload->PayloadSize << ", " << totalByteRecv << " / " << serializedSize << std::endl;

                    if (totalByteRecv >= serializedSize) {
                        // now receiver has received all the data to rebuild an
                        // original image by deserialization.
                        received = true;
                        senderTick = payload->Timestamp;

                        return serializedSize;
                    }
                }
            }
        }
    }

    return serializedSize;
}

void svlVideoCodecUDP::ReportResults(void)
{
    std::filebuf fb;
    char fileName[20];

    sprintf(fileName, (CodecType == UDP_SENDER ? "result_write.txt" : "result_read.txt"));

    fb.open(fileName, std::ios::out);
    std::ostream os(&fb);
#ifdef USE_COMPRESSION
    os << "Compression: " << COMPRESSION_ARG << std::endl;
#else
    os << "No compression" << std::endl;
#endif
    os << "FrameNo,FrameSize,FPS,TimeSerialization,TimeProcessing,Timestamp" << std::endl;

    // Put experiment results to log file
    ExperimentResultElementsType::const_iterator it = ExperimentResultElements.begin();
    const ExperimentResultElementsType::const_iterator itEnd = ExperimentResultElements.end();
    for (; it != itEnd; ++it) {
        os << it->FrameNo << "," << it->FrameSize << "," << it->FPS << ",";
        if (CodecType == UDP_SENDER) {
            os << it->TimeSerialization;
        } else {
            os << it->TimeDeSerialization;
        }
        os << "," << it->TimeProcessing << "," << it->Timestamp << std::endl;
    }
        
    fb.close();
}

int svlVideoCodecUDP::SendUDP(const unsigned char * serializedImage, const size_t serializedImageSize)
{
    // Send video stream data via UDP
    // First, send MSG_HEADER message to let a client know the size of the
    // frame. Then, send serialized image data with fragmentation of size
    // 1300 bytes (predefined but tunnable value).

    // Send MSG_HEADER message
    static unsigned int frameSeq = 0;

    MSG_HEADER header;
    header.FrameSeq = ++frameSeq;
    header.SerializedSize = serializedImageSize;
    header.Timestamp = osaGetTime();

    int ret = sendto(SocketSend, (const char*) &header, sizeof(header), 0, (struct sockaddr *) &SendToAddr, sizeof(SendToAddr));
    if (ret != sizeof(header)) {
        std::cerr << "svlVideoCodecUDP: failed to send MSG_HEADER" << std::endl;
        return SVL_FAIL;
    } 
#ifdef _DEBUG_
    else {
        std::cout << "Sent MSG_HEADER" << std::endl;
        header.Print();
    }
    //std::cout << "######## " << sizeof(MSG_HEADER) << ", " << sizeof(MSG_PAYLOAD) << std::endl;
#endif


    // Send image data
    MSG_PAYLOAD payload;
    payload.FrameSeq = frameSeq;

    unsigned int byteSent = 0, n;
    while (byteSent < serializedImageSize) {
        n = (UNIT_MESSAGE_SIZE > (serializedImageSize - byteSent) ? (serializedImageSize - byteSent) : UNIT_MESSAGE_SIZE);
        
        memcpy(payload.Payload, reinterpret_cast<const char*>(serializedImage + byteSent), n);
        payload.PayloadSize = n;
        payload.Timestamp = osaGetTime();

        ret = sendto(SocketSend, (const char *)&payload, sizeof(payload), 0, (struct sockaddr *) &SendToAddr, sizeof(SendToAddr));
        if (ret < 0) {
            std::cerr << "svlVideoCodecUDP: failed to send UDP message: " 
                << byteSent << " / " << serializedImageSize << ", errono: ";
#if (CISST_OS == CISST_WINDOWS)
            std::cerr << WSAGetLastError() << std::endl;
#else
            std::cerr << strerror(errno) << std::endl;
#endif
            break;
        } else {
#ifdef _DEBUG_
            printf("Sent (%u) : %u / %u\n", ret, byteSent, serializedImageSize);
#endif
            byteSent += n;
        }

        //osaSleep(0.5 * cmn_ms);
        //osaSleep(1 * cmn_ms);
        //osaSleep(1 * cmn_s);
    }
#ifdef _DEBUG_
    printf("Send complete: %u / %u\n", byteSent, serializedImageSize);
#endif

    return serializedImageSize;
}

/*
int svlVideoCodecUDP::SetCompression(const svlVideoIO::Compression *compression)
{
    //if (Opened || !compression || compression->size < sizeof(svlVideoIO::Compression)) return SVL_FAIL;
    if (!compression || compression->size < sizeof(svlVideoIO::Compression)) return SVL_FAIL;

    std::string extensionlist(GetExtensions());
    std::string extension(compression->extension);
    extension += ";";
    if (extensionlist.find(extension) == std::string::npos) {
        // Codec parameters do not match this codec
        return SVL_FAIL;
    }

    svlVideoIO::ReleaseCompression(Codec);
    Codec = reinterpret_cast<svlVideoIO::Compression*>(new unsigned char[sizeof(svlVideoIO::Compression)]);

    std::string name("UDP Stream");
    memset(&(Codec->extension[0]), 0, 16);
    memset(&(Codec->name[0]), 0, 64);
    memcpy(&(Codec->name[0]), name.c_str(), std::min(static_cast<int>(name.length()), 63));
    Codec->size = sizeof(svlVideoIO::Compression);
    Codec->supports_timestamps = true;
    Codec->datasize = 1;
    if (compression->data[0] <= 9) Codec->data[0] = compression->data[0];
    else Codec->data[0] = 4;

    return SVL_OK;
}

/*
int svlVideoCodecUDP::DialogCompression()
{
    // TODO: Check this
    return SVL_OK;

    if (Opened) return SVL_FAIL;

    std::cout << std::endl << " # Enter compression level [0-99]: ";
    int level;
    std::cin >> level;
    if (level < 0) level = 0;
    else if (level > 99) level = 99;
    std::cout << level << std::endl;

    svlVideoIO::ReleaseCompression(Codec);
    Codec = reinterpret_cast<svlVideoIO::Compression*>(new unsigned char[sizeof(svlVideoIO::Compression)]);

    std::string name("JPEG Compression");
    memset(&(Codec->extension[0]), 0, 16);
    memcpy(&(Codec->extension[0]), ".udp", 4);
    memset(&(Codec->name[0]), 0, 64);
    memcpy(&(Codec->name[0]), name.c_str(), std::min(static_cast<int>(name.length()), 63));
    Codec->size = sizeof(svlVideoIO::Compression);
    Codec->supports_timestamps = true;
    Codec->datasize = 1;
    Codec->data[0] = static_cast<unsigned char>(level);

	return SVL_OK;
}

svlVideoIO::Compression* svlVideoCodecUDP::GetCompression() const
{
    // TODO: Check this
    return SVL_OK;

    if (!Codec) return 0;
    // Make a copy and return the pointer to it
    // The caller will need to release it by calling the
    // svlVideoIO::ReleaseCompression() method
    svlVideoIO::Compression* compression = reinterpret_cast<svlVideoIO::Compression*>(new unsigned char[Codec->size]);
    memcpy(compression, Codec, Codec->size);
    return compression;
}
//*/
