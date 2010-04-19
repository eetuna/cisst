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
#include <cisstStereoVision/svlConverters.h>
#include <cisstOSAbstraction/osaSleep.h>
#include <cisstOSAbstraction/osaGetTime.h>
#include <cisstOSAbstraction/osaTimeServer.h>
#include <iostream>
#include <fstream>

//#define _DEBUG_

#define USE_CISST_SERIALIZATION
#define USE_COMPRESSION
#ifdef USE_COMPRESSION
//#define COMPRESSION_ARG 10
//#define COMPRESSION_ARG 50
//#define COMPRESSION_ARG 75
#define COMPRESSION_ARG 95
#endif

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
std::stringstream SerializationBuffer;
std::stringstream DeSerializationBuffer;

CMN_IMPLEMENT_SERVICES(svlVideoCodecUDP)

svlVideoCodecUDP::svlVideoCodecUDP() :
    svlVideoCodecBase(),
    cmnGenericObject(),
    Width(0), Height(0), BegPos(-1), EndPos(0), Pos(-1),
    CurrentSeq(0),
    Writing(false),
    Opened(false),
    StatFPS(0), StatOverhead(0), StatDelay(0) 
{
    SetName("UDP Stream");
    SetExtensionList(".udp;");
    SetMultithreaded(false);

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
    Serializer = new cmnSerializer(SerializationBuffer);
    DeSerializer = new cmnDeSerializer(DeSerializationBuffer);

    // For statistics
    FrameCountPerSecond = 0;
    LastFPSTick = 0.0;
    LastDelayTick = 0.0;
}

svlVideoCodecUDP::~svlVideoCodecUDP()
{
    Close();

    SocketCleanup();

    if (Serializer) delete Serializer;
    if (DeSerializer) delete DeSerializer;
    if (StatFPS) delete StatFPS;
    if (StatDelay) delete StatDelay;
    if (StatOverhead) delete StatOverhead;
}

// Create a pair of UDP sockets (client/server)
bool svlVideoCodecUDP::CreateSocket(const UDP_TYPE type)
{
    if (type == UDP_RECEIVER) {
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

    // Extract receiver ip and port information from the selected .udp file

    std::string ip(filename);

    return SVL_OK;
}

int svlVideoCodecUDP::Write(svlProcInfo* CMN_UNUSED(procInfo), const svlSampleImageBase &image, const unsigned int CMN_UNUSED(videoch))
{
    // For testing
    static int frameNo = 0;
    if (frameNo == 1)
        osaSleep(1.0);

    //if (frameNo == 200)
    //    exit(1);

    //if (frameNo++ == 550) {
    if (frameNo++ == 100) {
        // Generate log file
        std::filebuf fb;
        fb.open("Write.txt", std::ios::out);
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
            os << it->FrameNo << "," << it->FrameSize << "," << it->FPS << "," << it->TimeSerialization
               << "," << it->TimeProcessing << "," << it->Timestamp << std::endl;
        }
        fb.close();

        exit(1);
    }

    if (SocketSend == 0) {
        if (!CreateSocket(UDP_SENDER)) {
            std::cout << "svlVideoCodecUDP: failed to create socket" << std::endl;
            return SVL_FAIL;
        }
    }

    ExperimentResultElement result;
    result.FrameNo = frameNo;
    result.FPS = 0; // will be updated later
    result.Timestamp = osaGetTime();

    const double ticProcessing = osaGetTime();

    // temporary compression
#ifdef USE_COMPRESSION
    //const_cast<svlSampleImageBase&>(image).SetEncoder("jpg", 25);
    const_cast<svlSampleImageBase&>(image).SetEncoder("jpg", COMPRESSION_ARG);
    //const_cast<svlSampleImageBase&>(image).SetEncoder("png", 9);
#endif

    // Serialize image data
    unsigned int serializedSize;
#ifdef USE_CISST_SERIALIZATION
    const double tic = osaGetTime();
    std::string s;
    Serialize(image, s);
    const double toc = osaGetTime();

    serializedSize = s.size();
    result.FrameSize = serializedSize; //image.GetDataSize();
    const char * dest = s.c_str();
    //std::cout << serializedSize << std::endl;

    result.TimeSerialization = toc - tic;
#else
    const unsigned char * dest = image.GetUCharPointer(videoch);
    serializedSize = image.GetDataSize(videoch);
    
    result.TimeSerialization = 0.0;
#endif

    // Send video stream data via UDP
    // First, send MSG_HEADER message to let a client know the size of the
    // frame. Then, send serialized image data with fragmentation of size
    // 1300 bytes (predefined but tunnable value).

    // Send MSG_HEADER message
    static unsigned int frameSeq = 0;

    MSG_HEADER header;
    header.FrameSeq = ++frameSeq;
    header.SerializedSize = serializedSize;
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
    std::cout << "######## " << sizeof(MSG_HEADER) << ", " << sizeof(MSG_PAYLOAD) << std::endl;
#endif


    // Send image data
    MSG_PAYLOAD payload;
    payload.FrameSeq = frameSeq;

    unsigned int byteSent = 0, n;
    while (byteSent < serializedSize) {
        n = (UNIT_MESSAGE_SIZE > (serializedSize - byteSent) ? (serializedSize - byteSent) : UNIT_MESSAGE_SIZE);
        
        memcpy(payload.Payload, reinterpret_cast<const char*>(dest + byteSent), n);
        payload.PayloadSize = n;
        payload.Timestamp = osaGetTime();

        ret = sendto(SocketSend, (const char *)&payload, sizeof(payload), 0, (struct sockaddr *) &SendToAddr, sizeof(SendToAddr));
        if (ret < 0) {
            std::cerr << "svlVideoCodecUDP: failed to send UDP message: " 
                << byteSent << " / " << serializedSize << ", errono: ";
#if (CISST_OS == CISST_WINDOWS)
            std::cerr << WSAGetLastError() << std::endl;
#else
            std::cerr << strerror(errno) << std::endl;
#endif
            break;
        } else {
#ifdef _DEBUG_
            printf("Sent (%u) : %u / %u\n", ret, byteSent, serializedSize);
#endif
            byteSent += n;
        }

        osaSleep(0.5 * cmn_ms);
        //osaSleep(1 * cmn_ms);
        //osaSleep(1 * cmn_s);
    }
#ifdef _DEBUG_
    printf("Send complete: %u / %u\n", byteSent, serializedSize);
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

    ExperimentResultElements.push_back(result);

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
        svlSampleImageBase * image = dynamic_cast<svlSampleImageBase *>(object->Services()->Create());
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
    width = 1920 * 2;
    height = 1080;
    //*/

    framerate = -1.0;
    Opened = true;

    return SVL_OK;
}

int svlVideoCodecUDP::Read(svlProcInfo* CMN_UNUSED(procInfo), svlSampleImageBase & image, const unsigned int CMN_UNUSED(videoch), const bool CMN_UNUSED(noresize))
{
    // for testing
    static int frameNo = 0;
    if (frameNo++ == 100) {
        // Generate log file
        std::filebuf fb;
        fb.open("Read.txt", std::ios::out);
        std::ostream os(&fb);
#ifdef USE_COMPRESSION
        os << "Compression: " << COMPRESSION_ARG << std::endl;
#else
        os << "No compression" << std::endl;
#endif
        os << "FrameNo,FrameSize,FPS,TimeDeSerialization,TimeProcessing,Timestamp" << std::endl;

        // Put experiment results to log file
        ExperimentResultElementsType::const_iterator it = ExperimentResultElements.begin();
        const ExperimentResultElementsType::const_iterator itEnd = ExperimentResultElements.end();
        for (; it != itEnd; ++it) {
            os << it->FrameNo << "," << it->FrameSize << "," << it->FPS << "," << it->TimeDeSerialization
                << "," << it->TimeProcessing << "," << it->Timestamp << std::endl;
        }
        fb.close();

        exit(1);
    }

    ExperimentResultElement result;
    result.FrameNo = frameNo;
    result.FPS = 0; // will be updated later
    result.Timestamp = osaGetTime();

    const double ticProcessing = osaGetTime();

    // Receive video stream data via UDP
    double senderTick;
    unsigned int serializedSize = GetOneImage(senderTick);
    if (serializedSize == 0) {
        return SVL_FAIL;
    }
    result.FrameSize = serializedSize;

    // Deserialize data
#ifdef USE_CISST_SERIALIZATION
    const double tic = osaGetTime();
    std::string serializedData(ReceiveBuffer, serializedSize);
    DeSerialize(serializedData, image);
    const double toc = osaGetTime();

    result.TimeDeSerialization = toc - tic;
#else
    unsigned char * ptr = image.GetUCharPointer(videoch);
    const unsigned int size = image.GetDataSize(videoch);
    std::cout << "serialized: " << serializedSize << ", size: " << size << std::endl;
    memcpy(ptr, ReceiveBuffer, serializedSize);
#endif

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

    return SVL_OK;
}

int svlVideoCodecUDP::Close()
{
    SocketCleanup();

    //std::cout << "Close called" << std::endl;

    return SVL_OK;
}

int svlVideoCodecUDP::SetPos(const int CMN_UNUSED(pos))
{
    std::cout << "SetPos called" << std::endl;

    return SVL_OK;
}

void svlVideoCodecUDP::Serialize(const cmnGenericObject & originalObject, std::string & serializedObject) 
{
    try {
        SerializationBuffer.str("");
        Serializer->Serialize(originalObject);
        serializedObject = SerializationBuffer.str();
    } catch (std::runtime_error e) {
        std::cerr << "Serialization failed: " << originalObject.ToString() << std::endl;
        std::cerr << e.what() << std::endl;
        serializedObject = "";
    }
}

void svlVideoCodecUDP::DeSerialize(const std::string & serializedObject, cmnGenericObject & originalObject) 
{
    try {
        DeSerializationBuffer.str("");
        DeSerializationBuffer << serializedObject;
        DeSerializer->DeSerialize(originalObject);
    }  catch (std::runtime_error e) {
        std::cerr << "DeSerialization failed: " << e.what() << std::endl;
    }
}

cmnGenericObject * svlVideoCodecUDP::DeSerialize(const std::string & serializedObject) 
{
    cmnGenericObject * deserializedObject = 0;
    try {
        DeSerializationBuffer.str("");
        DeSerializationBuffer << serializedObject;
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

/*
int svlVideoCodecUDP::SetCompression(const svlVideoIO::Compression *compression)
{
    // TODO: Check this
    return SVL_OK;

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
