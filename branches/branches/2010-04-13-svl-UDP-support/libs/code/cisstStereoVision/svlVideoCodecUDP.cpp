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

// Compression level
const unsigned int ZLibCompressionLevel = 3;

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

// For testing purpose
#define TEMPORAL_DIFF
#ifdef TEMPORAL_DIFF
static char * imagePrev = NULL;
static char * imageDiff = NULL;
#endif

//-------------------------------------------------------------------------
//  Constant Definitions
//-------------------------------------------------------------------------
// Network support
#define UDP_RECEIVER_IP "lcsr-minyang.compscidhcp.jhu.edu"
#define UDP_RECV_PORT   20705

struct sockaddr_in SendToAddr;

// Receive buffer
#define RECEIVE_BUFFER_SIZE (10 * 1024 * 1024) // 20 MB
char ReceiveBuffer[RECEIVE_BUFFER_SIZE];

// Internal buffer for serialization and deserialization
std::stringstream SerializationStreamBuffer;
std::stringstream DeSerializationStreamBuffer;

CMN_IMPLEMENT_SERVICES(svlVideoCodecUDP)

svlVideoCodecUDP::svlVideoCodecUDP() :
    svlVideoCodecBase(),
    cmnGenericObject(),
    Width(0), Height(0),
    CurrentSeq(0),
    SerializedClassService(0), SerializedClassServiceSize(0),
    ProcessCount(0),
    BufferYUV(0), BufferYUVSize(0), BufferCompression(0), BufferCompressionSize(0),
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
}

svlVideoCodecUDP::~svlVideoCodecUDP()
{
    Close();

    if (Serializer) delete Serializer;
    if (DeSerializer) delete DeSerializer;
    if (SerializedClassService) delete [] SerializedClassService;

    if (BufferYUV) delete [] BufferYUV;
    if (BufferCompression) delete [] BufferCompression;

#ifdef TEMPORAL_DIFF
    if (imagePrev) delete [] imagePrev;
    if (imageDiff) delete [] imageDiff;
#endif
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
    if (width < 1 || height < 1) return SVL_FAIL;

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

    // Set image width and height
    Width = width;
    Height = height;
    std::cout << "svlVideoCodecUDP: image width: " << Width << ", height: " << Height << std::endl;

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

    // Allocate YUV buffer if not done yet
    unsigned int size = Width * Height * 2;
    if (!BufferYUV) {
        BufferYUV = new unsigned char[size];
        BufferYUVSize = size;
    }
    else if (BufferYUV && BufferYUVSize < size) {
        delete [] BufferYUV;
        BufferYUV = new unsigned char[size];
        BufferYUVSize = size;
    }

    // Allocate compression buffer if not done yet
    size = Width * Height * 3;
    size += size / 100 + 4096;
    if (!BufferCompression) {
        BufferCompression = new unsigned char[size];
        BufferCompressionSize = size;
    }
    else if (BufferCompression && BufferCompressionSize < size) {
        delete [] BufferCompression;
        BufferCompression = new unsigned char[size];
        BufferCompressionSize = size;
    }

    return SVL_OK;
}

int svlVideoCodecUDP::Write(svlProcInfo* procInfo, const svlSampleImageBase &image, const unsigned int videoch)
{
    static int frameNo = 0;
    
    const unsigned int procId = procInfo->id;
    const unsigned int imageWidth = image.GetWidth();
    const unsigned int imageHeight = image.GetHeight();
    const unsigned char * imageCurr = image.GetUCharPointer();
    bool err = false;
    
    _OnSingleThread(procInfo)
    {
        // Remember total number of subimages
        if (ProcessCount == 0) {
            ProcessCount = procInfo->count;
        }

        if (frameNo == 0) {
            std::cout << "svlVideoCodecUDP: processor count: " << ProcessCount << std::endl;
            SubImageOffset.SetSize(procInfo->count);
            SubImageSize.SetSize(procInfo->count);

#ifdef TEMPORAL_DIFF
            const unsigned int imageSize = image.GetDataSize();
            std::cout << "svlVideoCodecUDP: buffered image size: " << imageSize << "(" 
                << imageWidth << "x" << imageHeight << ")" << std::endl;
            imagePrev = new char[imageSize];
            imageDiff = new char[imageSize];

            memset(imagePrev, 0, imageSize);

            // Sender:
            // 2. When a new image comes in, do diff and update previous image buffer
            // 3. Compress diff-ed image instead of the new image

            // Receiver:
            // 1. Create temporal image buffer
            // 2. When a new image comes in, update previous image buffer
            // 3. Recover original image after deserialization
#endif
        } else if (frameNo == 1) {
            // It takes time for UDP receiver to initialze (e.g. shows up output window)
            osaSleep(1.0);
        }

        // Get temporal image difference
#ifdef TEMPORAL_DIFF
        /*  TODO: Continue implementing after getting the idea of how svlImage is stored in memory.
        int idx = 0;
        for (int x = 0; x < imageWidth; ++x) {
            for (int y = 0; y < imageHeight; ++y) {
                idx = x + y * imageWidth;
                imageDiff[idx] = imageCurr[idx] - imagePrev[idx];
            }
        }
        */
#endif

        frameNo++;
    }

    _SynchronizeThreads(procInfo);

    // temporary compression
    //const_cast<svlSampleImageBase&>(image).SetEncoder("jpg", 95);

    const unsigned int procid = procInfo->id;
    const unsigned int proccount = procInfo->count;
    unsigned int start, end, size, offset;
    unsigned long comprsize;
    // Set zlib compression rate
    int compr = ZLibCompressionLevel;//Codec->data[0];

    // Parallelized(multi-threaded) compression
    while (1) {
        // Compute part size and offset
        size = Height / proccount + 1;
        comprsize = BufferCompressionSize / proccount;
        start = procid * size;
        if (start >= Height) break;
        end = start + size;
        if (end > Height) end = Height;
        offset = start * Width;
        size = Width * (end - start);
        SubImageOffset[procid] = procid * comprsize;

        // Convert RGB to YUV422 planar format
        svlConverter::RGB24toYUV422P(const_cast<unsigned char*>(image.GetUCharPointer(videoch)) + offset * 3, BufferYUV + offset * 2, size);

        // Compress part
        if (compress2(BufferCompression + SubImageOffset[procid], &comprsize, BufferYUV + offset * 2, size * 2, compr) != Z_OK) {
            err = true;
            break;
        }
        SubImageSize[procid] = comprsize;

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

        /* TODO: serialize these information as well

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
        */

        // Serialize cisst class service information
        const std::string str = SerializationStreamBuffer.str();
        // Allocate serialization buffer
        SerializedClassServiceSize = static_cast<unsigned int>(str.size());
        if (!SerializedClassService) {
            SerializedClassService = new char[SerializedClassServiceSize];
            std::cout << "svlVideoCodecUDP: size of serialized cisst class service: " << SerializedClassServiceSize << std::endl;
        }
        memcpy(SerializedClassService, str.c_str(), str.size());

        if (NetworkEnabled) {
            //std::cerr << "svlVideoCodecUDP: Sending frame no: " << frameNo << std::endl;
            if (SendUDP() == SVL_FAIL) {
                std::cerr << "svlVideoCodecUDP: failed to send UDP messages" << std::endl;
                return SVL_FAIL;
            }
        }
    }

    return SVL_OK;
}

int svlVideoCodecUDP::Open(const std::string &filename, unsigned int &width, unsigned int &height, double & CMN_UNUSED(framerate))
{
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

    double senderTick;
    unsigned int serializedSize = GetOneImage(senderTick);
    if (serializedSize == 0) {
        std::cout << "svlVideoCodecUDP: failed to read serialized image from socket" << std::endl;
        return SVL_FAIL;
    }

    // Set (return) image width and height
    width = Width;
    height = Height;

    // Allocate YUV buffer if not done yet
    unsigned int size = Width * Height * 2;
    if (!BufferYUV) {
        BufferYUV = new unsigned char[size];
        BufferYUVSize = size;
    } else if (BufferYUV && BufferYUVSize < size) {
        delete [] BufferYUV;
        BufferYUV = new unsigned char[size];
        BufferYUVSize = size;
    }

    // Allocate compression buffer if not done yet
    size = BufferYUVSize + BufferYUVSize / 100 + 4096;
    if (!BufferCompression) {
        BufferCompression = new unsigned char[size];
        BufferCompressionSize = size;
    } else if (BufferCompression && BufferCompressionSize < size) {
        delete [] BufferCompression;
        BufferCompression = new unsigned char[size];
        BufferCompressionSize = size;
    }

    std::cout << "svlVideoCodecUDP: width: " << width << ", height: " << height << std::endl;

    return SVL_OK;
}

int svlVideoCodecUDP::Read(svlProcInfo* procInfo, svlSampleImageBase & image, const unsigned int videoch, const bool CMN_UNUSED(noresize))
{
    // for testing
    static int frameNo = 0;

    std::cout << "svlVideoCodecUDP: Received frame no: " << ++frameNo << "\r";

    // Receive video stream data via UDP
    double senderTick;
    unsigned int serializedSize = GetOneImage(senderTick);
    if (serializedSize == 0) {
        return SVL_FAIL;
    }

    // Uses only a single thread
    if (procInfo && procInfo->id != 0) return SVL_OK;

    // Allocate image buffer if not done yet
    if (Width != image.GetWidth(videoch) || Height != image.GetHeight(videoch)) {
        image.SetSize(videoch, Width, Height);
    }

    // Deserialize cisst class service information
    DeSerializationStreamBuffer.str("");
    DeSerializationStreamBuffer.write(SerializedClassService, SerializedClassServiceSize);
    DeSerializer->DeSerialize(image, false);

    // Rebuild image frame
    unsigned char* img = image.GetUCharPointer(videoch);
    unsigned int i, compressedpartsize, offset, pos;
    unsigned long longsize;
    int ret = SVL_FAIL;

    offset = pos = 0;
    for (i = 0; i < ProcessCount; ++i) {
        compressedpartsize = SubImageSize[i];
#ifdef _DEBUG_
        std::cout << "[" << i << "] size: " << compressedpartsize << std::endl;
#endif

        // Decompress frame part
        longsize = BufferYUVSize - offset;
        if (uncompress(BufferYUV + offset, &longsize, (const Bytef*) (ReceiveBuffer + pos), compressedpartsize) != Z_OK) {
            std::cout << "ERROR: Uncompress failed" << std::endl;
            exit(1);
            return SVL_FAIL;
        }

        // Convert YUV422 planar to RGB format
        svlConverter::YUV422PtoRGB24(BufferYUV + offset, img + offset * 3 / 2, longsize >> 1);

        offset += longsize;
        pos += compressedpartsize;
    }

    return SVL_OK;
}

int svlVideoCodecUDP::Close()
{
    SocketCleanup();

    return SVL_OK;
}

void svlVideoCodecUDP::DeSerialize(const std::string & serializedObject, cmnGenericObject & originalObject)
{
    try {
        DeSerializationStreamBuffer.str("");
        DeSerializationStreamBuffer << serializedObject;
        DeSerializer->DeSerialize(originalObject);
    }  catch (std::runtime_error e) {
        std::cerr << "svlVideoCodecUDP: DeSerialization failed: " << e.what() << std::endl;
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
        std::cerr << "svlVideoCodecUDP: DeSerialization failed: " << e.what() << std::endl;
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
                // Check delimiter to make sure this is correct header
                MSG_HEADER * header = (MSG_HEADER*) buf;
                if (strncmp(DELIMITER_STRING, header->Delimiter, DELIMITER_STRING_SIZE) != 0) {
                    std::cout << "svlVideoCodecUDP: invalid header delimiter" << std::endl;
                    return 0;
                }

                // Frame sequence number
                CurrentSeq = header->FrameSeq;
                // Cisst class service
                SerializedClassServiceSize = header->CisstClassServiceSize;
                if (!SerializedClassService) {
                    SerializedClassService = new char[SerializedClassServiceSize];
                }
                memcpy(SerializedClassService, header->CisstClassService, SerializedClassServiceSize);
                // Set image width and height
                if (Width == 0) {
                    Width = header->Width;
                    Height = header->Height;
                    std::cout << "svlVideoCodecUDP: Set image width: " << Width << ", height: " << Height << std::endl;
                }
                // Subimages
                ProcessCount = header->SubImageCount;
                SubImageSize.SetSize(ProcessCount);
                for (unsigned int i = 0; i < ProcessCount; ++i) {
                    SubImageSize[i] = header->SubImageSize[i];
                }

                serializedSize = SubImageSize.SumOfElements();
                if (serializedSize == 0) {
                    std::cout << "svlVideoCodecUDP: incorrect serialized image size" << std::endl;
                    return 0;
                }

#ifdef _DEBUG_
                header->Print();
#endif
                continue;
            }

            //
            // Get payload
            //
            payload = reinterpret_cast<MSG_PAYLOAD *>(buf);
            if (CurrentSeq == 0) {
                continue;
            }

            // Incorrect frame has arrived.  Then, drop this frame.
            if (CurrentSeq != payload->FrameSeq) {
                std::cout << "svlVideoCodecUDP: frame with incorrect sequence number arrived: expected ("
                    << CurrentSeq << ") actual (" << payload->FrameSeq << ")" << std::endl;
                // TODO: Check if this is OK
                //CurrentSeq = 0;
                //totalByteRecv = 0;

                // drop all data received until now
                //totalByteRecv = 0;
                // start new image frame
                // waiting for header
                continue;
            }

            // Copy payload into receive buffer
            memcpy(ReceiveBuffer + totalByteRecv, payload->Payload, payload->PayloadSize);
            totalByteRecv += payload->PayloadSize;
#ifdef _DEBUG_
            std::cout << totalByteRecv << " / " << serializedSize << std::endl;
#endif

            if (totalByteRecv >= serializedSize) {
                // now receiver has received all the data to rebuild an original
                // image through deserialization.
                received = true;

                return serializedSize;
            }
        }
    }

    return serializedSize;
}

int svlVideoCodecUDP::SendUDP(void)
{
    // Send serialized video stream data via UDP.
    // First, send MSG_HEADER message to let a client know the comprehensive
    // information about the frame.  Then, send serialized image data using
    // 1300-byte UDP messages.

    // Get total size of all serialized subimages
    const unsigned int totalSubImageSize = SubImageSize.SumOfElements();

    // Build MSG_HEADER
    static unsigned int frameSeq = 0;

    MSG_HEADER header;
    header.FrameSeq = ++frameSeq;
    header.CisstClassServiceSize = SerializedClassServiceSize;
    memcpy(header.CisstClassService, SerializedClassService, SerializedClassServiceSize);
    header.Width = Width;
    header.Height = Height;
    header.SubImageCount = ProcessCount;
    for (unsigned int i = 0; i < ProcessCount; ++i) {
        header.SubImageSize[i] = SubImageSize[i];
    }

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
#endif

    // Send image data
    MSG_PAYLOAD payload;
    payload.FrameSeq = frameSeq;

    unsigned int byteSent = 0, totalByteSent = 0, n;
    unsigned char * subImagePtr = BufferCompression;
    for (size_t i = 0; i < ProcessCount; ++i) {
        byteSent = 0;
        subImagePtr = BufferCompression + SubImageOffset[i];

        while (byteSent < SubImageSize[i]) {
            n = (UNIT_MESSAGE_SIZE > (SubImageSize[i] - byteSent) ? (SubImageSize[i] - byteSent) : UNIT_MESSAGE_SIZE);

            memcpy(payload.Payload, reinterpret_cast<const char*>(subImagePtr + byteSent), n);
            payload.PayloadSize = n;

            ret = sendto(SocketSend, (const char *)&payload, sizeof(payload), 0, (struct sockaddr *) &SendToAddr, sizeof(SendToAddr));
            if (ret < 0) {
                std::cerr << "svlVideoCodecUDP: failed to send UDP message: " 
                    << byteSent << " / " << SubImageSize[i] << ", errono: ";
#if (CISST_OS == CISST_WINDOWS)
                std::cerr << WSAGetLastError() << std::endl;
#else
                std::cerr << strerror(errno) << std::endl;
#endif
                break;
            } else {
                byteSent += n;
#ifdef _DEBUG_
                printf("Sent (%u) : %u / %u\n", ret, byteSent, SubImageSize[i]);
#endif
            }

            //static int count = 0;
            //if (count++ == 10) {
                osaSleep(0.5 * cmn_ms);
            //    count = 0;
            //}
        }
        
        totalByteSent += byteSent;
    }
    
#ifdef _DEBUG_
    printf("Send complete: %u / %u\n", totalByteSent, SubImageSize.SumOfElements());
#endif

    return sizeof(header) + totalByteSent;
}

