/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: svlVideoCodecTCPStream.cpp 2095 2010-11-30 15:24:22Z adeguet1 $
  
  Author(s):  Balazs Vagvolgyi & Min Yang Jung
  Created on: 2010

  (C) Copyright 2006-2007 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#include "svlVideoCodecTCPStream.h"
#include <cisstCommon/cmnGetChar.h>
#include <cisstStereoVision/svlConverters.h>
#include <cisstStereoVision/svlSyncPoint.h>
#include <cisstOSAbstraction/osaSleep.h>

#include "zlib.h"

#if (CISST_OS == CISST_WINDOWS)
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #define WINSOCKVERSION MAKEWORD(2,2)

    static int inet_pton4(const char *src, unsigned char *dst)
    {
        static const char digits[] = "0123456789";
        int saw_digit, octets, ch;
        unsigned char tmp[4], *tp;

        saw_digit = 0;
        octets = 0;
        tp = tmp;
        *tp = 0;
        while ((ch = *src++) != '\0') {
            const char *pch;

            if ((pch = strchr(digits, ch)) != NULL) {
                unsigned int val = *tp * 10 + (unsigned int) (pch - digits);

                if (val > 255) return (0);
                *tp = (unsigned char) val;
                if (!saw_digit) {
                    if (++octets > 4) return (0);
                    saw_digit = 1;
                }
            } else if (ch == '.' && saw_digit) {
                if (octets == 4) return (0);
                *++tp = 0;
                saw_digit = 0;
            } else return (0);
        }
        if (octets < 4) return (0);
        memcpy(dst, tmp, 4);
        return (1);
    }

#else
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <netinet/in.h>
    #include <sys/socket.h>
    #include <sys/types.h>
    #include <errno.h>
#endif

#if (CISST_OS == CISST_WINDOWS)
    #define __errno         WSAGetLastError()
    #define __ECONNABORTED  WSAECONNABORTED
    #define __EAGAIN        WSATRY_AGAIN
#else
    #define __errno         errno
    #define __ECONNABORTED  ECONNABORTED
    #define __EAGAIN        EAGAIN
#endif

//#define _NET_VERBOSE_

#define MAX_CLIENTS         5
#define PACKET_SIZE         1400u
#define BROKEN_FRAME        1


/*************************************/
/*** svlVideoCodecTCPStream class ****/
/*************************************/

CMN_IMPLEMENT_SERVICES(svlVideoCodecTCPStream)

svlVideoCodecTCPStream::svlVideoCodecTCPStream() :
    svlVideoCodecBase(),
    cmnGenericObject(),
    CodecName("CISST Video Stream over TCP/IP"),
    FrameStartMarker("\r\nFrame\r\n"),
    File(0),
    PartCount(0),
    Width(0),
    Height(0),
    BegPos(-1),
    EndPos(-1),
    Pos(-1),
    Opened(false),
    Writing(false),
    Timestamp(-1.0),
    yuvBuffer(0),
    yuvBufferSize(0),
    comprBuffer(0),
    comprBufferSize(0),
    PacketData(0),
    streamingBuffer(0),
    streamingBufferSize(0),
    ServerSocket(-1),
    ServerThread(0),
    ServerInitEvent(0),
    ServerInitialized(false),
    ReceiveSocket(-1),
    ReceiveThread(0),
    ReceiveInitEvent(0),
    ReceiveInitialized(false)
{
    SetName("CISST Video Stream over TCP/IP");
    SetExtensionList(".ncvi;");
    SetMultithreaded(true);
    SetVariableFramerate(true);

    SockAddr = new char[sizeof(sockaddr_in)];
    PacketData = new char[PACKET_SIZE];
}

svlVideoCodecTCPStream::~svlVideoCodecTCPStream()
{
    Close();
    if (yuvBuffer) delete [] yuvBuffer;
    if (comprBuffer) delete [] comprBuffer;
    if (streamingBuffer) delete streamingBuffer;
    if (SockAddr) delete [] SockAddr;
    if (PacketData) delete [] PacketData;
}

int svlVideoCodecTCPStream::Open(const std::string &filename, unsigned int &width, unsigned int &height, double &framerate)
{
    if (Opened || ParseFilename(filename) != SVL_OK) return SVL_FAIL;

    while (1) {

        Opened = true;
        Writing = false;

        // Releasing existing buffers
        delete [] yuvBuffer;
        yuvBuffer = 0;
        yuvBufferSize = 0;
        delete streamingBuffer;
        streamingBuffer = 0;
        streamingBufferSize = 0;
        Width = Height = 0;

        // Start data receiving thread
        ReceiveInitialized = false;
        KillReceiveThread = false;
        ReceiveThread = new osaThread;
        ReceiveInitEvent = new osaThreadSignal;
        ReceiveThread->Create<svlVideoCodecTCPStream, int>(this, &svlVideoCodecTCPStream::ReceiveProc, 0);
        ReceiveInitEvent->Wait();
        if (ReceiveInitialized == false) break;

        // Wait until header is received
        while (Width == 0 || Height == 0) osaSleep(0.1);

        Pos = BegPos = EndPos = 0;
        width = Width;
        height = Height;
        framerate = -1.0;

        return SVL_OK;
    }

    Close();
    return SVL_FAIL;
}

int svlVideoCodecTCPStream::Create(const std::string &filename, const unsigned int width, const unsigned int height, const double CMN_UNUSED(framerate))
{
	if (Opened || ParseFilename(filename) != SVL_OK ||
        width < 1 || width > 4096 || height < 1 || height > 4096) return SVL_FAIL;

    if (!Codec) {
        // Set default compression level to 4
        Codec = reinterpret_cast<svlVideoIO::Compression*>(new unsigned char[sizeof(svlVideoIO::Compression)]);
        std::string name("CISST Video Stream (CVI)");
        memset(&(Codec->extension[0]), 0, 16);
        memcpy(&(Codec->extension[0]), ".ncvi", 5);
        memset(&(Codec->name[0]), 0, 64);
        memcpy(&(Codec->name[0]), name.c_str(), std::min(static_cast<int>(name.length()), 63));
        Codec->size = sizeof(svlVideoIO::Compression);
        Codec->datasize = 1;
        Codec->data[0] = 4;
    }

    unsigned int size;

    while (1) {

        Opened = true;
	    Writing = true;

        // Allocate YUV buffer if not done yet
        size = width * height * 2;
        if (!yuvBuffer) {
            yuvBuffer = new unsigned char[size];
            yuvBufferSize = size;
        }
        else if (yuvBuffer && yuvBufferSize < size) {
            delete [] yuvBuffer;
            yuvBuffer = new unsigned char[size];
            yuvBufferSize = size;
        }

        // Allocate compression buffer if not done yet
        size = width * height * 3;
        size += size / 100 + 4096;
        if (!comprBuffer) {
            comprBuffer = new unsigned char[size];
            comprBufferSize = size;
        }
        else if (comprBuffer && comprBufferSize < size) {
            delete [] comprBuffer;
            comprBuffer = new unsigned char[size];
            comprBufferSize = size;
        }

        // Allocate streaming buffers if not done yet
        if (!streamingBuffer) {
            streamingBuffer = new svlBufferMemory(size);
            streamingBufferSize = size;
        }
        else if (streamingBuffer && streamingBufferSize < size) {
            delete streamingBuffer;
            streamingBuffer = new svlBufferMemory(size);
            streamingBufferSize = size;
        }

        StreamingBufferUsedSize = 0;
        StreamingBufferUsedID = 0;

        // Start data saving thread
        ServerInitialized = false;
        KillServerThread = false;
        ServerThread = new osaThread;
        ServerInitEvent = new osaThreadSignal;
        ServerThread->Create<svlVideoCodecTCPStream, unsigned short>(this, &svlVideoCodecTCPStream::ServerProc, SocketPort);
        ServerInitEvent->Wait();
        if (ServerInitialized == false) break;

        BegPos = EndPos = Pos = 0;
        Width = width;
        Height = height;

        return SVL_OK;
    }

	Close();
	return SVL_FAIL;
}

int svlVideoCodecTCPStream::Close()
{
    if (Opened) {
        if (Writing) {
            // Shut down server
            if (ServerThread && ServerInitialized) {
                KillServerThread = true;
                ServerThread->Wait();
                delete ServerThread;
                ServerThread = 0;
                delete ServerInitEvent;
                ServerInitEvent = 0;
            }
        }
        else {
            // Shut down receiver
            if (ReceiveThread && ReceiveInitialized) {
                KillReceiveThread = true;
                ReceiveThread->Wait();
                delete ReceiveThread;
                ReceiveThread = 0;
                delete ReceiveInitEvent;
                ReceiveInitEvent = 0;
            }
        }
    }

    Width = 0;
    Height = 0;
    BegPos = -1;
    EndPos = -1;
    Pos = -1;
    Writing = false;
    Opened = false;

    return SVL_OK;
}

int svlVideoCodecTCPStream::GetBegPos() const
{
    return BegPos;
}

int svlVideoCodecTCPStream::GetEndPos() const
{
    return EndPos;
}

int svlVideoCodecTCPStream::GetPos() const
{
    return Pos;
}

svlVideoIO::Compression* svlVideoCodecTCPStream::GetCompression() const
{
    // The caller will need to release it by calling the
    // svlVideoIO::ReleaseCompression() method
    unsigned int size = sizeof(svlVideoIO::Compression);
    svlVideoIO::Compression* compression = reinterpret_cast<svlVideoIO::Compression*>(new unsigned char[size]);

    std::string name("CISST Video Stream over TCP/IP");
    memset(&(compression->extension[0]), 0, 16);
    memcpy(&(compression->extension[0]), ".ncvi", 5);
    memset(&(compression->name[0]), 0, 64);
    memcpy(&(compression->name[0]), name.c_str(), std::min(static_cast<int>(name.length()), 63));
    compression->size = size;
    compression->datasize = 1;
    if (Codec) compression->data[0] = Codec->data[0];
    else compression->data[0] = 4; // Set default compression level to 4

    return compression;
}

int svlVideoCodecTCPStream::SetCompression(const svlVideoIO::Compression *compression)
{
    if (Opened || !compression || compression->size < sizeof(svlVideoIO::Compression)) return SVL_FAIL;

    std::string extensionlist(GetExtensions());
    std::string extension(compression->extension);
    extension += ";";
    if (extensionlist.find(extension) == std::string::npos) {
        // Codec parameters do not match this codec
        return SVL_FAIL;
    }

    svlVideoIO::ReleaseCompression(Codec);
    Codec = reinterpret_cast<svlVideoIO::Compression*>(new unsigned char[sizeof(svlVideoIO::Compression)]);

    std::string name("CISST Video Stream over TCP/IP");
    memset(&(Codec->extension[0]), 0, 16);
    memset(&(Codec->name[0]), 0, 64);
    memcpy(&(Codec->name[0]), name.c_str(), std::min(static_cast<int>(name.length()), 63));
    Codec->size = sizeof(svlVideoIO::Compression);
    Codec->datasize = 1;
    if (compression->data[0] <= 9) Codec->data[0] = compression->data[0];
    else Codec->data[0] = 4;

    return SVL_OK;
}

int svlVideoCodecTCPStream::DialogCompression()
{
    if (Opened) return SVL_FAIL;

    std::cout << std::endl << " # Enter compression level [0-9]: ";
    int level = 0;
    while (level < '0' || level > '9') level = cmnGetChar();
    level -= '0';
    std::cout << level << std::endl;

    svlVideoIO::ReleaseCompression(Codec);
    Codec = reinterpret_cast<svlVideoIO::Compression*>(new unsigned char[sizeof(svlVideoIO::Compression)]);

    std::string name("CISST Video Stream over TCP/IP");
    memset(&(Codec->extension[0]), 0, 16);
    memcpy(&(Codec->extension[0]), ".ncvi", 5);
    memset(&(Codec->name[0]), 0, 64);
    memcpy(&(Codec->name[0]), name.c_str(), std::min(static_cast<int>(name.length()), 63));
    Codec->size = sizeof(svlVideoIO::Compression);
    Codec->datasize = 1;
    Codec->data[0] = static_cast<unsigned char>(level);

	return SVL_OK;
}

double svlVideoCodecTCPStream::GetTimestamp() const
{
    if (!Opened || Writing) return -1.0;
    return Timestamp;
}

int svlVideoCodecTCPStream::SetTimestamp(const double timestamp)
{
    if (!Opened || !Writing) return SVL_FAIL;
    Timestamp = timestamp;
    return SVL_OK;
}

int svlVideoCodecTCPStream::Read(svlProcInfo* procInfo, svlSampleImage &image, const unsigned int videoch, const bool noresize)
{
    if (videoch >= image.GetVideoChannels()) return SVL_FAIL;
    if (!Opened || Writing) return SVL_FAIL;

    // Uses only a single thread
    if (procInfo && procInfo->id != 0) return SVL_OK;

    unsigned int i, width, height, partcount, compressedpartsize, offset = 0, strmoffset;
    unsigned long longsize;
    unsigned char *img, *strmbuf = 0;
    int ret = SVL_FAIL;

    // Wait until new frame is received
    while (!strmbuf) {
        strmbuf = streamingBuffer->Pull(true, 0.1);
    }

    while (1) {
        // file start marker
        strmoffset = FrameStartMarker.length();

        // frame data size after file start marker
        strmoffset += sizeof(unsigned int);

        // image width
        width = reinterpret_cast<unsigned int*>(strmbuf + strmoffset)[0];
        strmoffset += sizeof(unsigned int);

        // image height
        height = reinterpret_cast<unsigned int*>(strmbuf + strmoffset)[0];
        strmoffset += sizeof(unsigned int);

        // timestamp
        Timestamp = reinterpret_cast<double*>(strmbuf + strmoffset)[0];
        strmoffset += sizeof(double);
        // Do not return timestamp for now...
        Timestamp = -1;

        // part count
        partcount = reinterpret_cast<unsigned int*>(strmbuf + strmoffset)[0];
        strmoffset += sizeof(unsigned int);

        // Allocate image buffer if not done yet
        if (width != image.GetWidth(videoch) || height != image.GetHeight(videoch)) {
            if (noresize) break;
            image.SetSize(videoch, width, height);
        }

        img = image.GetUCharPointer(videoch);

        for (i = 0; i < partcount; i ++) {
            // compressed part size
            compressedpartsize = reinterpret_cast<unsigned int*>(strmbuf + strmoffset)[0];
            strmoffset += sizeof(unsigned int);

            // Decompress frame part
            longsize = yuvBufferSize - offset;
            if (uncompress(yuvBuffer + offset, &longsize, strmbuf + strmoffset, compressedpartsize) != Z_OK) break;

            // Convert YUV422 planar to RGB format
            svlConverter::YUV422PtoRGB24(yuvBuffer + offset, img + offset * 3 / 2, longsize >> 1);

            strmoffset += compressedpartsize;
            offset += longsize;
        }
        if (i < PartCount) break;

        ret = SVL_OK;

        break;
    }

    return ret;
}

int svlVideoCodecTCPStream::Write(svlProcInfo* procInfo, const svlSampleImage &image, const unsigned int videoch)
{
    if (videoch >= image.GetVideoChannels()) return SVL_FAIL;
    if (!Opened || !Writing) return SVL_FAIL;
	if (Width != image.GetWidth(videoch) || Height != image.GetHeight(videoch)) return SVL_FAIL;

    bool err = false;

    if (Pos == 0) {

        _OnSingleThread(procInfo)
        {
            // Initialize multithreaded processing
            ComprPartOffset.SetSize(procInfo->count);
            ComprPartSize.SetSize(procInfo->count);
        }

        // Synchronize threads
        _SynchronizeThreads(procInfo);

        if (err) return SVL_FAIL;
    }

    const unsigned int procid = procInfo->id;
    const unsigned int proccount = procInfo->count;
    unsigned int i, start, end, size, offset;
    unsigned char* strmbuf = streamingBuffer->GetPushBuffer();
    unsigned long comprsize;
    int compr = Codec->data[0];

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

    // Synchronize threads
    _SynchronizeThreads(procInfo);

    if (err) return SVL_FAIL;

    // Single threaded data serialization phase
    _OnSingleThread(procInfo)
    {
        const double timestamp = image.GetTimestamp();

        // Add "frame start marker"
        memcpy(strmbuf, FrameStartMarker.c_str(), FrameStartMarker.length());
        StreamingBufferUsedSize = FrameStartMarker.length();

        // Add "data size after frame start marker"
        size = sizeof(unsigned int) * (4 + proccount) + sizeof(double);
        for (i = 0; i < proccount; i ++) size += ComprPartSize[i];
        memcpy(strmbuf + StreamingBufferUsedSize, &size, sizeof(unsigned int));
        StreamingBufferUsedSize += sizeof(unsigned int);

        // Add "width"
        memcpy(strmbuf + StreamingBufferUsedSize, &Width, sizeof(unsigned int));
        StreamingBufferUsedSize += sizeof(unsigned int);

        // Add "height"
        memcpy(strmbuf + StreamingBufferUsedSize, &Height, sizeof(unsigned int));
        StreamingBufferUsedSize += sizeof(unsigned int);

        // Add "timestamp"
        memcpy(strmbuf + StreamingBufferUsedSize, &timestamp, sizeof(double));
        StreamingBufferUsedSize += sizeof(double);

        // Add "partcount"
        memcpy(strmbuf + StreamingBufferUsedSize, &proccount, sizeof(unsigned int));
        StreamingBufferUsedSize += sizeof(unsigned int);

        for (i = 0; i < proccount; i ++) {
            // Add "compressed part size"
            memcpy(strmbuf + StreamingBufferUsedSize, &(ComprPartSize[i]), sizeof(unsigned int));
            StreamingBufferUsedSize += sizeof(unsigned int);

            // Add compressed frame
            memcpy(strmbuf + StreamingBufferUsedSize, comprBuffer + ComprPartOffset[i], ComprPartSize[i]);
            StreamingBufferUsedSize += ComprPartSize[i];
        }

        // Signal data saving thread to start writing
        streamingBuffer->Push();

		EndPos ++; Pos ++;
    }

	return SVL_OK;
}

void* svlVideoCodecTCPStream::ServerProc(unsigned short port)
{
    unsigned int clientid;

#if (CISST_OS == CISST_WINDOWS)
    bool wsa_running = false;
#endif
    bool socket_open = false;

    while (1) {

#if (CISST_OS == CISST_WINDOWS)
        WSADATA wsaData;
        if (WSAStartup(WINSOCKVERSION, &wsaData) != 0) {
#ifdef _NET_VERBOSE_
            std::cerr << "svlVideoCodecTCPStream::ServerProc - WSAStartup failed" << std::endl;
#endif
            break;
        }
        wsa_running = true;
#ifdef _NET_VERBOSE_
        std::cerr << "svlVideoCodecTCPStream::ServerProc - WSAStartup success" << std::endl;
#endif
#endif

        ServerSocket = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (ServerSocket < 0) {
#ifdef _NET_VERBOSE_
            std::cerr << "svlVideoCodecTCPStream::ServerProc - cannot create socket" << std::endl;
#endif
            break;
        }
        socket_open = true;
#ifdef _NET_VERBOSE_
        std::cerr << "svlVideoCodecTCPStream::ServerProc - socket created" << std::endl;
#endif

        int reuse = 1;
#if (CISST_OS == CISST_WINDOWS)
        if (setsockopt(ServerSocket, SOL_SOCKET, SO_REUSEADDR, (const char*)(&reuse), sizeof(int)) == 0) {
#else
        if (setsockopt(ServerSocket, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(int)) == 0) {
#endif
#ifdef _NET_VERBOSE_
            std::cerr << "svlVideoCodecTCPStream::ServerProc - setsockopt success" << std::endl;
#endif
        }
        else {
#ifdef _NET_VERBOSE_
            std::cerr << "svlVideoCodecTCPStream::ServerProc - setsockopt failed" << std::endl;
#endif
        }

        sockaddr_in address;
        memset(&address, 0, sizeof(sockaddr_in));

        address.sin_family = AF_INET;
        address.sin_port = htons(port);
        address.sin_addr.s_addr = INADDR_ANY;

        if (bind(ServerSocket, (sockaddr*)(&address), sizeof(sockaddr_in)) < 0) {
#ifdef _NET_VERBOSE_
            std::cerr << "svlVideoCodecTCPStream::ServerProc - bind failed (port=" << port << ")" << std::endl;
#endif
            break;
        }
#ifdef _NET_VERBOSE_
        std::cerr << "svlVideoCodecTCPStream::ServerProc - bind success (port=" << port << ")" << std::endl;
#endif

        if (listen(ServerSocket, MAX_CLIENTS) < 0)
        {
#ifdef _NET_VERBOSE_
            std::cerr << "svlVideoCodecTCPStream::ServerProc - listen failed" << std::endl;
#endif
            break;
        }
#ifdef _NET_VERBOSE_
        std::cerr << "svlVideoCodecTCPStream::ServerProc - listen success" << std::endl;
#endif

        ServerInitialized = true;
        ServerInitEvent->Raise();

        int connection;

        // Create client pool
        SendThread.SetSize(MAX_CLIENTS);
        SendConnection.SetSize(MAX_CLIENTS);
        KillSendThread.SetSize(MAX_CLIENTS);
        for (clientid = 0; clientid < MAX_CLIENTS; clientid ++) {
            SendThread[clientid]      = new osaThread;
            SendConnection[clientid]  = -1;
            KillSendThread[clientid]  = false;
        }

        fd_set fds;
        timeval tv;

        while (!KillServerThread) {

            // Wait 0.5s for new connection, then continue
            FD_ZERO(&fds); FD_SET(ServerSocket, &fds);
            tv.tv_sec  = 0; tv.tv_usec = 500000;
            if (select(ServerSocket + 1, &fds, 0, 0, &tv) <= 0 || !FD_ISSET(ServerSocket, &fds)) {
                continue;
            }
#ifdef _NET_VERBOSE_
            std::cerr << "svlVideoCodecTCPStream::ServerProc - incoming connection..." << std::endl;
#endif
            
            connection = accept(ServerSocket, 0, 0);
            if (connection < 0) {
#ifdef _NET_VERBOSE_
                std::cerr << "svlVideoCodecTCPStream::ServerProc - accept failed" << std::endl;
#endif
                continue;
            }
#ifdef _NET_VERBOSE_
            std::cerr << "svlVideoCodecTCPStream::ServerProc - accept success" << std::endl;
#endif

            // Find free client in the pool
            clientid = 0;
            while (clientid < MAX_CLIENTS && SendConnection[clientid] >= 0) clientid ++;
            if (clientid >= MAX_CLIENTS) {
#ifdef _NET_VERBOSE_
                std::cerr << "svlVideoCodecTCPStream::ServerProc - too many clients" << std::endl;
#endif
#if (CISST_OS == CISST_WINDOWS)
                shutdown(connection, SD_BOTH);
#else
                shutdown(connection, SHUT_RDWR);
#endif
#if (CISST_OS == CISST_WINDOWS)
                closesocket(connection);
#else
                close(connection);
#endif
                continue;
            }
#ifdef _NET_VERBOSE_
            std::cerr << "svlVideoCodecTCPStream::ServerProc - thread assigned to client" << std::endl;
#endif

            KillSendThread[clientid] = false;
            SendConnection[clientid] = connection;
            SendThread[clientid]->Create<svlVideoCodecTCPStream, unsigned int>(this, &svlVideoCodecTCPStream::SendProc, clientid);
            // The thread will manage the shutdown and closing of the connection when done sending or terminated
        }

        break;
    }

    if (ServerInitialized) {
#ifdef _NET_VERBOSE_
        std::cerr << "svlVideoCodecTCPStream::ServerProc - shutting down all connections" << std::endl;
#endif
        // Destroy client pool
        for (clientid = 0; clientid < SendThread.size(); clientid ++) {
            if (SendConnection[clientid] >= 0) {
                KillSendThread[clientid] = true;
                SendThread[clientid]->Wait();
            }
            delete SendThread[clientid];
        }
        ServerInitialized = false;
    }
    else {
#ifdef _NET_VERBOSE_
        std::cerr << "svlVideoCodecTCPStream::ServerProc - server failed to initialize" << std::endl;
#endif
        ServerInitEvent->Raise();
    }

    if (socket_open) {
#if (CISST_OS == CISST_WINDOWS)
        closesocket(ServerSocket);
#else
        close(ServerSocket);
#endif
    }
#if (CISST_OS == CISST_WINDOWS)
    if (wsa_running) WSACleanup();
#endif

    ServerSocket = -1;

    return this;
}

void* svlVideoCodecTCPStream::SendProc(unsigned int clientid)
{
    int ret, size;
    unsigned char* strmbuf;

    while (!KillSendThread[clientid]) {

        // Wait until new frame is acquired
        strmbuf = 0;
        while (!strmbuf && !KillSendThread[clientid]) {
            strmbuf = streamingBuffer->Pull(true, 0.1);
        }
        size = static_cast<int>(StreamingBufferUsedSize);

#ifdef _NET_VERBOSE_
        std::cerr << "svlVideoCodecTCPStream::SendProc - sending frame (" << size << " bytes)" << std::endl;
#endif

        while (size > 0) {
            ret = send(SendConnection[clientid], reinterpret_cast<const char*>(strmbuf), size, 0);

            if (ret < 0) {
#ifdef _NET_VERBOSE_
                std::cerr << "svlVideoCodecTCPStream::SendProc - send failed" << std::endl;
#endif
                KillSendThread[clientid] = true;
                break;
            }

            strmbuf += ret;
            size -= ret;
        }
    }

#if (CISST_OS == CISST_WINDOWS)
    shutdown(SendConnection[clientid], SD_BOTH);
#else
    shutdown(SendConnection[clientid], SHUT_RDWR);
#endif
#if (CISST_OS == CISST_WINDOWS)
    closesocket(SendConnection[clientid]);
#else
    close(SendConnection[clientid]);
#endif
    SendConnection[clientid] = -1;

#ifdef _NET_VERBOSE_
    std::cerr << std::endl << "svlVideoCodecTCPStream::SendProc - client (" << clientid << ") shut down" << std::endl;
#endif

    return this;
}

void* svlVideoCodecTCPStream::ReceiveProc(int CMN_UNUSED(param))
{
    int ret;
#if (CISST_OS == CISST_WINDOWS)
    bool wsa_running      = false;
#endif
    bool socket_open      = false;
    bool socket_connected = false;

    while (1) {

#if (CISST_OS == CISST_WINDOWS)
        WSADATA wsaData;
        if (WSAStartup(WINSOCKVERSION, &wsaData) != 0) {
#ifdef _NET_VERBOSE_
            std::cerr << "svlVideoCodecTCPStream::ReceiveProc - WSAStartup failed" << std::endl;
#endif
            break;
        }
        wsa_running = true;
#ifdef _NET_VERBOSE_
        std::cerr << "svlVideoCodecTCPStream::ReceiveProc - WSAStartup success" << std::endl;
#endif
#endif

        ReceiveSocket = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (ReceiveSocket < 0) {
#ifdef _NET_VERBOSE_
            std::cerr << "svlVideoCodecTCPStream::ReceiveProc - cannot create socket" << std::endl;
#endif
            break;
        }
        socket_open = true;
#ifdef _NET_VERBOSE_
        std::cerr << "svlVideoCodecTCPStream::ReceiveProc - socket created" << std::endl;
#endif

        sockaddr_in address;
        memset(&address, 0, sizeof(sockaddr_in));
        address.sin_family = AF_INET;
        address.sin_port = htons(SocketPort);

#if (CISST_OS == CISST_WINDOWS)
        ret = inet_pton4(SocketAddress.c_str(), (unsigned char*)(&address.sin_addr));
#else
        ret = inet_pton(AF_INET, SocketAddress.c_str(), &address.sin_addr);
#endif
        if (ret <= 0) {
#ifdef _NET_VERBOSE_
            std::cerr << "svlVideoCodecCVINet::ReceiveProc - inet_pton error (" << ret  << ")" << std::endl;
#endif
            break;
        }
#ifdef _NET_VERBOSE_
        std::cerr << "svlVideoCodecCVINet::ReceiveProc - inet_pton success (" << SocketAddress << ":"  << SocketPort << ")" << std::endl;
#endif

        ret = connect(ReceiveSocket, (sockaddr*)(&address), sizeof(sockaddr_in));
        if (ret < 0) {
#ifdef _NET_VERBOSE_
            std::cerr << "svlVideoCodecCVINet::ReceiveProc - connect failed (" << __errno << ")" << std::endl;
#endif
            break;
        }
        socket_connected = true;
#ifdef _NET_VERBOSE_
        std::cerr << "svlVideoCodecCVINet::ReceiveProc - connect success" << std::endl;
#endif

        ReceiveInitialized = true;
        ReceiveInitEvent->Raise();

        while (!KillReceiveThread) {

            ret = Receive();
            if (ret == SVL_FAIL) {
                break;
            }
            else if (ret == BROKEN_FRAME) {

                // Check if socket is still connected
                int optval;
                socklen_t optlen = sizeof(optval);
#if (CISST_OS == CISST_WINDOWS)
                ret = getsockopt(ReceiveSocket, SOL_SOCKET, SO_ERROR, (char*)(&optval), &optlen);
#else
                ret = getsockopt(ReceiveSocket, SOL_SOCKET, SO_ERROR, &optval, &optlen);
#endif

                if (optval || ret) {
#ifdef _NET_VERBOSE_
                    std::cerr << "svlVideoCodecTCPStream::ReceiveProc - connection lost" << std::endl;
#endif
                    ReceiveThreadError = true;
                    break;
                }
                else {
#ifdef _NET_VERBOSE_
                    std::cerr << "svlVideoCodecTCPStream::ReceiveProc - corrupt frame" << std::endl;
#endif
                    osaSleep(0.1);
                }
            }
        }

        break;
    }

    if (!socket_open) {
#ifdef _NET_VERBOSE_
        std::cerr << "svlVideoCodecTCPStream::ReceiveProc - client failed to initialize" << std::endl;
#endif
        ReceiveInitEvent->Raise();
    }

    if (socket_connected) {
#if (CISST_OS == CISST_WINDOWS)
        shutdown(ReceiveSocket, SD_BOTH);
#else
        shutdown(ReceiveSocket, SHUT_RDWR);
#endif
    }
    if (socket_open) {
#if (CISST_OS == CISST_WINDOWS)
        closesocket(ReceiveSocket);
#else
        close(ReceiveSocket);
#endif
    }
#if (CISST_OS == CISST_WINDOWS)
    if (wsa_running) WSACleanup();
#endif

    ReceiveSocket = -1;

    return this;
}

int svlVideoCodecTCPStream::Receive()
{
    const unsigned char* fsm = reinterpret_cast<const unsigned char*>(FrameStartMarker.c_str());
    const unsigned int fsm_len = FrameStartMarker.length();
    unsigned char* buffer = reinterpret_cast<unsigned char*>(PacketData);
    unsigned int size = PACKET_SIZE;
    unsigned int framesize = 0;
    bool framestart = false;
    bool started = false;
    fd_set mask;
    int ret;


    while (1) {

        // Wait for new packet
        FD_ZERO(&mask);
        FD_SET(ReceiveSocket, &mask);
        ret = select(ReceiveSocket + 1, &mask, 0, 0, 0);
        if (ret < 0) {
#ifdef _NET_VERBOSE_
            std::cerr << "svlVideoCodecTCPStream::Receive - select failed (" << __errno << ")" << std::endl;
#endif
            break;
        }

        // Receive packet
        if (FD_ISSET(ReceiveSocket, &mask) == 0) {
#ifdef _NET_VERBOSE_
            std::cerr << "svlVideoCodecTCPStream::Receive - FD_ISSET failed" << std::endl;
#endif
            continue;
        }

        ret = recv(ReceiveSocket,
                   reinterpret_cast<char*>(buffer),
                   std::min(size, PACKET_SIZE),
                   0);
        if (ret == 0) {
#ifdef _NET_VERBOSE_
                    std::cerr << "svlVideoCodecTCPStream::Receive - connection terminated" << std::endl;
#endif
            break;
        }
        else if (ret > 0) {
            // Check if it starts with a "frame start marker"
            framestart = CompareData(buffer, fsm, fsm_len);

            if (!framestart) {
                if (!started) {
                    // Looking for "frame start marker"
                    continue;
                }
            }
            else {
                if (started) {
#ifdef _NET_VERBOSE_
                    std::cerr << "svlVideoCodecTCPStream::Receive - broken frame" << std::endl;
#endif
                    return BROKEN_FRAME;
                }
                else {
                    // Extract dimensions from header
                    int offset = fsm_len + sizeof(unsigned int);
                    const unsigned int w = reinterpret_cast<unsigned int*>(buffer + offset)[0];
                    offset += sizeof(unsigned int);
                    const unsigned int h = reinterpret_cast<unsigned int*>(buffer + offset)[0];

                    size = w * h * 2;
                    if (yuvBuffer && yuvBufferSize < size) {
                        delete [] yuvBuffer;
                        yuvBuffer = 0;
                        yuvBufferSize = 0;
                    }
                    if (yuvBuffer == 0 || yuvBufferSize == 0) {
                        yuvBuffer = new unsigned char[size];
                        yuvBufferSize = size;
                    }
                    size += size / 100 + 4096;
                    if (streamingBuffer && streamingBufferSize < size) {
                        delete streamingBuffer;
                        streamingBuffer = 0;
                        streamingBufferSize = 0;
                    }
                    if (streamingBuffer == 0 || streamingBufferSize == 0) {
                        streamingBuffer = new svlBufferMemory(size);
                        streamingBufferSize = size;
                    }

                    buffer = streamingBuffer->GetPushBuffer();
                    memcpy(buffer, PacketData, ret);

                    Width = w;
                    Height = h;

                    // Get "data size after frame start marker"
                    framesize = reinterpret_cast<unsigned int*>(buffer + fsm_len)[0] + fsm_len;
                    // Don't request the rest of the data if buffer size is too small
                    if (framesize > size) {
#ifdef _NET_VERBOSE_
                        std::cerr << "svlVideoCodecTCPStream::Receive - received data larger than buffer" << std::endl;
#endif
                        break;
                    }

                    // Signal "frame start"
                    size = 0;
                    started = true;
                }
            }

            buffer += ret;
            size += ret;

            if (framesize == size) {
#ifdef _NET_VERBOSE_
                std::cerr << "svlVideoCodecTCPStream::Receive - frame received (" << framesize << ")" << std::endl;
#endif
                StreamingBufferUsedSize = framesize;
                streamingBuffer->Push();
                return SVL_OK;
            }
        }
        else {
            int err = __errno;
            if (err == __EAGAIN) {
#ifdef _NET_VERBOSE_
                std::cerr << "svlVideoCodecTCPStream::Receive - recv failed, retrying..." << std::endl;
#endif
            }
            else {
                if (err == __ECONNABORTED) {
#ifdef _NET_VERBOSE_
                    std::cerr << "svlVideoCodecTCPStream::Receive - recv failed, connection aborted" << std::endl;
#endif
                }
                else {
#ifdef _NET_VERBOSE_
                    std::cerr << "svlVideoCodecTCPStream::Receive - recv failed (" << err << ")" << std::endl;
#endif
                }
                KillReceiveThread = true;
                break;
            }
        }
    }

    return SVL_FAIL;
}

bool svlVideoCodecTCPStream::CompareData(const unsigned char* data1, const unsigned char* data2, unsigned int size)
{
    while (size) {
        if (data1[size - 1] != data2[size - 1]) break;
        size --;
    }
    return size ? false : true;
}

int svlVideoCodecTCPStream::ParseFilename(const std::string & filename)
{
    // Port is "CISST" on numerical keypad by default
    SocketAddress.clear();
    SocketPort = 24778;

    size_t pos = filename.find('@');
    if (pos != std::string::npos) {
        // Extract address string (substring before '@')
        SocketAddress = filename.substr(0, pos);

        // Extract port (numerical string after '@')
        pos += 1;
        unsigned int ui_port = 0;
        while (ui_port < 65536 && filename[pos] >= '0' && filename[pos] <= '9') {
            ui_port *= 10;
            ui_port += filename[pos] - '0';
            pos ++;
        }
        if (ui_port > 0 && ui_port < 65536) SocketPort = static_cast<unsigned short>(ui_port);
    }
    else {
        // If no port is specified, then extract
        // the whole string before the extension
        pos = filename.rfind('.');
        if (pos != std::string::npos) SocketAddress = filename.substr(0, pos);
    }

    return SVL_OK;
}
