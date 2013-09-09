/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Peter Kazanzides
  Created on: 2013-08-06

  (C) Copyright 2013 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


/*!
  \file
  \brief Declaration of mtsSocketProxyClient
  \ingroup cisstMultiTask
*/

#ifndef _mtsSocketProxyClient_h
#define _mtsSocketProxyClient_h

#include <cisstOSAbstraction/osaSocket.h>
#include <cisstOSAbstraction/osaMutex.h>
#include <cisstMultiTask/mtsTaskContinuous.h>

#include <cisstMultiTask/mtsForwardDeclarations.h>

class CommandWrapperBase;
class mtsCommandBase;

#include <cisstMultiTask/mtsExport.h>

class CISST_EXPORT mtsSocketProxyClientConstructorArg : public mtsGenericObject
{
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION, CMN_LOG_ALLOW_DEFAULT);
public:
    std::string Name;
    std::string IP;
    short Port;

    mtsSocketProxyClientConstructorArg() : mtsGenericObject() {}
    mtsSocketProxyClientConstructorArg(const std::string &name, const std::string &ip, short port) :
        mtsGenericObject(), Name(name), IP(ip), Port(port) {}
    mtsSocketProxyClientConstructorArg(const mtsSocketProxyClientConstructorArg &other) : mtsGenericObject(),
        Name(other.Name), IP(other.IP), Port(other.Port) {}
    ~mtsSocketProxyClientConstructorArg() {}

    void SerializeRaw(std::ostream & outputStream) const;
    void DeSerializeRaw(std::istream & inputStream);

    void ToStream(std::ostream & outputStream) const;

    /*! Raw text output to stream */
    virtual void ToStreamRaw(std::ostream & outputStream, const char delimiter = ' ',
                             bool headerOnly = false, const std::string & headerPrefix = "") const;

    /*! Read from an unformatted text input (e.g., one created by ToStreamRaw).
      Returns true if successful. */
    virtual bool FromStreamRaw(std::istream & inputStream, const char delimiter = ' ');
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsSocketProxyClientConstructorArg);

class CISST_EXPORT mtsSocketProxyClient : public mtsTaskContinuous
{
    CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION_ONEARG, CMN_LOG_ALLOW_DEFAULT);

 protected:

    osaSocket Socket;
    // SocketMutex is a quick fix to deal with the fact that Read and QualifiedRead commands
    // are not queued, and therefore occur asynchronously with respect to the Run method.
    osaMutex SocketMutex;
    mtsProxySerializer *InternalSerializer;

    // For memory cleanup
    std::vector<CommandWrapperBase *> CommandWrappers;
    std::vector<mtsCommandBase *> EventGenerators;

    /*! \brief Create client proxy
      \param providedInterfaceDescription Complete information about provided
      interface to be created with arguments serialized
      \return True if success, false otherwise */
    bool CreateClientProxy(const std::string & providedInterfaceName);

    // For use by MulticastCommandVoidProxy and MulticastCommandWriteProxy
    bool EventOperation(const std::string &command, const std::string &eventName, const char *handle);

    friend class MulticastCommandVoidProxy;
    friend class MulticastCommandWriteProxy;

 public:
    /*! Constructor
        \param name Name of the client proxy component
        \param ip IP address for corresponding server proxy
        \param port Port for corresponding server proxy (UDP socket)
    */
    mtsSocketProxyClient(const std::string &name, const std::string &ip, short port);

    mtsSocketProxyClient(const mtsSocketProxyClientConstructorArg &arg);
    

    /*! Destructor */
    virtual ~mtsSocketProxyClient();

    void Configure(const std::string &) {}

    void Startup(void);

    void Run(void);

    void Cleanup(void);

};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsSocketProxyClient)

#endif // _mtsSocketProxyClient_h
