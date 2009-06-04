/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsTaskGlobal.h 142 2009-03-11 23:02:34Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-04-26

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#ifndef _mtsTaskGlobal_h
#define _mtsTaskGlobal_h

#include <cisstMultiTask/mtsTaskManager.h>
#include <cisstMultiTask/mtsProxyBaseServer.h>
#include <cisstMultiTask/mtsTaskManagerProxy.h>
#include <cisstMultiTask/mtsMap.h>

#include <cisstMultiTask/mtsExport.h>

/*!
  \ingroup cisstMultiTask

  TODO: add class summary here
*/

/*! Information on an added task. */
class mtsTaskGlobal {
    /*! Information on a provided interface. */
    class GenericInterfaceInfo {
    public:
        std::string InterfaceName;

        GenericInterfaceInfo(const std::string & interfaceName)
            : InterfaceName(interfaceName)
        {}

        const std::string GetInterfaceName() const { return InterfaceName; }
    };

    class ProvidedInterfaceInfo : public GenericInterfaceInfo {
    public:
        // Provided-interface-proxy-server connection information
        std::string AdapterName;
        std::string EndpointInfo;
        std::string CommunicatorID;
    
        ProvidedInterfaceInfo(const std::string & adapterName,
                              const std::string & endpointInfo,
                              const std::string & communicatorID,
                              const std::string & interfaceName):
            GenericInterfaceInfo(interfaceName),
            AdapterName(adapterName),
            EndpointInfo(endpointInfo), 
            CommunicatorID(communicatorID)
        {}

        /*! Initialize this object. */
        //void Init() {
        //    AdapterName = ""; EndpointInfo = ""; CommunicatorID = ""; InterfaceName = "";
        //}

        void GetData(mtsTaskManagerProxy::ProvidedInterfaceInfo & info) {
            info.adapterName = AdapterName;
            info.endpointInfo = EndpointInfo;
            info.communicatorID = CommunicatorID;
            info.interfaceName = InterfaceName;
        }

        void InitData(mtsTaskManagerProxy::ProvidedInterfaceInfo & info) {
            info.adapterName = "";
            info.endpointInfo = "";
            info.communicatorID = "";
            info.interfaceName = "";
        }
    };

    /*! Information on a required interface. */
    class RequiredInterfaceInfo : public GenericInterfaceInfo {
    public:
        RequiredInterfaceInfo(const std::string & interfaceName)
            : GenericInterfaceInfo(interfaceName)
        {}
    };

protected:
    const std::string TaskName;
    const std::string TaskManagerID;

    /*! List of connected peer tasks' names. */
    std::vector<std::string> ConnectedTaskList;

    /*! map: (provided interface name, its information) */
    typedef std::map<std::string, ProvidedInterfaceInfo> ProvidedInterfaceMapType;
    ProvidedInterfaceMapType ProvidedInterfaces;

    /*! map: (required interface name, its information) */
    typedef std::map<std::string, RequiredInterfaceInfo> RequiredInterfaceMapType;
    RequiredInterfaceMapType RequiredInterfaces;

public:
    mtsTaskGlobal(const std::string& taskName, const std::string & taskManagerID) 
        : TaskName(taskName), TaskManagerID(taskManagerID) {}

    std::string ShowTaskInfo();

    /*! Register a new provided interface. */
    bool AddProvidedInterface(const ::mtsTaskManagerProxy::ProvidedInterfaceInfo &);

    /*! Register a new required interface. */
    bool AddRequiredInterface(const ::mtsTaskManagerProxy::RequiredInterfaceInfo &);

    /*! Return true if the provided interface has been registered. */
    const bool IsRegisteredProvidedInterface(const std::string providedInterfaceName) const;

    /*! Return true if the required interface has been registered. */
    const bool IsRegisteredRequiredInterface(const std::string requiredInterfaceName) const;

    /*! Return the access information of the specified provided interface. */
    const bool GetProvidedInterfaceInfo(const std::string & providedInterfaceName,
                                        mtsTaskManagerProxy::ProvidedInterfaceInfo & info);
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsTaskGlobal)

#endif // _mtsTaskGlobal_h
