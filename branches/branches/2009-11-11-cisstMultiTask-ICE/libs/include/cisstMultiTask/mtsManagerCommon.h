/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerCommon.h 794 2009-09-01 21:43:56Z pkazanz1 $

  Author(s):  Min Yang Jung
  Created on: 2009-12-08

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

/*!
  \file
  \brief Definition of mtsManagerCommon
  \ingroup cisstMultiTask

  This class defines an interface used by the global component manager to 
  communicate with local component managers. The interface is defined as a pure 
  abstract class because there are two different configurations:

  Standalone mode: Inter-thread communication, no ICE.  A local component manager 
    directly connects to the global component manager that runs in the same process. 
    In this case, the global component manager keeps only one instance of 
    mtsManagerLocal.

  Network mode: Inter-process communication, ICE enabled.  Local component 
    managers connect to the global component manager via a proxy.
    In this case, the global component manager handles instances of 
    mtsManagerLocalProxyClient.

  \note Please refer to mtsManagerLocal and mtsManagerLocalProxyClient for details.
*/

#ifndef _mtsManagerCommon_h
#define _mtsManagerCommon_h

#include <cisstCommon/cmnGenericObject.h>

class CISST_EXPORT mtsManagerCommon : public cmnGenericObject {

    friend class mtsManagerCommonTest;

protected:
    //-------------------------------------------------------------------------
    //  Definition of Provided Interface Summary
    //-------------------------------------------------------------------------
    /*! Command and event object definition */
	struct CommandVoidElement {
        std::string Name;
	};
	
	struct CommandWriteElement {
		std::string Name;
        std::string ArgumentPrototypeSerialized;
	};
	
	struct CommandReadElement {
		std::string Name;
        std::string ArgumentPrototypeSerialized;
	};
	
	struct CommandQualifiedReadElement {
		std::string Name;
        std::string Argument1PrototypeSerialized;
        std::string Argument2PrototypeSerialized;
	};
	
	struct EventVoidElement {
		std::string Name;
	};
	
	struct EventWriteElement {
        std::string Name;
        std::string ArgumentPrototypeSerialized;
	};

    typedef std::vector<CommandVoidElement>          CommandVoidVector;
	typedef std::vector<CommandWriteElement>         CommandWriteVector;
	typedef std::vector<CommandReadElement>          CommandReadVector;
	typedef std::vector<CommandQualifiedReadElement> CommandQualifiedReadVector;
    typedef std::vector<EventVoidElement>            EventVoidVector;
    typedef std::vector<EventWriteElement>           EventWriteVector;
	
	class ProvidedInterfaceDescription {
    public:
		// Interface name
        std::string ProvidedInterfaceName;
		
		// Commands
		CommandVoidVector          CommandsVoid;
		CommandWriteVector         CommandsWrite;
		CommandReadVector          CommandsRead;
		CommandQualifiedReadVector CommandsQualifiedRead;
        
        // Events
		EventVoidVector  EventsVoid;
		EventWriteVector EventsWrite;
	};

    //-------------------------------------------------------------------------
    //  Definition of Required Interface Summary
    //-------------------------------------------------------------------------
    typedef std::vector<std::string> CommandPointerNames;
    typedef CommandVoidVector  EventHandlerVoidVector;
    typedef CommandWriteVector EventHandlerWriteVector;

    class RequiredInterfaceDescription {
    public:
        // Interface name
        std::string RequiredInterfaceName;

        // Functions (i.e., command pointers)
        CommandPointerNames FunctionVoidNames;
        CommandPointerNames FunctionWriteNames;
        CommandPointerNames FunctionReadNames;
        CommandPointerNames FunctionQualifiedReadNames;

        // Event handlers
        EventHandlerVoidVector  EventHandlersVoid;
		EventHandlerWriteVector EventHandlersWrite;
    };
};

#endif // _mtsManagerCommon_h

