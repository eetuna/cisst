/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsTaskInterfaceProxy.ice 2009-03-16 mjung5 $
  
  Author(s):  Min Yang Jung
  Created on: 2009-04-24
  
  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

//
// This Slice file defines the communication between a provided interface
// and a required interfaces. 
// A provided interfaces act as a server while a required interface does 
// as a client.
//

#ifndef _mtsTaskInterfaceProxy_ICE_h
#define _mtsTaskInterfaceProxy_ICE_h

#include <Ice/Identity.ice>

module mtsTaskInterfaceProxy
{
	//-----------------------------------------------------------------------------
	//	Data Structure Definition
	//-----------------------------------------------------------------------------
	struct CommandVoidInfo { 
		string Name;
	};
	
	struct CommandWriteInfo { 
		string Name;
		string ArgumentTypeName;
	};
	
	struct CommandReadInfo { 
		string Name;
		string ArgumentTypeName;
	};
	
	struct CommandQualifiedReadInfo { 
		string Name;
		string Argument1TypeName;
		string Argument2TypeName;
	};
	
	/*
	struct EventVoidInfo { 
		string Name;
	};
	
	struct EventWriteInfo { 
		string Name;
	};
	*/

	sequence<CommandVoidInfo>          CommandVoidSeq;
	sequence<CommandWriteInfo>         CommandWriteSeq;
	sequence<CommandReadInfo>          CommandReadSeq;
	sequence<CommandQualifiedReadInfo> CommandQualifiedReadSeq;
    //sequence<EventVoidInfo> EventVoidSeq;
    //sequence<EventWriteInfo> EventWriteSeq;
    
	// Data structure definition
	struct ProvidedInterfaceSpecification {
		// Identity
		string interfaceName;
		
		// Flag to determine the type of this provided interface.
		// true, if this interface is of mtsTaskInterface type.
		// false, if this interface is of mtsDeviceInterface type.
		bool providedInterfaceForTask;
		
		// Commands
		CommandVoidSeq          commandsVoid;
		CommandWriteSeq         commandsWrite;
		CommandReadSeq          commandsRead;
		CommandQualifiedReadSeq commandsQualifiedRead;
		//EventVoidSeq eventsVoid;
		//EventWriteSeq eventsWrite;
		
		// Events: this isn't supported at this time. Event handling will be implemented.
		//sequence<EventVoidInfo> eventsVoid;
		//sequence<EventWriteInfo> eventsWrite;
	};
	
	sequence<ProvidedInterfaceSpecification> ProvidedInterfaceSpecificationSeq;

	/*! Typedef for a map of (command name, command object id). 
        This map is populated at a client task and is sent to a server task.
        (see mtsCommandBase::CommandUID) */
    struct CommandProxyInfo {
		string Name;
		int    ID;
    };
    sequence<CommandProxyInfo> CommandProxyInfoSeq;
    
	//-----------------------------------------------------------------------------
	// Interface for Required Interface (Proxy Client)
	//-----------------------------------------------------------------------------
	interface TaskInterfaceClient
	{
	};

	//-----------------------------------------------------------------------------
	// Interface for Provided Interface (Proxy Server)
	//-----------------------------------------------------------------------------
	interface TaskInterfaceServer
	{
		// from clients
		void AddClient(Ice::Identity ident);
		
		["cpp:const"] idempotent bool GetProvidedInterfaceSpecification(
			out ProvidedInterfaceSpecificationSeq providedInterfaceSpecifications);
			
		void SendCommandProxyInfo(CommandProxyInfoSeq commandProxyInfos);
	};

};

#endif // _mtsTaskInterfaceProxy_ICE_h
