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
	
	//------------------------------------- 1 -----------------------------------//
	struct CommandVoidInfo { 
		string Name;
        int CommandSID;
	};
	
	struct CommandWriteInfo { 
		string Name;
		string ArgumentTypeName;
        int CommandSID;
	};
	
	struct CommandReadInfo { 
		string Name;
		string ArgumentTypeName;
        int CommandSID;
	};
	
	struct CommandQualifiedReadInfo { 
		string Name;
		string Argument1TypeName;
		string Argument2TypeName;
        int CommandSID;
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
	
	//------------------------------------- 2 -----------------------------------//
	sequence<ProvidedInterfaceSpecification> ProvidedInterfaceSpecificationSeq;

	/*! Typedef for a map of (command name, command object id). 
        This map is populated at a client task and is sent to a server task.
        (see mtsCommandBase::CommandUID) */
    struct CommandProxyElement {
		string Name;
		int    ID;
    };
    sequence<CommandProxyElement> CommandProxyElementSeq;
    
    struct CommandProxyInfo {
		// MJUNG: Currently it is assumed that one required interface connects to only
		// one provided interface. If a required interface connects to more than
		// one provided interface, the following field (ConnectedProvidedInterfaceName)
		// should be vectorized.
		string ConnectedProvidedInterfaceName;
		
		CommandProxyElementSeq	CommandProxyVoidSeq;
		CommandProxyElementSeq	CommandProxyWriteSeq;
		CommandProxyElementSeq	CommandProxyReadSeq;
		CommandProxyElementSeq	CommandProxyQualifiedReadSeq;
    };
    
    //------------------------------------- 3 -----------------------------------//
    
    
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
			
		//void SendCommandProxyInfo(CommandProxyInfo commandProxyInformation);
		
        //bool ConnectAtServerSide(string providedInterfaceName, string requiredInterfaceName);
        
		// Execute command objects across networks
		// Here 'int' type is used instead of 'unsigned int' because SLICE does not
		// support unsigned type.
		// (see http://zeroc.com/doc/Ice-3.3.1/manual/Slice.5.8.html)
		void ExecuteCommandVoid(int CommandSID);
        void ExecuteCommandWrite(int CommandSID, double argument);
        void ExecuteCommandRead(int CommandSID, out double argument);
        void ExecuteCommandQualifiedRead(int CommandSID, double argument1, out double argument2);
	};

};

#endif // _mtsTaskInterfaceProxy_ICE_h
