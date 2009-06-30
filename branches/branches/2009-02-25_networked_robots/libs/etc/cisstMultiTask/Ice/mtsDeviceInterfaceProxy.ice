/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsDeviceInterfaceProxy.ice 2009-03-16 mjung5 $
  
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

#ifndef _mtsDeviceInterfaceProxy_ICE_h
#define _mtsDeviceInterfaceProxy_ICE_h

#include <Ice/Identity.ice>

module mtsDeviceInterfaceProxy
{
	//-----------------------------------------------------------------------------
	//	Command and Event Object Definition
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
	
	struct EventVoidInfo { 
		string Name;
	};
	
	struct EventWriteInfo { 
		string Name;
        string ArgumentTypeName;
	};

	sequence<CommandVoidInfo>          CommandVoidSequence;
	sequence<CommandWriteInfo>         CommandWriteSequence;
	sequence<CommandReadInfo>          CommandReadSequence;
	sequence<CommandQualifiedReadInfo> CommandQualifiedReadSequence;
    sequence<EventVoidInfo>            EventVoidSequence;
    sequence<EventWriteInfo>           EventWriteSequence;

    //-----------------------------------------------------------------------------
	//	Provided Interface Related Definition
	//-----------------------------------------------------------------------------	
	// Data structure definition
	struct ProvidedInterfaceInfo {
		// Interface name
		string InterfaceName;
		
		// Commands
		CommandVoidSequence          CommandsVoid;
		CommandWriteSequence         CommandsWrite;
		CommandReadSequence          CommandsRead;
		CommandQualifiedReadSequence CommandsQualifiedRead;
        
        // Events
		EventVoidSequence            EventsVoid;
		EventWriteSequence           EventsWrite;
	};

    /*! List of provided interfaces */
    //sequence<ProvidedInterfaceInfo> ProvidedInterfaceSequence;

    //-----------------------------------------------------------------------------
	//	Function Proxy Related Definition
	//-----------------------------------------------------------------------------	
    struct FunctionProxyInfo {
        string Name;
        int FunctionProxyPointer;
    };

    sequence<FunctionProxyInfo> FunctionProxySequence;

    struct FunctionProxySet {
        // Name of the server task proxy. This is used as a key to find a server 
        // task proxy at client side.
        string ServerTaskProxyName;
        
        // Name of the provided interface proxy that has command proxies at client side.
        string ProvidedInterfaceProxyName;

        // Sets of function proxy pointers
        FunctionProxySequence FunctionVoidProxies;
        FunctionProxySequence FunctionWriteProxies;
        FunctionProxySequence FunctionReadProxies;
        FunctionProxySequence FunctionQualifiedReadProxies;
    };

	//-----------------------------------------------------------------------------
	// Interface for Required Interface (Proxy Client)
	//-----------------------------------------------------------------------------
	interface DeviceInterfaceClient
	{
        /*! Update CommandId. This updates the command id of command proxies
        at client side, which is a critical step regarding thread synchronization. */
        //["cpp:const"] idempotent
        //void UpdateCommandId(FunctionProxySet functionProxies);
	};

	//-----------------------------------------------------------------------------
	// Interface for Provided Interface (Proxy Server)
	//-----------------------------------------------------------------------------
	interface DeviceInterfaceServer
	{
		/*! Replacement for OnConnect event. */
		void AddClient(Ice::Identity ident);

        /*! Get the information about the provided interface which will be used to 
            create a provided interface proxy at client side. */
        ["cpp:const"] idempotent 
        bool GetProvidedInterfaceInfo(string providedInterfaceName,
                                      out ProvidedInterfaceInfo info);

        /*! Call mtsTaskManager::Connect() at server side. */
        bool ConnectServerSide(string userTaskName, string requiredInterfaceName,
			                   string resourceTaskName, string providedInterfaceName);

        /*! Update CommandId. This updates the command id of command proxies
        at client side, which is a critical step regarding thread synchronization. */
        ["cpp:const"] idempotent
        void GetCommandId(out FunctionProxySet functionProxies);

		/*! Execute command objects across networks. */
		// Here 'int' type is used instead of 'unsigned int' because SLICE does not
		// support unsigned type.
		// (see http://zeroc.com/doc/Ice-3.3.1/manual/Slice.5.8.html)
		// (Also see http://www.zeroc.com/doc/Ice-3.3.1/manual/Cpp.7.6.html for
		// Mapping for simple built-in types)
		void ExecuteCommandVoid(int CommandId);
        void ExecuteCommandWriteSerialized(int CommandId, string argument);        
        void ExecuteCommandReadSerialized(int CommandId, out string argument);
        void ExecuteCommandQualifiedReadSerialized(int CommandId, string argument1, out string argument2);
	};

};

#endif // _mtsDeviceInterfaceProxy_ICE_h
