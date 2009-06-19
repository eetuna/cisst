/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsTaskManagerProxy.ice 2009-03-16 mjung5 $
  
  Author(s):  Min Yang Jung
  Created on: 2009-03-16
  
  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

//
// This Slice file defines the communication specification between 
// Task Manager server and Task Manager client across networks.
//

#ifndef _mtsTaskManagerProxy_ICE_h
#define _mtsTaskManagerProxy_ICE_h

#include <Ice/Identity.ice>

module mtsTaskManagerProxy
{
	//-----------------------------------------------------------------------------
	//	Data Structure Definition
	//
	//  TODO: Multiple [provided, required] interface => USE dictionary (SLICE)!!!
	//	refer to: mtsTask.h
	//-----------------------------------------------------------------------------
	sequence<string> TaskNameSeq;

	struct TaskList {
		string taskManagerID;
		TaskNameSeq taskNames;	// task name (Unicode supported)
	};

	struct ProvidedInterfaceInfo {
		string taskName;
		string interfaceName;
		string adapterName;
		string endpointInfo;
		string communicatorID;
	};

	struct RequiredInterfaceInfo {
		string taskName;
		string interfaceName;
	};

	//-----------------------------------------------------------------------------
	// Exception Definition
	//
	// TODO: define and use exceptions at application layer.
	//
	//-----------------------------------------------------------------------------
	//exception InvalidTaskNameError {
		//string msg1;
		//string msg2;
	//};

	//-----------------------------------------------------------------------------
	// Interface for TaskManager client
	//
	// - This interface defines the interface of the task manager clients to be
	//	 called by the global task manager
	//-----------------------------------------------------------------------------
	interface TaskManagerClient
	{
		// Test method
		void ReceiveData(int num);
		
		bool ConnectAtServerSide(
			string userTaskName, string interfaceRequiredName,
			string resourceTaskName, string providedInterfaceName);
		
	};

	//-----------------------------------------------------------------------------
	// Interface for TaskManager server
	//
	// - This interface defines the interface of the global task server to receive
	//	 events from the task manager clients.
	//-----------------------------------------------------------------------------
	interface TaskManagerServer
	{		
		void AddClient(Ice::Identity ident); // throws InvalidTaskNameError;
	    
		void UpdateTaskManager(TaskList localTaskInfo);

	    // from task manager client which manages a server task
		bool AddProvidedInterface(ProvidedInterfaceInfo newProvidedInterfaceInfo);
		
		// from task manager client which manages a client task
		bool AddRequiredInterface(RequiredInterfaceInfo newRequiredInterfaceInfo);

		["cpp:const"] idempotent bool IsRegisteredProvidedInterface(
			string taskName, string providedInterfaceName);

		["cpp:const"] idempotent bool GetProvidedInterfaceInfo(
			string taskName, string providedInterfaceName, out ProvidedInterfaceInfo info);
			
		void NotifyInterfaceConnectionResult(
			bool isServerTask, bool isSuccess,
			string userTaskName, string interfaceRequiredName,
			string resourceTaskName, string providedInterfaceName);
	};

};

#endif // _mtsTaskManagerProxy_ICE_h
