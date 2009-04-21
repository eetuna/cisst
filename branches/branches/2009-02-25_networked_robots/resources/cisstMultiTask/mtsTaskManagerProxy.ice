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
//	This Slice file defines the communication specification between 
//	mtsTaskManager objects across networks.
//

#ifndef _mtsTaskManagerProxy_h
#define _mtsTaskManagerProxy_h

#include <Ice/Identity.ice>

module mtsTaskManagerProxy
{
	sequence<string> TaskNameSeq;
	
	struct TaskInfo {
		//string taskManagerID;
		TaskNameSeq taskNames;	// task name (Unicode supported)
	};
	
	interface TaskManagerCommunicator {
        // TMclient --> TMserver
        void SendMyTaskInfo(TaskInfo clientTaskInfo);
        
        // TMserver --> TMclient
		void SendCurrentTaskInfo(TaskInfo clientTaskInfo, out TaskInfo serverTaskInfo);
	};

/*
	interface CallbackReceiver
	{
		void cbReceiveTaskCount(int num);	// cb represents 'call back'
	};

	interface CallbackSender
	{
		void AddTask(Ice::Identity ident);
	};
*/

	
//	interface Printer {
//		void printString(string s);
//	};
};

#endif // _mtsTaskManagerProxy_h
