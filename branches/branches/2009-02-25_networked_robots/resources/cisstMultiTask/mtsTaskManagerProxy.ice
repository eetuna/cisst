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

sequence<string> TaskNameSeq;

struct TaskInfo {
    //string taskManagerID;
    TaskNameSeq taskNames;	// task name (Unicode supported)
};

//-----------------------------------------------------------------------------
// Interface for TaskManager client
//-----------------------------------------------------------------------------
interface TaskManagerClient
{
    // passive (callback)
    void ReceiveData(int num);
    
    // active
    void SendMyTaskInfo(TaskInfo clientTaskInfo);
};

//-----------------------------------------------------------------------------
// Interface for TaskManager server
//-----------------------------------------------------------------------------
interface TaskManagerServer
{
    // passive
    void AddClient(Ice::Identity ident);

    // active
    void SendCurrentTaskInfo();//TaskInfo clientTaskInfo, out TaskInfo serverTaskInfo);
};

};

#endif // _mtsTaskManagerProxy_ICE_h
