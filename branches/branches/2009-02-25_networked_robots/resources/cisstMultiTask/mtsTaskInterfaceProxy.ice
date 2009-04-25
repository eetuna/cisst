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
// Interface for Required Interface (client)
//-----------------------------------------------------------------------------
interface TaskInterfaceClient
{
};

//-----------------------------------------------------------------------------
// Interface for Provided Interface (server)
//-----------------------------------------------------------------------------
interface TaskInterfaceServer
{
    // from clients
    void AddClient(Ice::Identity ident);
};

};

#endif // _mtsTaskInterfaceProxy_ICE_h
