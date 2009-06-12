/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsDeviceProxy.h 291 2009-04-28 01:49:13Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-05-06

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


/*!
  \file
  \brief Defines a periodic task.
*/

// A server task proxy is of mtsDevice type, not of mtsTask because we are assuming
// that there is only one required interface at client side, which means there is
// only one user thread. Thus, we don't need to worry about thread synchronization
// issues. 
// However, if there are more than one required interface at client side, we should
// need to consider the thread synchronization issues. Moreover, if one required
// interface can connect to more than one provided interface, things get more 
// complicated. (However, the current design doesn't consider such case.)
#ifndef _mtsDeviceProxy_h
#define _mtsDeviceProxy_h

#include <cisstMultiTask/mtsDevice.h>

class mtsDeviceProxy : public mtsDevice {

public:
    mtsDeviceProxy(const std::string & deviceName) {}
    ~mtsDeviceProxy() {};

    void Configure(const std::string & deviceName) {};
};

#endif // _mtsDeviceProxy_h

