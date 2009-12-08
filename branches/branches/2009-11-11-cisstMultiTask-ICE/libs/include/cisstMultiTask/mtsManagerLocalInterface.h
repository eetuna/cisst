/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerLocalInterface.h 794 2009-09-01 21:43:56Z pkazanz1 $

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
  \brief Definition of mtsManagerLocalInterface
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

#ifndef _mtsManagerLocalInterface_h
#define _mtsManagerLocalInterface_h

#include <cisstCommon/cmnGenericObject.h>

class CISST_EXPORT mtsManagerLocalInterface : public cmnGenericObject {

public:
    /*! Getters */
    virtual const std::string GetProcessName() const = 0;
};

#endif // _mtsManagerLocalInterface_h

