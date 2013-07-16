/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id: VirtualFixture.h 3148 2013-06-26 15:46:31Z oozgune1 $

Author(s):	Orhan Ozguner
Created on:	2013-06-26

(C) Copyright 2013 Johns Hopkins University (JHU), All Rights
Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/
#include <cisstVector.h>
#include <cisstParameterTypes/prmFixtureGainCartesianSet.h>

#ifndef virtualfixture_h
#define virtualfixture_h

/*! @brief Virtual Fixture base class.
*
*   This clas is the base class for the different kinds of virtual fixtures.
*   All different virtual fixture classes have to extend the base class and 
*   override the methods in the base class.
*/
class VirtualFixture{
public:
    /*! @brief Updates the virtual fixture paramters with the given current position.
    *
    *   This function is responsibe for the math. It takes current position and orientation 
    *   of the manipulator, calculates the virtual fixture parameters.
    *
    *   @param pos current position and orientation of the MTM
    *   @param vfParams address of the virtual fixture parameters.
    */
    virtual void update(const vctFrm3 & position , prmFixtureGainCartesianSet & vfParams) = 0;

};

#endif
