/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id: PointVirtualFixture.h 3148 2013-06-26 15:46:31Z oozgune1 $

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
#include "VirtualFixture.h"
#include <cisstVector.h>

#ifndef pointvirtualfixture_h
#define pointvirtualfixture_h

/*!@brief Point Virtual Fixture Class.
*
*  This class guides the MTM to the given point. Purpose of the type is 
*  to find a specific location when it is necessary.
*/

class PointVirtualFixture: public VirtualFixture {

public:

    /*! @brief Constructor takes no argument
    */
    PointVirtualFixture();

    /*! @brief Constructor takes vf point position and sets the point
    */
    PointVirtualFixture(const vct3 pt);

    /*! @brief Update point virtual fixture parameters.
    *
    *   This method takes the current MTM position and orientation matrix and 
    *   virtual fixture parameter reference, it calculates and assigns virtual fixture
    *   parameters. For the point virtual fixture, we set the given point as the force position.
    *   Since we have only a point, we do not need to calculate the force orientation matrix.
    *   We set all x,y,z position stiffness forces as the same.
    *
    *   @param mtmPos current manipulator position and orientation.
    *   @param vfparams virtual fixture parameters.
    */
    void update(const vctFrm3 & mtmPos , prmFixtureGainCartesianSet & vfParams);

    /*! @brief Set the point position.
    *
    *   @param p given point position.
    */
    void setPoint(const vct3 & p);

    /*! @brief Return the point position.
    *
    *   @return p point position.
    */
    vct3 getPoint(void);

protected:
    vct3 point; //!< Point position.

};

#endif