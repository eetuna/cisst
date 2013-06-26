/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id: DoubleSphereVirtualFixture.h 3148 2013-06-26 15:46:31Z oozgune1 $

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
#include <cisst3DUserInterface/VirtualFixture.h>
#include <cisstVector.h>

#ifndef doublespherevirtualfixture_h
#define doublespherevirtualfixture_h

/*! @brief Double Sphere Virtual Fixture.
*
*   This class extends the base virtual fixture class and overloads the
*   two methods in the base class.This is the type of a sphere virtual fixture that 
*   keeps the MTM in the middle of two spheres.Inside of the double sphere is shell sphere virtual
*   fixture, outside of the sphere is inner sphere virtual fixture. This will lead the user
*   not to penetrate a delicate median and not to leave bounded area.
*   In this type of virtual fixture, forbidden regions are outside of the inner sphere and outer sphere.
*   If the user tries to penetrate the inner sphere, there will be force against the 
*   user with the same direction. Same force will be aplied when the user tries to leave the outer sphere.
*/
class DoubleSphereVirtualFixture: public VirtualFixture {

public:

    /*! @brief Update Double Sphere Virtual Fixture parameters.
    *
    *   This method takes the current MTM position and orientation matrix and 
    *   the virtual fixture parameter reference, it calculates and assigns virtual fixture
    *   parameters. For the double sphere virtual fixture, we decide which sphere radius will
    *   be used based on the closest distance to the spheres. For example, if we are close to the 
    *   inner sphere, we will be using inner sphere radius to calculate force position
    *   and orientation. We add the given radius to the sphere center positon and assign 
    *   as the force position. using the current position,we calulate the norm vector and allign 
    *   the norm as z-axis and form the force orientation matrix. Since we allign the norm as z-axis, 
    *   we only apply positive force on the z-axis. Other force values remain as 0.
    *
    *   @param mtmPos current manipulator position and orientation.
    *   @param vfparams Virtual fixture parameters.
    */
    void update(const vctFrm3 & mtmPos , prmFixtureGainCartesianSet & vfParams);


    /*!@brief Helper method to find two orthogonal vectors to the given vector.
    *
    *  This method takes 3 vectors. First one is the given vector, other two stores
    *  the orthogonal vectors. We use one arbitrary vector(Y- axis in this case).
    *  We first cross product the given vector and the arbitrary vector to find the 
    *  first orthogonal vector to the given vector. Then we cross product the given vector
    *  and the first orthogonal vector to find the second orthogonal vector.
    *
    *  @param in given vector.
    *  @param out1 the first orthogonal vector to the given vector.
    *  @param out2 the second orthogonal vector to the given vector.
    */
    void findOrthogonal(vct3 in, vct3 &out1, vct3 &out2);

    /*! @brief Sets the center position for the spheres.
    *
    *   @param c given center position.
    */
    void setCenter(const vct3 & c);

    /*! @brief Sets the inner sphere radius.
    *
    *   @param r1 given center position.
    */
    void setRadius1(const double & r1);

    /*! @brief Sets the outer sphere radius.
    *
    *   @param r2 given center position.
    */
    void setRadius2(const double & r2);

    /*! @brief Returns the center for the two spheres.
    * 
    *   @return center position.
    */
    vct3 getCenter(void);

    /*! @brief Returns the radius for the inner sphere.
    *
    *   @return radius1 inner sphere radius.
    */
    double getRadius1(void);

    /*! @brief Returns the radius for the outer sphere.
    *
    *   @return radius2 outer sphere radius.
    */
    double getRadius2(void);

protected:
    vct3 center;    //!< Center position of the spheres.
    double radius1; //!< Radius of the inner sphere.
    double radius2; //!< Radius of the outer sphere.

};

#endif