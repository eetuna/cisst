/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id: ShellSphereVirtualFixture.h 3148 2013-06-26 15:46:31Z oozgune1 $

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

#ifndef shellvirtualfixture_h
#define shellvirtualfixture_h

/*! @brief Shell Sphere Virtual Fixture.
*
*   This class extends the base virtual fixture class and overloads the
*   two methods in the base class.This is the type of a sphere virtual fixture that 
*   protects inside of the sphere. In this type, forbidden region is the inside.
*   If the user tries to penetrate into the sphere, there will be force
*   against the user with the same direction as penetration orientation.
*/

class ShellSphereVirtualFixture: public VirtualFixture {

public:

    /*! @brief Update shell sphere virtual fixture parameters.
    *
    *   This method takes the current MTM position and orientation matrix and 
    *   virtual fixture parameter reference, it calculates and assigns virtual fixture
    *   parameters. For the shell sphere virtual fixture, we add the given radius to the 
    *   sphere center positon and assign as the force position. using the current position,
    *   we calulate the norm vector and allign the norm as z-axis and form the force orientation
    *   matrix. Since we allign the norm as z-axis, we only apply negative position stiffness
    *   force on the z-axis. Other force values remains as 0.
    *
    *   @param mtmPos current manipulator position and orientation.
    *   @param vfparams virtual fixture parameters.
    */
    void update(const vctFrm3 & mtmPos , prmFixtureGainCartesianSet & vfParams);

    /*! @brief Return the Force Compliance Frame
    */
    vctFrm3 getForceComplianceFrame(void);

    /*!@brief Helper method to find two orthogonal vectors.
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

    /*! @brief Constructor.
    *
    *   Constructor takes no argument
    */
    ShellSphereVirtualFixture(void);

    /*! @brief Constructor.
    *
    *   Constructor takes center position and radius of the sphere
    */
    ShellSphereVirtualFixture(vct3 &center, double &radius);

    /*! @brief Set the center position for the sphere.
    *
    *   @param c given center position.
    */
    void setCenter(const vct3 & c);

    /*! @brief Set the sphere radius.
    *
    *   @param r given radius for the sphere.
    */
    void setRadius(const double & r);

    /*! @brief Set the Force Compliance Frame.
    *
    *   @param forceF given force compliance frame.
    */
    void setComplianceFrame(const vctFrm3 & compliance);

    /*! @brief Return the radius of the sphere.
    *
    *   @return radius.
    */
    double getRadius(void);

    /*! @brief Return the center position for the sphere.
    *
    *   @return center.
    */
    vct3 getCenter(void);

    /*! @brief Returns the Force Compliance Frame
    *
    *   @return forceComplianceFrame.
    */
    vctFrm3 getComplianceFrame(void);


protected:
    vct3 center;  //!< Center position of the sphere.
    double radius; //!< Radius of the sphere.
    vctFrm3 complianceFrame; //!<Force compliance frame

};

#endif