/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id: DoublePlaneVirtualFixture.h 3148 2013-06-26 15:46:31Z oozgune1 $

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

#ifndef doubleplanevirtualfixture_h
#define doubleplanevirtualfixture_h

/*!@brief Double Plane Virtual Fixture class.
*
*  This class extends the base virtual fixture class. Behaviour of this type
*  of virtual fixture is smilar to the plane virtual fixture. The difference is
*  there are two planes, one above and one below. When the user gets close to the
*  above one, there will be force to keep the user below the plane. When the user gets
*  close to the below one, there will be force to keep the user above the plane.
*/

class DoublePlaneVirtualFixture: public VirtualFixture {
public:

    /*! @brief Update double plane virtual fixture parameters.
    *
    *   This method takes the current MTM position and orientation matrix and 
    *   the virtual fixture parameter reference, it calculates and assigns virtual fixture
    *   parameters. For the double plane virtual fixture, we calculate the closest points to the planes
    *   and set the point as the force position(which one the we are close to). 
    *   Using the closest point, we calulate the norm vector and allign the norm as z-axis and
    *   form the force orientation matrix. Since we allign the norm as z-axis, we only apply 
    *   positive force on the z-axis. We check the distance from the current position to the planes. 
    *   If the user is above the first plane, we set positive force on the z-axis. If the user
    *   is below the second plane, we set the negative force on the z-axis.
    *
    *   @param mtmPos current manipulator position and orientation.
    *   @param vfparams virtual fixture parameters.
    */
    void update(const vctFrm3 & mtmPos , prmFixtureGainCartesianSet & vfParams);

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

    /*!@brief Find the closest point to the plane.
    *
    *  @param p current position.
    *  @param norm unit normal vector to the plane.
    *  @param base base point in the plane.
    *  @return closest point to the plane.
    */
    vct3 closestPoint(vct3 p , vct3 norm , vct3 base);

    /*!@brief Find the closest distance to the plane.
    *
    *  @param p current position.
    *  @param norm unit normal vector to the plane.
    *  @param base base point in the plane.
    *  @return closest distance to the plane.
    */
    double shortestDistance(vct3 p , vct3 norm, vct3 base);

    /*!@brief This is a helper function to get a point from user.
    *
    *  @param position vector that contains position obtained from user.
    */
    void getPositionFromUser(vct3 & position);


    /*!@brief Set base point vectorfor the first plane.
    *
    *  @param c base point.
    */
    void setBasePoint1(const vct3 & c);

    /*!@brief Set unit normal vector for the first plane.
    *
    *  @param n unit normal vector.
    */
    void setNormVector1(const vct3 & n);

    /*!@brief Set base point vector for the second plane.
    *
    *  @param c base point.
    */
    void setBasePoint2(const vct3 & c);

    /*!@brief Set unit normal vector for the second plane.
    *
    *  @param n unit normal vector.
    */
    void setNormVector2(const vct3 & n);

    /*!@brief Returns base point for the upper plane.
    *
    *  @return basePoint1
    */

    vct3 getBasePoint1(void);

    /*!@brief Returns base point for the lower plane.
    *
    *  @return basePoint2
    */
    vct3 getBasePoint2(void);

    /*!@brief Returns norm vector for the upper plane.
    *
    *  @return normVector1
    */
    vct3 getNormVector1(void);

    /*!@brief Returns norm vector for the lower plane.
    *
    *  @return normVector2
    */
    vct3 getNormVector2(void);

protected:
    vct3 basePoint1;   //!< Base point position for the first plane.
    vct3 basePoint2;   //!< Base point position for the second plane. 
    vct3 normVector1;  //!< Unit normal vector for the first plane.
    vct3 normVector2;  //!< Unit normal vector for the second plane.

};

#endif