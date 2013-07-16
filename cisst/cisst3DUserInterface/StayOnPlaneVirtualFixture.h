/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id: StayOnPlaneVirtualFixture.h 3148 2013-06-26 15:46:31Z oozgune1 $

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

#ifndef stayonplanevirtualfixture_h
#define stayonplanevirtualfixture_h

/*!@brief StayOnPlane Virtual Fixture.
*
*  This class extends the base virtual fixture class and overloads the
*  methods in the base class. The stay on a plane virtual fixture keeps the handle 
*  on a desired plane. Guidance region is desired plane based on
*  the user need.
*/

class StayOnPlaneVirtualFixture: public VirtualFixture {
public:

    /*! @brief Constructor takes no argument
    */
    StayOnPlaneVirtualFixture(void);

    /*! @brief Constructor takes the base point position and the plane normal
    *   and sets them.
    */
    StayOnPlaneVirtualFixture(const vct3 basePoint, const vct3 planeNormal);

    /*! @brief Update plane virtual fixture parameters.
    *
    *   This method takes the current MTM position and orientation matrix and 
    *   the virtual fixture parameter reference, it calculates and assigns virtual fixture
    *   parameters. For the stay on plane virtual fixture, we calculate the closest point to the plane
    *   and set the point as the force position. Usind the closest point, we calulate the norm
    *   vector and allign the norm as z-axis and form the force orientation matrix. 
    *   Since we allign the norm as z-axis perpendicular to the plane, we only apply positive and negative
    *   force on the z-axis at the same time.
    *
    *   @param mtmPos current manipulator position and orientation.
    *   @param vfparams virtual fixture parameters.
    */
    void update(const vctFrm3 & mtmPos , prmFixtureGainCartesianSet & vfParams);

    /*! @brief Helper method to find two orthogonal vectors.
    *
    *   This method takes 3 vectors. First one is the given vector, other two stores
    *   the orthogonal vectors. We use one arbitrary vector(Y- axis in this case).
    *   We first cross product the given vector and the arbitrary vector to find the 
    *   first orthogonal vector to the given vector. Then we cross product the given vector
    *   and the first orthogonal vector to find the second orthogonal vector.
    *
    *   @param in given vector.
    *   @param out1 the first orthogonal vector to the given vector.
    *   @param out2 the second orthogonal vector to the given vector.
    */
    void findOrthogonal(vct3 in, vct3 &out1, vct3 &out2);

    /*! @brief Finds the closest point to the plane.
    *
    *   @param p current position.
    *   @return closest point.
    */
    vct3 closestPoint(vct3 p);

    /*! @brief Finds the closest distance to the plane.
    *
    *   @param p current position.
    *   @return closest distance to the plane.
    */
    double shortestDistance(vct3 p);

    /*! @brief Sets the base point vector.
    *
    *   @param base base point
    */
    void setBasePoint(const vct3 & base);

    /*! @brief Sets the plane unit normal vector.
    *
    *  s@param normal unit normal vector.
    */
    void setPlaneNormal(const vct3 & normal);

    /*! @brief Returns the base point vector.
    *
    *   @return basePoint.
    */
    vct3 getBasePoint(void);

    /*! @brief Returns the plane unit normal vector.
    *
    *   @return normVector.
    */
    vct3 getPlaneNormal(void);

    /*! @brief Sets the given positive position stiffness constant.
    *
    *   @param stiffPos positive position stiffness constant.
    */
    void setPositionStiffnessPositive(const vct3 stiffPos);

    /*! @brief Sets the given negative position stiffness constant.
    *
    *   @param stiffNeg negative position stiffness constant.
    */
    void setPositionStiffnessNegative(const vct3 stiffNeg);

    /*! @brief Sets the given positive position damping constant.
    *
    *   @param dampPos positive position damping constant.
    */
    void setPositionDampingPositive(const vct3 dampPos);

    /*! @brief Sets the given negative position damping constant.
    *
    *   @param dampNeg negative position damping constant.
    */
    void setPositionDampingNegative(const vct3 dampNeg);

    /*! @brief Sets the given positive force bias constant.
    *
    *   @param biasPos positive force bias constant.
    */
    void setForceBiasPositive(const vct3 biasPos);

    /*! @brief Sets the given negative force bias constant.
    *
    *   @param biasNeg negative force bias constant.
    */
    void setForceBiasNegative(const vct3 biasNeg);

    /*! @brief Sets the given positive orientation stiffness constant.
    *
    *   @param orientStiffPos positive orientation stiffness constant.
    */
    void setOrientationStiffnessPositive(const vct3 orientStiffPos);

    /*! @brief Sets the given negative orientation stiffness constant.
    *
    *   @param orientStiffNeg negative orientation stiffness constant.
    */
    void setOrientationStiffnessNegative(const vct3 orientStiffNeg);

    /*! @brief Sets the given positive orientation damping constant.
    *
    *   @param orientDampPos positive orientation damping constant.
    */
    void setOrientationDampingPositive(const vct3 orientDampPos);

    /*! @brief Sets the given negative orientation damping constant.
    *
    *   @param orientDampNeg negative orientation damping constant.
    */
    void setOrientationDampingNegative(const vct3 orientDampNeg);

    /*! @brief Sets the given positive torque bias constant.
    *
    *   @param torqueBiasPos positive torque bias constant.
    */
    void setTorqueBiasPositive(const vct3 torqueBiasPos);

    /*! @brief Sets the given negative torque bias constant.
    *
    *   @param torqueBiasNeg negative torque bias constant.
    */
    void setTorqueBiasNegative(const vct3 torqueBiasNeg);

    /*! @brief Returns the positive position stiffness constant.
    *
    *   @return PositionStiffnessPositive positive position stiffness constant.
    */
    vct3 getPositionStiffnessPositive(void);

    /*! @brief Returns the negative position stiffness constant.
    *
    *   @return PositionStiffnessNegative negative position stiffness constant.
    */
    vct3 getPositionStiffnessNegative(void);

    /*! @brief Returns the positive position damping constant.
    *
    *   @return PositionDampingPositive positive position damping constant.
    */
    vct3 getPositionDampingPositive(void);

    /*! @brief Returns the negative position damping constant.
    *
    *   @return PositionStiffnessNegative negative position damping constant.
    */
    vct3 getPositionDampingNegative(void);

    /*! @brief Returns the positive force bias constant.
    *
    *   @return ForceBiasPositive positive force bias constant.
    */
    vct3 getForceBiasPositive(void);

    /*! @brief Returns the negative force bias constant.
    *
    *   @return ForceBiasNegative negative force bias constant.
    */
    vct3 getForceBiasNegative(void);

    /*! @brief Returns the positive orientation stiffness constant.
    *
    *   @return OrientationStiffnessPositive positive orientation stiffness constant.
    */
    vct3 getOrientationStiffnessPositive(void);

    /*! @brief Returns the negative orientation stiffness constant.
    *
    *   @return OrientationStiffnessNegative negative orientation stiffness constant.
    */
    vct3 getOrientationStiffnessNegative(void);

    /*! @brief Returns the positive orientation damping constant.
    *
    *   @return OrientationDampingPositive positive orientation damping constant.
    */
    vct3 getOrientationDampingPositive(void);

    /*! @brief Returns the negative orientation damping constant.
    *
    *   @return OrientationDampingNegative negative orientation damping constant.
    */
    vct3 getOrientationDampingNegative(void);

    /*! @brief Returns the positive torque bias constant.
    *
    *   @return TorqueBiasPositive positive torque bias constant.
    */
    vct3 getTorqueBiasPositive(void);

    /*! @brief Returns the negative torque bias constant.
    *
    *   @return TorqueBiasNegative negative torque bias constant.
    */
    vct3 getTorqueBiasNegative(void);

protected:
    vct3 basePoint; //!< Base point position vector.
    vct3 planeNormal; //!< Plane unit normal vector.
    vct3 PositionStiffnessPositive; //<! Positive position stiffness constant.
    vct3 PositionStiffnessNegative; //<! Negative position stiffness constant.
    vct3 PositionDampingPositive; //<! Positive position damping constant.
    vct3 PositionDampingNegative; //<! Negative position damping constant.
    vct3 ForceBiasPositive; //<! Positive force bias constant.
    vct3 ForceBiasNegative; //<! Negative force bias constant.
    vct3 OrientationStiffnessPositive; //<! Positive orientation stiffness constant.
    vct3 OrientationStiffnessNegative; //<! Negative orientation stiffness constant.
    vct3 OrientationDampingPositive; //<! Positive orientation damping constant.
    vct3 OrientationDampingNegative; //<! Negative orientation damping constant.
    vct3 TorqueBiasPositive; //<! Positive torque bias constant.
    vct3 TorqueBiasNegative; //<! Negative torque bias constant.

};

#endif