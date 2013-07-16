/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id: CurveVirtualFixture.h 3148 2013-06-26 15:46:31Z oozgune1 $

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
#include <cisstParameterTypes/prmFixtureGainCartesianSet.h>
#include <cisstVector.h>

#ifndef curvevirtualfixture_h
#define curvevirtualfixture_h

/*!@brief Curve Virtual Fixture class.
*
*  This type of virtual fixture is designed to move along a given
*  line segment with a desired orientation. Line segment has start and end positions. 
*  This virtual fixture allows us to move along the line and applies force if the user 
*  tries to leave the line. At the same time, applies torque to keep the user in the
*  desired orientation.
*/
class CurveVirtualFixture: public VirtualFixture {

public:

    /*!@brief Constructor that takes no argument
    */
    CurveVirtualFixture(void);

    /*! @brief Constructor that takes curve points.
    *
    *   In the constructor, we set the points that constructs the
    *   the curve.
    */
    CurveVirtualFixture(std::vector<vct3>);

    /*! @brief Update Curve Virtual Fixture parameterss.
    *
    *   We find the closest point to the line and set this point as a force frame
    *   position. Using the closest point, we allign the z-axis parallel 
    *   to the line. We apply force on x and y axis. This way there will be force
    *   if the user tries to leave the curve. Also this method will ensures that
    *   the user moves along the curve freely.
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

    /*!@brief Find the closest point to the line fragment.
    *
    *   This method uses linear search for the initial testing purpose. In the future
    *   we will implement more sophisticated searching methods to improve the 
    *   search performance.
    *
    *   @param pos current position.
    *   @return closest point to the line.
    */
    void findClosestPointToLine(vct3 &pos, vct3 &start, vct3 &end);

    /*! @brief Sets line points into a dynamic array.
    *  
    *   This function takes an array of vectors and stores them.
    *   Using this function, we can define the line any time during the operation.
    *   It allows to change the line dynamicaly.
    *
    *   @param points array contains line points.
    */
    void setLinePoints(std::vector<vct3> &points);

    /*! @brief Set closest point to the line.
    *
    *   @param cp closest point.
    */
    void setClosestPoint(const vct3 &cp);

    /*! @brief Return closest point to the line.
    *  
    *   @return closestPoint.
    */
    vct3 getClosestPoint(void);

    /*! @brief Sets given points for the desired curve.
    *
    *   @param points given points
    */
    void setPoints(const std::vector<vct3> &points); 

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

    std::vector<vct3> LinePoints; //!< Dynmamic array of vector that holds curve points.

protected:
    vct3 closestPoint; //!< Closest point to the curve.
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