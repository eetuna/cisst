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
#include <cisst3DUserInterface/VirtualFixture.h>
#include <cisstVector.h>

#ifndef curvevirtualfixture_h
#define curvevirtualfixture_h

/*!@brief Curve Virtual Fixture class.
*
*  This type of virtual fixture is designed to move along a given
*  curve with a desired orientation. Line segment has start and end positions. 
*  This virtual fixture allows us to move along the line and applies force if the user 
*  tries to leave the line. At the same time, applies torque to keep the user in the
*  desired orientation.
*/
class CurveVirtualFixture: public VirtualFixture {

public:

    /*!@brief Constructor with no argument
    */
    CurveVirtualFixture(void);

    /*! @brief Constructor.
    *
    *   In the constructor, we call the function called DefineLine.
    */
    CurveVirtualFixture(std::vector<vct3>);

    /*! @brief Update Line Segment Virtual Fixture parameterss.
    *
    *  We find the closest point to the line and set this point as a force frame
    *  position. Using the closest point, we allign the z-axis parallel 
    *  to the line. We apply force on x and y axes. This way there will be force
    *  if the user tries to leave the line. Also this method will ensures that
    *  the user moves along the line.
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

    /*!@brief Find the closest point to the line.
    *
    *   This method uses linear search for the initial testing purpose. In the future
    *   we will implement more sophisticated searching methods to improve the 
    *   search performance.
    *
    *  @param pos current position.
    *  @return closest point to the line.
    */
    void findClosestPointToLine(vct3 &pos, vct3 &start, vct3 &end);

    /*!@brief Sets line points into a dynamic array.
    *  
    *  This function takes an array of vectors and stores them.
    *  Using this function, we can define the line any time during the operation.
    *
    *  @param points array contains line points.
    */
    void setLinePoints(std::vector<vct3> &points); //sets line points localy

    /*!@brief Set closest point to the line.
    *
    *  @param cp closest point.
    */
    void setClosestPoint(const vct3 &cp);

    /*!@brief Return closest point to the line.
    *  
    *  @return closestPoint.
    */
    vct3 getClosestPoint(void);

    /*!@brief Sets given points for the desired line
    *
    *  @param points given points
    */
    void setPoints(const std::vector<vct3> &points); 

    std::vector<vct3> LinePoints; //!< Dynmamic array of vector that holds line points.

protected:
    vct3 closestPoint; //!< Closest point to the line.
};

#endif