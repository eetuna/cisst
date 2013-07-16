/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id: PlaneVirtualFixture.cpp 3148 2013-06-26 15:46:31Z oozgune1 $

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

#include <cisst3DUserInterface/PlaneVirtualFixture.h>
#include <cisstVector.h>

// PlaneVirtualFixture contructor that takes no argument.
PlaneVirtualFixture::PlaneVirtualFixture(void){
    PositionStiffnessPositive.SetAll(0.0);
    PositionStiffnessNegative.SetAll(-500.0);
    PositionDampingPositive.SetAll(0.0);
    PositionDampingNegative.SetAll(0.0);
    ForceBiasPositive.SetAll(0.0);
    ForceBiasNegative.SetAll(0.0);
    OrientationStiffnessPositive.SetAll(0.0);
    OrientationStiffnessNegative.SetAll(0.0);
    OrientationDampingPositive.SetAll(0.0);
    OrientationDampingNegative.SetAll(0.0);
    TorqueBiasPositive.SetAll(0.0);
    TorqueBiasNegative.SetAll(0.0);
}

/* PlaneVirtualFixture contructor that takes the plane normal vector and 
the base point and sets them.
*/
PlaneVirtualFixture::PlaneVirtualFixture(const vct3 basePoint, const vct3 planeNormal){
    setBasePoint(basePoint); //set base point for the plane
    setPlaneNormal(planeNormal); //set plane normal vector
    PositionStiffnessPositive.SetAll(0.0);
    PositionStiffnessNegative.SetAll(-500.0);
    PositionDampingPositive.SetAll(0.0);
    PositionDampingNegative.SetAll(0.0);
    ForceBiasPositive.SetAll(0.0);
    ForceBiasNegative.SetAll(0.0);
    OrientationStiffnessPositive.SetAll(0.0);
    OrientationStiffnessNegative.SetAll(0.0);
    OrientationDampingPositive.SetAll(0.0);
    OrientationDampingNegative.SetAll(0.0);
    TorqueBiasPositive.SetAll(0.0);
    TorqueBiasNegative.SetAll(0.0);
}

// Finds two orthogonal vectors to the given vector.
void PlaneVirtualFixture::findOrthogonal(vct3 in, vct3 &out1, vct3 &out2){
    vct3 vec1, vec2;
    vct3 axisY(0.0 , 1.0 , 0.0);
    vct3 axisZ(0.0 , 0.0 , 1.0);
    double len1, len2, len3;
    // Find 1st orthogonal vector
    len1 = in.Norm(); 
    in = in.Divide(len1); 
    // Use Y-axis unit vector to find first orthogonal
    vec1.CrossProductOf(in,axisY); 
    // Check to make sure the Y-axis unit vector is not too close to input unit vector,
    // if they are close dot product will be large and then use different arbitrary unit vector
    if ( vctDotProduct(in, vec1) >= 0.98){
        vec1.CrossProductOf(in,axisZ); 
    }
    //TODO find a better way to handle
    if(vec1.X()==0.0 && vec1.Y()==0.0 && vec1.Z()== 0.0){
        vec1.CrossProductOf(in,axisZ);
    }
    // Now find 2nd orthogonal vector
    vec2.CrossProductOf(in,vec1); 
    len2 = vec1.Norm(); 
    len3 = vec2.Norm(); 
    //orthogonal vectors to the given vector
    out1 = vec1.Divide(len2);
    out2 = vec2.Divide(len3);
}

// Finds the closest point to the plane.
vct3 PlaneVirtualFixture::closestPoint(vct3 p){
    return p-getPlaneNormal()*(vctDotProduct((p-getBasePoint()),getPlaneNormal()));
}

// Finds the closest distance to the plane.
double PlaneVirtualFixture::shortestDistance(vct3 p){
    return abs(vctDotProduct((p-getBasePoint()),getPlaneNormal()));
}

// Updates the plane virtual fixture parameters.
void PlaneVirtualFixture::update(const vctFrm3 & pos , prmFixtureGainCartesianSet & vfParams) {
    vct3 position;  //<! Force position
    vctMatRot3 rotation;  //<! Force orientation
    vct3 pos_error; //<! Position error between current and the closest point position
    vct3 currentPosition; //<! Current MTM position
    vct3 closest; //<! Closest point from MTM current position to the plane
    vct3 norm_vector; //<! normal vector 
    vct3 ortho1(0.0); //<! Orthogonal vector to the normal vector to form force/torque orientation matrix
    vct3 ortho2(0.0); //<! Orthogonal vector to the normal vector to form force/torque orientation matrix
    vct3 stiffnessPos, stiffnessNeg; //<! Position stiffness constant (positive and negative)
    double distance; //<! Distance between current position and the closest point position
    stiffnessNeg.SetAll(0.0);
    stiffnessPos.SetAll(0.0);

    //get curent MTM position
    currentPosition = pos.Translation();
    //calculate closest point from MTM to the plane
    closest = closestPoint(currentPosition);
    //calculate position error
    pos_error = (closest-currentPosition);
    //find distance
    distance = pos_error.Norm();
    //calculate normal vector
    norm_vector = pos_error.Divide(distance);
    norm_vector = getPlaneNormal(); //use plane normal from user
    //find 2 orthogonal vectors to the norm vector
    findOrthogonal(norm_vector,ortho1,ortho2);
    //form force orientation using orthogonal vectors and normal vector
    rotation.Column(0).Assign(ortho2);
    rotation.Column(1).Assign(-ortho1);
    rotation.Column(2).Assign(norm_vector);
    //position should be closest point to the plane
    position = closest;
    //set force position
    vfParams.SetForcePosition(position);
    //set force orientation
    vfParams.SetForceOrientation(rotation);
    //set torque orientation
    vfParams.SetTorqueOrientation(rotation);

    //in order to protect abowe the plane comment out the following line
    //stiffnessPos.Z() = PositionStiffnessPositive.Z();

    //in order to protect below the plane comment out the following line
    stiffnessNeg.Z() = PositionStiffnessNegative.Z();

    //set Position Stiffness
    vfParams.SetPositionStiffnessPos(stiffnessPos);
    vfParams.SetPositionStiffnessNeg(stiffnessNeg);

    //Temporary hard code solution ask Anton for better way
    vfParams.SetPositionDampingPos(PositionDampingPositive);
    vfParams.SetPositionDampingNeg(PositionDampingNegative);
    vfParams.SetForceBiasPos(ForceBiasPositive);
    vfParams.SetForceBiasNeg(ForceBiasNegative);
    vfParams.SetOrientationStiffnessPos(OrientationStiffnessPositive);
    vfParams.SetOrientationStiffnessNeg(OrientationStiffnessNegative);
    vfParams.SetOrientationDampingPos(OrientationDampingPositive);
    vfParams.SetOrientationDampingNeg(OrientationDampingNegative);
    vfParams.SetTorqueBiasPos(TorqueBiasPositive);
    vfParams.SetTorqueBiasNeg(TorqueBiasNegative);
}

// Sets the base point for the plane.
void PlaneVirtualFixture::setBasePoint(const vct3 & base){
    basePoint = base;
}

// Sets the plane normal vector for the plane.
void PlaneVirtualFixture::setPlaneNormal(const vct3 & normal){
    planeNormal = normal;
}

// Returns the base point for the plane.
vct3 PlaneVirtualFixture::getBasePoint(void){
    return basePoint;
}

// Returns the plane normal vector.
vct3 PlaneVirtualFixture::getPlaneNormal(void){
    return planeNormal;
}
