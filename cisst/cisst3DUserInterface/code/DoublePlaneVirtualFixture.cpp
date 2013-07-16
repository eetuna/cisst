/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id: DoublePlaneVirtualFixture.cpp 3148 2013-06-26 15:46:31Z oozgune1 $

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
#include <cisst3DUserInterface/DoublePlaneVirtualFixture.h>
#include <cisstVector.h>

// Constructor that takes no argument.
DoublePlaneVirtualFixture::DoublePlaneVirtualFixture(void){
    PositionStiffnessPositive.SetAll(-500.0);
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
    TorqueBiasNegative.SetAll(0.0);}

// Constructor that takes plane base points and plane normal vectors and sets them.
DoublePlaneVirtualFixture::DoublePlaneVirtualFixture(vct3 base1, vct3 base2, vct3 planeNormal1, vct3 planeNormal2){
    setBasePoint1(base1); //set base1
    setBasePoint2(base2); //set base 2
    setPlaneNormal1(planeNormal1/(planeNormal1.Norm())); //normalize and set plane normal
    setPlaneNormal2(planeNormal2/(planeNormal2.Norm())); //normalize and set plane normal

    PositionStiffnessPositive.SetAll(-500.0);
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
void DoublePlaneVirtualFixture::findOrthogonal(vct3 in, vct3 &out1, vct3 &out2){
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
vct3 DoublePlaneVirtualFixture::closestPoint(vct3 point , vct3 norm , vct3 base){
    return point-norm*(vctDotProduct((point-base),norm));
}

// Finds the shortest distance to the plane.
double DoublePlaneVirtualFixture::shortestDistance(vct3 p ,vct3 norm , vct3 base){
    return abs(vctDotProduct((p-base),norm));
}

// Updates double plane virtual fixture parameters.
void DoublePlaneVirtualFixture::update(const vctFrm3 & pos , prmFixtureGainCartesianSet & vfParams) {
    vct3 position;  //<! Force position
    vctMatRot3 rotation; //<! Force orientation
    vct3 currentPosition; //<! Current MTM position
    double distance1, distance2; //<! Distance between current position and the closest point position on the planes
    vct3 closest1, closest2; //<! Closest point from MTM current position to the planes
    vct3 norm_vector; //<! Normal vector to create force orientation
    vct3 ortho1(0.0); //<! Orthogonal vector to the norm vector to form force/torque orientation
    vct3 ortho2(0.0); //<! Orthogonal vector to the norm vector to form force/torque orientation
    vct3 stiffnessPos, stiffnessNeg;    //position stiffness constants

    //reset all values
    rotation.SetAll(0.0);
    position.SetAll(0.0);
    stiffnessNeg.SetAll(0.0);
    stiffnessPos.SetAll(0.0);

    //get curent MTM position
    currentPosition = pos.Translation();
    //calculate closest point from MTM to the plane 1
    closest1 = closestPoint(currentPosition,getPlaneNormal1(),getBasePoint1());
    //calculate closest point from MTM to the plane 2
    closest2 = closestPoint(currentPosition,getPlaneNormal2(),getBasePoint2());
    //calculate position error for the plane 1
    distance1 = (closest1-currentPosition).Norm(); 
    //calculate position error for the plane 2
    distance2 = (currentPosition-closest2).Norm();

    //decide which plane to use
    if(distance1<=distance2){
        position = closest1;
        //norm_vector = pos_error1.Divide(distance1);
        norm_vector = getPlaneNormal1();
        stiffnessPos.Z() = PositionStiffnessPositive.Z();
    }else{
        position = closest2;
        //norm_vector = pos_error2.Divide(distance2);
        norm_vector = getPlaneNormal2();
        stiffnessNeg.Z() = PositionStiffnessNegative.Z();
    }

    //find 2 orthogonal vectors to the norm vector
    findOrthogonal(norm_vector,ortho1,ortho2);
    //form rotation using orthogonal vectors and norm vector
    rotation.Column(0).Assign(ortho2);
    rotation.Column(1).Assign(-ortho1);
    rotation.Column(2).Assign(norm_vector);
    //set force position
    vfParams.SetForcePosition(position);
    //set force orientation
    vfParams.SetForceOrientation(rotation);
    //set torque orientation
    vfParams.SetTorqueOrientation(rotation);
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

// Sets the base point for the first plane.
void DoublePlaneVirtualFixture::setBasePoint1(const vct3 & base1){
    basePoint1 = base1;
}

// Sets the base point for the second plane.
void DoublePlaneVirtualFixture::setBasePoint2(const vct3 & base2){
    basePoint2 = base2;
}

// Sets the plane normal vector for the first plane.
void DoublePlaneVirtualFixture::setPlaneNormal1(const vct3 & normal1){
    planeNormal1 = normal1;
}

// Sets the plane normal vector for the second plane.
void DoublePlaneVirtualFixture::setPlaneNormal2(const vct3 & normal2){
    planeNormal2 = normal2;
}

// Returns the base plane for the first plane.
vct3 DoublePlaneVirtualFixture::getBasePoint1(void){
    return basePoint1;
}

// Returns the firt plane's plane normal vector.
vct3 DoublePlaneVirtualFixture::getPlaneNormal1(void){
    return planeNormal1;
}

// Returns the base plane for the second plane.
vct3 DoublePlaneVirtualFixture::getBasePoint2(void){
    return basePoint2;
}

// Returns the second plane's plane normal vector.
vct3 DoublePlaneVirtualFixture::getPlaneNormal2(void){
    return planeNormal2;
}

// Sets the given positive position stiffness constant.
void DoublePlaneVirtualFixture::setPositionStiffnessPositive(const vct3 stiffPos){
    this->PositionStiffnessPositive = stiffPos;
}

// Sets the given negative position stiffness constant.
void DoublePlaneVirtualFixture::setPositionStiffnessNegative(const vct3 stiffNeg){
    this->PositionStiffnessNegative = stiffNeg;
}

// Sets the given positive position damping constant.
void DoublePlaneVirtualFixture::setPositionDampingPositive(const vct3 dampPos){
    this->PositionDampingPositive = dampPos;
}

// Sets the given negative position damping constant.
void DoublePlaneVirtualFixture::setPositionDampingNegative(const vct3 dampNeg){
    this->PositionStiffnessNegative = dampNeg;
}

// Sets the given positive force bias constant.
void DoublePlaneVirtualFixture::setForceBiasPositive(const vct3 biasPos){
    this->ForceBiasPositive = biasPos;
}

// Sets the given negative force bias constant.
void DoublePlaneVirtualFixture::setForceBiasNegative(const vct3 biasNeg){
    this->ForceBiasNegative = biasNeg;
}

// Sets the given positive orientation stiffness constant.
void DoublePlaneVirtualFixture::setOrientationStiffnessPositive(const vct3 orientStiffPos){
    this->OrientationStiffnessPositive = orientStiffPos;
}

// Sets the given negative orientation stiffness constant.
void DoublePlaneVirtualFixture::setOrientationStiffnessNegative(const vct3 orientStiffNeg){
    this->OrientationStiffnessNegative = orientStiffNeg;
}

// Sets the given positive orientation damping constant.
void DoublePlaneVirtualFixture::setOrientationDampingPositive(const vct3 orientDampPos){
    this->OrientationDampingPositive = orientDampPos;
}

// Sets the given negative orientation damping constant.
void DoublePlaneVirtualFixture::setOrientationDampingNegative(const vct3 orientDampNeg){
    this->OrientationDampingNegative = orientDampNeg;
}

// Sets the given positive torque bias constant.
void DoublePlaneVirtualFixture::setTorqueBiasPositive(const vct3 torqueBiasPos){
    this->TorqueBiasPositive = torqueBiasPos;
}

// Sets the given negative torque bias constant.
void DoublePlaneVirtualFixture::setTorqueBiasNegative(const vct3 torqueBiasNeg){
    this->TorqueBiasNegative = torqueBiasNeg;
}

// Returns the positive position stiffness constant.
vct3 DoublePlaneVirtualFixture::getPositionStiffnessPositive(void){
    return this->PositionStiffnessPositive;
}

// Returns the negative position stiffness constant.
vct3 DoublePlaneVirtualFixture::getPositionStiffnessNegative(void){
    return this->PositionStiffnessNegative;
}

// Returns the positive position damping constant.
vct3 DoublePlaneVirtualFixture::getPositionDampingPositive(void){
    return this->PositionDampingPositive;
}

// Returns the negative position damping constant.
vct3 DoublePlaneVirtualFixture::getPositionDampingNegative(void){
    return this->PositionStiffnessNegative;
}

// Returns the positive force bias constant.
vct3 DoublePlaneVirtualFixture::getForceBiasPositive(void){
    return this->ForceBiasPositive;
}

// Returns the negative force bias constant.
vct3 DoublePlaneVirtualFixture::getForceBiasNegative(void){
    return this->ForceBiasNegative;
}

// Returns the positive orientation stiffness constant.
vct3 DoublePlaneVirtualFixture::getOrientationStiffnessPositive(void){
    return this->OrientationStiffnessPositive;
}

// Returns the negative orientation stiffness constant.
vct3 DoublePlaneVirtualFixture::getOrientationStiffnessNegative(void){
    return this->OrientationStiffnessNegative;
}

// Returns the positive orientation damping constant.
vct3 DoublePlaneVirtualFixture::getOrientationDampingPositive(void){
    return this->OrientationDampingPositive;
}

// Returns the negative orientation damping constant.
vct3 DoublePlaneVirtualFixture::getOrientationDampingNegative(void){
    return this->OrientationDampingNegative;
}

// Returns the positive torque bias constant.
vct3 DoublePlaneVirtualFixture::getTorqueBiasPositive(void){
    return this->TorqueBiasPositive;
}

// Returns the negative torque bias constant.
vct3 DoublePlaneVirtualFixture::getTorqueBiasNegative(void){
    return this->TorqueBiasNegative;
}
