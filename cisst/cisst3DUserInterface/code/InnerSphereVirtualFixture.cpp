/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id: InnerSphereVirtualFixture.cpp 3148 2013-06-26 15:46:31Z oozgune1 $

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
#include <cisst3DUserInterface/InnerSphereVirtualFixture.h>
#include <cisstVector.h>

// InnerSphereVirtualFixture contructor that takes no argument.
InnerSphereVirtualFixture::InnerSphereVirtualFixture(void){
    PositionStiffnessPositive.SetAll(-500.0);
    PositionStiffnessNegative.SetAll(0.0);
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

/* InnerPlaneVirtualFixture contructor that takes the center of the sphere 
and the radius and sets them.
*/
InnerSphereVirtualFixture::InnerSphereVirtualFixture(const vct3 center, const double radius){
    setCenter(center); //set sphere center
    setRadius(radius); //set sphere radius
    PositionStiffnessPositive.SetAll(-500.0);
    PositionStiffnessNegative.SetAll(0.0);
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
void InnerSphereVirtualFixture::findOrthogonal(vct3 in, vct3 &out1, vct3 &out2){
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

// Updates inner sphere virtual fixture parameters.
void InnerSphereVirtualFixture::update(const vctFrm3 & pos ,prmFixtureGainCartesianSet & vfParams) {
    vct3 position;  //<! Force position
    vctMatRot3 rotation; //<! Force orientation
    vct3 pos_error; //<! Position error between current and center position
    vct3 currentPosition; //<! Current MTM position
    vct3 norm_vector; //<! Normal vector to create froce/torque orientation matrix 
    vct3 scaled_norm_vector; //<! Norm vector scaled with -radius
    vct3 ortho1(0.0); //<! Orthogonal vector to the normal vector to form force/torque orientation matrix
    vct3 ortho2(0.0); //<! Orthogonal vector to the normal vector to form force/torque orientation matrix
    double distance; //<! Distance between current position and center position
    vct3 stiffnessPos; //<! Positive position stiffness
    stiffnessPos.SetAll(0.0);

    //get curent position
    currentPosition = pos.Translation();
    //calculate position error
    pos_error = (center-currentPosition);
    //find distance (norm of pos_error)
    distance = pos_error.Norm();
    //scale pos_error to calculate norm vector
    norm_vector = pos_error.Divide(distance);
    //find 2 orthogonal vectors to the norm vector
    findOrthogonal(norm_vector,ortho1,ortho2);
    //form rotation using orthogonal vectors
    rotation.Column(0).Assign(ortho2);
    rotation.Column(1).Assign(-ortho1);
    rotation.Column(2).Assign(norm_vector);
    //scale norm vector with radius
    scaled_norm_vector = norm_vector.Multiply(-radius);
    //add scaled_norm_vector to the sphere center
    position = center+scaled_norm_vector;

    //set force position
    vfParams.SetForcePosition(position);
    //set force orientation
    vfParams.SetForceOrientation(rotation);
    //set torque orientation
    vfParams.SetTorqueOrientation(rotation);

    //set Position Stiffness
    stiffnessPos.Z() = PositionStiffnessPositive.Z();
    vfParams.SetPositionStiffnessPos(stiffnessPos);
    //Temporary hard code solution ask Anton for better way
    vfParams.SetPositionStiffnessNeg(PositionStiffnessNegative);
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

// Sets the sphere center position.
void InnerSphereVirtualFixture::setCenter(const vct3 & c){
    center = c;
}

// Sets the sphere radius.
void InnerSphereVirtualFixture::setRadius(const double & r){
    radius = r;
}

// Returns the sphere radius.
double InnerSphereVirtualFixture::getRadius(void){
    return radius;
}

// Returns the center position of the sphere.
vct3 InnerSphereVirtualFixture::getCenter(void){
    return center;
}

// Sets the given positive position stiffness constant.
void InnerSphereVirtualFixture::setPositionStiffnessPositive(const vct3 stiffPos){
    this->PositionStiffnessPositive = stiffPos;
}

// Sets the given negative position stiffness constant.
void InnerSphereVirtualFixture::setPositionStiffnessNegative(const vct3 stiffNeg){
    this->PositionStiffnessNegative = stiffNeg;
}

// Sets the given positive position damping constant.
void InnerSphereVirtualFixture::setPositionDampingPositive(const vct3 dampPos){
    this->PositionDampingPositive = dampPos;
}

// Sets the given negative position damping constant.
void InnerSphereVirtualFixture::setPositionDampingNegative(const vct3 dampNeg){
    this->PositionStiffnessNegative = dampNeg;
}

// Sets the given positive force bias constant.
void InnerSphereVirtualFixture::setForceBiasPositive(const vct3 biasPos){
    this->ForceBiasPositive = biasPos;
}

// Sets the given negative force bias constant.
void InnerSphereVirtualFixture::setForceBiasNegative(const vct3 biasNeg){
    this->ForceBiasNegative = biasNeg;
}

// Sets the given positive orientation stiffness constant.
void InnerSphereVirtualFixture::setOrientationStiffnessPositive(const vct3 orientStiffPos){
    this->OrientationStiffnessPositive = orientStiffPos;
}

// Sets the given negative orientation stiffness constant.
void InnerSphereVirtualFixture::setOrientationStiffnessNegative(const vct3 orientStiffNeg){
    this->OrientationStiffnessNegative = orientStiffNeg;
}

// Sets the given positive orientation damping constant.
void InnerSphereVirtualFixture::setOrientationDampingPositive(const vct3 orientDampPos){
    this->OrientationDampingPositive = orientDampPos;
}

// Sets the given negative orientation damping constant.
void InnerSphereVirtualFixture::setOrientationDampingNegative(const vct3 orientDampNeg){
    this->OrientationDampingNegative = orientDampNeg;
}

// Sets the given positive torque bias constant.
void InnerSphereVirtualFixture::setTorqueBiasPositive(const vct3 torqueBiasPos){
    this->TorqueBiasPositive = torqueBiasPos;
}

// Sets the given negative torque bias constant.
void InnerSphereVirtualFixture::setTorqueBiasNegative(const vct3 torqueBiasNeg){
    this->TorqueBiasNegative = torqueBiasNeg;
}

// Returns the positive position stiffness constant.
vct3 InnerSphereVirtualFixture::getPositionStiffnessPositive(void){
    return this->PositionStiffnessPositive;
}

// Returns the negative position stiffness constant.
vct3 InnerSphereVirtualFixture::getPositionStiffnessNegative(void){
    return this->PositionStiffnessNegative;
}

// Returns the positive position damping constant.
vct3 InnerSphereVirtualFixture::getPositionDampingPositive(void){
    return this->PositionDampingPositive;
}

// Returns the negative position damping constant.
vct3 InnerSphereVirtualFixture::getPositionDampingNegative(void){
    return this->PositionStiffnessNegative;
}

// Returns the positive force bias constant.
vct3 InnerSphereVirtualFixture::getForceBiasPositive(void){
    return this->ForceBiasPositive;
}

// Returns the negative force bias constant.
vct3 InnerSphereVirtualFixture::getForceBiasNegative(void){
    return this->ForceBiasNegative;
}

// Returns the positive orientation stiffness constant.
vct3 InnerSphereVirtualFixture::getOrientationStiffnessPositive(void){
    return this->OrientationStiffnessPositive;
}

// Returns the negative orientation stiffness constant.
vct3 InnerSphereVirtualFixture::getOrientationStiffnessNegative(void){
    return this->OrientationStiffnessNegative;
}

// Returns the positive orientation damping constant.
vct3 InnerSphereVirtualFixture::getOrientationDampingPositive(void){
    return this->OrientationDampingPositive;
}

// Returns the negative orientation damping constant.
vct3 InnerSphereVirtualFixture::getOrientationDampingNegative(void){
    return this->OrientationDampingNegative;
}

// Returns the positive torque bias constant.
vct3 InnerSphereVirtualFixture::getTorqueBiasPositive(void){
    return this->TorqueBiasPositive;
}

// Returns the negative torque bias constant.
vct3 InnerSphereVirtualFixture::getTorqueBiasNegative(void){
    return this->TorqueBiasNegative;
}
