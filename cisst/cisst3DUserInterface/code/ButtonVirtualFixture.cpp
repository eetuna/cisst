/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id: ButtonVirtualFixture.cpp 3148 2013-06-26 15:46:31Z oozgune1 $

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

#include <cisst3DUserInterface/ButtonVirtualFixture.h>
#include <cisstVector.h>

// ButtonVirtualFixture contructor that takes no argument.
ButtonVirtualFixture::ButtonVirtualFixture(void){
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

// PointVirtualFixture contructor that takes the point position and sets it.
ButtonVirtualFixture::ButtonVirtualFixture(const vct3 pt){
    setButtonPosition(pt); //set point position

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

// Updates the point virtual fixture parameters.
void ButtonVirtualFixture::update(const vctFrm3 & pos , prmFixtureGainCartesianSet & vfParams) {
    vct3 position; //<! Force position.
    vctMatRot3 rotation; //<! Force orientation.
    const double buttonTop = 3.0; //<! Button top limit
    const double buttonBottom = 6.0; //<! Button bottom limit
    double depthZ; //<!Button depth
    double buttonSurface; //<! Button surface
    vct3 dampP,dempN; //<! Positive and negative damping constant
    //we set rotation to the initial condition
    //we do not need force orientation
    rotation = vctMatRot3::Identity(); 
    //force position is the given position
    position = getButtonPosition();
    //set force position
    vfParams.SetForcePosition(position);
    //set force orientation
    vfParams.SetForceOrientation(rotation);
    //set torque orientation
    vfParams.SetTorqueOrientation(rotation);
    vct3 negatifStiffness(-1000.0,-1000.0,-20.0);
    depthZ = pos.Translation().Z();
    buttonSurface = button.Z();

    if (depthZ > (buttonSurface + buttonTop)) { 
        double depth = depthZ - (buttonSurface + buttonTop);
        negatifStiffness.Z() = 1.0 - depth * 50.0;
        dampP.SetAll(-10.0);
        dempN.SetAll(-10.0);
    } else {
        dampP.SetAll(0.0);
        dempN.SetAll(0.0);
        negatifStiffness.Z() = -500.0;

    }

    //set Position Stiffness
    vfParams.SetPositionStiffnessPos(PositionStiffnessPositive);
    vfParams.SetPositionStiffnessNeg(negatifStiffness);
    //Temporary hard code solution ask Anton for better way
    vfParams.SetPositionDampingPos(dampP);
    vfParams.SetPositionDampingNeg(dempN);
    vfParams.SetForceBiasPos(ForceBiasPositive);
    vfParams.SetForceBiasNeg(ForceBiasNegative);
    vfParams.SetOrientationStiffnessPos(OrientationStiffnessPositive);
    vfParams.SetOrientationStiffnessNeg(OrientationStiffnessNegative);
    vfParams.SetOrientationDampingPos(OrientationDampingPositive);
    vfParams.SetOrientationDampingNeg(OrientationDampingNegative);
    vfParams.SetTorqueBiasPos(TorqueBiasPositive);
    vfParams.SetTorqueBiasNeg(TorqueBiasNegative);
}

// Sets the button position.
void ButtonVirtualFixture::setButtonPosition(const vct3 & pt){
    button = pt;
}

// Returns the button position.
vct3 ButtonVirtualFixture::getButtonPosition(void){
    return button;
}

// Sets the given positive position stiffness constant.
void ButtonVirtualFixture::setPositionStiffnessPositive(const vct3 stiffPos){
    this->PositionStiffnessPositive = stiffPos;
}

// Sets the given negative position stiffness constant.
void ButtonVirtualFixture::setPositionStiffnessNegative(const vct3 stiffNeg){
    this->PositionStiffnessNegative = stiffNeg;
}

// Sets the given positive position damping constant.
void ButtonVirtualFixture::setPositionDampingPositive(const vct3 dampPos){
    this->PositionDampingPositive = dampPos;
}

// Sets the given negative position damping constant.
void ButtonVirtualFixture::setPositionDampingNegative(const vct3 dampNeg){
    this->PositionStiffnessNegative = dampNeg;
}

// Sets the given positive force bias constant.
void ButtonVirtualFixture::setForceBiasPositive(const vct3 biasPos){
    this->ForceBiasPositive = biasPos;
}

// Sets the given negative force bias constant.
void ButtonVirtualFixture::setForceBiasNegative(const vct3 biasNeg){
    this->ForceBiasNegative = biasNeg;
}

// Sets the given positive orientation stiffness constant.
void ButtonVirtualFixture::setOrientationStiffnessPositive(const vct3 orientStiffPos){
    this->OrientationStiffnessPositive = orientStiffPos;
}

// Sets the given negative orientation stiffness constant.
void ButtonVirtualFixture::setOrientationStiffnessNegative(const vct3 orientStiffNeg){
    this->OrientationStiffnessNegative = orientStiffNeg;
}

// Sets the given positive orientation damping constant.
void ButtonVirtualFixture::setOrientationDampingPositive(const vct3 orientDampPos){
    this->OrientationDampingPositive = orientDampPos;
}

// Sets the given negative orientation damping constant.
void ButtonVirtualFixture::setOrientationDampingNegative(const vct3 orientDampNeg){
    this->OrientationDampingNegative = orientDampNeg;
}

// Sets the given positive torque bias constant.
void ButtonVirtualFixture::setTorqueBiasPositive(const vct3 torqueBiasPos){
    this->TorqueBiasPositive = torqueBiasPos;
}

// Sets the given negative torque bias constant.
void ButtonVirtualFixture::setTorqueBiasNegative(const vct3 torqueBiasNeg){
    this->TorqueBiasNegative = torqueBiasNeg;
}

// Returns the positive position stiffness constant.
vct3 ButtonVirtualFixture::getPositionStiffnessPositive(void){
    return this->PositionStiffnessPositive;
}

// Returns the negative position stiffness constant.
vct3 ButtonVirtualFixture::getPositionStiffnessNegative(void){
    return this->PositionStiffnessNegative;
}

// Returns the positive position damping constant.
vct3 ButtonVirtualFixture::getPositionDampingPositive(void){
    return this->PositionDampingPositive;
}

// Returns the negative position damping constant.
vct3 ButtonVirtualFixture::getPositionDampingNegative(void){
    return this->PositionStiffnessNegative;
}

// Returns the positive force bias constant.
vct3 ButtonVirtualFixture::getForceBiasPositive(void){
    return this->ForceBiasPositive;
}

// Returns the negative force bias constant.
vct3 ButtonVirtualFixture::getForceBiasNegative(void){
    return this->ForceBiasNegative;
}

// Returns the positive orientation stiffness constant.
vct3 ButtonVirtualFixture::getOrientationStiffnessPositive(void){
    return this->OrientationStiffnessPositive;
}

// Returns the negative orientation stiffness constant.
vct3 ButtonVirtualFixture::getOrientationStiffnessNegative(void){
    return this->OrientationStiffnessNegative;
}

// Returns the positive orientation damping constant.
vct3 ButtonVirtualFixture::getOrientationDampingPositive(void){
    return this->OrientationDampingPositive;
}

// Returns the negative orientation damping constant.
vct3 ButtonVirtualFixture::getOrientationDampingNegative(void){
    return this->OrientationDampingNegative;
}

// Returns the positive torque bias constant.
vct3 ButtonVirtualFixture::getTorqueBiasPositive(void){
    return this->TorqueBiasPositive;
}

// Returns the negative torque bias constant.
vct3 ButtonVirtualFixture::getTorqueBiasNegative(void){
    return this->TorqueBiasNegative;
}
