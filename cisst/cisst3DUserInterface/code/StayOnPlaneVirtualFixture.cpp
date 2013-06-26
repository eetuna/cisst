/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id: StayOnPlaneVirtualFixture.cpp 3148 2013-06-26 15:46:31Z oozgune1 $

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

#include <cisst3DUserInterface/StayOnPlaneVirtualFixture.h>
#include <cisstVector.h>
#include <iostream>
#include <conio.h>
#include <cstdio>
#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include<windows.h>


StayOnPlaneVirtualFixture::StayOnPlaneVirtualFixture(void){}

StayOnPlaneVirtualFixture::StayOnPlaneVirtualFixture(vct3 basePoint, vct3 planeNormal)
{
    setBasePoint(basePoint);
    setPlaneNormal(planeNormal/planeNormal.Norm());
}

void StayOnPlaneVirtualFixture::getPositionFromUser(vct3 & position){
    std::cin>>position.X()>>position.Y()>>position.Z();
}

void StayOnPlaneVirtualFixture::findOrthogonal(vct3 in, vct3 &out1, vct3 &out2){
    vct3 vec1, vec2;
    vct3 axisY(0.0 , 1.0 , 0.0);
    vct3 axizZ(0.0 , 0.0 , 1.0);
    double len1, len2, len3;

    // Find 1st orthogonal vector
    len1 = in.Norm(); 
    in = in.Divide(len1); 
    // Use Y-axis unit vector to find first orthogonal
    vec1.CrossProductOf(in,axisY); 

    // Check to make sure the Y-axis unit vector is not too close to input unit vector,
    // if they are close dot product will be large and then use different arbitrary unit vector
    if ( vctDotProduct(in, vec1) >= 0.98){
        vec1.CrossProductOf(in,axizZ); 
        std::cout<<"Something is not good..."<<std::endl;
    }

    // Now find 2nd orthogonal vector
    vec2.CrossProductOf(in,vec1); 
    len2 = vec1.Norm(); 
    len3 = vec2.Norm(); 

    out1 = vec1.Divide(len2);
    out2 = vec2.Divide(len3);
}


vct3 StayOnPlaneVirtualFixture::closestPoint(vct3 p){
    return p-getPlaneNormal()*(vctDotProduct((p-getBasePoint()),getPlaneNormal()));
}

double StayOnPlaneVirtualFixture::shortestDistance(vct3 p){
    return abs(vctDotProduct((p-getBasePoint()),getPlaneNormal()));
}
void StayOnPlaneVirtualFixture::update(const vctFrm3 & pos , prmFixtureGainCartesianSet & vfParams) {
    vct3 position;  //<!final force position
    vctMatRot3 rotation;  //<!final force orientation
    vct3 pos_error; //<!position error between current and the closest point position
    vct3 currentPosition; //<! current MTM position
    vct3 closest; //<!closest point from MTM current position to the plane
    vct3 norm_vector; //<!normal vector 
    vct3 ortho1(0.0); //<!orthogonal vector to the norm vector
    vct3 ortho2(0.0); //<!orthogonal vector to the norm vector
    vct3 stiffnessPos; //<!position stiffness constant (positive)
    vct3 stiffnessNeg; //<!position stiffness constant (negative)
    vct3 dampingPos;  //<!Positive damping constant
    vct3 dampingNeg;  //<!Negative damping constant
    double distance; //<!distance between current position and the closest point position

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

    stiffnessPos.SetAll(0.0);
    stiffnessPos.Z() = -500.0;
    stiffnessNeg.SetAll(0.0);
    stiffnessNeg.Z() = -500.0;
    dampingPos.SetAll(0.0);
    dampingPos.X() = -25.0;
    dampingPos.Y() = -25.0;
    dampingNeg.SetAll(0.0);
    dampingNeg.X() = -25.0;
    dampingNeg.Y() = -25.0;

    //set Position Stiffness
    vfParams.SetPositionStiffnessPos(stiffnessPos);
    vfParams.SetPositionStiffnessNeg(stiffnessNeg);
    vfParams.SetPositionDampingPos(dampingPos);
    vfParams.SetPositionDampingNeg(dampingNeg);
    //Temporary hard code solution ask Anton for better way
    vct3 temp;
    temp.SetAll(0.0);
    vfParams.SetForceBiasPos(temp);
    vfParams.SetForceBiasNeg(temp);
    vfParams.SetOrientationStiffnessPos(temp);
    vfParams.SetOrientationStiffnessNeg(temp);
    vfParams.SetOrientationDampingPos(temp);
    vfParams.SetOrientationDampingNeg(temp);
    vfParams.SetTorqueBiasPos(temp);
    vfParams.SetTorqueBiasNeg(temp);
}

void StayOnPlaneVirtualFixture::setBasePoint(const vct3 & b){
    basePoint = b;
}
void StayOnPlaneVirtualFixture::setPlaneNormal(const vct3 & n){
    planeNormal = n;
}

vct3 StayOnPlaneVirtualFixture::getBasePoint(void){
    return basePoint;
}

vct3 StayOnPlaneVirtualFixture::getPlaneNormal(void){
    return planeNormal;
}
