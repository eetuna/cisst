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
#include <iostream>
#include <conio.h>
#include <cstdio>
#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include<windows.h>


void DoublePlaneVirtualFixture::getPositionFromUser(vct3 & position){
    std::cin>>position.X()>>position.Y()>>position.Z();
}
void DoublePlaneVirtualFixture::findOrthogonal(vct3 in, vct3 &out1, vct3 &out2){
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

vct3 DoublePlaneVirtualFixture::closestPoint(vct3 p , vct3 norm , vct3 base){
    return p-norm*(vctDotProduct((p-base),norm));
}

double DoublePlaneVirtualFixture::shortestDistance(vct3 p ,vct3 norm , vct3 base){
    return abs(vctDotProduct((p-base),norm));
}
void DoublePlaneVirtualFixture::update(const vctFrm3 & pos , prmFixtureGainCartesianSet & vfParams) {
    vct3 position;  //<! final force position
    vctMatRot3 rotation; //<! final force orientation
    vct3 currentPosition; //<! current MTM position
    double distance1, distance2; //<! distance between current position and the closest point position on the planes
    vct3 closest1, closest2; //<! closest point from MTM current position to the planes
    vct3 norm_vector; //<! norm vector
    vct3 ortho1(0.0); //<! orthogonal vector to the norm vector
    vct3 ortho2(0.0); //<! orthogonal vector to the norm vector
    vct3 stiffnessPos, stiffnessNeg; //<! position stiffness constant

    stiffnessNeg.SetAll(0.0);
    stiffnessPos.SetAll(0.0);

    //get curent MTM position
    currentPosition = pos.Translation();
    //calculate closest point from MTM to the plane 1
    closest1 = closestPoint(currentPosition,getNormVector1(),getBasePoint1());
    //calculate closest point from MTM to the plane 2
    closest2 = closestPoint(currentPosition,getNormVector2(),getBasePoint2());
    //calculate position error for the plane 1
    distance1 = (closest1-currentPosition).Norm(); 
    //calculate position error for the plane 2
    distance2 = (currentPosition-closest2).Norm();

    if(distance1<=distance2){
        position = closest1;
        //norm_vector = pos_error1.Divide(distance1);
        norm_vector = getNormVector1();
        stiffnessPos.Z() = -500.0;
    }else{
        position = closest2;
        //norm_vector = pos_error2.Divide(distance2);
        norm_vector = getNormVector2();
        stiffnessNeg.Z() = -500.0;
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
    vct3 temp;
    temp.SetAll(0.0);
    vfParams.SetPositionDampingPos(temp);
    vfParams.SetPositionDampingNeg(temp);
    vfParams.SetForceBiasPos(temp);
    vfParams.SetForceBiasNeg(temp);
    vfParams.SetOrientationStiffnessPos(temp);
    vfParams.SetOrientationStiffnessNeg(temp);
    vfParams.SetOrientationDampingPos(temp);
    vfParams.SetOrientationDampingNeg(temp);
    vfParams.SetTorqueBiasPos(temp);
    vfParams.SetTorqueBiasNeg(temp);

}

void DoublePlaneVirtualFixture::setBasePoint1(const vct3 & b){
    basePoint1 = b;
}
void DoublePlaneVirtualFixture::setBasePoint2(const vct3 & b){
    basePoint2 = b;
}
void DoublePlaneVirtualFixture::setNormVector1(const vct3 & n){
    normVector1 = n;
}
void DoublePlaneVirtualFixture::setNormVector2(const vct3 & n){
    normVector2 = n;
}

vct3 DoublePlaneVirtualFixture::getBasePoint1(void){
    return basePoint1;
}

vct3 DoublePlaneVirtualFixture::getNormVector1(void){
    return normVector1;
}

vct3 DoublePlaneVirtualFixture::getBasePoint2(void){
    return basePoint2;
}

vct3 DoublePlaneVirtualFixture::getNormVector2(void){
    return normVector2;
}



