/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id: PointVirtualFixture.cpp 3148 2013-06-26 15:46:31Z oozgune1 $

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

#include <cisst3DUserInterface/PointVirtualFixture.h>
#include <cisstVector.h>
#include <iostream>
#include <conio.h>
#include <cstdio>
#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include<windows.h>

//Constructor takes no argument
PointVirtualFixture::PointVirtualFixture(){}


PointVirtualFixture::PointVirtualFixture(vct3 point)
{
    setPoint(point);
}

void PointVirtualFixture::update(const vctFrm3 & pos , prmFixtureGainCartesianSet & vfParams) {
    vct3 position;  //final force position
    vctMatRot3 rotation;  //final force orientation
    vct3 stiffnessPos; //positive position stiffness constant
    vct3 stiffnessNeg; //negative position stiffness constant

    //we set rotation to the initial condition
    rotation = pos.Rotation();
    //force position is the given position
    position = getPoint();
    //set force position
    vfParams.SetForcePosition(position);
    //set force orientation
    vfParams.SetForceOrientation(rotation);
    //set torque orientation
    vfParams.SetTorqueOrientation(rotation);

    //set Position Stiffness
    stiffnessPos.SetAll(-150.0);
    stiffnessNeg.SetAll(-150.0);
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

void PointVirtualFixture::setPoint(const vct3 & p){
    point = p;
}

vct3 PointVirtualFixture::getPoint(void){
    return point;
}



