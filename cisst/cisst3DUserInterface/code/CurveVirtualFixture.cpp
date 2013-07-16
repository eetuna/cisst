/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id: CurveVirtualFixture.cpp 3148 2013-06-26 15:46:31Z oozgune1 $

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

#include <cisst3DUserInterface/CurveVirtualFixture.h>
#include <cisstVector.h>


#define MAX_VALUE 1000000000000 

//constructor with no argumnet
CurveVirtualFixture::CurveVirtualFixture(void){
    PositionStiffnessPositive.X() = -300.0;
    PositionStiffnessPositive.Y() = -300.0;
    PositionStiffnessPositive.Z() = -50;
    PositionStiffnessNegative.X() = -300.0;
    PositionStiffnessNegative.Y() = -300.0;
    PositionStiffnessNegative.Z() = -50.0;
    PositionDampingPositive.SetAll(0.0);
    PositionDampingNegative.SetAll(0.0);
    ForceBiasPositive.SetAll(0.0);
    ForceBiasNegative.SetAll(0.0);
    OrientationStiffnessPositive.SetAll(-0.05);
    OrientationStiffnessNegative.SetAll(-0.05);
    OrientationDampingPositive.SetAll(-0.001);
    OrientationDampingNegative.SetAll(-0.001);
    TorqueBiasPositive.SetAll(0.0);
    TorqueBiasNegative.SetAll(0.0);
}

//constructor takes set of curve points as argument and sets them.
CurveVirtualFixture::CurveVirtualFixture(std::vector<vct3> myPoints){
    setPoints(myPoints);

    PositionStiffnessPositive.X() = -300.0;
    PositionStiffnessPositive.Y() = -300.0;
    PositionStiffnessPositive.Z() = -50;
    PositionStiffnessNegative.X() = -300.0;
    PositionStiffnessNegative.Y() = -300.0;
    PositionStiffnessNegative.Z() = -50.0;
    PositionDampingPositive.SetAll(0.0);
    PositionDampingNegative.SetAll(0.0);
    ForceBiasPositive.SetAll(0.0);
    ForceBiasNegative.SetAll(0.0);
    OrientationStiffnessPositive.SetAll(-0.05);
    OrientationStiffnessNegative.SetAll(-0.05);
    OrientationDampingPositive.SetAll(-0.001);
    OrientationDampingNegative.SetAll(-0.001);
    TorqueBiasPositive.SetAll(0.0);
    TorqueBiasNegative.SetAll(0.0);
}

// Finds two orthogonal vectors to a given vector
void CurveVirtualFixture::findOrthogonal(vct3 in, vct3 &out1, vct3 &out2){
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

// Finds the closest point from current position to the curve 
void CurveVirtualFixture::findClosestPointToLine(vct3 &pos, vct3 &start, vct3 &end){
    vct3 v; //<! Line between start and end point of the line.
    vct3 w; //<! ine between current point and start point of the line.
    vct3 closestPoint; //<! Closest point on the curve.
    vct3 point; //<! Temporary point to hold closest point.
    int startIndex = 0; //<! start index of the line segment
    int endIndex = 0; //<! end index of the line segment
    double length1; //<! length of v+w
    double length2; //<! length of v (start point to end point length)
    double minDist = MAX_VALUE; 
    closestPoint.SetAll(0.0);
    //linear search to find the closest point and line segment indexes.
    for(unsigned int i=0 ; i<LinePoints.size()-1; i++){
        v = LinePoints.at(i+1)-LinePoints.at(i);
        w = pos-LinePoints.at(i);
        length1 = vctDotProduct(v,w);
        length2 = vctDotProduct(v,v);
        if(length2 == 0){
            CMN_LOG_INIT_ERROR <<"CurveVirtualFixture class: findClosestPointToLine: Error, same point."<< std::endl;
        }
        if(length1<=0){ //before start point
            point = LinePoints.at(i);
        }else if(length2<=length1){ //after end point
            point = LinePoints.at(i+1);
        }else{ //on the line
            point = LinePoints.at(i)+(length1/length2)*v;
        }
        //check minimum distance.
        if(minDist > (point-pos).Norm()){
            closestPoint = point;
            minDist = (point-pos).Norm();
            startIndex = i;
            endIndex = i+1;
        }
    }
    setClosestPoint(closestPoint);
    start = LinePoints.at(startIndex);
    end = LinePoints.at(endIndex);
}

// Updates curve virtual fixture parameters
void CurveVirtualFixture::update(const vctFrm3 & pos , prmFixtureGainCartesianSet & vfParams) {
    vct3 position; //<! Force position
    vctMatRot3 rotation; //<! Force orientation
    vctMatRot3 currRot; //current rotation
    vct3 norm_dir; //direction of the normal
    vct3 currentPosition; // current MTM position
    //vct3 closestPoint; //closest point on the first line to the current point
    vct3 norm_vector;    //norm vector 
    vct3 ortho1(0.0);     //orthogonal vector to the norm vector
    vct3 ortho2(0.0);     //orthogonal vector to the norm vector
    vct3 stiffnessPos, stiffnessNeg;  //positive and negative position stiffness
    vct3 orientStiffPos,orientStiffNeg; //positive and negative orientation stiffness
    vct3 orientDampPos, orientDampNeg; //positive and negative orientation damping
    vct3 lineSegmentStart, lineSegmentEnd;

    double distance;      //distance between the current position and the closest point on the line

    stiffnessPos.SetAll(0.0);
    stiffnessNeg.SetAll(0.0);
    orientStiffPos.SetAll(0.0);
    orientStiffNeg.SetAll(0.0);
    orientDampPos.SetAll(0.0);
    orientDampNeg.SetAll(0.0);

    //get curent position
    currentPosition = pos.Translation();
    //calculate closest point to the current point on the first line
    //sets start,end and closest points
    findClosestPointToLine(currentPosition,lineSegmentStart,lineSegmentEnd);

    if(getClosestPoint()== lineSegmentStart){ //out of the line segment
        stiffnessNeg.Z() = PositionStiffnessNegative.Z();
    }else if(getClosestPoint()== lineSegmentEnd){ //out of the line segment
        stiffnessPos.Z() = PositionStiffnessPositive.Z();
    }else{ //on the line segment
        stiffnessPos.Z() = 0.0;
        stiffnessNeg.Z() = 0.0;
    }

    //force position is the closest point on the curve
    position = getClosestPoint();
    //normal vector direction which is parallel to the line
    norm_dir = (lineSegmentStart-lineSegmentEnd);
    //lebgth of the line segment
    distance = norm_dir.Norm();
    //scale pos_error to calculate norm vector
    norm_vector = norm_dir.Divide(distance);
    //find 2 orthogonal vectors to the norm vector
    findOrthogonal(norm_vector,ortho1,ortho2);
    //form rotation using orthogonal vectors
    rotation.Column(0).Assign(ortho2);
    rotation.Column(1).Assign(-ortho1);
    rotation.Column(2).Assign(norm_vector);
    //set force position
    vfParams.SetForcePosition(position);
    //set force orientation
    vfParams.SetForceOrientation(rotation);
    //set torque orientation
    vfParams.SetTorqueOrientation(rotation);
    //since we assign z-axis parallel to the line, z axis will be free
    //there will be force on x and y axes
    //we set positive and negative farces equal to lock the handle on x and y axes
    stiffnessPos.X() = PositionStiffnessPositive.X();
    stiffnessPos.Y() = PositionStiffnessPositive.Y();
    stiffnessNeg.X() = PositionStiffnessNegative.X();
    stiffnessNeg.Y() = PositionStiffnessNegative.Y();
    //Set position forces
    vfParams.SetPositionStiffnessPos(stiffnessPos);
    vfParams.SetPositionStiffnessNeg(stiffnessNeg);

    // Set orientation torques
    vfParams.SetOrientationStiffnessPos(OrientationStiffnessPositive);
    vfParams.SetOrientationStiffnessNeg(OrientationStiffnessNegative);

    // Set orientation damping
    vfParams.SetOrientationDampingPos(OrientationDampingPositive);
    vfParams.SetOrientationDampingNeg(OrientationDampingNegative);

    //Temporary hard code solution ask Anton for better way
    vfParams.SetPositionDampingPos(PositionDampingPositive);
    vfParams.SetPositionDampingNeg(PositionDampingNegative);
    vfParams.SetForceBiasPos(ForceBiasPositive);
    vfParams.SetForceBiasNeg(ForceBiasNegative);
    vfParams.SetTorqueBiasPos(TorqueBiasPositive);
    vfParams.SetTorqueBiasNeg(TorqueBiasPositive);
}

// Sets the curve points.
void CurveVirtualFixture::setPoints(const std::vector<vct3> &points){
    LinePoints.resize(0); //clear all
    LinePoints = points; //set points
}

// Sets the closest point on the curve.
void CurveVirtualFixture::setClosestPoint(const vct3 & cp){
    closestPoint = cp;
}

// Returns the closest point.
vct3 CurveVirtualFixture::getClosestPoint(void){
    return closestPoint;
}

// Sets the given positive position stiffness constant.
void CurveVirtualFixture::setPositionStiffnessPositive(const vct3 stiffPos){
    this->PositionStiffnessPositive = stiffPos;
}

// Sets the given negative position stiffness constant.
void CurveVirtualFixture::setPositionStiffnessNegative(const vct3 stiffNeg){
    this->PositionStiffnessNegative = stiffNeg;
}

// Sets the given positive position damping constant.
void CurveVirtualFixture::setPositionDampingPositive(const vct3 dampPos){
    this->PositionDampingPositive = dampPos;
}

// Sets the given negative position damping constant.
void CurveVirtualFixture::setPositionDampingNegative(const vct3 dampNeg){
    this->PositionStiffnessNegative = dampNeg;
}

// Sets the given positive force bias constant.
void CurveVirtualFixture::setForceBiasPositive(const vct3 biasPos){
    this->ForceBiasPositive = biasPos;
}

// Sets the given negative force bias constant.
void CurveVirtualFixture::setForceBiasNegative(const vct3 biasNeg){
    this->ForceBiasNegative = biasNeg;
}

// Sets the given positive orientation stiffness constant.
void CurveVirtualFixture::setOrientationStiffnessPositive(const vct3 orientStiffPos){
    this->OrientationStiffnessPositive = orientStiffPos;
}

// Sets the given negative orientation stiffness constant.
void CurveVirtualFixture::setOrientationStiffnessNegative(const vct3 orientStiffNeg){
    this->OrientationStiffnessNegative = orientStiffNeg;
}

// Sets the given positive orientation damping constant.
void CurveVirtualFixture::setOrientationDampingPositive(const vct3 orientDampPos){
    this->OrientationDampingPositive = orientDampPos;
}

// Sets the given negative orientation damping constant.
void CurveVirtualFixture::setOrientationDampingNegative(const vct3 orientDampNeg){
    this->OrientationDampingNegative = orientDampNeg;
}

// Sets the given positive torque bias constant.
void CurveVirtualFixture::setTorqueBiasPositive(const vct3 torqueBiasPos){
    this->TorqueBiasPositive = torqueBiasPos;
}

// Sets the given negative torque bias constant.
void CurveVirtualFixture::setTorqueBiasNegative(const vct3 torqueBiasNeg){
    this->TorqueBiasNegative = torqueBiasNeg;
}

// Returns the positive position stiffness constant.
vct3 CurveVirtualFixture::getPositionStiffnessPositive(void){
    return this->PositionStiffnessPositive;
}

// Returns the negative position stiffness constant.
vct3 CurveVirtualFixture::getPositionStiffnessNegative(void){
    return this->PositionStiffnessNegative;
}

// Returns the positive position damping constant.
vct3 CurveVirtualFixture::getPositionDampingPositive(void){
    return this->PositionDampingPositive;
}

// Returns the negative position damping constant.
vct3 CurveVirtualFixture::getPositionDampingNegative(void){
    return this->PositionStiffnessNegative;
}

// Returns the positive force bias constant.
vct3 CurveVirtualFixture::getForceBiasPositive(void){
    return this->ForceBiasPositive;
}

// Returns the negative force bias constant.
vct3 CurveVirtualFixture::getForceBiasNegative(void){
    return this->ForceBiasNegative;
}

// Returns the positive orientation stiffness constant.
vct3 CurveVirtualFixture::getOrientationStiffnessPositive(void){
    return this->OrientationStiffnessPositive;
}

// Returns the negative orientation stiffness constant.
vct3 CurveVirtualFixture::getOrientationStiffnessNegative(void){
    return this->OrientationStiffnessNegative;
}

// Returns the positive orientation damping constant.
vct3 CurveVirtualFixture::getOrientationDampingPositive(void){
    return this->OrientationDampingPositive;
}

// Returns the negative orientation damping constant.
vct3 CurveVirtualFixture::getOrientationDampingNegative(void){
    return this->OrientationDampingNegative;
}

// Returns the positive torque bias constant.
vct3 CurveVirtualFixture::getTorqueBiasPositive(void){
    return this->TorqueBiasPositive;
}

// Returns the negative torque bias constant.
vct3 CurveVirtualFixture::getTorqueBiasNegative(void){
    return this->TorqueBiasNegative;
}
