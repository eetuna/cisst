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
#include <iostream>
#include <conio.h>
#include <cstdio>
#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include<windows.h>

#define MAX_VALUE 1000000000000 

CurveVirtualFixture::CurveVirtualFixture(void){}

CurveVirtualFixture::CurveVirtualFixture(std::vector<vct3> myPoints)
{
    setPoints(myPoints);
}

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
        //std::cout<<"Something is not good..."<<std::endl;
    }
    //TODO find a better way to handle
    if(vec1.X()==0.0 && vec1.Y()==0.0 && vec1.Z()== 0.0){
        vec1.CrossProductOf(in,axisZ);
        //std::cout<<"Vec1:(in if) "<<vec1<<std::endl;
    }
    // Now find 2nd orthogonal vector
    vec2.CrossProductOf(in,vec1); 
    len2 = vec1.Norm(); 
    len3 = vec2.Norm(); 

    out1 = vec1.Divide(len2);
    out2 = vec2.Divide(len3);

}
void CurveVirtualFixture::findClosestPointToLine(vct3 &pos, vct3 &start, vct3 &end){
    vct3 v; //<! line between start and end point of the line
    vct3 w; //<! line between current point and start point of the line
    vct3 closestPoint; //<! closest point on the line
    vct3 point;
    int startIndex = 0; //<! line segment start index
    int endIndex = 0; //<! line segment end index
    double length1; //length of v+w
    double length2; //length of v
    double minDist = MAX_VALUE;
    closestPoint.SetAll(0.0);

    for(int i=0 ; i<LinePoints.size()-1; i++){
        v = LinePoints.at(i+1)-LinePoints.at(i);
        w = pos-LinePoints.at(i);
        length1 = vctDotProduct(v,w);
        length2 = vctDotProduct(v,v);

        if(length2 == 0){
            std::cout<<"Something is wrong "<<std::endl;
        }
        if(length1<=0){ //before start point
            point = LinePoints.at(i);
        }else if(length2<=length1){ //after end point
            point = LinePoints.at(i+1);
        }else{ //on the line
            point = LinePoints.at(i)+(length1/length2)*v;
        }
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

void CurveVirtualFixture::update(const vctFrm3 & pos , prmFixtureGainCartesianSet & vfParams) {
    vct3 position;  //<! final force position
    vctMatRot3 rotation; //<! final force orientation
    vctMatRot3 currRot; //<! current rotation
    vct3 norm_dir; //<! direction of the normal
    vct3 currentPosition; //<! current MTM position
    //vct3 closestPoint; //closest point on the first line to the current point
    vct3 norm_vector; //<! norm vector 
    vct3 ortho1(0.0); //<! orthogonal vector to the norm vector
    vct3 ortho2(0.0); //<! orthogonal vector to the norm vector
    vct3 stiffnessPos, stiffnessNeg;  //<! positive and negative position stiffness
    vct3 orientStiffPos,orientStiffNeg; //<! positive and negative orientation stiffness
    vct3 orientDampPos, orientDampNeg; //<! positive and negative orientation damping
    vct3 lineSegmentStart, lineSegmentEnd;
    vctMatRot3 robotHomeOrientation;
    double distance; //<! distance between the current position and the closest point on the line

    //get curent position
    currentPosition = pos.Translation();
    currRot = pos.Rotation();
    //calculate closest point to the current point on the first line
    //sets start,end and closest points
    findClosestPointToLine(currentPosition,lineSegmentStart,lineSegmentEnd);

    if(getClosestPoint()== lineSegmentStart){
        stiffnessNeg.Z() = -50;
    }else if(getClosestPoint()== lineSegmentEnd){
        stiffnessPos.Z() = -50;
    }else{
        stiffnessPos.Z() = 0.0;
        stiffnessNeg.Z() = 0.0;
    }

    position = getClosestPoint();
    norm_dir = (lineSegmentStart-lineSegmentEnd);
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

    //set torque orientation as robot home orientation
    robotHomeOrientation.Column(0).Assign(0.997660, -0.0611869, 0.0303909);
    robotHomeOrientation.Column(1).Assign(0.00284961, 0.48027, 0.877079);
    robotHomeOrientation.Column(2).Assign(-0.0695227, -0.874843, 0.479249);
    //vfParams.setR_o(robotHomeOrientation);

    //since we assign z-axis parallel to the line, z axis will be free
    //there will be force on x and y axes
    //we set positive and negative farces equal to lock the handle on x and y axes
    stiffnessPos.X() = -300.0;
    stiffnessPos.Y() = -300.0;
    stiffnessNeg.X() = -300.0;
    stiffnessNeg.Y() = -300.0;

    //Set position forces
    vfParams.SetPositionStiffnessPos(stiffnessPos);
    vfParams.SetPositionStiffnessNeg(stiffnessNeg);

    //Positive orientation gain
    orientStiffPos.SetAll(-0.09);
    orientStiffNeg.SetAll(-0.09);
    vfParams.SetOrientationStiffnessPos(orientStiffPos);
    vfParams.SetOrientationStiffnessNeg(orientStiffNeg);

    //Positive orientation damping
    orientDampPos.SetAll(-0.001);
    //Neagtive orientation damping
    orientDampNeg.SetAll(-0.001);
    // Set orientation damping
    vfParams.SetOrientationDampingPos(orientDampPos);
    vfParams.SetOrientationDampingNeg(orientDampNeg);

    //Temporary hard code solution ask Anton for better way
    vct3 temp;
    temp.SetAll(0.0);
    vfParams.SetPositionDampingPos(temp);
    vfParams.SetPositionDampingNeg(temp);
    vfParams.SetForceBiasPos(temp);
    vfParams.SetForceBiasNeg(temp);
    vfParams.SetTorqueBiasPos(temp);
    vfParams.SetTorqueBiasNeg(temp);
}

void CurveVirtualFixture::setPoints(const std::vector<vct3> &points){
    LinePoints.resize(0); //clear all
    LinePoints = points; //set points
}

void CurveVirtualFixture::setClosestPoint(const vct3 & cp){
    closestPoint = cp;
}

vct3 CurveVirtualFixture::getClosestPoint(void){
    return closestPoint;
}
