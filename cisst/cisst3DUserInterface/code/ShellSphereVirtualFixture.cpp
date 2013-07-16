/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id: ShellSphereVirtualFixture.cpp 3148 2013-06-26 15:46:31Z oozgune1 $

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

#include <cisst3DUserInterface/ShellSphereVirtualFixture.h>
#include <cisstVector.h>

// ShellSphereVirtualFixture contructor that takes no argument.
ShellSphereVirtualFixture::ShellSphereVirtualFixture(void){
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

/* ShellPlaneVirtualFixture contructor that takes the center
of the sphere and the radius and sets them.
*/
ShellSphereVirtualFixture::ShellSphereVirtualFixture(const vct3 center, const double radius){
    setCenter(center); //set center position
    setRadius(radius); //set radius of the sphere
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
void ShellSphereVirtualFixture::findOrthogonal(vct3 in, vct3 &out1, vct3 &out2){
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

// Updates the shell virtual fixture parameters.
void ShellSphereVirtualFixture::update(const vctFrm3 & pos , prmFixtureGainCartesianSet & vfParams) {
    vct3 position; //<! Force position
    vctMatRot3 rotation; //<! Force orientation
    vct3 pos_error; //<! Position error between current and center position
    vct3 currentPosition; //<! Current MTM position
    vct3 norm_vector; //<! Normal vector to create froce/torque orientation matrix 
    vct3 scaled_norm_vector; //<! Norm vector scaled with -radius
    vct3 ortho1(0.0); //<! Orthogonal vector to the normal vector to form force/torque orientation matrix
    vct3 ortho2(0.0); //<! Orthogonal vector to the normal vector to form force/torque orientation matrix
    double distance; //<! Distance between current position and center position
    vct3 stiffnessNeg; //<! Negative position stiffness constant
    stiffnessNeg.SetAll(0.0);

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
    //form rotation using orthogonal vectors and norm vector
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

    //set Negative Position Stiffness
    stiffnessNeg.Z() = PositionStiffnessNegative.Z();
    vfParams.SetPositionStiffnessNeg(stiffnessNeg);

    //Temporary hard code solution ask Anton for better way
    vfParams.SetPositionStiffnessPos(PositionStiffnessPositive);
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

// Sets the center position of the sphere.
void ShellSphereVirtualFixture::setCenter(const vct3 & c){
    center = c;
}

// Sets the radius of the sphere.
void ShellSphereVirtualFixture::setRadius(const double & r){
    radius = r;
}

// Returns the radius of the sphere.
double ShellSphereVirtualFixture::getRadius(void){
    return radius;
}

// Returns the center position of the sphere.
vct3 ShellSphereVirtualFixture::getCenter(void){
    return center;
}