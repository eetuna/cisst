#include <cisst3DUserInterface/PlaneVirtualFixture.h>
#include <cisstVector.h>
#include <iostream>
#include <conio.h>
#include <cstdio>
#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include<windows.h>


PlaneVirtualFixture::PlaneVirtualFixture(void){}

PlaneVirtualFixture::PlaneVirtualFixture(vct3 basePoint, vct3 planeNormal)
{
	setBasePoint(basePoint);
    setPlaneNormal(planeNormal/planeNormal.Norm());
}

void PlaneVirtualFixture::findOrthogonal(vct3 in, vct3 &out1, vct3 &out2){
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


vct3 PlaneVirtualFixture::closestPoint(vct3 p){
	 return p-getPlaneNormal()*(vctDotProduct((p-getBasePoint()),getPlaneNormal()));
}

double PlaneVirtualFixture::shortestDistance(vct3 p){
	return abs(vctDotProduct((p-getBasePoint()),getPlaneNormal()));
}
void PlaneVirtualFixture::update(const vctFrm3 & pos , prmFixtureGainCartesianSet & vfParams) {
	vct3 position;  //final force position
	vctMatRot3 rotation;  //final force orientation
	vct3 pos_error; //position error between current and the closest point position
	vct3 currentPosition; // current MTM position
	vct3 closest;    //closest point from MTM current position to the plane
	vct3 norm_vector;    //normal vector 
	vct3 ortho1(0.0);     //orthogonal vector to the norm vector
	vct3 ortho2(0.0);     //orthogonal vector to the norm vector
	vct3 stiffnessPos;    //position stiffness constant (positive)
	vct3 stiffnessNeg;	  //position stiffness constant (negative)
	double distance;      //distance between current position and the closest point position

	//vfParams.Reset();
    vct3 temp;
    temp.SetAll(0.0);

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
	stiffnessPos.SetAll(0.0);
	//stiffnessPos.Z() = -500.0;

	//in order to protect below the plane comment out the following line
	stiffnessNeg.SetAll(0.0);
	stiffnessNeg.Z() = -500.0;

	//set Position Stiffness
	vfParams.SetPositionStiffnessPos(stiffnessPos);
	vfParams.SetPositionStiffnessNeg(stiffnessNeg);

    //Temporary hard code solution ask Anton for better way
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

void PlaneVirtualFixture::getUserData(void) {
	//three different points to form the plane
	vct3 point1, point2,point3;
	//lines to find the plane normal vector
	vct3 line12, line23;
	vct3 base;     //base point
	vct3 unitVec;   //unit vector
	bool pointCheck = false; //flag to control the given points

	do{
	

	//set base point
	base = (point1+point2+point3)/3.0;
	
	line12 = point1-point2;
	line23 = point3-point2;

	unitVec.CrossProductOf(line12,line23);

	//if given points are on the same line,
	//then crossproduct of the created lines from given points is zero
	//TODO look for exceptions
	if(unitVec.X() == 0.0 && unitVec.Y() == 0.0 && unitVec.Z() == 0.0){
		std::cerr<<"Given points are on the same line!!!\n"<<std::endl;
		std::cout<<"Please enter new sets of points!!!\n"<<std::endl;
		pointCheck = false;
	}else{
		pointCheck = true;
	}

	}while(pointCheck == false);
	
	//since points are fine we can set base point and 
	//norm vector
	setBasePoint(base);
	//we ste the plane normal vector
	setPlaneNormal(unitVec/(unitVec.Norm()));

}

void PlaneVirtualFixture::setBasePoint(const vct3 & b){
	basePoint = b;
}
void PlaneVirtualFixture::setPlaneNormal(const vct3 & n){
	normVector = n;
}

vct3 PlaneVirtualFixture::getBasePoint(void){
	return basePoint;
}

vct3 PlaneVirtualFixture::getPlaneNormal(void){
	return normVector;
}
