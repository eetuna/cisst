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

	//vfParams.Reset(); //reset all virtual fixture paramter values
    vct3 temp;
    temp.SetAll(0.0);

	//we set rotation to the initial condition
	//we do not need force orientation
	//rotation = pos.Rotation();
    rotation = vctMatRot3::Identity();
	//force position is the given position
	position = getPoint();

    //set force position
	vfParams.SetForcePosition(position);

	//set force orientation
	vfParams.SetForceOrientation(rotation);

	//set torque orientation
	vfParams.SetTorqueOrientation(rotation);

	

	//set Position Stiffness

    //Hard on Y and Z but soft on X 
	//stiffnessPos.SetAll(-400.0);
    stiffnessPos.X() = -150.0;
    stiffnessPos.Y() = -150.0;
    stiffnessPos.Z() = -150.0;
    
	//stiffnessNeg.SetAll(-400.0);
    //Hard on Y and Z but soft on X 
    stiffnessNeg.X() = -150.0;
    stiffnessNeg.Y() = -150.0;
    stiffnessNeg.Z() = -150.0;
    
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

void PointVirtualFixture::getUserData(void) {
	vct3 p; //point position

	//asks user to type center
	std::cout<<"Input Point x y z location:  "<<std::endl;
	std::cin >> p.X() >> p.Y() >> p.Z();

	//set Center position
	setPoint(p);
	
}

void PointVirtualFixture::setPoint(const vct3 & p){
	point = p;
}


vct3 PointVirtualFixture::getPoint(void){
	return point;
}



