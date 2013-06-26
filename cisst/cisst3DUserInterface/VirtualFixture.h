#include <cisstVector.h>
//#include "VfParameters.h"
#include <cisstParameterTypes/prmFixtureGainCartesianSet.h>

#ifndef virtualfixture_h
#define virtualfixture_h

/*! @brief Virtual Fixture base class.
*
*   This clas is the base class for the different kinds of virtual fixtures.
*   All different virtual fixture classes have to extend the base class and 
*   override the methods in the base class.
*/
class VirtualFixture{
public:
	/*! @brief Updates the virtual fixture paramters with the given current position.
	*
	*   This function is responsibe for the math. It takes current position and orientation 
	*   of the manipulator, calculates the virtual fixture parameters.
	*
	*   @param pos current position and orientation of the MTM
	*   @param vfParams address of the virtual fixture parameters.
	*/
	virtual void update(const vctFrm3 & pos , prmFixtureGainCartesianSet & vfParams) = 0;

	/*! @brief Gets the user data.
	*
	*   This function gets the user data based on the virtual fixture type and
	*   it sets those values. We add this function to test implemented virtual
	*   fixtures.
	*/
	virtual void getUserData(void) = 0;

};

#endif
