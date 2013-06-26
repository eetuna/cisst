#include "VirtualFixture.h"
#include <cisstVector.h>

#ifndef pointvirtualfixture_h
#define pointvirtualfixture_h

/*!@brief Point Virtual Fixture Class.
*
*  This class guides the MTM to the given point. Purpose of the type is 
*  to find a specific location when it is necessary.
*/

class PointVirtualFixture: public VirtualFixture {

public:

    /*! @brief Constructor takes no argument
    */
    PointVirtualFixture();

    /*! @brief Constructor takes vf point position and sets the point
    */
    PointVirtualFixture(const vct3 pt);
    
    /*! @brief Update point virtual fixture parameters.
    *
    *   This method takes the current MTM position and orientation matrix and 
    *   virtual fixture parameter reference, it calculates and assigns virtual fixture
    *   parameters. For the point virtual fixture, we set the given point as the force position.
    *   Since we have only a point, we do not need to calculate the force orientation matrix.
    *   We set all x,y,z position stiffness forces as the same.
    *
    *   @param mtmPos current manipulator position and orientation.
    *   @param vfparams virtual fixture parameters.
    */
    void update(const vctFrm3 & mtmPos , prmFixtureGainCartesianSet & vfParams);
    
    /*! @brief Get the necessary user data.
    *
    *   This function asks user a point position.
    *   When the user types the necessary infromation, we store it.
    */
    void getUserData(void);
    
    /*! @brief Set the point position.
	*
	*   @param p given point position.
	*/
	void setPoint(const vct3 & p);

	/*! @brief Return the point position.
	*
	*   @return p point position.
	*/
	vct3 getPoint(void);

protected:
	vct3 point; //!< Point position.

};

#endif