#include "VirtualFixture.h"
#include <cisstVector.h>

#ifndef stayonplanevirtualfixture_h
#define stayonplanevirtualfixture_h

/*!@brief StayOnPlane Virtual Fixture.
*
*  This class extends the base virtual fixture class and overloads the
*  two methods in the base class. The stay on a plane virtual fixture keeps the handle 
*  on a desired plane. Guidance region is desired plane based on
*  the user need.
*/

class StayOnPlaneVirtualFixture: public VirtualFixture {
public:

    /*! @brief Constructor takes no argument
    */
    StayOnPlaneVirtualFixture(void);

    /*! @brief Constructor takes vf base point position and plane normal
    * and sets them
    */
    StayOnPlaneVirtualFixture(const vct3 basePoint, const vct3 planeNormal);
    
    /*! @brief Update plane virtual fixture parameters.
    *
    *   This method takes the current MTM position and orientation matrix and 
    *   the virtual fixture parameter reference, it calculates and assigns virtual fixture
    *   parameters. For the plane virtual fixture, we calculate the closest point to the plane
    *   and set the point as the force position. Usind the closest point, we calulate the norm
    *   vector and allign the norm as z-axis and form the force orientation matrix. 
    *   Since we allign the norm as z-axis, we only apply positive force on the z-axis. 
    *   We check the distance from the current position to the plane. If it is negative
    *   we apply force.
    *
    *   @param mtmPos current manipulator position and orientation.
    *   @param vfparams virtual fixture parameters.
    */
    void update(const vctFrm3 & mtmPos , prmFixtureGainCartesianSet & vfParams);
    
    /*!@brief Get the necessary user data.
    *
    *  This function asks three different points from the user.
    *  We check points are on the same line. If they are on the same line, we ask user to
    *  type new point sets. Using these three different points, we calculate the base point and
    *  the normal vector and set them.
    */  
    void getUserData(void);
    
    /*!@brief Helper method to find two orthogonal vectors.
    *
    *  This method takes 3 vectors. First one is the given vector, other two stores
    *  the orthogonal vectors. We use one arbitrary vector(Y- axis in this case).
    *  We first cross product the given vector and the arbitrary vector to find the 
    *  first orthogonal vector to the given vector. Then we cross product the given vector
    *  and the first orthogonal vector to find the second orthogonal vector.
    *
    *  @param in given vector.
    *  @param out1 the first orthogonal vector to the given vector.
    *  @param out2 the second orthogonal vector to the given vector.
    */
    void findOrthogonal(vct3 in, vct3 &out1, vct3 &out2);
    
    /*!@brief Find the closest point to the plane.
    *
    *  @param p current position.
    *  @return closest point.
    */
    vct3 closestPoint(vct3 p);
    
    /*!@brief Find the closest distance the plane.
    *
    *  @param p current position.
    *  @return closest distance to the plane.
    */
    double shortestDistance(vct3 p);
    
    /*@brief This is a helper method for input/output.
    *
    * @param position
    */
    void getPositionFromUser(vct3 & position);

	/*!@brief Set base point vector.
	*
	*  @param c base point
	*/
	void setBasePoint(const vct3 & c);

	/*!@brief Set plane unit normal vector.
	*
	*  @param n unit normal vector.
	*/
	void setPlaneNormal(const vct3 & n);

	/*!@brief Return base point vector.
	*
	*  @return basePoint.
	*/
	vct3 getBasePoint(void);

	/*!@brief Return plane unit normal vector.
	*
	*  @return normVector.
	*/
	vct3 getPlaneNormal(void);

protected:
	vct3 basePoint; //!< Base point position vector.
	vct3 planeNormal; //!< Plane unit normal vector.

};

#endif