/*

  Author(s): Simon Leonard
  Created on: Nov 11 2009

  (C) Copyright 2008 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstCommon/cmnLogger.h>
#include <cisstRobot/robLinear.h>
#include <cisstRobot/robTrajectory.h>
#include <iostream>
#include <typeinfo>

robLinear::robLinear( robSpace::Basis codomain,
		      double t1, double y1, double t2, double y2 ) :
  // initialize the base class R^1->R^n
  robFunction( robSpace::TIME, 
	       codomain & ( robSpace::JOINTS_POS | robSpace::TRANSLATION ) ) {

  // Check that the time values are greater than zero and that t1 < t2
  if( (t1 < 0) || (t2 < 0) || (t2 < t1) ){
    CMN_LOG_RUN_ERROR << CMN_LOG_DETAILS 
		      << ": " << t1 << " must be less than " << t2 << "." 
		      << std::endl;
  }
  
  this->m = (y2-y1)/(t2-t1); // compute the slope
  this->b = y1 - m*t1;       // compute the 0-offset

  this->tmin = t1;           // copy the domain
  this->tmax = t2;

}

robFunction::Context robLinear::GetContext( const robVariable& input ) const{
 
  // Test the input is time
  if( !input.IsTimeEnabled() ){
    CMN_LOG_RUN_ERROR << CMN_LOG_DETAILS 
		      << ": Expected time input." 
		      << std::endl;
    return robFunction::CUNDEFINED;
  }
  
  // Check the context
  double t = input.time;
  if( this->tmin <= t && t <= this->tmax ) { return robFunction::CDEFINED; }
  else                                     { return robFunction::CUNDEFINED; }
}

robFunction::Errno robLinear::Evaluate( const robVariable& input, 
					robVariable& output ){

  // Test the context
  if( GetContext( input ) != robFunction::CDEFINED ){
    CMN_LOG_RUN_ERROR << CMN_LOG_DETAILS 
		      << ": Function is undefined for the input." 
		      << std::endl;
    return robFunction::EUNDEFINED;
  }

  double t = input.time;
  double y = m*t+b;
  double yd = m;
  double ydd = 0.0;
  
  output.IncludeBasis( Codomain().GetBasis(), y, yd, ydd );

  return robFunction::ESUCCESS;
}
