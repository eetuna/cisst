/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCollectorGUI.cpp 188 2009-03-20 17:07:32Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-03-20

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#include <cisstMultiTask/mtsCollectorGUI.h>

#define _UI_TEST_CODE_

CMN_IMPLEMENT_SERVICES(mtsCollectorGUI)

//-------------------------------------------------------
//	Constructor, Destructor, and Initializer
//-------------------------------------------------------
mtsCollectorGUI::mtsCollectorGUI(MULTIPLOT * graphPane)
    : GraphPane(graphPane)
{
    Initialize();
}

void mtsCollectorGUI::Initialize()
{
    GraphPane->set_scrolling(50);
    GraphPane->set_grid(MP_LINEAR_GRID, MP_LINEAR_GRID, true);
}

void mtsCollectorGUI::UpdateUI(const double newValue)
{
#ifdef _UI_TEST_CODE_
    static unsigned int x = 0;

    GraphPane->add(0, PLOT_POINT(x, newValue));

    //GraphPane->add(0, PLOT_POINT(x, 0.1*x*sin(x/6.0),1,1,0));
    //GraphPane->add(1, PLOT_POINT(x, 0.1*x*sin(x/3.0),1,0,0));

    //if((x % 200)==0)mygl->clear();	// clear the traces of plot-window 3

    x++;
    // force the redrawing of the windows. 
    // this could be done less often, for example every tenth timestep,  to speed up calculations.
    //m1.redraw();
    //m2.redraw();
    GraphPane->redraw();
#endif
}