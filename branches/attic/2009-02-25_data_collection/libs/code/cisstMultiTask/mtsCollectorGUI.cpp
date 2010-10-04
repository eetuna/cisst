/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCollectorGUI.cpp 188 2009-03-20 17:07:32Z mjung5 $

  Author(s):  Min Yang Jung
  Created on: 2009-07-24

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

#define AQUA    0, 1, 1
#define BLACK   0, 0, 0
#define BLUE    0, 0, 1
#define FUCHSIA 1, 0, 1
#define GRAY    0.5, 0.5, 0.5
#define GREEN   0, 0.5, 0
#define LIME    0, 1, 0
#define MAROON  0.5, 0, 0
#define NAVY    0, 0, 0.5
#define OLIVE   0.5, 0.5, 0
#define PURPLE  0.5, 0, 0.5
#define RED     1,  0, 0
#define SILVER  0.75, 0.75, 0.75
#define TEAL    0, 0.5, 0.5
#define WHITE   1, 1, 1
#define YELLOW  1, 1, 0

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
    GraphPane->set_scrolling(75);    
    GraphPane->set_grid(MP_LINEAR_GRID, MP_LINEAR_GRID, true);
}

void mtsCollectorGUI::UpdateUI(const double newValue)
{
#ifdef _UI_TEST_CODE_
    static unsigned int x = 0;
    
    for (int i=0; i<=12; ++i) {
        if (i <= 7) {
            GraphPane->set_pointsize(i, 3.0);
            GraphPane->set_linewidth(i, 0.0);
        } else {
            GraphPane->set_pointsize(i, 3.0);
            GraphPane->set_linewidth(i, 1.0);
        }
    }

    GraphPane->add(0, PLOT_POINT(x, newValue * 1.0, AQUA));
    GraphPane->add(1, PLOT_POINT(x, newValue * 1.1, BLUE));
    GraphPane->add(2, PLOT_POINT(x, newValue * 1.2, FUCHSIA));
    GraphPane->add(3, PLOT_POINT(x, newValue * 1.3, GREEN));
    GraphPane->add(4, PLOT_POINT(x, newValue * 1.4, LIME));
    GraphPane->add(5, PLOT_POINT(x, newValue * 1.5, MAROON));
    GraphPane->add(6, PLOT_POINT(x, newValue * 1.6, NAVY));
    GraphPane->add(7, PLOT_POINT(x, newValue * 1.7, OLIVE));
    GraphPane->add(8, PLOT_POINT(x, newValue * 1.8, PURPLE));
    GraphPane->add(9, PLOT_POINT(x, newValue * 1.9, RED));
    GraphPane->add(10, PLOT_POINT(x, newValue * 2.0, TEAL));
    GraphPane->add(11, PLOT_POINT(x, newValue * 2.1, WHITE));
    GraphPane->add(12, PLOT_POINT(x, newValue * 2.2, YELLOW));

    //GraphPane->add(0, PLOT_POINT(x, 0.1*x*sin(x/6.0),1,1,0));
    //GraphPane->add(1, PLOT_POINT(x, 0.1*x*sin(x/3.0),1,0,0));

    //if((x % 200)==0)mygl->clear();	// clear the traces of plot-window 3

    ++x;
    // force the redrawing of the windows. 
    // this could be done less often, for example every tenth timestep,  to speed up calculations.
    //m1.redraw();
    //m2.redraw();
    GraphPane->redraw();
#endif
}