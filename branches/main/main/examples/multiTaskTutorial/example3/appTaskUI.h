/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */
/* $Id: appTaskUI.h 75 2009-02-24 16:47:20Z adeguet1 $ */

#ifndef _appTaskUI_h
#define _appTaskUI_h

#include <FL/Fl.H>
#include <cisstMultiTask/mtsForwardDeclarations.h>
#include <cisstOSAbstraction/osaMutex.h>

class Fl_Double_Window;
class Fl_Value_Output;
class Fl_Value_Input;
class Fl_Check_Button;
class Fl_Output;

class AppTaskUI {
protected:
    static osaMutex Mutex;
    mtsTask * Task;
    Fl_Double_Window * Window;
    Fl_Value_Output * PositionControlled1;
    Fl_Value_Output * PositionControlled2;
    Fl_Value_Output * PositionObserved1;
    Fl_Value_Output * PositionObserved2;
    Fl_Value_Input * GoalControlled1;
    Fl_Value_Input * GoalControlled2;
    Fl_Value_Output * Ticks;
    Fl_Output * Moving;

    typedef void (*CloseCB)(mtsTask*);
    CloseCB closeCB;

    std::string ControlledName;
    std::string ObservedName;

    // UI callbacks
    static void cb_Close(Fl_Widget * CMN_UNUSED(widget), AppTaskUI * object);
    static void cb_MoveControlled(Fl_Widget * CMN_UNUSED(widget), AppTaskUI * object);
    
public:
    AppTaskUI(const std::string & controlledRobot,
              const std::string & observedRobot,
              mtsTask * task);
    ~AppTaskUI();
    void Show(void);
    void Hide(void);
    bool MoveControlledPressed;
    void GetGoalControlled(double & joint1, double & joint2) const;
    /*! Update the timestamp (t), position1 (p1) and position2 (p2). */
    void Update(unsigned long ticks,
                double positionControlled1, double positionControlled2,
                double positionObserved1, double positionObserved2);
    void SetCloseHandler(CloseCB func) {
        Mutex.Lock();
        closeCB = func;
        Mutex.Unlock();
    }
    void ShowMoving(bool isMoving);
};

#endif // _appTaskUI_h

/*
  Author(s):  Peter Kazanzides, Anton Deguet
  Created on: 2004-04-30

  (C) Copyright 2004-2008 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/
