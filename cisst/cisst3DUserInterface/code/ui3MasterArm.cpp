/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Balazs Vagvolgyi, Simon DiMaio, Anton Deguet
  Created on: 2008-05-23

  (C) Copyright 2008-2012 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisst3DUserInterface/ui3MasterArm.h>

#include <cisstMultiTask/mtsTaskManager.h>
#include <cisstMultiTask/mtsInterfaceRequired.h>
#include <cisst3DUserInterface/ui3Manager.h>
#include <cisst3DUserInterface/ui3Selectable.h>
#include <cisst3DUserInterface/StayOnPlaneVirtualFixture.h>
#include <cisst3DUserInterface/PointVirtualFixture.h>

#include <cisstParameterTypes/prmPositionCartesianSet.h>

CMN_IMPLEMENT_SERVICES(ui3MasterArm)


ui3MasterArm::ui3MasterArm(const std::string & name):
    Name(name),
    Cursor(0),
    ButtonPressed(false),
    ButtonReleased(false),
    ScaleFactor(1.0),
    Clutched(false),
    Manager(0),
    Selected(0),
    WasOnMenu(false),
    WasOnButton(false),
    ActiveVF(false)
{
}

ui3MasterArm::~ui3MasterArm()
{
}


bool ui3MasterArm::SetInput(mtsDevice * positionDevice, const std::string & positionInterface,
                            mtsDevice * buttonDevice, const std::string & buttonInterface,
                            mtsDevice * clutchDevice, const std::string & clutchInterface,
                            const RoleType & role)
{
    return this->SetInput(positionDevice->GetName(), positionInterface,
                          buttonDevice->GetName(), buttonInterface,
                          clutchDevice->GetName(), clutchInterface,
                          role);
}


bool ui3MasterArm::SetInput(const std::string & positionDevice, const std::string & positionInterface,
                            const std::string & buttonDevice, const std::string & buttonInterface,
                            const std::string & clutchDevice, const std::string & clutchInterface,
                            const RoleType & role)
{
    if (this->Manager == 0) {
        CMN_LOG_CLASS_INIT_ERROR << "SetInput: can not setup input for master arm \""
                                 << this->Name << "\" before adding it to a ui3Manager"
                                 << std::endl;
        return false;
    }

    this->Role = role;

    // add required interface for master arm to Manager
    mtsInterfaceRequired * requiredInterface;

    // setup master arm required interface
    requiredInterface = this->Manager->AddInterfaceRequired(this->Name);
    if (requiredInterface) {
        // bound the mtsFunction to the command provided by the interface
        requiredInterface->AddFunction("GetPositionCartesian", this->GetCartesianPosition, MTS_REQUIRED);
    } else {
        CMN_LOG_CLASS_INIT_ERROR << "SetInput: failed to add \""
                                 << this->Name
                                 << "\" interface, are you trying to set this arm twice?"
                                 << std::endl;
        return false;
    }
    // connect the master device to the master required interface
    this->Manager->ComponentManager->Connect(this->Manager->GetName(), this->Name,
                                             positionDevice, positionInterface);

    //setup master arm required interface for write commands
    requiredInterface = this->Manager->AddInterfaceRequired(this->Name + "Wrt", MTS_OPTIONAL);
    if (requiredInterface) {
        // bound the mtsFunctions to the command provided by the interface
        // enable/disable and set
        requiredInterface->AddFunction("EnableVirtualFixture", this->FunctionVoidEnableVF, MTS_OPTIONAL);
        requiredInterface->AddFunction("DisableVirtualFixture", this->FunctionVoidDisableVF, MTS_OPTIONAL);
        requiredInterface->AddFunction("SetVirtualFixture", this->FunctionWriteSetVF, MTS_OPTIONAL);
    } else {
        CMN_LOG_CLASS_INIT_ERROR << "SetInput: failed to add \""
                                 << this->Name
                                 << "\" interface, are you trying to set this arm twice?"
                                 << std::endl;
        return false;
    }

    // connect the master device to the master required interface
    this->Manager->ComponentManager->Connect(this->Manager->GetName(), this->Name+"Wrt",
                                             positionDevice, positionInterface+"Wrt");

    // setup master select button required interface
    requiredInterface = this->Manager->AddInterfaceRequired(this->Name + "Select");
    if (requiredInterface) {
        requiredInterface->AddEventHandlerWrite(&ui3MasterArm::ButtonEventHandler, this,
                                                "Button");
    } else {
        CMN_LOG_CLASS_INIT_ERROR << "SetInput: failed to add \""
                                 << this->Name
                                 << "\" interface, are you trying to set this arm twice?"
                                 << std::endl;
        return false;
    }
    // connect the master button device to the master button required interface
    this->Manager->ComponentManager->Connect(this->Manager->GetName(), this->Name + "Select",
                                             buttonDevice, buttonInterface);

    // setup master clutch button required interface
    requiredInterface = this->Manager->AddInterfaceRequired(this->Name + "Clutch");
    if (requiredInterface) {
        requiredInterface->AddEventHandlerWrite(&ui3MasterArm::ClutchEventHandler, this,
                                                "Button");
    } else {
        CMN_LOG_CLASS_INIT_ERROR << "SetInput: failed to add \""
                                 << this->Name
                                 << "\" interface, are you trying to set this arm twice?"
                                 << std::endl;
        return false;
    }
    // connect the master clutch device to the master clutch required interface
    this->Manager->ComponentManager->Connect(this->Manager->GetName(), this->Name + "Clutch",
                                             clutchDevice, clutchInterface);

    return true;
}


bool ui3MasterArm::SetTransformation(const vctFrm3 & transformation,
                                     double scale)
{
    // keep the transformation and scale
    this->Transformation.Assign(transformation);
    this->Scale = scale;
    return true;
}


void ui3MasterArm::SetCursorPosition(const prmPositionCartesianSet & position)
{
    SetCursorPosition(position.Goal().Translation());
}


void ui3MasterArm::SetCursorPosition(const vctDouble3 & desiredCursorPosition)
{
    // get the current arm position
    this->GetCartesianPosition(this->ArmPosition);
    // apply transformation and scale
    vctDouble3 actualCursorPosition;
    this->Transformation.ApplyTo(this->ArmPosition.Position().Translation(), actualCursorPosition);
    actualCursorPosition.Multiply(this->Scale * this->ScaleFactor);
    // compute difference and apply to inverse of transformation
    vctDouble3 differenceInScene;
    differenceInScene.DifferenceOf(desiredCursorPosition, actualCursorPosition);
    differenceInScene.Divide(this->Scale * this->ScaleFactor);
    // create a transformation corresponding to the difference
    vctFrm3 cursorTransformation;
    cursorTransformation.Translation().Assign(differenceInScene);
    // and apply it (compound) to the overall transformation.  May be we should keep these as separate data members
    vctFrm3 newTransformation;
    cursorTransformation.ApplyTo(this->Transformation, newTransformation);
    this->Transformation.Assign(newTransformation);
    // apply transformation and scale
    this->Transformation.ApplyTo(this->ArmPosition.Position(), this->CursorPosition);
    this->CursorPosition.Translation().Multiply(this->Scale * this->ScaleFactor);
}


bool ui3MasterArm::SetCursor(ui3CursorBase * cursor)
{
    this->Cursor = cursor;
    return true;
}


void ui3MasterArm::SetScaleFactor(const mtsDouble & factor)
{
    const vctDouble3 currentPosition = this->CursorPosition.Translation();
    this->ScaleFactor = factor;
    this->SetCursorPosition(currentPosition);
}



void ui3MasterArm::ButtonEventHandler(const prmEventButton & buttonEvent)
{
    if (buttonEvent.Type() == prmEventButton::PRESSED) {
        this->Cursor->SetPressed(true);
        this->ButtonPressed = true;
        this->PressedOverMenu = IsOverMenu;
    } else {
        this->Cursor->SetPressed(false);
        this->ButtonReleased = true;
    }

    if (this->Manager->ActiveBehavior != this->Manager) {
        bool blockEvent = ((buttonEvent.Type() == prmEventButton::RELEASED) && PressedOverMenu)
                           || ((buttonEvent.Type() == prmEventButton::PRESSED) && IsOverMenu);
        if (!blockEvent) {
            this->Manager->DispatchButtonEvent(this->Role, buttonEvent);
        }
    }
}


void ui3MasterArm::ClutchEventHandler(const prmEventButton & buttonEvent)
{
    // position when user clutched out/in
    vctDouble3 clutchedInPosition;
    // placeholder to retrieve position from device
    if (buttonEvent.Type() == prmEventButton::PRESSED) {
        // user is using it's clutch
        this->Clutched = true;
        this->Cursor->SetClutched(true); // render differently
        this->GetCartesianPosition(ArmPosition); // get the current position
        this->Transformation.ApplyTo(ArmPosition.Position().Translation(), this->ClutchedOutPosition);
    } else {
        // end of clutch
        this->Clutched = false;
        this->Cursor->SetClutched(false); // render differently
        this->GetCartesianPosition(ArmPosition); // get the current position
        // compute the updated transformation between device and cursor
        this->Transformation.ApplyTo(ArmPosition.Position().Translation(), clutchedInPosition);
        this->Transformation.Translation().Add(this->ClutchedOutPosition);
        this->Transformation.Translation().Subtract(clutchedInPosition);
    }
}


void ui3MasterArm::PreRun(void)
{
    this->ButtonReleased = false;
    this->ButtonPressed = false;
    this->HighestIntention = 0.0;
    this->ToBeSelected = 0;
}



void ui3MasterArm::UpdateCursorPosition(void)
{
    // if not clutched, update the position from device
    if (!this->Clutched) {
        this->GetCartesianPosition(ArmPosition);
        // apply transformation and scale
        this->Transformation.ApplyTo(ArmPosition.Position(), this->CursorPosition);
        this->CursorPosition.Translation().Multiply(this->Scale * this->ScaleFactor);
    }
    // store position for state table
    this->CartesianPosition.Position().Assign(this->CursorPosition);
    // update cursor position
    this->Cursor->SetTransformation(this->CursorPosition);
}


void ui3MasterArm::Hide(void)
{
    if (this->Cursor) {
        this->Cursor->GetVisibleObject()->Hide();
    }
}


void ui3MasterArm::Show(void)
{
    if (this->Cursor) {
        this->Cursor->GetVisibleObject()->Show();
    }
}


void ui3MasterArm::UpdateIntention(ui3Selectable * selectable)
{
    double intention = selectable->GetIntention(this->CursorPosition);
    if (intention > HighestIntention) {
        HighestIntention = intention;
        ToBeSelected = selectable;
    }
    selectable->UpdateOverallIntention(intention);
}


void ui3MasterArm::SetOnMenu(bool onMenu, ui3MenuBar * menuBar, ui3MenuButton * selectedButton)
{   
    IsOverMenu = onMenu;

    if (!onMenu) {
        if (WasOnMenu || WasOnButton) {
            WasOnMenu = false;
            WasOnButton = false;
        }
    } else {
        if (!WasOnMenu) {
            // transition, we are now on the menu
            if (!ActiveVF) {
                ActiveVF = true;
                FunctionVoidEnableVF();
            }

            WasOnMenu = true;
            //store default menu position
            defaultDepth = ArmPosition.Position();

            if (!selectedButton) {
                //base point and plane normal
                vct3 basePoint, planeNormal;
        
                //Since plane is on the Z plane
                planeNormal.X() = 0.0;
                planeNormal.Y() = 0.0;
                planeNormal.Z() = 1.0;
        
                //Point on the Z plane
                basePoint.X() = 0.0;
                basePoint.Y() = 0.0;
                basePoint.Z() = defaultDepth.Translation().Z();

                PlaneVF.setPlaneNormal(planeNormal);
                PlaneVF.setBasePoint(basePoint);

                //vf = new StayOnPlaneVirtualFixture(basePoint,planeNormal);
                
                PlaneVF.update(ArmPosition.Position(), vfParams);
            }
        }

        if (selectedButton) {
            this->SetScaleFactor(0.2);
            if (this->ButtonReleased && this->PressedOverMenu) {
                // todo, add error code check
                selectedButton->Callable->Execute();
            }
            if (!WasOnButton) {
                vct3 buttonPositionUI;
                buttonPositionUI.Assign(menuBar->GetButtonCenter(*selectedButton));
                // this->Transformation.ApplyInverseTo(buttonPositionUI, buttonPositionArm);
                this->SetCursorPosition(buttonPositionUI);
                vct3 vfPoint = this->ArmPosition.Position().Translation();
                vfPoint.Z() = defaultDepth.Translation().Z();
                PointVF.setPoint(vfPoint);
                PointVF.update(defaultDepth, vfParams);
                ActiveVF = true;
                WasOnButton = true;
            } 
        } else {
            this->SetScaleFactor(1.0);
            if (WasOnButton) {
                WasOnButton = false;                
                WasOnMenu = false;
            }
        }
    }

    if (ActiveVF) {
         FunctionWriteSetVF(vfParams);
    }
    if (!onMenu && ActiveVF) {
        // transition off menu
        FunctionVoidDisableVF();
        ActiveVF = false;
    }
}
