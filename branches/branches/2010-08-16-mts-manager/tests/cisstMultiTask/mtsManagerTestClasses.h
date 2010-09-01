/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Min Yang Jung
  Created on: 2009-12-08

  (C) Copyright 2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#ifndef _mtsManagerTestClasses_h
#define _mtsManagerTestClasses_h

#include <cppunit/TestCase.h>
#include <cppunit/extensions/HelperMacros.h>

#include <cisstMultiTask.h>
#include <cisstCommon/cmnUnits.h>
#include <cisstOSAbstraction/osaSleep.h>

/*
    Following component definitions are described in the project wiki page.
    (see https://trac.lcsr.jhu.edu/cisst/wiki/Private/cisstMultiTaskNetwork)
*/

//-----------------------------------------------------------------------------
//  Provided Interface and Required Interface Definition
//-----------------------------------------------------------------------------
class mtsManagerTestInterfaceProvided
{
private:
    mtsInt Value;
    double ExecutionDelay; // to test blocking commands

public:
    mtsFunctionVoid  EventVoid;
    mtsFunctionWrite EventWrite;

    mtsManagerTestInterfaceProvided(double executionDelay = 0.0):
        ExecutionDelay(executionDelay)
    {
        Value.Data = -1;   // initial value = -1;
    }

    void CommandVoid(void) {
        osaSleep(ExecutionDelay);
        Value.Data = 0;
    }

    void CommandWrite(const mtsInt & argument) {
        osaSleep(ExecutionDelay);
        Value = argument;
    }

    void CommandRead(mtsInt & argument) const {
        argument = Value;
    }

    void CommandQualifiedRead(const mtsInt & argumentIn, mtsInt & argumentOut) const {
        argumentOut.Data = argumentIn.Data + 1;
    }

    int GetValue(void) const {
        return Value.Data;
    }

    void PopulateExistingInterface(mtsInterfaceProvided * provided) {
        provided->AddCommandVoid(&mtsManagerTestInterfaceProvided::CommandVoid, this, "Void");
        provided->AddCommandWrite(&mtsManagerTestInterfaceProvided::CommandWrite, this, "Write");
        provided->AddCommandRead(&mtsManagerTestInterfaceProvided::CommandRead, this, "Read");
        provided->AddCommandQualifiedRead(&mtsManagerTestInterfaceProvided::CommandQualifiedRead, this, "QualifiedRead");
        provided->AddEventVoid(this->EventVoid, "EventVoid");
        provided->AddEventWrite(this->EventWrite, "EventWrite", mtsInt(-1));
    }
};

class mtsManagerTestInterfaceRequired
{
private:
    mtsInt Value;

public:
    mtsFunctionVoid          CommandVoid;
    mtsFunctionWrite         CommandWrite;
    mtsFunctionRead          CommandRead;
    mtsFunctionQualifiedRead CommandQualifiedRead;

    mtsManagerTestInterfaceRequired() {
        Value.Data = -1;   // initial value = -1;
    }

    void EventVoidHandler(void) {
        Value.Data = 0;
    }

    void EventWriteHandler(const mtsInt & argument) {
        Value.Data = argument.Data;
    }

    int GetValue(void) const {
        return Value.Data;
    }

    void PopulateExistingInterface(mtsInterfaceRequired * required) {
        required->AddFunction("Void", this->CommandVoid);
        required->AddFunction("Write", this->CommandWrite);
        required->AddFunction("Read", this->CommandRead);
        required->AddFunction("QualifiedRead", this->CommandQualifiedRead);
        required->AddEventHandlerVoid(&mtsManagerTestInterfaceRequired::EventVoidHandler, this, "EventVoid", MTS_EVENT_NOT_QUEUED);
        required->AddEventHandlerWrite(&mtsManagerTestInterfaceRequired::EventWriteHandler, this, "EventWrite", MTS_EVENT_NOT_QUEUED);
    }
};

//-----------------------------------------------------------------------------
//  Periodic1: (P1:Periodic1:r1 - P2:Continuous1:p1), (P1:Periodic1:r2 - P2:Continuous1:p2)
//  - provided interface: none
//  - required interface: r1, r2
//-----------------------------------------------------------------------------
class mtsManagerTestPeriodic1 : public mtsTaskPeriodic
{
public:
    mtsManagerTestInterfaceRequired InterfaceRequired1, InterfaceRequired2;

    mtsManagerTestPeriodic1() : mtsTaskPeriodic("Periodic1Task", 10 * cmn_ms)
    {
        mtsInterfaceRequired * required;

        // Define required interface: r1
        required = AddInterfaceRequired("r1");
        if (required) {
            InterfaceRequired1.PopulateExistingInterface(required);
        }
        // Define required interface: r2
        required = AddInterfaceRequired("r2");
        if (required) {
            InterfaceRequired2.PopulateExistingInterface(required);
        }
    }

    void Run(void) {}
};

class mtsManagerTestDevice1 : public mtsComponent
{
public:
    mtsManagerTestInterfaceRequired InterfaceRequired1, InterfaceRequired2;

    mtsManagerTestDevice1() : mtsComponent("Device1")
    {
        mtsInterfaceRequired * required;

        // Define required interface: r1
        required = AddInterfaceRequired("r1");
        if (required) {
            InterfaceRequired1.PopulateExistingInterface(required);
        }
        // Define required interface: r2
        required = AddInterfaceRequired("r2");
        if (required) {
            InterfaceRequired2.PopulateExistingInterface(required);
        }
    }

    void Configure(const std::string & CMN_UNUSED(filename) = "") {}
};

//-----------------------------------------------------------------------------
//  Continuous1: (P1:Continuous1:r1 - P2:Continuous1:p2)
//  - provided interface: p1, p2
//  - required interface: r1
//-----------------------------------------------------------------------------
class mtsManagerTestContinuous1 : public mtsTaskContinuous
{
public:
    mtsManagerTestInterfaceProvided InterfaceProvided1, InterfaceProvided2;
    mtsManagerTestInterfaceRequired InterfaceRequired1;

    mtsManagerTestContinuous1() : mtsTaskContinuous("Continuous1Task")
    {
        mtsInterfaceRequired * required;
        mtsInterfaceProvided * provided;

        // Define provided interface: p1
        provided = AddInterfaceProvided("p1");
        if (provided) {
            InterfaceProvided1.PopulateExistingInterface(provided);
        }

        // Define provided interface: p2
        provided = AddInterfaceProvided("p2");
        if (provided) {
            InterfaceProvided2.PopulateExistingInterface(provided);
        }

        // Define required interface: r1
        required = AddInterfaceRequired("r1");
        if (required) {
            InterfaceRequired1.PopulateExistingInterface(required);
        }
    }

    void Run(void) {
        osaSleep(1.0 * cmn_ms);
    }
};

class mtsManagerTestDevice2 : public mtsComponent
{
public:
    mtsManagerTestInterfaceProvided InterfaceProvided1, InterfaceProvided2;
    mtsManagerTestInterfaceRequired InterfaceRequired1;

    mtsManagerTestDevice2() : mtsComponent("Device2")
    {
        mtsInterfaceRequired * required;
        mtsInterfaceProvided * provided;

        // Define provided interface: p1
        provided = AddInterfaceProvided("p1");
        if (provided) {
            InterfaceProvided1.PopulateExistingInterface(provided);
        }

        // Define provided interface: p2
        provided = AddInterfaceProvided("p2");
        if (provided) {
            InterfaceProvided2.PopulateExistingInterface(provided);
        }

        // Define required interface: r1
        required = AddInterfaceRequired("r1");
        if (required) {
            InterfaceRequired1.PopulateExistingInterface(required);
        }
    }

    void Configure(const std::string & CMN_UNUSED(filename) = "") {}
};

//-----------------------------------------------------------------------------
//  FromCallback1: (P2:FromCallback1:r1 - P2:C2:p2)
//  - provided interface: none
//  - required interface: r1
//-----------------------------------------------------------------------------
class mtsManagerTestFromCallback1 : public mtsTaskFromCallback
{
public:
    mtsManagerTestInterfaceRequired InterfaceRequired1;

    // Counters to test Create()
    int CounterCreateCall;

    mtsManagerTestFromCallback1() : mtsTaskFromCallback("FromCallback1Task"), CounterCreateCall(0)
    {
        mtsInterfaceRequired * required;

        // Define required interface: r1
        required = AddInterfaceRequired("r1");
        if (required) {
            InterfaceRequired1.PopulateExistingInterface(required);
        }
    }

    void Run(void) {}
};


class mtsManagerTestCallbackTrigger
{
    osaThread Thread;
    mtsTaskFromCallback * Task;
    bool Running;
public:
    mtsManagerTestCallbackTrigger(mtsTaskFromCallback * task):
        Task(task),
        Running(true)
    {
        Thread.Create<mtsManagerTestCallbackTrigger, int>(this, &mtsManagerTestCallbackTrigger::Run,
                                                          0, "TstCb");
    }

    ~mtsManagerTestCallbackTrigger() {
        Thread.Wait();
    }

    void Stop(void) {
        this->Running = false;
    }

    void * Run(int CMN_UNUSED(data)) {
        while (this->Running) {
            Task->DoCallback(0);
            osaSleep(1.0 * cmn_ms);
        }
        // stop the thread
        osaCurrentThreadYield();
        return 0;
    }
};


class mtsManagerTestDevice3 : public mtsComponent
{
public:
    mtsManagerTestInterfaceRequired InterfaceRequired1;

    mtsManagerTestDevice3() : mtsComponent("Device3")
    {
        mtsInterfaceRequired * required;

        // Define required interface: r1
        required = AddInterfaceRequired("r1");
        if (required) {
            InterfaceRequired1.PopulateExistingInterface(required);
        }
    }

    void Configure(const std::string & CMN_UNUSED(filename) = "") {}
};


class mtsManagerTestFromSignal1 : public mtsTaskFromSignal
{
public:
    mtsManagerTestInterfaceRequired InterfaceRequired1;

    mtsManagerTestFromSignal1() : mtsTaskFromSignal("FromSignal1Task")
    {
        mtsInterfaceRequired * required;

        // Define required interface: r1
        required = AddInterfaceRequired("r1");
        if (required) {
            InterfaceRequired1.PopulateExistingInterface(required);
        }
    }

    void Run(void) {}
};

#endif
