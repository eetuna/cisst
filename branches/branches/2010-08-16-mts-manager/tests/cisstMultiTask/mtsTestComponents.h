/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Min Yang Jung, Anton Deguet
  Created on: 2009-12-08

  (C) Copyright 2009-2010 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#ifndef _mtsTestComponents_h
#define _mtsTestComponents_h

#include <cppunit/TestCase.h>
#include <cppunit/extensions/HelperMacros.h>

#include <cisstMultiTask.h>
#include <cisstCommon/cmnUnits.h>
#include <cisstOSAbstraction/osaSleep.h>

/*
    Following component definitions are described in the project wiki page.
    (see https://trac.lcsr.jhu.edu/cisst/wiki/Private/cisstMultiTaskNetwork)

    All interfaces match, i.e. any provided can be connected to any required.
    Available components are:
      mtsTestPeriodic1: p1, r1, r2
      mtsTestDevice1: p1, r1, r2
      mtsTestContinuous1: p1, p2, r1
      mtsTestDevice2: p1, p2, r1
      mtsTestFromCallback1: p1, r1 (use with mtsTestCallbackTrigger to run)
      mtsTestDevice3: p1, r1
      mtsTestFromSignal1: p1, r1

   Be aware, all required interfaces are MTS_OPTIONAL.

   Execution delay is used to make the Void and Write commands
   artificially slow and test blocking commands.
*/

//-----------------------------------------------------------------------------
//  Provided Interface and Required Interface Definition
//-----------------------------------------------------------------------------
class mtsTestInterfaceProvided
{
private:
    mtsInt Value;
    double ExecutionDelay; // to test blocking commands

public:
    mtsFunctionVoid EventVoid;
    mtsFunctionWrite EventWrite;

    mtsTestInterfaceProvided(double executionDelay = 0.0):
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
        provided->AddCommandVoid(&mtsTestInterfaceProvided::CommandVoid, this, "Void");
        provided->AddCommandWrite(&mtsTestInterfaceProvided::CommandWrite, this, "Write");
        provided->AddCommandRead(&mtsTestInterfaceProvided::CommandRead, this, "Read");
        provided->AddCommandQualifiedRead(&mtsTestInterfaceProvided::CommandQualifiedRead, this, "QualifiedRead");
        provided->AddEventVoid(this->EventVoid, "EventVoid");
        provided->AddEventWrite(this->EventWrite, "EventWrite", mtsInt(-1));
    }
};

class mtsTestInterfaceRequired
{
private:
    mtsInt Value;

public:
    mtsFunctionVoid CommandVoid;
    mtsFunctionWrite CommandWrite;
    mtsFunctionRead CommandRead;
    mtsFunctionQualifiedRead CommandQualifiedRead;

    mtsTestInterfaceRequired() {
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
        required->AddEventHandlerVoid(&mtsTestInterfaceRequired::EventVoidHandler, this, "EventVoid", MTS_EVENT_NOT_QUEUED);
        required->AddEventHandlerWrite(&mtsTestInterfaceRequired::EventWriteHandler, this, "EventWrite", MTS_EVENT_NOT_QUEUED);
    }
};

//-----------------------------------------------------------------------------
//  Periodic1: (P1:Periodic1:r1 - P2:Continuous1:p1), (P1:Periodic1:r2 - P2:Continuous1:p2)
//  - provided interface: none
//  - required interface: r1, r2
//-----------------------------------------------------------------------------
class mtsTestPeriodic1: public mtsTaskPeriodic
{
public:
    mtsTestInterfaceProvided InterfaceProvided1;
    mtsTestInterfaceRequired InterfaceRequired1, InterfaceRequired2;

    mtsTestPeriodic1(const std::string & name = "mtsTestPeriodic1",
                     double executionDelay = 0.0):
        mtsTaskPeriodic(name, 1.0 * cmn_ms),
        InterfaceProvided1(executionDelay)
    {
        mtsInterfaceProvided * provided;
        // Define provided interface: p1
        provided = AddInterfaceProvided("p1");
        if (provided) {
            InterfaceProvided1.PopulateExistingInterface(provided);
        }

        mtsInterfaceRequired * required;
        // Define required interface: r1
        required = AddInterfaceRequired("r1", MTS_OPTIONAL);
        if (required) {
            InterfaceRequired1.PopulateExistingInterface(required);
        }
        // Define required interface: r2
        required = AddInterfaceRequired("r2", MTS_OPTIONAL);
        if (required) {
            InterfaceRequired2.PopulateExistingInterface(required);
        }
    }

    void Run(void) {
        ProcessQueuedCommands();
        ProcessQueuedEvents();
    }
};

class mtsTestDevice1: public mtsComponent
{
public:
    mtsTestInterfaceProvided InterfaceProvided1;
    mtsTestInterfaceRequired InterfaceRequired1, InterfaceRequired2;

    mtsTestDevice1(const std::string & name = "mtsTestDevice1",
                   double executionDelay = 0.0):
        mtsComponent(name),
        InterfaceProvided1(executionDelay)
    {
        mtsInterfaceProvided * provided;
        // Define provided interface: p1
        provided = AddInterfaceProvided("p1");
        if (provided) {
            InterfaceProvided1.PopulateExistingInterface(provided);
        }

        mtsInterfaceRequired * required;
        // Define required interface: r1
        required = AddInterfaceRequired("r1", MTS_OPTIONAL);
        if (required) {
            InterfaceRequired1.PopulateExistingInterface(required);
        }
        // Define required interface: r2
        required = AddInterfaceRequired("r2", MTS_OPTIONAL);
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
class mtsTestContinuous1: public mtsTaskContinuous
{
public:
    mtsTestInterfaceProvided InterfaceProvided1, InterfaceProvided2;
    mtsTestInterfaceRequired InterfaceRequired1;

    mtsTestContinuous1(const std::string & name = "mtsTestContinuous1",
                       double executionDelay = 0.0):
        mtsTaskContinuous(name),
        InterfaceProvided1(executionDelay),
        InterfaceProvided2(executionDelay)
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
        required = AddInterfaceRequired("r1", MTS_OPTIONAL);
        if (required) {
            InterfaceRequired1.PopulateExistingInterface(required);
        }
    }

    void Run(void) {
        osaSleep(1.0 * cmn_ms);
        ProcessQueuedCommands();
        ProcessQueuedEvents();
    }
};


class mtsTestDevice2: public mtsComponent
{
public:
    mtsTestInterfaceProvided InterfaceProvided1, InterfaceProvided2;
    mtsTestInterfaceRequired InterfaceRequired1;

    mtsTestDevice2(const std::string & name = "mtsTestDevice2",
                   double executionDelay = 0.0):
        mtsComponent(name),
        InterfaceProvided1(executionDelay),
        InterfaceProvided2(executionDelay)
    {

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

        mtsInterfaceRequired * required;
        // Define required interface: r1
        required = AddInterfaceRequired("r1", MTS_OPTIONAL);
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
class mtsTestFromCallback1: public mtsTaskFromCallback
{
public:
    mtsTestInterfaceProvided InterfaceProvided1;
    mtsTestInterfaceRequired InterfaceRequired1;

    // Counters to test Create()
    int CounterCreateCall;

    mtsTestFromCallback1(const std::string & name = "mtsTestFromCallback1",
                         double executionDelay = 0.0):
        mtsTaskFromCallback(name),
        InterfaceProvided1(executionDelay),
        CounterCreateCall(0)
    {
        mtsInterfaceProvided * provided;
        // Define provided interface: p1
        provided = AddInterfaceProvided("p1");
        if (provided) {
            InterfaceProvided1.PopulateExistingInterface(provided);
        }

        mtsInterfaceRequired * required;
        // Define required interface: r1
        required = AddInterfaceRequired("r1", MTS_OPTIONAL);
        if (required) {
            InterfaceRequired1.PopulateExistingInterface(required);
        }
    }

    void Run(void) {
        ProcessQueuedCommands();
        ProcessQueuedEvents();
    }
};


class mtsTestCallbackTrigger
{
    osaThread Thread;
    mtsTaskFromCallback * Task;
    bool Running;
public:
    mtsTestCallbackTrigger(mtsTaskFromCallback * task):
        Task(task),
        Running(true)
    {
        Thread.Create<mtsTestCallbackTrigger, int>(this, &mtsTestCallbackTrigger::Run,
                                                          0, "TstCb");
    }

    ~mtsTestCallbackTrigger() {
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


class mtsTestDevice3: public mtsComponent
{
public:
    mtsTestInterfaceProvided InterfaceProvided1;
    mtsTestInterfaceRequired InterfaceRequired1;

    mtsTestDevice3(const std::string & name = "mtsTestDevice3",
                   double executionDelay = 0.0):
        mtsComponent(name),
        InterfaceProvided1(executionDelay)
    {
        mtsInterfaceProvided * provided;
        // Define provided interface: p1
        provided = AddInterfaceProvided("p1");
        if (provided) {
            InterfaceProvided1.PopulateExistingInterface(provided);
        }

        mtsInterfaceRequired * required;
        // Define required interface: r1
        required = AddInterfaceRequired("r1", MTS_OPTIONAL);
        if (required) {
            InterfaceRequired1.PopulateExistingInterface(required);
        }
    }

    void Configure(const std::string & CMN_UNUSED(filename) = "") {}
};


class mtsTestFromSignal1: public mtsTaskFromSignal
{
public:
    mtsTestInterfaceProvided InterfaceProvided1;
    mtsTestInterfaceRequired InterfaceRequired1;

    mtsTestFromSignal1(const std::string & name = "mtsTestFromSignal1",
                       double executionDelay = 0.0):
        mtsTaskFromSignal(name),
        InterfaceProvided1(executionDelay)
    {
        mtsInterfaceProvided * provided;
        // Define provided interface: p1
        provided = AddInterfaceProvided("p1");
        if (provided) {
            InterfaceProvided1.PopulateExistingInterface(provided);
        }

        mtsInterfaceRequired * required;
        // Define required interface: r1
        required = AddInterfaceRequired("r1", MTS_OPTIONAL);
        if (required) {
            InterfaceRequired1.PopulateExistingInterface(required);
        }
    }

    void Run(void) {
        ProcessQueuedCommands();
        ProcessQueuedEvents();
    }
};

#endif // _mtsTestComponents_h
