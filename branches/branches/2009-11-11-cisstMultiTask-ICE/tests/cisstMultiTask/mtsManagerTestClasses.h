/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsManagerTestClasses.h 2009-03-05 mjung5 $
  
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

/*
    Following component definitions are described in the project wiki page.
    (see https://trac.lcsr.jhu.edu/cisst/wiki/Private/cisstMultiTaskNetwork)
*/

//-----------------------------------------------------------------------------
//  Provided Interface and Required Interface Definition
//-----------------------------------------------------------------------------
class mtsManagerTestProvidedInterface
{
private:
    mtsInt Value;

public:
    mtsFunctionVoid  EventVoid;
    mtsFunctionWrite EventWrite;

    mtsManagerTestProvidedInterface() {
        Value.Data = -1;   // initial value = -1;
    }

    void CommandVoid(void) { 
        Value.Data = 0;
    }

    void CommandWrite(const mtsInt & argument) {
        Value = argument;
    }

    void CommandRead(mtsInt & argument) const {
        argument = Value;
    }

    void CommandQualifiedRead(const mtsInt & argumentIn, mtsInt & argumentOut) const {
        argumentOut = argumentIn;
    }

    const int GetValue() const {
        return Value.Data;
    }
};

class mtsManagerTestRequiredInterface
{
private:
    mtsInt Value;

public:
    mtsFunctionVoid          CommandVoid;
    mtsFunctionWrite         CommandWrite;
    mtsFunctionRead          CommandRead;
    mtsFunctionQualifiedRead CommandQualifiedRead;

    mtsManagerTestRequiredInterface() {
        Value.Data = -1;   // initial value = -1;
    }

    void EventVoidHandler(void) {
        Value.Data = 0;
    }

    void EventWriteHandler(const mtsInt & argument) {
        Value.Data = argument.Data;
    }

    const int GetValue() const {
        return Value.Data;
    }
};

//-----------------------------------------------------------------------------
//  C1: (P1:C1:r1 - P2:C2:p1), (P1:C1:r2 - P2:C2:p2)
//  - provided interface: none 
//  - required interface: r1, r2
//-----------------------------------------------------------------------------
class mtsManagerTestC1 : public mtsTaskPeriodic
{
    mtsManagerTestRequiredInterface R1, R2;

public:
    mtsManagerTestC1() : mtsTaskPeriodic("C1", 10 * cmn_ms)
    {
        mtsRequiredInterface * required;

        // Define required interface: r1
        required = AddRequiredInterface("r1");
        if (required) {
            required->AddFunction("Void", R1.CommandVoid);
            required->AddFunction("Write", R1.CommandWrite);
            required->AddFunction("Read", R1.CommandRead);
            required->AddFunction("QualifiedRead", R1.CommandQualifiedRead);
            required->AddEventHandlerVoid(&mtsManagerTestRequiredInterface::EventVoidHandler, &R1, "EventVoid");
            required->AddEventHandlerWrite(&mtsManagerTestRequiredInterface::EventWriteHandler, &R1, "EventWrite");
        }
        // Define required interface: r2
        required = AddRequiredInterface("r2");
        if (required) {
            required->AddFunction("Void", R2.CommandVoid);
            required->AddFunction("Write", R2.CommandWrite);
            required->AddFunction("Read", R2.CommandRead);
            required->AddFunction("QualifiedRead", R2.CommandQualifiedRead);
            required->AddEventHandlerVoid(&mtsManagerTestRequiredInterface::EventVoidHandler, &R2, "EventVoid");
            required->AddEventHandlerWrite(&mtsManagerTestRequiredInterface::EventWriteHandler, &R2, "EventWrite");
        }
    }

    void Run(void) {}
};

class mtsManagerTestC1Device : public mtsDevice
{
    mtsManagerTestRequiredInterface R1, R2;

public:
    mtsManagerTestC1Device() : mtsDevice("C1")
    {
        mtsRequiredInterface * required;

        // Define required interface: r1
        required = AddRequiredInterface("r1");
        if (required) {
            required->AddFunction("Void", R1.CommandVoid);
            required->AddFunction("Write", R1.CommandWrite);
            required->AddFunction("Read", R1.CommandRead);
            required->AddFunction("QualifiedRead", R1.CommandQualifiedRead);
            required->AddEventHandlerVoid(&mtsManagerTestRequiredInterface::EventVoidHandler, &R1, "EventVoid");
            required->AddEventHandlerWrite(&mtsManagerTestRequiredInterface::EventWriteHandler, &R1, "EventWrite");
        }
        // Define required interface: r2
        required = AddRequiredInterface("r2");
        if (required) {
            required->AddFunction("Void", R2.CommandVoid);
            required->AddFunction("Write", R2.CommandWrite);
            required->AddFunction("Read", R2.CommandRead);
            required->AddFunction("QualifiedRead", R2.CommandQualifiedRead);
            required->AddEventHandlerVoid(&mtsManagerTestRequiredInterface::EventVoidHandler, &R2, "EventVoid");
            required->AddEventHandlerWrite(&mtsManagerTestRequiredInterface::EventWriteHandler, &R2, "EventWrite");
        }
    }

    void Configure(const std::string & filename = "") {}
};

//-----------------------------------------------------------------------------
//  C2: (P1:C2:r1 - P2:C2:p2)
//  - provided interface: p1, p2
//  - required interface: r1
//-----------------------------------------------------------------------------
class mtsManagerTestC2 : public mtsTaskContinuous
{
    mtsManagerTestProvidedInterface P1, P2;
    mtsManagerTestRequiredInterface R1;

public:
    mtsManagerTestC2() : mtsTaskContinuous("C2")
    {
        mtsRequiredInterface * required;
        mtsProvidedInterface * provided;

        // Define provided interface: p1
        provided = AddProvidedInterface("p1");
        if (provided) {
            provided->AddCommandVoid(&mtsManagerTestProvidedInterface::CommandVoid, &P1, "Void");
            provided->AddCommandWrite(&mtsManagerTestProvidedInterface::CommandWrite, &P1, "Write");
            provided->AddCommandRead(&mtsManagerTestProvidedInterface::CommandRead, &P1, "Read");            
            provided->AddCommandQualifiedRead(&mtsManagerTestProvidedInterface::CommandQualifiedRead, &P1, "QualifiedRead");
            provided->AddEventVoid(P1.EventVoid, "EventVoid");
            provided->AddEventWrite(P1.EventWrite, "EventWrite", mtsInt(-1));
        }

        // Define provided interface: p2
        provided = AddProvidedInterface("p2");
        if (provided) {
            provided->AddCommandVoid(&mtsManagerTestProvidedInterface::CommandVoid, &P2, "Void");
            provided->AddCommandWrite(&mtsManagerTestProvidedInterface::CommandWrite, &P2, "Write");
            provided->AddCommandRead(&mtsManagerTestProvidedInterface::CommandRead, &P2, "Read");            
            provided->AddCommandQualifiedRead(&mtsManagerTestProvidedInterface::CommandQualifiedRead, &P2, "QualifiedRead");
            provided->AddEventVoid(P2.EventVoid, "EventVoid");
            provided->AddEventWrite(P2.EventWrite, "EventWrite", mtsInt(-1));
        }

        // Define required interface: r1
        required = AddRequiredInterface("r1");
        if (required) {
            required->AddFunction("Void", R1.CommandVoid);
            required->AddFunction("Write", R1.CommandWrite);
            required->AddFunction("Read", R1.CommandRead);
            required->AddFunction("QualifiedRead", R1.CommandQualifiedRead);
            required->AddEventHandlerVoid(&mtsManagerTestRequiredInterface::EventVoidHandler, &R1, "EventVoid");
            required->AddEventHandlerWrite(&mtsManagerTestRequiredInterface::EventWriteHandler, &R1, "EventWrite");
        }
    }

    void Run(void) {}
};

class mtsManagerTestC2Device : public mtsDevice
{
    mtsManagerTestProvidedInterface P1, P2;
    mtsManagerTestRequiredInterface R1;

public:
    mtsManagerTestC2Device() : mtsDevice("C2")
    {
        mtsRequiredInterface * required;
        mtsProvidedInterface * provided;

        // Define provided interface: p1
        provided = AddProvidedInterface("p1");
        if (provided) {
            provided->AddCommandVoid(&mtsManagerTestProvidedInterface::CommandVoid, &P1, "Void");
            provided->AddCommandWrite(&mtsManagerTestProvidedInterface::CommandWrite, &P1, "Write");
            provided->AddCommandRead(&mtsManagerTestProvidedInterface::CommandRead, &P1, "Read");            
            provided->AddCommandQualifiedRead(&mtsManagerTestProvidedInterface::CommandQualifiedRead, &P1, "QualifiedRead");
            provided->AddEventVoid(P1.EventVoid, "EventVoid");
            provided->AddEventWrite(P1.EventWrite, "EventWrite", mtsInt(-1));
        }

        // Define provided interface: p2
        provided = AddProvidedInterface("p2");
        if (provided) {
        }

        // Define required interface: r1
        required = AddRequiredInterface("r1");
        if (required) {
            required->AddFunction("Void", R1.CommandVoid);
            required->AddFunction("Write", R1.CommandWrite);
            required->AddFunction("Read", R1.CommandRead);
            required->AddFunction("QualifiedRead", R1.CommandQualifiedRead);
            required->AddEventHandlerVoid(&mtsManagerTestRequiredInterface::EventVoidHandler, &R1, "EventVoid");
            required->AddEventHandlerWrite(&mtsManagerTestRequiredInterface::EventWriteHandler, &R1, "EventWrite");
        }
    }

    void Configure(const std::string & filename = "") {}
};

//-----------------------------------------------------------------------------
//  C3: (P2:C3:r1 - P2:C2:p2)
//  - provided interface: none
//  - required interface: r1
//-----------------------------------------------------------------------------
class mtsManagerTestC3 : public mtsTaskFromCallback
{
    mtsManagerTestRequiredInterface R1;

public:
    // Counters to test Create()
    int CounterCreateCall;

    mtsManagerTestC3() : mtsTaskFromCallback("C3"), CounterCreateCall(0)
    {
        mtsRequiredInterface * required;

        // Define required interface: r1
        required = AddRequiredInterface("r1");
        if (required) {
            required->AddFunction("Void", R1.CommandVoid);
            required->AddFunction("Write", R1.CommandWrite);
            required->AddFunction("Read", R1.CommandRead);
            required->AddFunction("QualifiedRead", R1.CommandQualifiedRead);
            required->AddEventHandlerVoid(&mtsManagerTestRequiredInterface::EventVoidHandler, &R1, "EventVoid");
            required->AddEventHandlerWrite(&mtsManagerTestRequiredInterface::EventWriteHandler, &R1, "EventWrite");
        }
    }

    void Run(void) {}
    //void Create(void *data) { ++CounterCreateCall; }
};

class mtsManagerTestC3Device : public mtsDevice
{
    mtsManagerTestRequiredInterface R1;

public:
    mtsManagerTestC3Device() : mtsDevice("C3")
    {
        mtsRequiredInterface * required;

        // Define required interface: r1
        required = AddRequiredInterface("r1");
        if (required) {
            required->AddFunction("Void", R1.CommandVoid);
            required->AddFunction("Write", R1.CommandWrite);
            required->AddFunction("Read", R1.CommandRead);
            required->AddFunction("QualifiedRead", R1.CommandQualifiedRead);
            required->AddEventHandlerVoid(&mtsManagerTestRequiredInterface::EventVoidHandler, &R1, "EventVoid");
            required->AddEventHandlerWrite(&mtsManagerTestRequiredInterface::EventWriteHandler, &R1, "EventWrite");
        }
    }

    void Configure(const std::string & filename = "") {}
};

#endif