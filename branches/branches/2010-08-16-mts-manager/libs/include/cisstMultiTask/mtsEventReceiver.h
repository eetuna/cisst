/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Peter Kazanzides
  Created on: 2010-09-24

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


#ifndef _mtsEventReceiver_h
#define _mtsEventReceiver_h

#include <cisstMultiTask/mtsInterfaceRequired.h>
class osaThreadSignal;

// EventReceivers must be added before Bind (add check for InterfaceProvidedOrOutput==0)
// EventHandlers can be added at any time.
// When Bind called, 
//    if no EventReceiver, directly add EventHandler
//    if EventReceiver, set handler in it

class mtsEventReceiverBase {
protected:
    std::string Name;
    mtsInterfaceRequired *Required;   // Pointer to the required interface
    osaThreadSignal *EventSignal;
    bool Waiting;
    bool OwnEventSignal;   // true if we created our own thread signal

    virtual bool mtsEventReceiverBase::CheckRequired() const;

public:
    mtsEventReceiverBase();
    virtual ~mtsEventReceiverBase();

    virtual std::string GetName() const { return Name; }

    // Called from mtsInterfaceRequired::AddEventReceiver
    virtual void SetRequired(mtsInterfaceRequired *req);

    // wait for event to be issued (could also add timeout).
    // Returns true if successful, false if failed.
    virtual bool Wait();

    virtual void Detach();

    /*! Human readable output to stream. */
    virtual void ToStream(std::ostream & outputStream) const = 0;
};


/*! Stream out operator. */
inline std::ostream & operator << (std::ostream & output,
                                   const mtsEventReceiverBase & receiver) {
    receiver.ToStream(output);
    return output;
}

class mtsEventReceiverVoid : public mtsEventReceiverBase {
protected:
    mtsCommandVoidBase *Command;      // Command object for calling EventHandler method
    mtsCommandVoidBase *UserHandler;  // User supplied event handler

    // This one always gets added non-queued
    void EventHandler(void);

public:
    mtsEventReceiverVoid();
    ~mtsEventReceiverVoid();

    // Called from mtsInterfaceRequired::BindCommandsAndEvents
    mtsCommandVoidBase *GetCommand();

    // Called from mtsInterfaceRequired::AddEventHandlerVoid
    void SetHandlerCommand(mtsCommandVoidBase *cmdHandler);

    // Same functionality as mtsInterfaceRequired::AddEventHandlerVoid.
    template <class __classType>
    inline mtsCommandVoidBase * SetHandler(void (__classType::*method)(void),
                                           __classType * classInstantiation,
                                           mtsEventQueuingPolicy queuingPolicy = MTS_INTERFACE_EVENT_POLICY) {
        return CheckRequired() ? (Required->AddEventHandlerVoid(method, classInstantiation, this->GetName(), queuingPolicy)) : 0;
    }

#if 0
    inline mtsCommandVoidBase * SetHandler(void (*function)(void),
                                                mtsEventQueuingPolicy = MTS_INTERFACE_EVENT_POLICY) {
        return CheckRequired() ? (Required->AddEventHandlerVoid(function, this->GetName(), queuingPolicy)) : 0;
    }
#endif

    void ToStream(std::ostream & outputStream) const;
};

class mtsEventReceiverWrite : public mtsEventReceiverBase {
protected:
    mtsCommandWriteBase *Command;      // Command object for calling EventHandler method
    mtsCommandWriteBase *UserHandler;  // User supplied event handler
    mtsGenericObject *ArgPtr;

    // This one always gets added non-queued
    void EventHandler(const mtsGenericObject &arg);

public:
    mtsEventReceiverWrite();
    ~mtsEventReceiverWrite();

    // Called from mtsInterfaceRequired::BindCommandsAndEvents
    mtsCommandWriteBase *GetCommand();

    // Called from mtsInterfaceRequired::AddEventHandlerWrite
    void SetHandlerCommand(mtsCommandWriteBase *cmdHandler);

    // Same functionality as mtsInterfaceRequired::AddEventHandlerWrite.
    template <class __classType, class __argumentType>
    inline mtsCommandWriteBase * SetHandler(void (__classType::*method)(const __argumentType &),
                                            __classType * classInstantiation,
                                            mtsEventQueuingPolicy queuingPolicy = MTS_INTERFACE_EVENT_POLICY) {
        return CheckRequired() ? (Required->AddEventHandlerWrite(method, classInstantiation, this->GetName(), queuingPolicy)) : 0;
    }

    // PK: Do we need the "generic" version (AddEventHandlerWriteGeneric)?

    // Wait and return received argument. 
    // A false return value could mean that the wait failed, or that the wait succeeded but the return value (obj)
    // is invalid.
    bool Wait(mtsGenericObject &obj);

    //PK: might be nice to have this
    //const mtsGenericObject *GetArgumentPrototype() const;

    void ToStream(std::ostream & outputStream) const;
};


#endif // _mtsEventReceiver_h
