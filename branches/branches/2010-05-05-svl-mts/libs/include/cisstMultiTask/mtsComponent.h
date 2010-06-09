/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Ankur Kapoor, Peter Kazanzides, Anton Deguet
  Created on: 2004-04-30

  (C) Copyright 2004-2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/


#ifndef _mtsComponent_h
#define _mtsComponent_h

#include <cisstCommon/cmnPortability.h>
#include <cisstCommon/cmnGenericObject.h>
#include <cisstCommon/cmnClassRegisterMacros.h>
#include <cisstCommon/cmnNamedMap.h>

#include <cisstOSAbstraction/osaThread.h>

#include <cisstMultiTask/mtsCommandBase.h>
#include <cisstMultiTask/mtsForwardDeclarations.h>
#include <cisstMultiTask/mtsMulticastCommandVoid.h>
#include <cisstMultiTask/mtsMulticastCommandWrite.h>

// Always include last
#include <cisstMultiTask/mtsExport.h>

/*!
  \file
  \brief Declaration of mtsComponent
 */


/*!
  \ingroup cisstMultiTask

  mtsComponent should be used to write wrappers around existing
  devices or resources.  This class allows to interact with existing
  devices as one would interact with a task (as in mtsTask and
  mtsTaskPeriodic).  To do so, the component maintains a list of
  provided interfaces (of type mtsDeviceInterface) which contains
  commands.

  The main differences are that the base component class doesn't have
  a thread and is stateless.  Since the component doesn't have any
  thread, the commands are not queued and the class doesn't add any
  thread safety mechanism.  The component class doesn't maintain a state
  as it relies on the underlying device to do so.  It is basically a
  pass-thru or wrapper.
 */
class CISST_EXPORT mtsComponent: public cmnGenericObject
{
    CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, CMN_LOG_LOD_RUN_ERROR);

    friend class mtsManagerLocal;
    friend class mtsComponentProxy;

 protected:

    /*! A string identifying the 'Name' of the component. */
    std::string Name;

    /*! Default constructor. Protected to prevent creation of a component
      without a name. */
    mtsComponent(void) {}

    /*! Add an already existing interface required to the interface,
      the user must pay attention to mailbox (or lack of) used to
      create the required interface.  By default, mtsComponent uses a
      required interface without mailbox (i.e. doesn't queue), mtsTask
      uses an interface with a mailbox and mtsTaskFromSignal uses an
      interface with a mailbox with a post command queued command. */
    mtsInterfaceRequired * AddInterfaceRequiredExisting(const std::string & interfaceRequiredName,
                                                        mtsInterfaceRequired * interfaceRequired);

    /*! Create and add a required interface with an existing mailbox.
      If the creation or addition failed (name already exists), the
      caller must make sure he/she deletes the unused mailbox. */
    mtsInterfaceRequired * AddInterfaceRequiredUsingMailbox(const std::string & interfaceRequiredName,
                                                            mtsMailBox * mailBox);

 public:

    /*! Default constructor. Sets the name. */
    mtsComponent(const std::string & deviceName);

    /*! Default destructor. Does nothing. */
    virtual ~mtsComponent() {}

    /*! Returns the name of the component. */
    const std::string & GetName(void) const;

    /*! Set name.  This method is useful to perform dynamic creation
      using the default constructor and then set the name. */
    void SetName(const std::string & componentName);

    /*! The virtual method so that the interface or tasks can
      configure themselves */
    virtual void Configure(const std::string & filename = "");

    /*! Virtual method called after components are connected and
      before they get started.  Use to place initialization code. */
    virtual void Start(void);

    /*! Method to add an interface to the component.  This method is
      virtual so that mtsTaskBase can redefine it and generate the
      appropriate type of interface, i.e. mtsTaskInterface as opposed
      to mtsDeviceInterface for mtsComponent. */
    virtual mtsDeviceInterface * AddProvidedInterface(const std::string & newInterfaceName);

    /*! Return the list of provided interfaces.  This returns a list
      of names.  To retrieve the actual interface, use
      GetProvidedInterface with the provided interface name. */
    virtual std::vector<std::string> GetNamesOfProvidedInterfaces(void) const;

    /*! Get a pointer on a provided interface defined by its name.
      This can be used by a "user" task to find out which interfaces
      are provided and then retrieve commands/events from that
      interface. */
    mtsDeviceInterface * GetProvidedInterface(const std::string & interfaceName) const;

    /*! Add a required interface.  This interface will later on be
      connected to another task and use the provided interface of the
      other task.  The required interface created also contains a list
      of event handlers to be used as observers. */
    virtual mtsInterfaceRequired * AddInterfaceRequired(const std::string & interfaceRequiredName);
    
    // provided for backward compatibility
    inline CISST_DEPRECATED mtsRequiredInterface * AddRequiredInterface(const std::string & requiredInterfaceName) {
        return this->AddInterfaceRequired(requiredInterfaceName);
    }

    /*! Provide a list of existing required interfaces (by names) */
    std::vector<std::string> GetNamesOfInterfacesRequiredOrInput(void) const;

    /*! Get a pointer on the provided interface that has been
      connected to a given required interface (defined by its name).
      This method will return a null pointer if the required interface
      has not been connected.  See mtsTaskManager::Connect. */
    const mtsDeviceInterface * GetProvidedInterfaceFor(const std::string & interfaceRequiredName);

    /*! Get the required/input interface */
    mtsInterfaceRequiredOrInput * GetInterfaceRequiredOrInput(const std::string & interfaceRequiredOrInputName);

    /*! Get the required interface */
    mtsInterfaceRequired * GetInterfaceRequired(const std::string & interfaceRequired);

    /*! Connect a required interface, used by mtsTaskManager */
    bool ConnectInterfaceRequiredOrInput(const std::string & requiredInterfaceName,
                                         mtsDeviceInterface * providedInterface);

 protected:
    /*! Thread Id counter.  Used to count how many "user" tasks are
      connected from a single thread.  In most cases the count
      should be one. */
    //@{
    typedef std::pair<osaThreadId, unsigned int> ThreadIdCounterPairType;
    typedef std::vector<ThreadIdCounterPairType> ThreadIdCountersType;
    ThreadIdCountersType ThreadIdCounters;
    //@}

    /*! Map of provided interfaces.  Used to store pointers on all
      provided interfaces. */
    //@{
    typedef cmnNamedMap<mtsDeviceInterface> ProvidedInterfacesMapType;
    ProvidedInterfacesMapType ProvidedInterfaces;
    //@}

    /*! Map of required interfaces.  Used to store pointers on all
      required interfaces. */
    //@{
    typedef cmnNamedMap<mtsInterfaceRequiredOrInput> InterfacesRequiredOrInputMapType;
    InterfacesRequiredOrInputMapType InterfacesRequiredOrInput;
    //@}

 public:

    /*! Send a human readable description of the component. */
    void ToStream(std::ostream & outputStream) const;

    /*! Put in format suitable for graph visualization. */
    std::string ToGraphFormat(void) const;
};


// overload mtsObjectName to retrieve the actual name
inline std::string mtsObjectName(const mtsComponent * object) {
    return "mtsComponent: " + object->GetName();
}


CMN_DECLARE_SERVICES_INSTANTIATION(mtsComponent)


#endif // _mtsComponent_h

