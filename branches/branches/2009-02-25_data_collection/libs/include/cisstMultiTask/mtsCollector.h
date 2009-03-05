/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: mtsCollector.h 2009-03-02 mjung5

  Author(s):  Min Yang Jung
  Created on: 2009-02-25

  (C) Copyright 2008-2009 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

/*!
  \file
  \brief A data collection tool
*/

#ifndef _mtsCollector_h
#define _mtsCollector_h

#include <cisstCommon/cmnUnits.h>
#include <cisstMultiTask/mtsTaskPeriodic.h>
#include <cisstMultiTask/mtsExport.h>
#include <cisstMultiTask/mtsTaskManager.h>

#include <string>
#include <map>

// enable this macro for unit-test purposes only
#define	_OPEN_PRIVATE_FOR_UNIT_TEST_

/*!
  \ingroup cisstMultiTask

  This class provides a way to collect data from state table in real-time.
  Collected data can be either saved as a log file or displayed in GUI like an oscilloscope.
*/
class CISST_EXPORT mtsCollector : public mtsTaskPeriodic
{
	CMN_DECLARE_SERVICES(CMN_NO_DYNAMIC_CREATION, 5);

#ifdef _OPEN_PRIVATE_FOR_UNIT_TEST_
public:
#else
private:
#endif
	typedef std::multimap<std::string, std::string> SignalMap;	// (taskName, signalName)
	SignalMap SignalCollection;

	static unsigned int CollectorCount;
	static mtsTaskManager * taskManager;

public:
	mtsCollector(const std::string & collectorName, double period = 100 * cmn_ms);
	virtual ~mtsCollector();

	//------------ Thread management functions (called internally) -----------//
	/*! set some initial values */
	void Startup(void);

	/*! performed periodically */
    void Run(void);

	/*! clean-up */
    void Cleanup(void);

	//----------------- Signal registration for collection ------------------//
	/*! Add a signal to the list */
	bool AddSignal(const std::string & taskName, const std::string & signalName, 
				   const std::string & format);	// format is currently meaningless.

	/*! Remove a signal from the list */
	bool RemoveSignal(const std::string & taskName, const std::string & signalName);

	/*! Find a signal from the list */
	bool FindSignal(const std::string & taskName, const std::string & signalName);

	/*! Adjust the period of the collector task automatically */
	void AdjustPeriod(const double newPeriod);

	//
	// To be considered more or to be implemented
	//
	//void SetFileName("filename");
	//void SetTimeBase( double DeltaT, bool offsetToZero);
	//void Start(time);
	//void Stop(time);
	//void SetFormatInterpreter(callback);

	//---------------------- Miscellaneous functions ------------------------//
	inline static const unsigned int GetCollectorCount() { return CollectorCount; }

protected:

	/* Initialize this collector instance */
	void Init();
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsCollector)

#endif // _mtsCollector_h

