/*
  Author(s):  Min Yang Jung
  Created on: 2009-02-25
*/

/*!
  \file
  \brief A data collection tool
*/

#ifndef _mtsCollector_h
#define _mtsCollector_h

#include <cisstMultiTask/mtsTaskPeriodic.h>

#include <cisstMultiTask/mtsExport.h>

#include <string>

using namespace std;

/*!
  \ingroup cisstMultiTask
o
  This class provides a way to collect data from state table in real-time.
  Collected data can be either saved as a log file or displayed in GUI like an oscilloscope.
*/
class CISST_EXPORT mtsCollector : public mtsTaskPeriodic
{
private:
	static unsigned int CollectorCount;

public:
	mtsCollector(const std::string & collectorName, double period);
	virtual ~mtsCollector();

	//------------ Thread management functions (called internally) -----------//
	/* set some initial values */
	void Startup(void);

	/* performed periodically */
    void Run(void);

	/* clean-up */
    void Cleanup(void);

	//----------------- Signal registration for collection ------------------//
	/* Add signal */
	bool AddSignal(const string & taskName, const string & signalName, const string & format);

	//
	// To be considered more or to be implemented
	//
	//void SetFileName("filename");
	//void SetTimeBase( double DeltaT, bool offsetToZero);
	//void Start(time);
	//void Stop(time);
	//void SetFormatInterpreter(callback);

	//---------------------- Miscellaneous functions ------------------------//
	inline const unsigned int GetCollectorCount() const { return CollectorCount; }

protected:

	/* Initialize this collector instance */
	void Init();
};

CMN_DECLARE_SERVICES_INSTANTIATION(mtsCollector)

#endif // _mtsCollector_h

