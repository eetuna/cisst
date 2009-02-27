/*
  Author(s):  Min Yang Jung
  Created on: 2009-02-25
*/

#include <cisstMultiTask/mtsCollector.h>

unsigned int mtsCollector::CollectorCount = 0;

//-------------------------------------------------------
//	Constructor, Destructor, and Initializer
//-------------------------------------------------------
mtsCollector::mtsCollector(const std::string & collectorName, double period) : 
	mtsTaskPeriodic(collectorName, period, false)
{
	++CollectorCount;
}

mtsCollector::~mtsCollector()
{
	--CollectorCount;

	CMN_LOG_CLASS(5) << "Collector " << GetName() << " ends." << std::endl;
}

void mtsCollector::Init()
{
	//
}

//-------------------------------------------------------
//	Thread management functions (called internally)
//-------------------------------------------------------
void mtsCollector::Startup(void)
{
	// initialization
	Init();
}

void mtsCollector::Run(void)
{
	// periodic execution

	static int i = 0;

	CMN_LOG_CLASS(5) << "----- [ " << GetName() << " ] " << ++i << std::endl;
}

void mtsCollector::Cleanup(void)
{
	// clean up
}

//-------------------------------------------------------
//	Signal Registration
//-------------------------------------------------------
bool mtsCollector::AddSignal(const std::string & taskName, 
							 const std::string & signalName, 
							 const std::string & format)
{
	return true;
}