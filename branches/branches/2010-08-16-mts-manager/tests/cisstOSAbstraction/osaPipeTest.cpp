/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: $

  Author(s): Martin Kelly
  Created on: 2010-09-23

  (C) Copyright 2010 Johns Hopkins University (JHU), All Rights Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstCommon/cmnAssert.h>
#include <cisstOSAbstraction/osaPipe.h>

#include "osaPipeTest.h"

void osaPipeTest::TestPipeInternalsSize(void) {
	CPPUNIT_ASSERT(osaPipe::INTERNALS_SIZE >= osaPipe::SizeOfInternals());
}

void osaPipeTest::TestPipe() {
	osaPipe pipe;
	std::string testProgramName = std::string(CISST_BUILD_ROOT) + std::string("/tests/bin/osaPipeUtility");
	char * const command[] = {(char * const) testProgramName.c_str(), NULL};
	pipe.Open(command, "rw");

	// Generate random test string between 0 and 999 characters
	srand((unsigned int) time(NULL));
	const int length = rand() % 1000;
	char *testString = (char *) malloc((length+1)*sizeof(char));
	char *s;
	// Generate random ASCII characters while avoiding '\0'
	for (s = testString; s < testString+length; s++)
		*s = 1 + (rand() % 128);
	*s = '\0';

	// If this gives problems, wrap this in a loop similar to read
	int ret = pipe.Write(testString);
	CPPUNIT_ASSERT_EQUAL(length, ret);

	// Keep reading there there is still data to be read
	const int bufferLength = 1000;
	char buffer[bufferLength];
	char *bufferEnd = buffer + bufferLength + 1;
	s = buffer;
	int charsRead = 0;
	while (charsRead < length && s < bufferEnd) {
		ret = pipe.Read(s, bufferEnd-s);
		charsRead += ret;
		s += ret;
	}
	CPPUNIT_ASSERT(s < bufferEnd);
	*s = '\0';
	CPPUNIT_ASSERT_EQUAL(length, charsRead);
	CPPUNIT_ASSERT_EQUAL(std::string(testString), std::string(buffer));

	free(testString);
	pipe.Close();

	// Currently, this creates a debug error on Visual Studio 2008. I should look at this.
	ret = pipe.Write(testString);
	CPPUNIT_ASSERT_EQUAL(-1, ret);
	ret = pipe.Read(buffer, length);
	CPPUNIT_ASSERT_EQUAL(-1, ret);
}

CPPUNIT_TEST_SUITE_REGISTRATION(osaPipeTest);
