/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */
/* $Id: main.cpp 78 2009-02-25 16:13:12Z adeguet1 $ */

#include <cisstCommon.h>
#include <cisstOSAbstraction.h>
#include <cisstMultiTask.h>

#include "sineTask.h"
#include "displayTask.h"
#include "displayUI.h"

using namespace std;

int main(int argc, char * argv[])
{
    // log configuration
    cmnLogger::SetLoD(10);
    cmnLogger::GetMultiplexer()->AddChannel(cout, 10);
    // add a log per thread
    osaThreadedLogFile threadedLog("example1-");
    cmnLogger::GetMultiplexer()->AddChannel(threadedLog, 10);
    // specify a higher, more verbose log level for these classes
    cmnClassRegister::SetLoD("sineTask", 10);
    cmnClassRegister::SetLoD("displayTask", 10);
    cmnClassRegister::SetLoD("mtsTaskInterface", 10);
    cmnClassRegister::SetLoD("mtsTaskManager", 10);

    // create our two tasks
    const double PeriodSine = 1 * cmn_ms; // in milliseconds
    const double PeriodDisplay = 50 * cmn_ms; // in milliseconds

    mtsTaskManager * taskManager = mtsTaskManager::GetInstance();
    if (argc == 1) {
        taskManager->SetTaskManagerMode(mtsTaskManager::TASK_MANAGER_SERVER);
    } else {
        taskManager->SetTaskManagerMode(mtsTaskManager::TASK_MANAGER_CLIENT);
    }

    while (1) {
        osaSleep(10 * cmn_ms);
    }

    return 0;

    /*
    try {
        if (argc == 1) {
            taskManager->SetTaskManagerMode(mtsTaskManager::TASK_MANAGER_SERVER);
        } else {
            taskManager->SetTaskManagerMode(mtsTaskManager::TASK_MANAGER_CLIENT);
        }

        // SetTaskManagerMode() 명령어 하나만으로 server/client 모두 TaskManager thread가
        // 생성되어 이미 돌고 있어야 하며 서로 연결 또한 성립되어 있어야 한다.
        // 만약 연결이 안되거나 하는 상황이면 
        // 1. TaskManager에서 예외 처리를 해서 에러 리포트를 하고, 예외를 re-throw.
        // 2. 본 main()에서도 예외 처리를 해서 에러 로깅 및 에러 처리.
        // 3. 연결이 되었다면 
        //   3-1. mtsTaskManager 내부 구조체를 각 server/client worker thread에서 접근할 수 있는지
        //   3-2. TaskManager끼리 통신이 잘 되는지
        //   를 확인해야만 한다.
        // 4. 3번까지 확인이 되면 multiTaskTutorialExample1을 네트워크로 분리시켜서 데모하도록 한다.

        //sineTask * sineTaskObject =
        //    new sineTask("SIN", PeriodSine);
        //
        //displayTask * displayTaskObject =
        //    new displayTask("DISP", PeriodDisplay);
        //displayTaskObject->Configure();

        //// add the tasks to the task manager
        //taskManager->AddTask(sineTaskObject);

        //taskManager->AddTask(displayTaskObject);
        //// connect the tasks, task.RequiresInterface -> task.ProvidesInterface
        //taskManager->Connect("DISP", "DataGenerator", "SIN", "MainInterface");

        //// generate a nice tasks diagram
        //std::ofstream dotFile("example1.dot"); 
        //taskManager->ToStreamDot(dotFile);
        //dotFile.close();

        //// create the tasks, i.e. find the commands
        //taskManager->CreateAll();
        //// start the periodic Run
        //taskManager->StartAll();

        //// wait until the close button of the UI is pressed
        //while (1) {
        //    osaSleep(100.0 * cmn_ms); // sleep to save CPU
        //    if (displayTaskObject->GetExitFlag()) {
        //        break;
        //    }
        //}
        //// cleanup
        //taskManager->KillAll();

        //osaSleep(PeriodDisplay * 2);
        //while (!sineTaskObject->IsTerminated()) osaSleep(PeriodDisplay);
        //while (!displayTaskObject->IsTerminated()) osaSleep(PeriodDisplay);

        while (1) {
            osaSleep(10 * cmn_ms);
        }
    //} catch (const Ice::Exception& e) {
    //    std::cout << "====== Client proxy error: " << e << std::endl;
    //} catch (const char * msg) {
    //    std::cout << "====== Client proxy error: " << msg << std::endl;
    } catch (...) {
        std::cout << "====== WTH!!! " << std::endl;
    }

    return 0;
    */
}

/*
  Author(s):  Ankur Kapoor, Peter Kazanzides, Anton Deguet
  Created on: 2004-04-30

  (C) Copyright 2004-2008 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/
