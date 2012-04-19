/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s): Marcin Balicki
  Created on: 2011-02-10

  (C) Copyright 2011 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include "cdpPlayerPlot2D.h"


#include <math.h>
#include <QMenu>
#include <QFileDialog>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QtGui>
#include <cisstOSAbstraction/osaGetTime.h>


#include <iostream>
#include <sstream>

CMN_IMPLEMENT_SERVICES(cdpPlayerPlot2D);


// Add/Remove Signal Window
addSignalWindow::addSignalWindow(QWidget *parent)
: QWidget(parent)
{
    grid = new QGridLayout;
    setLayout(grid);
    
    setWindowTitle(tr("Group Boxes"));
    resize(480, 320);
}

void addSignalWindow::AddWidget(QWidget *widget, int x, int y){
    this->grid->addWidget(widget,x,y);
    setLayout(grid);
    return;
}


cdpPlayerPlot2D::cdpPlayerPlot2D(const std::string & name, double period):
    cdpPlayerBase(name, period)
{

    //Test
    
    // create the user interface
    ExWidget.setupUi(&Widget);
    mainWidget = new QWidget();
    ScaleZoom = new QDoubleSpinBox(mainWidget);
    ScaleZoom->setValue(1);
    ZoomInOut = new QLabel(mainWidget);
    ZoomInOut->setText("Set Visualization Scale");
    ScaleZoom->setMaximum(9999);
    
    SignalButton = new QPushButton(tr("&Add/Remove Signal"),mainWidget);

    // create the user interface
    Plot = new vctPlot2DOpenGLQtWidget(mainWidget);
    Plot->SetNumberOfPoints(100);

    
    // TODO: Set default Trace here, thifs should be removed for multi-Scale architecture
    cdpPlayerPlot2D::Plot2DScale * traceElement = new cdpPlayerPlot2D::Plot2DScale;   
    traceElement->scalePointer = Plot->AddScale("Scale");
    traceElement->TimeFieldName =  "TimeStamp";
    //traceElement->DataFieldName = "TipForceNorm_Nm";
    this->Scales.push_back(*traceElement);

    VerticalLinePointer = Plot->AddVerticalLine("Scale-X");

    CentralLayout = new QGridLayout(mainWidget);

    CentralLayout->setContentsMargins(0, 0, 0, 0);
    CentralLayout->setRowStretch(0, 1);
    CentralLayout->setColumnStretch(1, 1);
    CentralLayout->addWidget(Plot, 0, 0, 1, 2);
    CentralLayout->addWidget(ScaleZoom, 1, 1, 1, 2);
    CentralLayout->addWidget(ZoomInOut, 1, 0, 1, 1);
    CentralLayout->addWidget(SignalButton, 2,0,1,2);

    CentralLayout->addWidget(this->GetWidget(),3,0,4,4);
    mainWidget->resize(300,500);

    // Add elements to state table
    StateTable.AddData(ZoomScaleValue,  "ZoomScale");
    StateTable.AddData(VectorIndex,  "VectorIndex");

    mtsInterfaceProvided * provided = AddInterfaceProvided("Provides2DPlot");
    if (provided) {
        provided->AddCommandReadState(StateTable, ZoomScaleValue,         "GetZoomScale");
        provided->AddCommandReadState(StateTable, VectorIndex,        "GetVectorIndex");
        provided->AddCommandWrite(&cdpPlayerPlot2D::SetVectorIndex, this, "SetVectorIndex", mtsInt() );
    }
    // Connect to ourself, for Qt Thread

    mtsInterfaceRequired * interfaceRequired = AddInterfaceRequired("Get2DPlotStatus");
    if (interfaceRequired) {        
        interfaceRequired->AddFunction("GetZoomScale", Plot2DAccess.GetZoomScale);        
        interfaceRequired->AddFunction("GetVectorIndex", Plot2DAccess.GetVectorIndex);
        interfaceRequired->AddFunction("SetVectorIndex",  Plot2DAccess.WriteVectorIndex);        
    }

    ZoomScaleValue = 1;

    window.show();
    window.setVisible(false);
    //// Add Parser Thread
    //taskManager = mtsTaskManager::GetInstance();
    //taskManager->AddComponent(&Parser);
}


cdpPlayerPlot2D::~cdpPlayerPlot2D()
{
    // cleanup
    //taskManager->KillAll();
    //taskManager->Cleanup();
}


void cdpPlayerPlot2D::MakeQTConnections(void)
{
    QObject::connect(ExWidget.PlayButton, SIGNAL(clicked()),
                     this, SLOT(QSlotPlayClicked()));

    QObject::connect(ExWidget.TimeSlider, SIGNAL(sliderMoved(int)),
                     this, SLOT(QSlotSeekSliderMoved(int)));

    QObject::connect(ExWidget.SyncCheck, SIGNAL(clicked(bool)),
                     this, SLOT(QSlotSyncCheck(bool)));

    QObject::connect(ExWidget.StopButton, SIGNAL(clicked()),
                     this, SLOT(QSlotStopClicked()));

    QObject::connect(ExWidget.SetSaveStartButton, SIGNAL(clicked()),
                     this, SLOT(QSlotSetSaveStartClicked()));

    QObject::connect(ExWidget.SetSaveEndButton, SIGNAL(clicked()),
                     this, SLOT(QSlotSetSaveEndClicked()));

    QObject::connect(ExWidget.OpenFileButton, SIGNAL(clicked()),
                     this, SLOT(QSlotOpenFileClicked()));

    QObject::connect(ScaleZoom , SIGNAL(valueChanged(double)),
                     this, SLOT(QSlotSpinBoxValueChanged(double)));
    
    QObject::connect(SignalButton, SIGNAL(clicked()),
                     this, SLOT(QSlotAddRemoveSignalClicked()));
}


void cdpPlayerPlot2D::Configure(const std::string & CMN_UNUSED(filename))
{
    MakeQTConnections();
    Widget.setWindowTitle(QString::fromStdString(GetName()));
    Widget.show();
    mainWidget->show();
    ResetPlayer();
    // Start Parser Thread
    //taskManager->CreateAll();
    //taskManager->StartAll();
}


void cdpPlayerPlot2D::Startup(void)
{
    LoadData();
    UpdateLimits();
}

void cdpPlayerPlot2D::StateExecutor(vctPlot2DBase::Trace *tracePointer){

    //update the model (load data) etc.
    if (State == PLAY) {

        double currentTime = TimeServer.GetAbsoluteTimeInSeconds();
        Time = currentTime - PlayStartTime.Timestamp() + PlayStartTime.Data;

        if (Time.Data > PlayUntilTime.Data)  {
            Time = PlayUntilTime;
            State = STOP;
        }
        else {
            if((ZoomScaleValue)  > (TimeBoundary-Time )*(0.8) && TimeBoundary <  PlayUntilTime.Data){
                size_t traceIndex = 0;
                // TODO: Modify this part for multi-Scale, so far, it is multi-Signal/Trace only 
                // we assume we have only one scale here, loop for every trace
                for(traceIndex = 0; traceIndex < this->Scales.at(0).scalePointer->Traces.size(); traceIndex++){
                    vctPlot2DBase::Trace * traceP = this->Scales.at(0).scalePointer->Traces.at(traceIndex);
                    // check if the name already exists
                    std::string name = traceP->GetName();
                    const std::string delimiter("-");
                    std::string dataFieldName;
                    size_t delimiterPosition = name.find(delimiter);
                    
                    dataFieldName = name.substr(delimiterPosition+1, name.size());                    
                    Parser.SetDataFieldForSearch(dataFieldName);
                    Parser.SetTimeFieldForSearch(this->Scales.at(0).TimeFieldName);
                    Parser.LoadDataFromFile(traceP, Time, ZoomScaleValue, false);                    
                }
                Parser.GetBoundary(tracePointer->GetParent(), TopBoundary, LowBoundary);
                TimeBoundary  =TopBoundary;
            }
            // update plot
            UpdatePlot();
        }
    }
    //make sure we are at the correct seek position.
    else if (State == SEEK) {
        //// Everything here should be moved to Qt thread since we have to re-alloc a new Plot object
        //size_t i = 0;
        if(LastTime.Data != Time.Data ){          
            LastTime = Time;
            PlayStartTime = Time;
            
            size_t traceIndex = 0;
            for(traceIndex = 0; traceIndex < this->Scales.at(0).scalePointer->Traces.size(); traceIndex++){
                vctPlot2DBase::Trace * traceP = this->Scales.at(0).scalePointer->Traces.at(traceIndex);
                std::string name = traceP->GetName();
                const std::string delimiter("-");
                std::string dataFieldName;
                size_t delimiterPosition = name.find(delimiter);
                
                dataFieldName = name.substr(delimiterPosition+1, name.size());                    
                Parser.SetDataFieldForSearch(dataFieldName);

                Parser.SetDataFieldForSearch(dataFieldName);
                Parser.SetTimeFieldForSearch(this->Scales.at(0).TimeFieldName);
                Parser.LoadDataFromFile(traceP, Time, ZoomScaleValue, true);                    
            }
            Parser.GetBoundary(tracePointer->GetParent(), TopBoundary, LowBoundary);
            
            TimeBoundary  =TopBoundary;
            // update plot
            UpdatePlot();
        }
    }
    else if (State == STOP) {
        //do Nothing

        //// update plot
        //UpdatePlot();
    }
    
}

void cdpPlayerPlot2D::Run(void)
{
    ProcessQueuedEvents();
    ProcessQueuedCommands();
    
    CS.Enter();

    size_t traceIndex;
    const size_t numberofTraces = this->Scales.at(0).scalePointer->Traces.size();//TracePointer->GetParent()->Traces.size();

    //TODO: This is one scale based, need to be changed for multi-Scale
    for(traceIndex = 0; traceIndex < numberofTraces; traceIndex++)
        this->StateExecutor(this->Scales.at(0).scalePointer->Traces[traceIndex]);
        //this->StateExecutor(TracePointer->GetParent()->Traces[traceIndex]);
    
    CS.Leave();
    //now display updated data in the qt thread space.
    if (Widget.isVisible()) {
        emit QSignalUpdateQT();
    }
}


void cdpPlayerPlot2D::UpdatePlot(void)
{
    double timeStamp = 0.0;
    double ScaleValue = 0.0; 
    mtsInt index;

    Plot2DAccess.GetVectorIndex(index);       
    Plot2DAccess.GetZoomScale(ScaleValue);

    Plot->SetContinuousFitX(false);    
    Plot->FitX(Time.Data-ScaleValue-PlayerDataInfo.DataStart() ,  Time.Data+ScaleValue-PlayerDataInfo.DataStart(), 0);
    VerticalLinePointer->SetX(Time.Data-PlayerDataInfo.DataStart());
    // UpdateGL should be called at Qt thread
}

//in QT thread space
void cdpPlayerPlot2D::UpdateQT(void)
{
    mtsDouble timevalue;
    CS.Enter();
    //BaseAccess.GetTime(timevalue);
    timevalue = Time;
    if (State == PLAY) {
        //Display the last datasample before Time.
        ExWidget.TimeSlider->setValue((int)timevalue.Data);
        //update Plot in Qt Thread
        if(Plot)
            Plot->updateGL();
    }    
    else if (State == STOP) {
        //Optional: Test if the data needs to be updated:
        ExWidget.TimeSlider->setValue((int)timevalue.Data);
        //update Plot in Qt Thread
        if(Plot)
            Plot->updateGL();
    }
    else if (State == SEEK) {     
        //Optional: Test if the data needs to be updated:
        ExWidget.TimeSlider->setValue((int)timevalue.Data);
        //update Plot in Qt Thread
        if(Plot)
            Plot->updateGL();
    }   
    CS.Leave();
    ExWidget.TimeLabel->setText(QString::number(timevalue.Data,'f', 3));
}


void cdpPlayerPlot2D::Play(const mtsDouble & time)
{
    if (Sync) {
        CMN_LOG_CLASS_RUN_DEBUG << "Play " << PlayStartTime << std::endl;
        State = PLAY;
        PlayUntilTime = PlayerDataInfo.DataEnd();
        PlayStartTime = time;
    }
}


void cdpPlayerPlot2D::Stop(const mtsDouble & time)
{
    if (Sync) {
        CMN_LOG_CLASS_RUN_DEBUG << "Stop " << time << std::endl;
        PlayUntilTime = time;
    }
}


void cdpPlayerPlot2D::Seek(const mtsDouble & time)
{
    static mtsDouble lasttime =0 ;
    if (Sync && lasttime.Data != time.Data) {
        CMN_LOG_CLASS_RUN_DEBUG << "Seek " << time << std::endl;

        State = SEEK;
        PlayUntilTime = PlayerDataInfo.DataEnd();
        // this will cause state table write command overflow
        //BaseAccess.WriteTime(time);
        Time = time;
        lasttime = time;
    }
}


void cdpPlayerPlot2D::Save(const cdpSaveParameters & saveParameters)
{
    if (Sync) {
        CMN_LOG_CLASS_RUN_DEBUG << "Save " << saveParameters << std::endl;
    }
}


void cdpPlayerPlot2D::Quit(void)
{
    CMN_LOG_CLASS_RUN_DEBUG << "Quit" << std::endl;
    this->Kill();
}



void cdpPlayerPlot2D::DataCheckBoxStateChanged(int CMN_UNUSED(state)){
    // Update Data Signal List 
    for(size_t i = 0; i < window.DataCheckBox.size();i++){
        // TODO: modify this part for multi-Scale, it is now multi-Signal/Tracs.
        if(window.DataCheckBox.at(i)->isChecked()){
            this->Scales.at(0).scalePointer->AddSignal(this->Scales.at(0).scalePointer->GetName() + "-" + DataList.at(i));
        }else{
            this->Scales.at(0).scalePointer->RemoveSignal(this->Scales.at(0).scalePointer->GetName() + "-" + DataList.at(i));
        }
    }
}
void cdpPlayerPlot2D::TimeRadioButtonStateChanged(int CMN_UNUSED(state)){
    // Change Time-base
    
}

void cdpPlayerPlot2D::QSlotPlayClicked(void)
{
    mtsDouble playTime;
    BaseAccess.GetTime(playTime);    
    playTime.Timestamp() = TimeServer.GetAbsoluteTimeInSeconds();

    if (Sync) {
        PlayRequest(playTime);
    } else {
        //not quite thread safe, if there is mts play call this can be corrupt.
        State = PLAY;
        PlayUntilTime = PlayerDataInfo.DataEnd();
        PlayStartTime = playTime;
    }
}


void cdpPlayerPlot2D::QSlotSeekSliderMoved(int c)
{
    mtsDouble t = c;
    static mtsDouble lasttime;
    if(lasttime.Data == t.Data)
        return; 
    else
        lasttime = t;

    if (Sync) {
        SeekRequest(t);       
    }         
    State = SEEK;      
    Time = t ;
    PlayUntilTime = PlayerDataInfo.DataEnd();
}


void cdpPlayerPlot2D::QSlotSyncCheck(bool checked)
{
    Sync = checked;
}


void cdpPlayerPlot2D::QSlotStopClicked(void)
{
    mtsDouble now = Time;

    if (Sync) {
        StopRequest(now);
    } else {
        PlayUntilTime = now;
    }
}

void cdpPlayerPlot2D::QSlotAddRemoveSignalClicked(void){    
    // Update Data Signal List 

    for(size_t j = 0; j < this->Scales.at(0).scalePointer->Traces.size(); j++){
        std::string traceName = this->Scales.at(0).scalePointer->Traces.at(j)->GetName();
        for(size_t i = 0; i < DataList.size();i++){
            const std::string delimiter("-");
            std::string dataFieldName;
            std::string checkBoxName = window.DataCheckBox.at(i)->text().toAscii().data();
            size_t delimiterPosition = traceName.find(delimiter);
            
            dataFieldName = traceName.substr(delimiterPosition+1, traceName.size()); 
            // TODO: modify this part for multi-Scale, it is now multi-Signal/Tracs.
            if(DataList.at(i).find(dataFieldName) != std::string::npos){
                window.DataCheckBox.at(i)->setChecked(true);
            }
        }
    }
    
    for(size_t i = 0 ; i < TimeList.size(); i++){
        if(TimeList.at(i).find(this->Scales.at(0).TimeFieldName) != std::string::npos)
            window.TimeRadio.at(i)->setChecked(true);
        
    }
    window.setVisible(true);
    
}


void cdpPlayerPlot2D::LoadData(void)
{
    //PlayerDataInfo.DataStart() = 1297723451.415;
    //PlayerDataInfo.DataEnd() = 1297723900.022;


    if (Time.Data < PlayerDataInfo.DataStart()) {
        Time = PlayerDataInfo.DataStart();
    }

    if (Time.Data > PlayerDataInfo.DataEnd()) {
        Time = PlayerDataInfo.DataEnd();
    }

    //This is the standard.
    PlayUntilTime = PlayerDataInfo.DataEnd();
    Time =  PlayerDataInfo.DataStart();

    UpdatePlayerInfo(PlayerDataInfo);
}

void cdpPlayerPlot2D::CreateSignalBox(void){
    QVBoxLayout *vboxData = new QVBoxLayout;
    QGroupBox *dataFieldAdded = new QGroupBox(tr("Data Field"));
    dataFieldAdded->setFlat(true);
    for(size_t i = 0; i < DataList.size(); i++){
        QCheckBox *checkBox = new QCheckBox(DataList.at(i).c_str());
        window.DataCheckBox.push_back(checkBox);
        
        QObject::connect(checkBox, SIGNAL(stateChanged(int)),
                         this, SLOT(DataCheckBoxStateChanged(int)));
        
        vboxData->addWidget(checkBox);
    }
    vboxData->addStretch(1);
    dataFieldAdded->setLayout(vboxData);
    window.AddWidget(dataFieldAdded,0,0);
    
 
    QVBoxLayout *vboxTime = new QVBoxLayout;
    QGroupBox *timeFieldAdded = new QGroupBox(tr("Time Field"));
    timeFieldAdded->setFlat(true);
    for(size_t i = 0; i < TimeList.size(); i++){
        QRadioButton *radio = new QRadioButton(TimeList.at(i).c_str());
        window.TimeRadio.push_back(radio);
        QObject::connect(radio, SIGNAL(stateChanged(int)),
                         this, SLOT(TimeRadioButtonStateChanged(int)));

        vboxTime->addWidget(radio);
    }
    vboxTime->addStretch(1);
    timeFieldAdded->setLayout(vboxTime);
    window.AddWidget(timeFieldAdded,0,1);
    
    //window.setVisible(true);
}


void cdpPlayerPlot2D::QSlotSetSaveStartClicked(void)
{
    mtsDouble timevalue;
    BaseAccess.GetTime(timevalue);
    ExWidget.SaveStartSpin->setValue(timevalue.Data);
}


void cdpPlayerPlot2D::QSlotSetSaveEndClicked(void)
{
    mtsDouble timevalue;
    BaseAccess.GetTime(timevalue);
    ExWidget.SaveEndSpin->setValue(timevalue.Data);
}


void cdpPlayerPlot2D::QSlotOpenFileClicked(void)
{
    // read data and update relatives
    OpenFile();
    LoadData();
    UpdateLimits();
    CreateSignalBox();
}


// Executed in Qt Thread
void cdpPlayerPlot2D::QSlotSpinBoxValueChanged(double value)
{
    ZoomScaleValue = value;
    ScaleZoom->setValue(ZoomScaleValue);    
    UpdatePlot();
}


// read data from file
void cdpPlayerPlot2D::OpenFile(void)
{
   QString result;


    result = QFileDialog::getOpenFileName(mainWidget, "Open File", tr("./"), tr("Desc (*.desc)"));
    if (!result.isNull()) {
        // read Data from file
	    ExtractDataFromStateTableCSVFile(result);

        Parser.GetBoundary(/*TracePointer*/this->Scales.at(0).scalePointer,TopBoundary,LowBoundary);
        TimeBoundary = TopBoundary;
        ResetPlayer();
        UpdatePlot();
        BaseAccess.WriteTime(LowBoundary);
    }
}


void cdpPlayerPlot2D::UpdateLimits()
{
    ExWidget.TimeSlider->setRange((int)PlayerDataInfo.DataStart(), (int)PlayerDataInfo.DataEnd());

    ExWidget.TimeStartLabel->setText(QString::number(PlayerDataInfo.DataStart(),'f', 3));
    ExWidget.TimeEndLabel->setText(QString::number(PlayerDataInfo.DataEnd(),'f', 3));

    ExWidget.SaveStartSpin->setRange(PlayerDataInfo.DataStart(), PlayerDataInfo.DataEnd());
    ExWidget.SaveEndSpin->setRange(PlayerDataInfo.DataStart(), PlayerDataInfo.DataEnd());
}


bool cdpPlayerPlot2D::ExtractDataFromStateTableCSVFile(QString & path){

    std::string Path(path.toStdString());
    size_t traceIndex = 0;

    // open header file
    Parser.ParseHeader(Path);
    Parser.GenerateIndex();
    // we sould name the file Path - .desc + .idx
    Parser.WriteIndexToFile("Parser.idx");
    Parser.GetHeaderField(TimeList, DataList);
    // Make the first element default value
    Scales.at(0).scalePointer->AddSignal(DataList.at(0));
    for(traceIndex = 0; traceIndex < this->Scales.at(0).scalePointer->Traces.size(); traceIndex++){
        vctPlot2DBase::Trace * traceP = this->Scales.at(0).scalePointer->Traces.at(traceIndex);
        std::string name = traceP->GetName();
        const std::string delimiter("-");
        std::string dataFieldName;
        size_t delimiterPosition = name.find(delimiter);
        
        dataFieldName = name.substr(delimiterPosition+1, name.size());                    
        
        Parser.SetDataFieldForSearch(dataFieldName);
        Parser.SetTimeFieldForSearch(this->Scales.at(0).TimeFieldName);
        Parser.LoadDataFromFile(traceP, 0.0, ZoomScaleValue, false);                    
    }
    
    Parser.GetBeginEndTime(PlayerDataInfo.DataStart(), PlayerDataInfo.DataEnd());        
    
    return true;
}

// reset player to initial state
//! TODO: make this function thread safe
void cdpPlayerPlot2D::ResetPlayer(void)
{    
    // set to maximun period we read
    //ZoomScaleValue = (PlayerDataInfo.DataStart() != 0) ? ((PlayerDataInfo.DataEnd() - PlayerDataInfo.DataStart()) / 2.0) : 1.0 ;    
    //if(TimeStamps->size() != 0)
    //    ZoomScaleValue = (TimeStamps->at(TimeStamps->size()-1) - TimeStamps->at(0))/2.0;   

    ScaleZoom->setValue(ZoomScaleValue);    
    BaseAccess.WriteTime(0.0);
    Plot2DAccess.WriteVectorIndex(0); 
}

