/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id$

Author(s):  Wen P. Liu, Anton Deguet
Created on: 2012-01-27

(C) Copyright 2009 Johns Hopkins University (JHU), All Rights
Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstNumerical/nmrRegistrationRigid.h>
#include <cisstParameterTypes/prmPositionJointGet.h>
#include <cisst3DUserInterface/ui3BehaviorBase.h>
#include <cisst3DUserInterface/ui3VisibleAxes.h>
#include <cisst3DUserInterface/ui3VTKStippleActor.h>
#include <cisst3DUserInterface/ui3VTKRenderer.h>
#include <cisst3DUserInterface/ui3ImagePlane.h>
#include <cisstStereoVision/svlFilterImageRegistrationGUI.h>
#include <cisstRobot/robManipulator.h>

#include <vtkActor.h>
#include <vtkAssembly.h>
#include <vtkAxesActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkSphereSource.h>
#include <vtkCubeSource.h>
#include <vtkContourFilter.h>
#include <vtkJPEGReader.h>
#include <vtkStripper.h>
#include <vtkVolumeReader.h>
#include <vtkPolyDataNormals.h>
#include <vtkVolume16Reader.h>
#include <vtkOutlineFilter.h>
#include <vtkImageActor.h>
#include <vtkLookupTable.h>
#include <vtkImageMapToColors.h>
#include <vtkPolyDataReader.h>
#include <vtkCellArray.h>
#include <vtkCubeSource.h>
#include <vtkCylinderSource.h>
#include <vtkFloatArray.h>
#include <vtkFollower.h>
#include <vtkPointData.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkCommand.h>
#include <vtkPointPicker.h>
#include <vtkPropPicker.h>
#include <vtkTexture.h>
#include <vtkTextActor3D.h>
#include <vtkTextProperty.h>
#include <vtkTextureMapToPlane.h>
#include <vtkVectorText.h>
#include <vtkLinearExtrusionFilter.h>


#include "vtkColorTransferFunction.h"
#include "vtkDICOMImageReader.h"
#include "vtkImageData.h"
#include "vtkImageResample.h"
#include "vtkMetaImageReader.h"
#include "vtkPiecewiseFunction.h"
#include "vtkPlanes.h"
#include "vtkProperty.h"
//#include "vtkVolume.h"
//#include "vtkVolumeProperty.h"
//#include "vtkXMLImageDataReader.h"
//#include "vtkSmartVolumeMapper.h"
#include <vtkSmartPointer.h>

// Always include last!
#include <ui3BehaviorsExport.h>

class ManualRegistrationSurfaceVisibleStippleObject;

class CISST_EXPORT ManualRegistration: public ui3BehaviorBase
{
public:
	enum VisibleObjectType {ALL = 0, NO_FIDUCIALS, MODEL, TUMOR, MODEL_ONLY, TUMOR_ONLY, TARGETS_REAL, TARGETS_VIRTUAL, FIDUCIALS_REAL, FIDUCIALS_VIRTUAL, CALIBRATION_REAL, CALIBRATION_VIRTUAL, TOOLS_REAL, NONE};
	enum BooleanFlagTypes {DEBUG = 0, VISIBLE, PREVIOUS_MAM, LEFT_BUTTON, RIGHT_BUTTON,
		CAMERA_PRESSED, CLUTCH_PRESSED, BOTH_BUTTON_PRESSED, UPDATE_FIDUCIALS, LEFT_BUTTON_RELEASED, RIGHT_BUTTON_RELEASED, TOOL_TRACKING_CORRECTION};
	enum Frame {WRIST=0, TIP, TOOLTIP, TOOLTOP, ECM, UI3, ECMRCM};
	enum ScopeType {ZERO=0, THIRTYUP, THIRTYDOWN};

	ManualRegistration(const std::string & name);
	~ManualRegistration();

	void Startup(void);
	void Cleanup(void) {}
	void ConfigureMenuBar(void);
	bool RunForeground(void);
	bool RunBackground(void);
	bool RunNoInput(void);
	void OnQuit(void);
	void OnStart(void);
	void Configure(const std::string & CMN_UNUSED(configFile)) {}
	bool SaveConfiguration(const std::string & CMN_UNUSED(configFile)) { return true; }
	inline ui3VisibleObject * GetVisibleObject(void) {
		return this->VisibleList;
	}
	VisibleObjectType ToggleVisibility(void);
	void PositionBack(void);
	void PositionHome(void);
	VisibleObjectType ToggleFiducials(void);
	void Register(VisibleObjectType type = ManualRegistration::ALL);
	VisibleObjectType GetFiducialToggle(void){return this->FiducialToggle;};
	VisibleObjectType GetVisibleToggle(void){return this->VisibleToggle;};
	VisibleObjectType SetVisibleToggle(VisibleObjectType visibility);

	void SetTransformation(vctFrm3 transformation)
	{
		ExternalTransformation = transformation;
		UpdateExternalTransformation = true;
	}

	void SetFiducial(vctFrm3 position, VisibleObjectType type, bool validity, int index = 0, bool overWrite = false);
	void UpdateOpacity(double delta);
	svlFilterImageRegistrationGUI *m_registrationGUIRight, *m_registrationGUILeft;
	bool SetContinuousRegistration(bool reg){ m_continuousRegister = reg; return m_continuousRegister;};
	bool SetVideoAugmentationVisibility(bool visible);
	bool GetVideoAugmentationVisibility(void){return m_videoAugmentationVisibility;};
	double SetWristTipOffset(bool flag);
	double UpdateExternalTransformationThreshold(double delta);
	void ComputeTRE(bool flag = true);

protected:
	void PrimaryMasterButtonCallback(const prmEventButton & event);
	void SecondaryMasterButtonCallback(const prmEventButton & event);
	void MasterClutchPedalCallback(const prmEventButton & payload);
	void UpdateButtonEvents(void);
	void ResetButtonEvents(void)
	{
		this->BooleanFlags[RIGHT_BUTTON] = false;
		this->BooleanFlags[LEFT_BUTTON] = false;
		this->BooleanFlags[BOTH_BUTTON_PRESSED] = false;
		this->BooleanFlags[RIGHT_BUTTON_RELEASED] = false;
		this->BooleanFlags[LEFT_BUTTON_RELEASED] = false;
	}
	void FollowMaster(void);
	void ComputeTransform(double pointa[3], double pointb[3],
		double point1[3], double point2[3],
		double object_displacement[3],
		double object_rotation[4]);
	void CameraControlPedalCallback(const prmEventButton & payload);
	void UpdateCameraPressed(void);
	void ToolTrackingCorrectionCallback(const vctFrm3 & payload);

private:
	void PositionDepth(void);
	void UpdateFiducials(void);

	vctFrm3 GetCurrentECMtoECMRCM(bool tool=false);
	vctFrm3 GetCurrentCartesianPositionSlave(void);
	vctFrm3 GetCurrentCorrectedCartesianPositionSlave(void);
	vctFrm3 GetCurrentPSM1byJointPositionDH();
	void UpdatePreviousPosition();
	bool ImportFiducialFile(const std::string & inputFile, VisibleObjectType type);
	void Tokenize(const std::string & str, std::vector<std::string> & tokens, const std::string & delimiters);
	void AddFiducial(vctFrm3 positionUI3, VisibleObjectType type);
	ManualRegistrationSurfaceVisibleStippleObject* FindClosestFiducial(vctFrm3 positionUI3, VisibleObjectType type, int& index, double distance = 4.0);
	bool RayRayIntersect(vctDouble3 p1,vctDouble3 p2,vctDouble3 p3,vctDouble3 p4,vctDouble3 &pa,vctDouble3 &pb);
	void GetFiducials(vctDynamicVector<vct3>& fiducialsVirtualECMRCM, vctDynamicVector<vct3>& fiducialsRealECMRCM,VisibleObjectType type, Frame frame,vctDynamicVector<vct3>& toolsRealECMRCM = vctDynamicVector<vct3>());
	void UpdateVisibleList(void);
	void OnStreamSample(svlSample* sample, int streamindex);

	StateType PreviousState;
	mtsFunctionRead GetCartesianPositionSlave;
	mtsFunctionRead GetCartesianPositionRCMSlave;
	mtsFunctionRead GetJointPositionPSM1;
	mtsFunctionRead GetJointPositionECM;
	typedef std::map<int, bool> FlagType;
	FlagType BooleanFlags;

	bool m_videoAugmentationVisibility, m_visionBasedTrackingVisibility;
	VisibleObjectType VisibleToggle;
	VisibleObjectType FiducialToggle;
	ScopeType EndoscopeType;
	double MeanTRE, MaxTRE, MeanToolTRE, MaxToolTRE, MeanTREProjection, MaxTREProjection, MeanTRETriangulation, MaxTRETriangulation;
	int TREFiducialCount;
	int calibrationCount;
	FILE *TRE, *TREProjection, *TRETool;

	vctDouble3 InitialMasterLeft, InitialMasterRight;
	typedef std::map<int, vctFrm3> ManualRegistrationFrameType;
	ManualRegistrationFrameType Frames;
	ui3VisibleAxes * Cursor;
	ui3VisibleList * VisibleList, * VisibleListECM, * VisibleListECMRCM, * VisibleListVirtual, * VisibleListVirtualGlobal, * VisibleListReal;
	typedef std::map<int, ManualRegistrationSurfaceVisibleStippleObject *> ManualRegistrationObjectType;
	ManualRegistrationObjectType VisibleObjects, VisibleObjectsVirtualFeedback, VisibleObjectsVirtualTumors, VisibleObjectsVirtualFiducials, VisibleObjectsRealFiducials, VisibleObjectsVirtualTargets, VisibleObjectsRealTargets,
		VisibleObjectsVirtualCalibration,VisibleObjectsRealCalibration,VisibleObjectsRealTool;

	// Provided Interfaces
	void SetupProvidedInterfaces(void);
	void UpdateProvidedInterfaces(void);
	mtsFunctionWrite SetPositionCartesianSlaveArm1;
	mtsFunctionRead GetPositionCartesianSlaveArm1;
	mtsFunctionWrite SetPositionCartesianECM_T_ECMRCM;
	mtsFunctionRead GetPositionCartesianECM_T_ECMRCM;
	mtsFunctionWrite SetPositionCartesianECMRCM_T_Virtual;
	mtsFunctionRead GetPositionCartesianECMRCM_T_Virtual;
	mtsFunctionWrite SetPositionCartesianECMRCM_T_VirtualGlobal;
	mtsFunctionRead GetPositionCartesianECMRCM_T_VirtualGlobal;

	void UpdateFiducialRegistration();
	void Update3DFiducialCameraPressed(void);
	void Follow();
	vctFrm3 ExternalTransformation;
	bool UpdateExternalTransformation;
	bool m_continuousRegister;

	//Tool tracking
	robManipulator *PSM1;
	vctFrm3 TTCorrectedTransformation;

	double wristToTipOffset;
	double m_externalTransformationThreshold;
};
