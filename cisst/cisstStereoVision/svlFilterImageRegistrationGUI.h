/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: svlFilterImageRegistrationGUI.h 3591 2012-04-05 04:52:30Z wliu25 $

  Author(s):  Wen P. Liu
  Created on: 2011

  (C) Copyright 2006-2007 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---

*/

#ifndef _svlFilterImageRegistrationGUI_h
#define _svlFilterImageRegistrationGUI_h

#include <cisstStereoVision/svlTypes.h>
#include <cisstStereoVision/svlFilterBase.h>
#include <cisstStereoVision/svlFilterInput.h>
#include <cisstStereoVision/svlOverlayObjects.h>
#include <cisstStereoVision/svlImageProcessing.h>
#include <limits>
#include <math.h>
#include <list>

// Always include last!
#include <cisstStereoVision/svlExport.h>

class CISST_EXPORT svlFilterImageRegistrationGUI : public svlOverlay
{
    //CMN_DECLARE_SERVICES(CMN_DYNAMIC_CREATION, CMN_LOG_ALLOW_DEFAULT);

public:
   enum VisibleObjectType {ALL = 0, NO_FIDUCIALS, MODEL, TUMOR, MODEL_ONLY, TUMOR_ONLY, TARGETS_REAL, TARGETS_VIRTUAL, FIDUCIALS_REAL, FIDUCIALS_VIRTUAL, CALIBRATION_REAL, CALIBRATION_VIRTUAL, TOOLS_REAL, NONE};
   enum FiducialColor {WHITE = 0, RED, GREEN, BLUE, YELLOW, BLACK, NUM_COLORS};

    typedef struct _ContourInternal {
	    int				ID;
		int				pixelMatchID;
		CvSeq*			contour;
		double			area;
		double			arcLength;
		//circularity = (cmnPI * currentArea) / (arclen * arclen) = 0.25
		CvBox2D			ellipse;
		cv::Point2f		pixelLocation;
		double			matchToCircle;
		double			matchToEllipse;
		CvScalar		meanColorRGB;
		CvScalar		meanColorHSV;
		FiducialColor   fiducialColor;
		CvScalar		color;
		bool			Valid;
		IplImage		*image; //used only for debug!!! remember to dealloc
    } ContourInternal;

    typedef struct _PointInternal {
	    int				ID;
		cv::Point2f		pixelLocation;
		svlPoint2D		sampleLocation;
		CvScalar		sampleHSV;
		CvScalar		sampleRGB;
		cv::Point3f		pointLocation;
        vct3x3          frame;
        double          timestamp;
		VisibleObjectType type;
		FiducialColor   fiducialColor;
		double			fiducialRadius;
		CvSeq*			largestContour;
		CvSeq*			fittedContour;
		CvRect			boundingBox;
		CvRect			expandedBoundingBox;

		svlPoint2D		Center;
		int				RadiusHoriz;
		int				RadiusVert;
		double			Angle;
		svlRGB			Color;
		bool			Fill;
		unsigned int	Thickness;
		unsigned int	VideoCh;
		bool			Valid;
		bool			isColor;

		cv::RotatedRect		ellipsoidRect; 

		ContourInternal	fiducialContour;
		std::list<ContourInternal> sortedContourList;
		cv::Point2f		projectedPixelLocation;
		cv::Point3f		kinematicPointLocation;

    } PointInternal;

    typedef struct _LineInternal {
	    int				ID;
		cv::Vec4i		pixelLocation;
		svlPoint2D		startPoint;
		svlPoint2D		endPoint;
		FiducialColor   fiducialColor;

		float			Angle;
		float			Length;
		unsigned int	VideoCh;
		bool			Valid;

    } LineInternal;

    typedef std::map<int, PointInternal> _PointCacheMap;
	typedef std::map<int, _LineInternal> _LineCacheMap;
	typedef std::map<int, _ContourInternal> _ContourCacheMap;

	svlFilterImageRegistrationGUI();
    svlFilterImageRegistrationGUI(unsigned int videoch,
                            bool visible);
    virtual ~svlFilterImageRegistrationGUI();

	void AddPoint(const svlPoint2D center,
            int radius_horiz,
            int radius_vert,
            double angle,
            svlRGB color,
			FiducialColor fidColor = WHITE,
			double fiducialRadius = 4.7625,//radii of 3/8" spheres
			unsigned int thickness = 1.0,
			VisibleObjectType type = FIDUCIALS_REAL,
            bool fill = true,
			unsigned int video_channel = SVL_LEFT);

	PointInternal GetPoint(int index, VisibleObjectType type = FIDUCIALS_REAL);
	std::map<int, PointInternal> GetPoints(void){return m_Points;};
	std::map<int, PointInternal> GetRegistrationPoints(void);
	std::map<int, PointInternal> GetCalibrationPoints(){return m_CalibrationPoints;};
	std::map<int, PointInternal> GetTargetPoints(){return m_TargetPoints;};
	int GetValidCount(VisibleObjectType type = FIDUCIALS_REAL);

	void SetCenter(const svlPoint2D center, int index = 0, VisibleObjectType type = FIDUCIALS_REAL);
	void SetPointLocation(const cv::Point3f location, int index = 0, VisibleObjectType type = FIDUCIALS_REAL);
	void SetKinematicPointLocation(const cv::Point3f location, int index = 0, VisibleObjectType type = FIDUCIALS_REAL);
	void SetProjectedPixelLocation(const cv::Point2f location, int index = 0, VisibleObjectType type = FIDUCIALS_REAL);
	void SetValidity(bool validity, int index = 0, VisibleObjectType type = FIDUCIALS_REAL);
	void SetValid(bool validity){this->Valid = validity;};

	bool ToggleUpdate(void)
	{
		if(this->updateToggle)
			this->updateToggle = false;
		else
			this->updateToggle = true;

		return this->updateToggle;
	};

	bool ToggleDebug(void)
	{
		if(this->m_debug)
			this->m_debug = false;
		else
			this->m_debug = true;

		return this->m_debug;
	};

	void SetCameraCalibration(svlCameraGeometry camera_geometry) {CameraGeometry = camera_geometry;};

	void ComputationalStereo(svlFilterImageRegistrationGUI* left);
	void Update3DPosition(const cv::Point3f location, int index);

	int UpdateThreshold(svlFilterImageRegistrationGUI::FiducialColor fidColor, int delta);
	double UpdateSearchRadius(svlFilterImageRegistrationGUI::FiducialColor fidColor, double delta);
	double UpdateHSVMinSaturation(svlFilterImageRegistrationGUI::FiducialColor fidColor, double delta);
	double UpdateHSVMaxSaturation(svlFilterImageRegistrationGUI::FiducialColor fidColor, double delta);
	double UpdateHSVMinHue(svlFilterImageRegistrationGUI::FiducialColor fidColor, double delta);
	double UpdateHSVMaxHue(svlFilterImageRegistrationGUI::FiducialColor fidColor, double delta);
	double UpdateHSVMinValue(svlFilterImageRegistrationGUI::FiducialColor fidColor, double delta);
	double UpdateHSVMaxValue(svlFilterImageRegistrationGUI::FiducialColor fidColor, double delta);
	double UpdateFiducialBoundaryThreshold(svlFilterImageRegistrationGUI::FiducialColor fidColor, double delta);
	double UpdateROIExpansion(double delta);
	int UpdateDilate(int delta){m_dilate+=delta; return m_dilate;};
	double Update3DThreshold(double delta){m_3DThreshold+=delta; return m_3DThreshold;};
	bool GetValidity(void){return Valid;};
	void ResetFiducialColors(void){this->m_resetPixelHSV = true;};
	void SetPixelHSV(const svlPoint2D center, int index, VisibleObjectType type = FIDUCIALS_REAL);

protected:
    virtual void DrawInternal(svlSampleImage* bgimage, svlSample* input);
	void Update(svlSampleImage* bgimage);
	bool IsValid(svlSampleImage* bgimage);
	bool check3DPositionValidity(void);

private:
	void updateSearch(svlSampleImage* bgimage, int index, double radius);
	void updatePoint(int index, std::vector<cv::Point2f> points);
	void getPixelRGB(svlSampleImage* bgimage, int x, int y, CvScalar &pixel);
	cv::Point2f computeMidpoint(int x1, int y1, int x2, int y2);
	svlFilterImageRegistrationGUI::_PointCacheMap::iterator getPointByColor(svlFilterImageRegistrationGUI::FiducialColor fidColor);
	bool isColorRGB(int threshold, unsigned char R, unsigned char G, unsigned char B, svlFilterImageRegistrationGUI::FiducialColor fidColor);
	float distanceBetweenTwoPoints ( int x1, int y1, int x2, int y2);
	float distanceBetweenTwoPoints ( int x1, int y1, int z1, int x2, int y2, int z2);
	IplImage* getThresholdedImage(svlSampleImage* bgimage, svlFilterImageRegistrationGUI::FiducialColor color);
	CvSeq* findLargestContour(svlSampleImage* bgimage, IplImage *thresholdImage, svlFilterImageRegistrationGUI::FiducialColor color);
	cv::Point2f findMoment(IplImage* imgThresh);
	cv::Point2f findMoment(CvArr* imgThresh);
	void updateFiducial(svlSampleImage* bgimage, svlFilterImageRegistrationGUI::FiducialColor color, CvRect* boundingBox, bool reset = false);
	CvRect expandBoundingBox(svlSampleImage* bgimage, CvRect input, double rescale);
	void getPixelHSV(svlSampleImage* bgimage, int x, int y, CvScalar &pixel);
	std::list<svlFilterImageRegistrationGUI::ContourInternal> categorizeContours(svlSampleImage* bgimage, IplImage *thresholdImage, svlFilterImageRegistrationGUI::FiducialColor fiducialColor);
	void samplePixelHSV(svlSampleImage* bgimage, svlFilterImageRegistrationGUI::PointInternal fiducial);
	void processContour(svlSampleImage* bgimage, CvSeq* approximateContour,svlFilterImageRegistrationGUI::ContourInternal* contour);
	void getContourColor(svlFilterImageRegistrationGUI::ContourInternal* contour, double R,double G,double B,double H, double S, double V);
	bool isColorHSV(int threshold, unsigned char H, unsigned char S, unsigned char V, svlFilterImageRegistrationGUI::FiducialColor fidColor);
	IplImage* svlFilterImageRegistrationGUI::getContourMask(svlSampleImage* bgimage, CvSeq* contour);
	void filterConcentricContours(std::list<svlFilterImageRegistrationGUI::ContourInternal>* contourList);
	vctDouble2 svlFilterImageRegistrationGUI::Wrld2Cam(const vctDouble3 & point3D);

	svlCameraGeometry CameraGeometry;
	bool updateToggle;

	_PointCacheMap		m_Points;
	_PointCacheMap		m_CalibrationPoints;
	_PointCacheMap		m_TargetPoints;
	_LineCacheMap		m_Lines;
	_ContourCacheMap	m_Contours;

	int m_thresholds[SVL_MAX_CHANNELS][svlFilterImageRegistrationGUI::NUM_COLORS];
	int m_tolerance[SVL_MAX_CHANNELS][svlFilterImageRegistrationGUI::NUM_COLORS];
	double m_searchRadii[SVL_MAX_CHANNELS][svlFilterImageRegistrationGUI::NUM_COLORS];
	double m_HSVMinSaturation[SVL_MAX_CHANNELS][svlFilterImageRegistrationGUI::NUM_COLORS];
	double m_HSVMaxSaturation[SVL_MAX_CHANNELS][svlFilterImageRegistrationGUI::NUM_COLORS];
	double m_fiducialBoundaryThreshold[SVL_MAX_CHANNELS][svlFilterImageRegistrationGUI::NUM_COLORS];
	double m_HSVMinHue[SVL_MAX_CHANNELS][svlFilterImageRegistrationGUI::NUM_COLORS];
	double m_HSVMaxHue[SVL_MAX_CHANNELS][svlFilterImageRegistrationGUI::NUM_COLORS];
	double m_HSVMinValue[SVL_MAX_CHANNELS][svlFilterImageRegistrationGUI::NUM_COLORS];
	double m_HSVMaxValue[SVL_MAX_CHANNELS][svlFilterImageRegistrationGUI::NUM_COLORS];
	double m_ROI_expansion[SVL_MAX_CHANNELS];
	int m_dilate;
	double m_3DThreshold;

	CvMemStorage *m_storageApprox;
	CvMemStorage *m_storage;
	
	bool Valid;
	bool m_resetPixelHSV;
	bool m_debug;
};

CMN_DECLARE_SERVICES_INSTANTIATION_EXPORT(svlFilterImageRegistrationGUI)

#endif // svlFilterImageRegistrationGUI_h

