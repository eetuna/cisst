/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
$Id: svlFilterImageRegistrationGUI.cpp 3618 2012-05-04 19:52:31Z bvagvol1 $

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

#include <cisstStereoVision/svlFilterImageRegistrationGUI.h>
#include "svlImageProcessingHelper.h"
#include <sstream>

const static bool VERBOSE = false;

/***************************************************/
/*** svlFilterImageRegistrationGUI class ***/
/***************************************************/

svlFilterImageRegistrationGUI::svlFilterImageRegistrationGUI(unsigned int videoch,
															 bool visible) :
svlOverlay(videoch, visible)
{
	updateToggle = true;
	Valid = false;

	//set up search radii & RGB disparity thresholds
	//{WHITE = 0, RED, GREEN, BLUE, YELLOW};
	for(unsigned int i=SVL_LEFT; i <= SVL_RIGHT; i++)
	{
		m_ROI_expansion[i] = 0.0;
		for(unsigned int j=WHITE; j < NUM_COLORS; j++)
		{
			m_searchRadii[i][j] = 100.0;
			m_fiducialBoundaryThreshold[i][j] = 116.0;
			m_tolerance[i][j] = 10.0;
		}

		m_thresholds[i][WHITE] = 200;
		m_thresholds[i][RED] = 29;
		m_thresholds[i][GREEN] = 25;
		m_thresholds[i][BLUE] = 25;
		m_thresholds[i][YELLOW] = 30;
		m_fiducialBoundaryThreshold[VideoCh][YELLOW] = 0.0;
		m_tolerance[i][WHITE] = 30.0;
		m_tolerance[i][BLACK] = 30.0;

		//Orange  0-22
		//Yellow 22- 38
		//Green 38-75
		//Blue 75-130
		//Violet 130-160
		//Red 160-179

		//WHITE
		m_HSVMinHue[VideoCh][WHITE] = 104.0;
		m_HSVMinSaturation[VideoCh][WHITE] = 0.0;
		m_HSVMinValue[VideoCh][WHITE] = 100.0;
		m_HSVMaxHue[VideoCh][WHITE] = 140.0;
		m_HSVMaxSaturation[VideoCh][WHITE] = 100;
		m_HSVMaxValue[VideoCh][WHITE] = 256.0;
		//BLUE
		m_HSVMinHue[VideoCh][BLUE] = 75;
		m_HSVMinSaturation[VideoCh][BLUE] = 0.0;
		m_HSVMinValue[VideoCh][BLUE] = 60;
		m_HSVMaxHue[VideoCh][BLUE] = 130.0;
		m_HSVMaxSaturation[VideoCh][BLUE] = 256;
		m_HSVMaxValue[VideoCh][BLUE] = 256.0;
		//GREEN
		m_HSVMinHue[VideoCh][GREEN] = 38;
		m_HSVMinSaturation[VideoCh][GREEN] = 30.0;
		m_HSVMinValue[VideoCh][GREEN] = 60.0;
		m_HSVMaxHue[VideoCh][GREEN] = 75.0;
		m_HSVMaxSaturation[VideoCh][GREEN] = 256;
		m_HSVMaxValue[VideoCh][GREEN] = 256.0;
		//YELLOW
		m_HSVMinHue[VideoCh][YELLOW] = 22;
		m_HSVMinSaturation[VideoCh][YELLOW] = 0.0;
		m_HSVMinValue[VideoCh][YELLOW] = 130.0;
		m_HSVMaxHue[VideoCh][YELLOW] = 40;
		m_HSVMaxSaturation[VideoCh][YELLOW] = 256;
		m_HSVMaxValue[VideoCh][YELLOW] = 256.0;
	}

	m_storage = cvCreateMemStorage(0); //storage area for all contours
	m_storageApprox = cvCreateMemStorage(0); //storage area for all contours
	m_resetPixelHSV = false;
	m_dilate = 0;
	m_debug = false;
	m_3DThreshold = 0.1;
}

svlFilterImageRegistrationGUI::~svlFilterImageRegistrationGUI()
{
	cvReleaseMemStorage(&m_storageApprox);
	cvReleaseMemStorage(&m_storage);
}


void svlFilterImageRegistrationGUI::AddPoint(const svlPoint2D center,
											 int radius_horiz,
											 int radius_vert,
											 double angle,
											 svlRGB color,
											 FiducialColor fidColor,
											 double fiducialRadius,
											 unsigned int thickness,
											 VisibleObjectType type,
											 bool fill,
											 unsigned int video_channel)
{
	PointInternal point;

	point.Center = center;
	point.type = type;
	point.pixelLocation = cv::Point2f(center.x, center.y);
	point.pointLocation = cv::Point3f(center.x, center.y,0.0);
	point.RadiusHoriz = radius_horiz;
	point.RadiusVert = radius_vert;
	point.Angle = angle;
	point.Color = color;
	point.fiducialColor = fidColor;
	point.fiducialRadius = fiducialRadius;
	point.Thickness = thickness;
	point.Fill = fill;
	point.VideoCh = video_channel;
	point.Valid = true;
	point.isColor = false;
	point.ellipsoidRect = cv::RotatedRect();
	point.largestContour = NULL;
	point.fittedContour = NULL;
	point.boundingBox = cvRect(0,0,0,0);
	point.expandedBoundingBox = cvRect(0,0,0,0);
	point.sampleHSV = cvScalar(	(m_HSVMinSaturation[VideoCh][fidColor]+m_HSVMaxSaturation[VideoCh][fidColor])/2,
		(m_HSVMinHue[VideoCh][fidColor]+m_HSVMaxHue[VideoCh][fidColor])/2,
		(m_HSVMinValue[VideoCh][fidColor]+m_HSVMaxValue[VideoCh][fidColor])/2);
	point.sampleRGB = cvScalar(	(m_HSVMinSaturation[VideoCh][fidColor]+m_HSVMaxSaturation[VideoCh][fidColor])/2,
		(m_HSVMinHue[VideoCh][fidColor]+m_HSVMaxHue[VideoCh][fidColor])/2,
		(m_HSVMinValue[VideoCh][fidColor]+m_HSVMaxValue[VideoCh][fidColor])/2);

	switch(type)
	{
	case(FIDUCIALS_REAL):
		{
			point.ID = m_Points.size();
			m_Points[m_Points.size()] = point;
			if(VERBOSE)
				std::cout << "Adding a registration fiducial at index " << point.ID << std::endl;

		}
		break;
	case(CALIBRATION_REAL):
		{
			point.ID = m_CalibrationPoints.size();
			m_CalibrationPoints[m_CalibrationPoints.size()] = point;
			if(VERBOSE)
				std::cout << "Adding a calibration fiducial at index " << point.ID << std::endl;
		}
		break;
	case(TARGETS_REAL):
		{
			point.ID = m_TargetPoints.size();
			m_TargetPoints[m_TargetPoints.size()] = point;
			if(VERBOSE)
				std::cout << "Adding a target fiducial at index " << point.ID << std::endl;
		}
		break;
	default:
		return;
	}
}

svlFilterImageRegistrationGUI::PointInternal svlFilterImageRegistrationGUI::GetPoint(int index, VisibleObjectType type)
{
	switch(type)
	{
	case(FIDUCIALS_REAL):
		{
			if(index < m_Points.size())
			{
				return m_Points[index];
			}
		}
		break;
	case(CALIBRATION_REAL):
		{
			if(index < m_CalibrationPoints.size())
			{
				return m_CalibrationPoints[index];
			}
		}
		break;
	case(TARGETS_REAL):
		{
			if(index < m_TargetPoints.size())
			{
				return m_TargetPoints[index];
			}
		}
		break;
	default:
		break;
	}
}

void svlFilterImageRegistrationGUI::SetPixelHSV(const svlPoint2D center, int index, VisibleObjectType type)
{
	switch(type)
	{
	case(FIDUCIALS_REAL):
		{
			if(index < m_Points.size())
			{
				m_Points[index].sampleLocation = center;
				m_Points[index].pixelLocation = cv::Point2f(center.x, center.y);
				m_Points[index].pointLocation = cv::Point3f(center.x, center.y,0.0);
			}
		}
		break;
	default:
		return;
	}
}

void svlFilterImageRegistrationGUI::SetCenter(const svlPoint2D center, int index, VisibleObjectType type)
{
	switch(type)
	{
	case(FIDUCIALS_REAL):
		{
			if(index < m_Points.size())
			{
				m_Points[index].Center = center;
				m_Points[index].pixelLocation = cv::Point2f(center.x, center.y);
				m_Points[index].pointLocation = cv::Point3f(center.x, center.y,0.0);
				if(VERBOSE)
					std::cout << "Setting center of a fiducial real at index " << index << std::endl;
			}
		}
		break;
	case(TARGETS_REAL):
		{
			if(index < m_TargetPoints.size())
			{
				m_TargetPoints[index].Center = center;
				m_TargetPoints[index].pixelLocation = cv::Point2f(center.x, center.y);
				m_TargetPoints[index].pointLocation = cv::Point3f(center.x, center.y,0.0);
				//if(VERBOSE)
					std::cout << "Setting center of a target real at index " << index << std::endl;
			}
		}
		break;
	default:
		return;
	}
}

void svlFilterImageRegistrationGUI::SetKinematicPointLocation(const cv::Point3f location, int index, VisibleObjectType type)
{
	switch(type)
	{
	case(FIDUCIALS_REAL):
		{
			if(index < m_Points.size()) 
				m_Points[index].kinematicPointLocation = location;
		}
		break;
	default:
		return;
	}
}

void svlFilterImageRegistrationGUI::SetPointLocation(const cv::Point3f location, int index, VisibleObjectType type)
{
	switch(type)
	{
	case(FIDUCIALS_REAL):
		{
			if(index < m_Points.size()) 
				m_Points[index].pointLocation = location;
		}
		break;
	case(CALIBRATION_REAL):
		{
			if(index < m_CalibrationPoints.size()) 
				m_CalibrationPoints[index].pointLocation = location;
		}
		break;
	case(TARGETS_REAL):
		{
			if(index < m_TargetPoints.size()) 
				m_TargetPoints[index].pointLocation = location;
		}
		break;
	default:
		return;
	}
}

void svlFilterImageRegistrationGUI::SetProjectedPixelLocation(const cv::Point2f location, int index, VisibleObjectType type)
{
	switch(type)
	{
	case(FIDUCIALS_REAL):
		{
			if(index < m_Points.size()) 
				m_Points[index].projectedPixelLocation = location;
		}
		break;
	default:
		return;
	}
}

void svlFilterImageRegistrationGUI::SetValidity(bool validity, int index, VisibleObjectType type)
{
	switch(type)
	{
	case(FIDUCIALS_REAL):
		{
			if(index < m_Points.size()) 
			{
				m_Points[index].Valid = validity;
			}
		}
		break;
	case(CALIBRATION_REAL):
		{
			if(index < m_CalibrationPoints.size()) 
			{
				m_CalibrationPoints[index].Valid = validity;
			}
		}
		break;
	case(TARGETS_REAL):
		{
			if(index < m_TargetPoints.size()) 
			{
				m_TargetPoints[index].Valid = validity;
			}
		}
		break;
	default:
		return;
	}
}

std::map<int, svlFilterImageRegistrationGUI::PointInternal> svlFilterImageRegistrationGUI::GetRegistrationPoints()
{
	svlFilterImageRegistrationGUI::_PointCacheMap	m_RegistrationPoints;
	//don't return GREEN	
	for (svlFilterImageRegistrationGUI::_PointCacheMap::iterator iter = m_Points.begin(); iter != m_Points.end(); iter++) 
	{
		//if(iter->second.Valid && iter->second.fiducialColor != GREEN)//(iter->second.fiducialColor == YELLOW || iter->second.fiducialColor == WHITE || iter->second.fiducialColor == BLUE))
			m_RegistrationPoints[m_RegistrationPoints.size()] = iter->second;
	}
	return m_RegistrationPoints;
}

int svlFilterImageRegistrationGUI::GetValidCount(VisibleObjectType type)
{
	int count = 0;
	switch(type)
	{
		case(FIDUCIALS_REAL):
		{
			for (_PointCacheMap::iterator iter = m_Points.begin(); iter != m_Points.end(); iter++) 
			{
				if(iter->second.Valid)
					count++;
			}
		}
		break;
		case(TARGETS_REAL):
		{
			for (_PointCacheMap::iterator iter = m_TargetPoints.begin(); iter != m_TargetPoints.end(); iter++) 
			{
				if(iter->second.Valid)
					count++;
			}
		}
		break;
		default:
			break;

	}
	return count;
}

svlFilterImageRegistrationGUI::_PointCacheMap::iterator svlFilterImageRegistrationGUI::getPointByColor(svlFilterImageRegistrationGUI::FiducialColor fidColor)
{
	int count = 0;
	_PointCacheMap::iterator iter;
	for (iter = m_Points.begin(); iter != m_Points.end(); iter++) 
	{
		if(iter->second.Valid && iter->second.fiducialColor == fidColor)
		{
			break;
		}
	}
	return iter;
}

bool svlFilterImageRegistrationGUI::isColorHSV(int threshold, unsigned char H, unsigned char S, unsigned char V, svlFilterImageRegistrationGUI::FiducialColor fidColor)
{
	bool hue, saturation, value;
	hue = (H >= m_HSVMinHue[VideoCh][fidColor] && H <= m_HSVMaxHue[VideoCh][fidColor]);
	saturation = (S >= m_HSVMinSaturation[VideoCh][fidColor] && S <= m_HSVMaxSaturation[VideoCh][fidColor]);
	value = (V >= m_HSVMinValue[VideoCh][fidColor] && V <= m_HSVMaxValue[VideoCh][fidColor]);

	return hue && saturation && value;
}

bool svlFilterImageRegistrationGUI::isColorRGB(int threshold, unsigned char R, unsigned char G, unsigned char B, svlFilterImageRegistrationGUI::FiducialColor fidColor)
{
	switch(fidColor)
	{
	case(WHITE):
		{
			bool val = ((R >= (unsigned char)m_thresholds[VideoCh][WHITE] && G >= (unsigned char)m_thresholds[VideoCh][WHITE] && B >= (unsigned char)m_thresholds[VideoCh][WHITE]) && ((abs(R - B) <= threshold) && (abs(R - G) <= threshold) && (abs(B - G) <= threshold)));
#if VERBOSE
			if(!val && VideoCh == SVL_LEFT)
				std::cout << " WHITE " << R << ", " << G << ", " << B << std::endl;
#endif
			return val;
			break;
		}
	case(BLACK):
		{

			bool val = ((R <= (unsigned char)m_thresholds[VideoCh][BLACK] && G <= (unsigned char)m_thresholds[VideoCh][BLACK] && B <= (unsigned char)m_thresholds[VideoCh][BLACK]) && ((abs(R - B) <= threshold) && (abs(R - G) <= threshold) && (abs(B - G) <= threshold)));
#if VERBOSE
			if(!val && VideoCh == SVL_LEFT)
				std::cout << " BLACK " << r << ", " << g << ", " << b << std::endl;
#endif
			return val;
			break;
		}
	case(RED):
		{
			return ((R > threshold + G) && (R > threshold + B));
			break;
		}
	case(GREEN):
		{
			return ((G > threshold + R) && (G > threshold + B));
			break;
		}
	case(BLUE):
		{
			return ((B > threshold + R) && (B > threshold + G));
			break;
		}
	case(YELLOW):
		{
			return ((G > threshold + R) && (B > threshold + R));
			break;
		}
	default:
		return false;
		break;
	}

}

cv::Point2f svlFilterImageRegistrationGUI::computeMidpoint(int x1, int y1, int x2, int y2)
{
	return cv::Point2f((x1+x2)/2,(y1+y2)/2);
}

float svlFilterImageRegistrationGUI::distanceBetweenTwoPoints ( int x1, int y1, int x2, int y2)
{
	return std::sqrt( (double)((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) ) ;
}

float svlFilterImageRegistrationGUI::distanceBetweenTwoPoints ( int x1, int y1, int z1, int x2, int y2, int z2)
{
	return std::sqrt( (double)((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2)) ) ;
}

void svlFilterImageRegistrationGUI::getPixelHSV(svlSampleImage* bgimage, int x, int y, CvScalar &pixel)
{

	IplImage* img = bgimage->IplImageRef(VideoCh);
	uchar *data;
	// Convert the image into an HSV image
	IplImage* imgHSV = cvCreateImage(cvGetSize(img), 8, 3);
	cvCvtColor(img, imgHSV, CV_BGR2HSV);

	pixel = cvGet2D(imgHSV,y,x);

	cvReleaseImage(&imgHSV);
}

void svlFilterImageRegistrationGUI::getPixelRGB(svlSampleImage* bgimage, int x, int y, CvScalar &pixel)
{
	//int height, width, step, channels, i, j;
	//uchar *data;
	//IplImage* iplImage = bgimage->IplImageRef(VideoCh);
	//height = iplImage->height;
	//width = iplImage->width;
	//step = iplImage->widthStep;
	//channels = iplImage->nChannels;
	//data = (uchar *)iplImage->imageData;
	//int R = data[y*step+x*channels+0];
	//int G = data[y*step+x*channels+1];
	//int B = data[y*step+x*channels+2];
	//std::cout << "ROrig " << R << " GOrig " << G << " BOrig " << B << std::endl;

	pixel = cvGet2D(bgimage->IplImageRef(VideoCh),y,x);
}

bool svlFilterImageRegistrationGUI::IsValid(svlSampleImage* bgimage)
{
	bool valMidpoint0, valMidpoint1, valMidpoint2;
	valMidpoint0 = false;
	valMidpoint1 = false;
	valMidpoint2 = false;
	bool valColor = true;
	float point_x, point_y, line_x1, line_y1, line_x2, line_y2;
	unsigned char pixelRed, pixelGreen, pixelBlue;
	int height, width, step, channels, i, j;
	uchar *data;
	IplImage* iplImage = bgimage->IplImageRef(VideoCh);
	height = iplImage->height;
	width = iplImage->width;
	step = iplImage->widthStep;
	channels = iplImage->nChannels;
	data = (uchar *)iplImage->imageData;
	for(int index=0; index<m_Points.size(); index++)
	{
		if(m_Points[index].Valid && m_Points[index].fiducialColor != GREEN)
		{
			switch(m_Points[index].fiducialColor)
			{
				i = (int)m_Points[index].pixelLocation.y;
				j = (int)m_Points[index].pixelLocation.x;	
				pixelRed = data[i*step+j*channels+0];
				pixelGreen = data[i*step+j*channels+1];
				pixelBlue = data[i*step+j*channels+2];
			case(BLACK):
				{
					point_x = m_Points[index].pixelLocation.x;
					point_y = m_Points[index].pixelLocation.y;
					valMidpoint0 = true;
					break;
				}
			case(WHITE):
				{
					line_x1 = m_Points[index].pixelLocation.x;
					line_y1 = m_Points[index].pixelLocation.y;
					valMidpoint1 = true;
					break;
				}
			case(YELLOW):
				{
					line_x2 = (float)m_Points[index].pixelLocation.x;
					line_y2 = (float)m_Points[index].pixelLocation.y;
					valMidpoint2 = true;
					break;
				}
			default:
				break;
			}

			valColor = valColor && m_Points[index].isColor;
			//center of the tracked fiducial may be saturated by light source and will not be the color of the bead
			//val = val && isColor, set when we find thresholds+contours
		}
	}

	// CHECK MIDPOINT OF FIDUCIALS
	// for now only if all GBY fiducials are present
	if(valMidpoint0 && valMidpoint1 && valMidpoint2)
	{
		// B & G
		cv::Point2f midpoint = computeMidpoint(point_x,point_y, line_x1, line_y1);
		i = midpoint.y;
		j = midpoint.x;
		pixelRed = data[i*step+j*channels+0];
		pixelGreen = data[i*step+j*channels+1];
		pixelBlue = data[i*step+j*channels+2];
		valMidpoint0 = isColorRGB(10, pixelRed, pixelGreen, pixelBlue, GREEN);

		// B & Y
		midpoint = computeMidpoint(point_x,point_y, line_x2, line_y2);
		i = midpoint.y;
		j = midpoint.x;
		pixelRed = data[i*step+j*channels+0];
		pixelGreen = data[i*step+j*channels+1];
		pixelBlue = data[i*step+j*channels+2];
		valMidpoint1 = isColorRGB(10, pixelRed, pixelGreen, pixelBlue, GREEN);

		// G & Y
		midpoint = computeMidpoint(line_x1,line_y1, line_x2, line_y2);
		i = midpoint.y;
		j = midpoint.x;
		pixelRed = data[i*step+j*channels+0];
		pixelGreen = data[i*step+j*channels+1];
		pixelBlue = data[i*step+j*channels+2];
		valMidpoint2 = isColorRGB(10, pixelRed, pixelGreen, pixelBlue, GREEN);
	}
	// valid if tracking on fiducial by color
	// and 2/3 midpoints are green
	return valColor && ((valMidpoint0 && valMidpoint1) || (valMidpoint1 && valMidpoint2) || (valMidpoint0 && valMidpoint2));
}

vctDouble2 svlFilterImageRegistrationGUI::Wrld2Cam(const vctDouble3 & point3D)
{
	vctDouble4 worldPoints(point3D[0],point3D[1],point3D[2],1.0);
    vctDouble4 result;
	vctFrm3 staticECM_T_UI3;
	vctDoubleFrm4x4 ECM_T_UI3;
	vctDouble2 resultPixel;
	svlCameraGeometry::Intrinsics in;
	svlCameraGeometry::Extrinsics ex;
    staticECM_T_UI3.Rotation().From(vctAxAnRot3(vctDouble3(1.0,0.0,0.0), cmnPI));
	ECM_T_UI3.From(staticECM_T_UI3.Rotation(),staticECM_T_UI3.Translation());


	ex = CameraGeometry.GetExtrinsics(SVL_RIGHT);
	if(VideoCh == SVL_LEFT)
	{
		in = CameraGeometry.GetIntrinsics(SVL_LEFT);
	}
	else
	{
		in = CameraGeometry.GetIntrinsics(SVL_RIGHT);

	}
    vctDoubleFrm4x4 extrinsics;
	extrinsics.Identity();
	
	if(VideoCh == SVL_LEFT)
	{
		//extrinsics.Element(0,3) =  -ex.frame.Translation()[0]/2;
		//extrinsics.Element(1,3) =  ex.frame.Translation()[1];
		//extrinsics.Element(2,3) =  ex.frame.Translation()[2];
		//std::cout << " extrinsics " << ex.frame << std::endl;
		extrinsics = ex.frame.Inverse();
		//std::cout << " extrinsics " << extrinsics << std::endl;
		extrinsics.Element(0,0) =  1.0;
		extrinsics.Element(0,1) =  0.0;
		extrinsics.Element(0,2) =  0.0;
		extrinsics.Element(0,3) =  -ex.frame.Translation()[0]/2;
		//extrinsics.Element(1,3) =  0.0;//-ex.frame.Translation()[1];
		//extrinsics.Element(2,3) =  0.0;//-ex.frame.Translation()[2];

		//std::cout << " extrinsics " << extrinsics << std::endl;
	}
	else
	{
		//extrinsics.Translation() = ex.frame.Translation();
		extrinsics.Element(0,3) =  ex.frame.Translation()[0]/2;//extrinsics.Element(0,3)/2;

	}

	result = extrinsics*ECM_T_UI3*worldPoints;
    resultPixel[0] = (in.fc[0] * result[0]/result[2] + in.cc[0])/2;
    resultPixel[1] = (in.fc[1] * result[1]/result[2] + in.cc[1])/2;

    return resultPixel;

}

void svlFilterImageRegistrationGUI::ComputationalStereo(svlFilterImageRegistrationGUI* left)
{

	double scale = 1.0;

	svlCameraGeometry::Intrinsics intrinsicsL = CameraGeometry.GetIntrinsics(SVL_LEFT);
	svlCameraGeometry::Intrinsics intrinsicsR = CameraGeometry.GetIntrinsics(SVL_RIGHT);
	svlCameraGeometry::Extrinsics extrinsicsL = CameraGeometry.GetExtrinsics(SVL_LEFT);
	svlCameraGeometry::Extrinsics extrinsicsR = CameraGeometry.GetExtrinsics(SVL_RIGHT);

	intrinsicsL.fc[0] = intrinsicsL.fc[0]*scale;
	intrinsicsL.fc[1] = intrinsicsL.fc[1]*scale;
	intrinsicsL.cc[0] = intrinsicsL.cc[0]*scale;
	intrinsicsL.cc[1] = intrinsicsL.cc[1]*scale;
	intrinsicsR.fc[0] = intrinsicsR.fc[0]*scale;
	intrinsicsR.fc[1] = intrinsicsR.fc[1]*scale;
	intrinsicsR.cc[0] = intrinsicsR.cc[0]*scale;
	intrinsicsR.cc[1] = intrinsicsR.cc[1]*scale;

	float bl           = static_cast<float>(extrinsicsL.T.X() - extrinsicsR.T.X());
	float rightcamposx = static_cast<float>(extrinsicsR.T.X());
	float fl           = static_cast<float>(intrinsicsR.fc[0]);
	float ppx          = static_cast<float>(intrinsicsR.cc[0]);
	float ppy          = static_cast<float>(intrinsicsR.cc[1]);
	float disp_corr    = static_cast<float>(intrinsicsL.cc[0]) - ppx;

	float x,y,z,disp, ratio;
	x = static_cast<float>(0.0);
	y = static_cast<float>(0.0);
	z = static_cast<float>(0.0);
	disp = static_cast<float>(0.0);
	ratio = static_cast<float>(0.0);

	cv::Point2f pointRight;
	cv::Point2f pointLeft;

	int index = 0;

	for (_PointCacheMap::iterator iter = m_Points.begin(); iter != m_Points.end(); iter++) 
	{
		bl           = static_cast<float>(extrinsicsL.T.X() - extrinsicsR.T.X());
		rightcamposx = static_cast<float>(extrinsicsR.T.X());
		fl           = static_cast<float>(intrinsicsR.fc[0]);
		ppx          = static_cast<float>(intrinsicsR.cc[0]);
		ppy          = static_cast<float>(intrinsicsR.cc[1]);
		disp_corr    = static_cast<float>(intrinsicsL.cc[0]) - ppx;

		pointRight = 2.0*(iter->second).pixelLocation;
		pointLeft = 2.0*left->GetPoint(index).pixelLocation;

		// assume rectified
		disp = static_cast<float>(pointLeft.x - pointRight.x);
		disp = disp - disp_corr;
		if (disp < 0.01f) disp = 0.01f;
		ratio = bl / disp;

		x = (float)((pointRight.x - ppx) * ratio - rightcamposx/2); // X
		y = (float)((pointRight.y - ppy) * ratio);               // Y
		z = (float)(fl         * ratio);                // Z

#if VERBOSE
		std::cout << "Index : " << index << std::endl;
		std::cout << "PRight (" << pointRight.x << ", " << pointRight.y << ")"<< std::endl;
		std::cout << "PLeft (" << pointLeft.x << ", " << pointLeft.y << ")"<< std::endl;
		std::cout << "rightcamposx " << rightcamposx << std::endl;
		std::cout << "baseline " << bl << std::endl;
		std::cout << "disparity " << disp << std::endl;
		std::cout << "disp_corr " << disp_corr << std::endl; 
		std::cout << "ratio " << ratio << std::endl; 
		std::cout << "Point (" << x << ", " << -y << ", " << -z<< ")"<< std::endl;
#endif
		//update
		(iter->second).pointLocation = cv::Point3f(x,-y,-z);
		//vctDouble2 projection = Wrld2Cam(vctDouble3((iter->second).pointLocation.x,(iter->second).pointLocation.y,(iter->second).pointLocation.z));
		//(iter->second).projectedPixelLocation = cv::Point2f(projection.X(),projection.Y());
		//std::cout << "Right (" << (iter->second).pixelLocation.x << ", " << (iter->second).pixelLocation.y << ")" << "projected (" << (iter->second).projectedPixelLocation.x << ", " << (iter->second).projectedPixelLocation.y << ")"<< std::endl;
		//projection = left->Wrld2Cam(vctDouble3((iter->second).pointLocation.x,(iter->second).pointLocation.y,(iter->second).pointLocation.z));
		//left->SetProjectedPixelLocation(cv::Point2f(projection.X(),projection.Y()),index);
		//std::cout << "Left (" << left->GetPoint(index).pixelLocation.x << ", " << left->GetPoint(index).pixelLocation.y << ")" << "projected (" << left->GetPoint(index).projectedPixelLocation.x << ", " << left->GetPoint(index).projectedPixelLocation.y << ")"<< std::endl;
		left->SetPointLocation(cv::Point3f(x,-y,-z),index);
		index++;

	}

	index = 0;
	for (_PointCacheMap::iterator iter = m_TargetPoints.begin(); iter != m_TargetPoints.end(); iter++) 
	{
		bl           = static_cast<float>(extrinsicsL.T.X() - extrinsicsR.T.X());
		rightcamposx = static_cast<float>(extrinsicsR.T.X());
		fl           = static_cast<float>(intrinsicsR.fc[0]);
		ppx          = static_cast<float>(intrinsicsR.cc[0]);
		ppy          = static_cast<float>(intrinsicsR.cc[1]);
		disp_corr    = static_cast<float>(intrinsicsL.cc[0]) - ppx;

		pointRight = 2.0*(iter->second).pixelLocation;
		pointLeft = 2.0*left->GetPoint(index,TARGETS_REAL).pixelLocation;

		// assume rectified
		disp = static_cast<float>(pointLeft.x - pointRight.x);
		disp = disp - disp_corr;
		if (disp < 0.01f) disp = 0.01f;
		ratio = bl / disp;

		x = (float)((pointRight.x - ppx) * ratio - rightcamposx/2); // X
		y = (float)((pointRight.y - ppy) * ratio);               // Y
		z = (float)(fl         * ratio);                // Z

#if VERBOSE
		std::cout << "Index : " << index << std::endl;
		std::cout << "PRight (" << pointRight.x << ", " << pointRight.y << ")"<< std::endl;
		std::cout << "PLeft (" << pointLeft.x << ", " << pointLeft.y << ")"<< std::endl;
		std::cout << "rightcamposx " << rightcamposx << std::endl;
		std::cout << "baseline " << bl << std::endl;
		std::cout << "disparity " << disp << std::endl;
		std::cout << "disp_corr " << disp_corr << std::endl; 
		std::cout << "ratio " << ratio << std::endl; 
		std::cout << "Point (" << x << ", " << -y << ", " << -z<< ")"<< std::endl;
#endif
		//update
		(iter->second).pointLocation = cv::Point3f(x,-y,-z);
		//vctDouble2 projection = Wrld2Cam(vctDouble3((iter->second).pointLocation.x,(iter->second).pointLocation.y,(iter->second).pointLocation.z));
		//(iter->second).projectedPixelLocation = cv::Point2f(projection.X(),projection.Y());
		//std::cout << "Right (" << (iter->second).pixelLocation.x << ", " << (iter->second).pixelLocation.y << ")" << "projected (" << (iter->second).projectedPixelLocation.x << ", " << (iter->second).projectedPixelLocation.y << ")"<< std::endl;
		//projection = left->Wrld2Cam(vctDouble3((iter->second).pointLocation.x,(iter->second).pointLocation.y,(iter->second).pointLocation.z));
		//left->SetProjectedPixelLocation(cv::Point2f(projection.X(),projection.Y()),index);
		//std::cout << "Left (" << left->GetPoint(index).pixelLocation.x << ", " << left->GetPoint(index).pixelLocation.y << ")" << "projected (" << left->GetPoint(index).projectedPixelLocation.x << ", " << left->GetPoint(index).projectedPixelLocation.y << ")"<< std::endl;
		left->SetPointLocation(cv::Point3f(x,-y,-z),index,TARGETS_REAL);
		index++;

	}

	bool valid3D = check3DPositionValidity();
	this->Valid = valid3D;
	left->SetValid(valid3D);
	if(this->Valid || this->m_resetPixelHSV)
	{
		//if(this->Valid && !this->m_resetPixelHSV)
		//{
		//	ResetFiducialColors();
		//	left->ResetFiducialColors();
		//}
		//project
		index = 0;
		for (_PointCacheMap::iterator iter = m_Points.begin(); iter != m_Points.end(); iter++) 
		{
			Update3DPosition((iter->second).pointLocation,index);
			left->Update3DPosition((iter->second).pointLocation,index);
			index++;
		}
	}
}

void svlFilterImageRegistrationGUI::Update3DPosition(const cv::Point3f location, int index)
{
	bool debugLocal = false;
	SetKinematicPointLocation(location,index);
	PointInternal point = GetPoint(index);
	vctDouble2 projection = Wrld2Cam(vctDouble3(point.kinematicPointLocation.x,point.kinematicPointLocation.y,point.kinematicPointLocation.z));
	SetProjectedPixelLocation(cv::Point2f(projection.X(),projection.Y()),index);
	point = GetPoint(index);
	if(debugLocal)
	{
		if(VideoCh == SVL_RIGHT)
		{
			std::cout << "Right 3D << "<< index << " (" << point.kinematicPointLocation.x << ", " << point.kinematicPointLocation.y << ", " << point.kinematicPointLocation.z << ")" << std::endl;
			std::cout << "Right << "<< index << " (" << point.pixelLocation.x << ", " << point.pixelLocation.y << ")" << "projected (" << point.projectedPixelLocation.x << ", " << point.projectedPixelLocation.y << ")"<< std::endl;
		}
		else
		{
			std::cout << "Left 3D << "<< index << " (" << point.kinematicPointLocation.x << ", " << point.kinematicPointLocation.y << ", " << point.kinematicPointLocation.z << ")" << std::endl;
			std::cout << "Left << "<< index << " (" << point.pixelLocation.x << ", " << point.pixelLocation.y << ")" << "projected (" << point.projectedPixelLocation.x << ", " << point.projectedPixelLocation.y << ")"<< std::endl;
		}
	}
	//projection = left->Wrld2Cam(vctDouble3(point.kinematicPointLocation.x,point.kinematicPointLocation.y,point.kinematicPointLocation.z));
	//left->SetProjectedPixelLocation(cv::Point2f(projection.X(),projection.Y()),index);
	//std::cout << "Left (" << left->GetPoint(index).pixelLocation.x << ", " << left->GetPoint(index).pixelLocation.y << ")" << "projected (" << left->GetPoint(index).projectedPixelLocation.x << ", " << left->GetPoint(index).projectedPixelLocation.y << ")"<< std::endl;
	//left->SetKinematicPointLocation(location,index);
}

bool svlFilterImageRegistrationGUI::check3DPositionValidity(void)
{
	bool value = false;
	int bx, by, bz, wx, wy, wz, yx, yy, yz;
	bool valMidpoint0, valMidpoint1, valMidpoint2;
	valMidpoint0 = false;
	valMidpoint1 = false;
	valMidpoint2 = false;

	// Draw points
	for (_PointCacheMap::iterator iter = m_Points.begin(); iter != m_Points.end(); iter++) 
	{
		if(iter->second.Valid)
		{
			switch((iter->second).fiducialColor)
			{
			case(BLACK):
				{
					bx = (iter->second).pointLocation.x;
					by = (iter->second).pointLocation.y;
					bz = (iter->second).pointLocation.z;
					valMidpoint0 = true;
					break;
				}
			case(WHITE):
				{
					wx = (iter->second).pointLocation.x;
					wy = (iter->second).pointLocation.y;
					wz = (iter->second).pointLocation.z;
					valMidpoint1 = true;
					break;
				}
			case(YELLOW):
				{
					yx = (iter->second).pointLocation.x;
					yy = (iter->second).pointLocation.y;
					yz = (iter->second).pointLocation.z;
					valMidpoint2 = true;
					break;
				}
			default:
				break;
			}
		}
	}

	float lengthBY = 10/2*std::sqrt((double)2.0);//mm
	float lengthBW = 10/2*std::sqrt((double)2.0);//mm
	float lengthYW = 10;//mm

	if(valMidpoint0 && valMidpoint1 && valMidpoint2)
	{
		float distanceBY = distanceBetweenTwoPoints (bx, by, bz, yx, yy, yz);
		float distanceBW = distanceBetweenTwoPoints (bx, by, bz, wx, wy, wz);
		float distanceYW = distanceBetweenTwoPoints (yx, yy, yz, wx, wy, wz);
		valMidpoint0 = abs(distanceBY - lengthBY) <= m_3DThreshold*lengthBY;
		valMidpoint1 = abs(distanceBW - lengthBW) <= m_3DThreshold*lengthBW;
		valMidpoint2 = abs(distanceYW - lengthYW) <= m_3DThreshold*lengthYW;
		value = (valMidpoint0 && valMidpoint1 && valMidpoint2) || 
			(valMidpoint0 && valMidpoint1 && abs(distanceYW - lengthYW) < lengthYW/2) || 
			(abs(distanceBY - lengthBY) < lengthBY/2 && valMidpoint1 && valMidpoint2) || 
			(valMidpoint0 && valMidpoint2 && abs(distanceBY - lengthBY) < lengthBY/2);

		//std::cout << "valMidpointOne " << valMidpoint0 << " valMidpointTwo " << valMidpoint1 << " valMidpointThree " << valMidpoint2 << std::endl;
		//std::cout << "distanceBY " << distanceBY << "distanceBW " << distanceBW << "distanceYW " << distanceYW << std::endl;
		//std::cout << "lengthBY " << lengthBY << "lengthBW " << lengthBW << "lengthYW " << lengthYW << std::endl;
		if(m_debug)
		{
			std::cout << "BY " << abs(distanceBY - lengthBY) << " BW " << abs(distanceBW - lengthBW) << " YW " << abs(distanceYW - lengthYW) << std::endl;
			std::cout << "valid " << value << " distanceBY " << m_3DThreshold*distanceBY << " distanceBW " << m_3DThreshold*distanceBW << " distanceYW " << m_3DThreshold*distanceYW << std::endl;
		}
	}

	return value;
}

int svlFilterImageRegistrationGUI::UpdateThreshold(svlFilterImageRegistrationGUI::FiducialColor fidColor, int delta)
{
	m_thresholds[VideoCh][fidColor] += delta;
	m_thresholds[VideoCh][fidColor] = std::max(m_thresholds[VideoCh][fidColor],0);
	return m_thresholds[VideoCh][fidColor];
}

double svlFilterImageRegistrationGUI::UpdateSearchRadius(svlFilterImageRegistrationGUI::FiducialColor fidColor, double delta)
{
	m_searchRadii[VideoCh][fidColor] += delta;
	m_searchRadii[VideoCh][fidColor] = std::max<double>((double)m_searchRadii[VideoCh][fidColor],(double)0.0);
	return m_searchRadii[VideoCh][fidColor];
}

double svlFilterImageRegistrationGUI::UpdateHSVMinSaturation(svlFilterImageRegistrationGUI::FiducialColor fidColor, double delta)
{
	m_HSVMinSaturation[VideoCh][fidColor] += delta;
	m_HSVMinSaturation[VideoCh][fidColor] = std::max<double>((double)m_HSVMinSaturation[VideoCh][fidColor],(double)0.0);
	return m_HSVMinSaturation[VideoCh][fidColor];
}

double svlFilterImageRegistrationGUI::UpdateHSVMaxSaturation(svlFilterImageRegistrationGUI::FiducialColor fidColor, double delta)
{
	m_HSVMaxSaturation[VideoCh][fidColor] += delta;
	m_HSVMaxSaturation[VideoCh][fidColor] = std::max<double>((double)m_HSVMaxSaturation[VideoCh][fidColor],(double)0.0);
	return m_HSVMaxSaturation[VideoCh][fidColor];
}

double svlFilterImageRegistrationGUI::UpdateHSVMinHue(svlFilterImageRegistrationGUI::FiducialColor fidColor, double delta)
{
	m_HSVMinHue[VideoCh][fidColor] += delta;
	m_HSVMinHue[VideoCh][fidColor] = std::max<double>((double)m_HSVMinHue[VideoCh][fidColor],(double)0.0);
	return m_HSVMinHue[VideoCh][fidColor];
}

double svlFilterImageRegistrationGUI::UpdateHSVMaxHue(svlFilterImageRegistrationGUI::FiducialColor fidColor, double delta)
{
	m_HSVMaxHue[VideoCh][fidColor] += delta;
	m_HSVMaxHue[VideoCh][fidColor] = std::max<double>((double)m_HSVMaxHue[VideoCh][fidColor],(double)0.0);
	return m_HSVMaxHue[VideoCh][fidColor];
}

double svlFilterImageRegistrationGUI::UpdateHSVMinValue(svlFilterImageRegistrationGUI::FiducialColor fidColor, double delta)
{
	m_HSVMinValue[VideoCh][fidColor] += delta;
	m_HSVMinValue[VideoCh][fidColor] = std::max<double>((double)m_HSVMinValue[VideoCh][fidColor],(double)0.0);
	return m_HSVMinValue[VideoCh][fidColor];
}

double svlFilterImageRegistrationGUI::UpdateHSVMaxValue(svlFilterImageRegistrationGUI::FiducialColor fidColor, double delta)
{
	m_HSVMaxValue[VideoCh][fidColor] += delta;
	m_HSVMaxValue[VideoCh][fidColor] = std::max<double>((double)m_HSVMaxValue[VideoCh][fidColor],(double)0.0);
	return m_HSVMaxValue[VideoCh][fidColor];
}
double svlFilterImageRegistrationGUI::UpdateFiducialBoundaryThreshold(svlFilterImageRegistrationGUI::FiducialColor fidColor, double delta)
{
	m_fiducialBoundaryThreshold[VideoCh][fidColor] += delta;
	m_fiducialBoundaryThreshold[VideoCh][fidColor] = std::max<double>((double)m_fiducialBoundaryThreshold[VideoCh][fidColor],(double)0.0);
	return m_fiducialBoundaryThreshold[VideoCh][fidColor];
}

double svlFilterImageRegistrationGUI::UpdateROIExpansion(double delta)
{
	m_ROI_expansion[VideoCh] += delta;
	return m_ROI_expansion[VideoCh];
}

IplImage* svlFilterImageRegistrationGUI::getThresholdedImage(svlSampleImage* bgimage, svlFilterImageRegistrationGUI::FiducialColor color)
{
	//Orange  0-22
	//Yellow 22- 38
	//Green 38-75
	//Blue 75-130
	//Violet 130-160
	//Red 160-179
	double minHue, minSaturation, minValue, maxHue, maxSaturation, maxValue;

	switch(color)
	{
	case(WHITE):
	case(BLACK):
	case(GREEN):
	case(YELLOW):
		{
			minHue = m_HSVMinHue[VideoCh][color];
			minSaturation =  m_HSVMinSaturation[VideoCh][color];
			minValue = m_HSVMinValue[VideoCh][color];
			maxHue = m_HSVMaxHue[VideoCh][color];
			maxSaturation =  m_HSVMaxSaturation[VideoCh][color];
			maxValue = m_HSVMaxValue[VideoCh][color];
			break;
		}
	default:
		break;
	}

	IplImage* img = bgimage->IplImageRef(VideoCh);
	// Convert the image into an HSV image
	IplImage* imgHSV = cvCreateImage(cvGetSize(img), 8, 3);
	cvCvtColor(img, imgHSV, CV_BGR2HSV);
	IplImage* imgThreshed = cvCreateImage(cvGetSize(img), 8, 1);

	// left scope for Imaging Lab 30 degree up is noisy
	//if(VideoCh)
	//	cvInRangeS(imgHSV, cvScalar(minHue, minSaturation, minValue), cvScalar(maxHue, maxSaturation, maxValue), imgThreshed);
	//else
	cvInRangeS(imgHSV, cvScalar(minHue, minSaturation, minValue), cvScalar(maxHue, maxSaturation, maxValue), imgThreshed);


	cvReleaseImage(&imgHSV);
	//cvReleaseImage(&imgThreshed);
	return imgThreshed;
}

bool compareLinesAngle(svlFilterImageRegistrationGUI::LineInternal first, svlFilterImageRegistrationGUI::LineInternal second)
{
	return first.Angle > second.Angle;
}

bool compareLinesLength(svlFilterImageRegistrationGUI::LineInternal first, svlFilterImageRegistrationGUI::LineInternal second)
{
	return first.Length > second.Length;
}

cv::Point2f svlFilterImageRegistrationGUI::findMoment(IplImage* imgThresh){
	// Calculate the moments of 'imgThresh'
	CvMoments *moments = (CvMoments*)malloc(sizeof(CvMoments));
	cvMoments(imgThresh, moments, 1);
	double moment10 = cvGetSpatialMoment(moments, 1, 0);
	double moment01 = cvGetSpatialMoment(moments, 0, 1);
	double area = cvGetCentralMoment(moments, 0, 0);
	int posX = 0;
	int posY = 0;  
	// if the area<1000, I consider that the there are no object in the image and it's because of the noise, the area is not zero 
	if(area>0){
		// calculate the position of the ball
		posX = moment10/area;
		posY = moment01/area;        
	}
	free(moments); 
	return cv::Point2f(posX,posY);
}

cv::Point2f svlFilterImageRegistrationGUI::findMoment(CvArr* imgThresh){
	// Calculate the moments of 'imgThresh'
	CvMoments *moments = (CvMoments*)malloc(sizeof(CvMoments));
	cvMoments(imgThresh, moments, 1);
	double moment10 = cvGetSpatialMoment(moments, 1, 0);
	double moment01 = cvGetSpatialMoment(moments, 0, 1);
	double area = cvGetCentralMoment(moments, 0, 0);
	int posX = 0;
	int posY = 0;  
	// if the area<1000, I consider that the there are no object in the image and it's because of the noise, the area is not zero 
	if(area>0){
		// calculate the position of the ball
		posX = moment10/area;
		posY = moment01/area;        
	}
	free(moments); 
	return cv::Point2f(posX,posY);
}

CvRect svlFilterImageRegistrationGUI::expandBoundingBox(svlSampleImage* bgimage, CvRect input, double rescale)
{
	CvRect output = input;
	if(rescale > 1.0)
	{
		output = cvRect(	std::max<int>((int)0,(int)input.x-input.width/rescale-m_ROI_expansion[VideoCh]),std::max<int>((int)0,(int)input.y-input.height/rescale-m_ROI_expansion[VideoCh]),
			std::min<int>((int)bgimage->GetWidth(),(int)rescale*(input.width+m_ROI_expansion[VideoCh])), std::min<int>((int)bgimage->GetHeight(),(int)rescale*(input.height+m_ROI_expansion[VideoCh])));
	}
	else
	{
		output = cvRect(	std::max<int>((int)0,(int)input.x-m_ROI_expansion[VideoCh]),std::max<int>((int)0,(int)input.y-m_ROI_expansion[VideoCh]),
			std::min<int>((int)bgimage->GetWidth(),(int)(input.width+2*m_ROI_expansion[VideoCh])), std::min<int>((int)bgimage->GetHeight(),(int)(input.height+2*m_ROI_expansion[VideoCh])));

	}
	return output;
}

void svlFilterImageRegistrationGUI::samplePixelHSV(svlSampleImage* bgimage, svlFilterImageRegistrationGUI::PointInternal fiducial)
{
	bool debugLocal = false;

	switch(fiducial.fiducialColor)
	{
	case(BLACK):
	case(WHITE):
		{
			//getPixelRGB(bgimage,fiducial.sampleLocation.x,fiducial.sampleLocation.y,fiducial.sampleRGB);
			fiducial.sampleRGB = fiducial.fiducialContour.meanColorRGB;
			double minRGB = std::min(fiducial.sampleRGB.val[0],std::min(fiducial.sampleRGB.val[1],fiducial.sampleRGB.val[2]));
			double maxRGB = std::max(fiducial.sampleRGB.val[0],std::max(fiducial.sampleRGB.val[1],fiducial.sampleRGB.val[2]));
			if(fiducial.fiducialColor == BLACK)
				m_thresholds[VideoCh][fiducial.fiducialColor] = maxRGB+5;
			else
				m_thresholds[VideoCh][fiducial.fiducialColor] = minRGB-5;
			m_tolerance[VideoCh][fiducial.fiducialColor] = abs(maxRGB-minRGB)+5;

			if(debugLocal)
			{
				std::cout << "Resample "<< fiducial.fiducialColor << " " << m_thresholds[VideoCh][fiducial.fiducialColor] << " with tolerance " << m_tolerance[VideoCh][fiducial.fiducialColor] << std::endl;
				std::cout << "R " << fiducial.sampleRGB.val[0] << " G " << fiducial.sampleRGB.val[1] << " B " << fiducial.sampleRGB.val[2] << std::endl;
				std::cout << "-------------------------------------------------------------------------" <<std::endl <<std::endl<< std::endl;
			}
			break;
		}
	case(YELLOW):
		//getPixelHSV(bgimage,fiducial.sampleLocation.x,fiducial.sampleLocation.y,fiducial.sampleHSV);
		fiducial.sampleHSV = fiducial.fiducialContour.meanColorHSV;
		//std::cout << VideoCh << " samplePixelHSV " << fiducial.fiducialColor << std::endl;
		//std::cout << " m_HSVMinHue " << m_HSVMinHue[VideoCh][fiducial.fiducialColor] << " " << fiducial.sampleHSV.val[0]-10 << " ";
		m_HSVMinHue[VideoCh][fiducial.fiducialColor] = std::max((double)0.0,(double)std::min((double)m_HSVMinHue[VideoCh][fiducial.fiducialColor],(double)fiducial.sampleHSV.val[0]-10));
		//std::cout << m_HSVMinHue[VideoCh][fiducial.fiducialColor] << std::endl;

		//std::cout << " m_HSVMinSaturation " <<	m_HSVMinSaturation[VideoCh][fiducial.fiducialColor] << " " << fiducial.sampleHSV.val[1]-10 << " ";
		m_HSVMinSaturation[VideoCh][fiducial.fiducialColor] = std::max((double)0.0,std::min((double)m_HSVMinSaturation[VideoCh][fiducial.fiducialColor],(double)fiducial.sampleHSV.val[1]-10));
		//std::cout << m_HSVMinSaturation[VideoCh][fiducial.fiducialColor] << std::endl;

		//std::cout << " m_HSVMinValue " <<	m_HSVMinValue[VideoCh][fiducial.fiducialColor] << " " << fiducial.sampleHSV.val[2]-10 << " ";
		m_HSVMinValue[VideoCh][fiducial.fiducialColor] = std::max((double)0.0,(double)std::min((double)m_HSVMinValue[VideoCh][fiducial.fiducialColor],(double)fiducial.sampleHSV.val[2]-10));
		//std::cout << m_HSVMinValue[VideoCh][fiducial.fiducialColor] << std::endl;

		//std::cout << " m_HSVMaxHue " <<	m_HSVMaxHue[VideoCh][fiducial.fiducialColor] << " " << fiducial.sampleHSV.val[0]+10 << " ";
		m_HSVMaxHue[VideoCh][fiducial.fiducialColor] = std::min((double) 256,std::max((double)m_HSVMaxHue[VideoCh][fiducial.fiducialColor],(double)fiducial.sampleHSV.val[0]+10));
		//std::cout << m_HSVMinValue[VideoCh][fiducial.fiducialColor] << std::endl;

		//std::cout << " m_HSVMaxSaturation " <<	m_HSVMaxSaturation[VideoCh][fiducial.fiducialColor] << " " << fiducial.sampleHSV.val[1]+10 << " ";
		m_HSVMaxSaturation[VideoCh][fiducial.fiducialColor] = std::min((double) 256,std::max((double)m_HSVMaxSaturation[VideoCh][fiducial.fiducialColor],(double)fiducial.sampleHSV.val[1]+10));
		//std::cout << m_HSVMaxSaturation[VideoCh][fiducial.fiducialColor] << std::endl;

		//std::cout << " m_HSVMaxValue " <<	m_HSVMaxValue[VideoCh][fiducial.fiducialColor] << " " << fiducial.sampleHSV.val[2]+10 << " ";
		m_HSVMaxValue[VideoCh][fiducial.fiducialColor] = std::min((double) 256, std::max((double)m_HSVMaxValue[VideoCh][fiducial.fiducialColor],(double)fiducial.sampleHSV.val[2]+10));
		//std::cout << m_HSVMaxValue[VideoCh][fiducial.fiducialColor] << std::endl;
		//std::cout << "-------------------------------------------------------------------------" <<std::endl <<std::endl<< std::endl;
		break;
	default:
		break;
	}
}

void svlFilterImageRegistrationGUI::DrawInternal(svlSampleImage* bgimage, svlSample* CMN_UNUSED(input))
{
	bool debug = true;
	bool draw = true;
	if(m_debug)
	{
		// Check for triangle ROI
		//IplImage *thresholdImageGreen;
		//thresholdImageGreen = getThresholdedImage(bgimage, GREEN); 

		//IplImage *thresholdImageBlue;
		//thresholdImageBlue = getThresholdedImage(bgimage, BLUE); 

		//IplImage *thresholdImageYellow;
		//thresholdImageYellow = getThresholdedImage(bgimage, YELLOW); 

		//IplImage *thresholdImageWhite;
		//thresholdImageWhite = getThresholdedImage(bgimage, WHITE);

		// Use ROI
		//Check for color moments
		if(VideoCh == 1)
		{
			cvNamedWindow("Green_R");
			//cvShowImage("Green_R", thresholdImageGreen);
			cvWaitKey(100);
			//cvNamedWindow("Blue_R");
			//cvShowImage("Blue_R", thresholdImageBlue);
			//cvWaitKey(100);
			//cvNamedWindow("Yellow_R");
			//cvShowImage("Yellow_R", thresholdImageYellow);
			//cvWaitKey(100);
			//cvNamedWindow("White_R");
			//cvShowImage("White_R", thresholdImageWhite);
			//cvNamedWindow("Contour_R_1");
			//cvWaitKey(100);
			//cvNamedWindow("Contour_R_2");
			//cvWaitKey(100);
			//cvNamedWindow("Contour_R_3");
			//cvWaitKey(100);
			//cvNamedWindow("Contour_R_4");
			//cvWaitKey(100);
		}else
		{
			cvNamedWindow("Green_L");
			//cvShowImage("Green_L", thresholdImageGreen);
			cvWaitKey(100);
			//cvNamedWindow("Blue_L");
			//cvShowImage("Blue_L", thresholdImageBlue);
			//cvWaitKey(100);
			//cvNamedWindow("Yellow_L");
			//cvShowImage("Yellow_L", thresholdImageYellow);
			//cvWaitKey(100);
			//cvNamedWindow("White_L");
			//cvShowImage("White_L", thresholdImageWhite);
			//cvNamedWindow("Contour_L_1");
			//cvWaitKey(100);
			//cvNamedWindow("Contour_L_2");
			//cvWaitKey(100);
			//cvNamedWindow("Contour_L_3");
			//cvWaitKey(100);
			//cvNamedWindow("Contour_L_4");
			//cvWaitKey(100);
		}
	}else
	{
		cvDestroyAllWindows();
	}

	int cx, cy, rx, ry, bx, by, gx, gy, yx, yy;
	bool valMidpoint0, valMidpoint1, valMidpoint2;
	valMidpoint0 = false;
	valMidpoint1 = false;
	valMidpoint2 = false;

	if(this->updateToggle)
	{
		Update(bgimage);
		//Valid = IsValid(bgimage);
	}

	// Draw points
	for (_PointCacheMap::iterator iter = m_Points.begin(); iter != m_Points.end(); iter++) 
	{
		if(iter->second.Valid)
		{
			switch((iter->second).fiducialColor)
			{
			case(BLACK):
				{
					bx = (iter->second).pixelLocation.x;
					by = (iter->second).pixelLocation.y;
					valMidpoint0 = true;
					break;
				}
			case(WHITE):
				{
					gx = (iter->second).pixelLocation.x;
					gy = (iter->second).pixelLocation.y;
					valMidpoint1 = true;
					break;
				}
			case(YELLOW):
				{
					yx = (iter->second).pixelLocation.x;
					yy = (iter->second).pixelLocation.y;
					valMidpoint2 = true;
					break;
				}
			default:
				break;
			}

			if (Transformed) {
				// Calculate translation
				cx = static_cast<int>(Transform.Element(0, 0) * (iter->second).pixelLocation.x +
					Transform.Element(0, 1) * (iter->second).pixelLocation.y +
					Transform.Element(0, 2));
				cy = static_cast<int>(Transform.Element(1, 0) * (iter->second).pixelLocation.x +
					Transform.Element(1, 1) * (iter->second).pixelLocation.y +
					Transform.Element(1, 2));
				// Calculate scale from rotation matrix norm
				vctFixedSizeMatrixRef<double, 2, 2, 3, 1> rot(Transform.Pointer());
				double norm = rot.Norm();
				rx = static_cast<int>(norm * (iter->second).RadiusHoriz);
				ry = static_cast<int>(norm * (iter->second).RadiusVert);

				// TO DO: need to take care of rotation as well!
			}
			else {
				cx = (iter->second).pixelLocation.x;    cy = (iter->second).pixelLocation.y;
				rx = (iter->second).RadiusHoriz; ry = (iter->second).RadiusVert;
			}
			if(VERBOSE)
				std::cout << "Drawing registration fiducial index " <<(iter->second).ID << std::endl;
			if(draw)
			{
				//svlDraw::Ellipse(bgimage, VideoCh,
				//	cx, cy, rx, ry,
				//	(iter->second).Color,
				//	0.0, 360.0,
				//	//(iter->second).Angle * 57.295779513, // Convert from radians to angle
				//	-1);

				svlDraw::Ellipse(bgimage, VideoCh,
					(iter->second).pixelLocation.x, (iter->second).pixelLocation.y, rx, ry,
					(iter->second).Color,
					0.0, 360.0,
					(iter->second).Angle * 57.295779513, // Convert from radians to angle
					((iter->second).Fill ? -1 : (iter->second).Thickness));

			}

			// Get sample from input
			if (draw) {
				IplImage* cvimg = bgimage->IplImageRef(VideoCh);
				char buff[100];
				sprintf(buff, "%d", (iter->second).ID);
				CvFont Font;
				cvInitFont(&Font, CV_FONT_HERSHEY_PLAIN, 2, 2, 0, 1, 4);
				cvPutText(cvimg, buff ,(cv::Point2f)(iter->second).projectedPixelLocation, &Font, cvScalar((iter->second).Color.r, (iter->second).Color.g, (iter->second).Color.b));
			}

			// draw contours
			CvScalar color = CV_RGB((iter->second).Color.r, (iter->second).Color.g, (iter->second).Color.b);//rand()&255, rand()&255, rand()&255 );
			//if(draw)
			//{
			//	if((iter->second).fiducialColor == GREEN)
			//		cvRectangle(bgimage->IplImageRef(VideoCh),cv::Point((iter->second).expandedBoundingBox.x,(iter->second).expandedBoundingBox.y),cv::Point((iter->second).expandedBoundingBox.x+(iter->second).expandedBoundingBox.width,(iter->second).expandedBoundingBox.y+(iter->second).expandedBoundingBox.height),color);
			//	else
			//		cvRectangle(bgimage->IplImageRef(VideoCh),cv::Point((iter->second).boundingBox.x,(iter->second).boundingBox.y),cv::Point((iter->second).boundingBox.x+(iter->second).boundingBox.width,(iter->second).boundingBox.y+(iter->second).boundingBox.height),color);
			//}
			// bound and approximate largest contour
			//if((iter->second).largestContour != NULL)
			//{
			//	cvDrawContours( bgimage->IplImageRef(VideoCh), (iter->second).largestContour, color, color, -1, CV_FILLED, 8 );
			//}
			//if((iter->second).fittedContour)
			//	cvDrawContours( bgimage->IplImageRef(VideoCh), (iter->second).fittedContour, color, color, -1,8,0);
			//}
			//if((iter->second).fiducialContour != NULL)
			if((iter->second).isColor)
				cvDrawContours( bgimage->IplImageRef(VideoCh), (iter->second).fiducialContour.contour, color, color, -1, 1 /*CV_FILLED*/, CV_AA );

		}
	}

	// Draw points
	for (_PointCacheMap::iterator iter = m_TargetPoints.begin(); iter != m_TargetPoints.end(); iter++) 
	{
		if(iter->second.Valid)
		{
			switch((iter->second).fiducialColor)
			{
			case(BLACK):
				{
					bx = (iter->second).pixelLocation.x;
					by = (iter->second).pixelLocation.y;
					valMidpoint0 = true;
					break;
				}
			case(WHITE):
				{
					gx = (iter->second).pixelLocation.x;
					gy = (iter->second).pixelLocation.y;
					valMidpoint1 = true;
					break;
				}
			case(YELLOW):
				{
					yx = (iter->second).pixelLocation.x;
					yy = (iter->second).pixelLocation.y;
					valMidpoint2 = true;
					break;
				}
			default:
				break;
			}

			if (Transformed) {
				// Calculate translation
				cx = static_cast<int>(Transform.Element(0, 0) * (iter->second).pixelLocation.x +
					Transform.Element(0, 1) * (iter->second).pixelLocation.y +
					Transform.Element(0, 2));
				cy = static_cast<int>(Transform.Element(1, 0) * (iter->second).pixelLocation.x +
					Transform.Element(1, 1) * (iter->second).pixelLocation.y +
					Transform.Element(1, 2));
				// Calculate scale from rotation matrix norm
				vctFixedSizeMatrixRef<double, 2, 2, 3, 1> rot(Transform.Pointer());
				double norm = rot.Norm();
				rx = static_cast<int>(norm * (iter->second).RadiusHoriz);
				ry = static_cast<int>(norm * (iter->second).RadiusVert);

				// TO DO: need to take care of rotation as well!
			}
			else {
				cx = (iter->second).pixelLocation.x;    cy = (iter->second).pixelLocation.y;
				rx = (iter->second).RadiusHoriz; ry = (iter->second).RadiusVert;
			}
			if(VERBOSE)
				std::cout << "Drawing registration fiducial index " <<(iter->second).ID << std::endl;
			if(draw)
			{
				//svlDraw::Ellipse(bgimage, VideoCh,
				//	cx, cy, rx, ry,
				//	(iter->second).Color,
				//	0.0, 360.0,
				//	//(iter->second).Angle * 57.295779513, // Convert from radians to angle
				//	-1);

				svlDraw::Ellipse(bgimage, VideoCh,
					(iter->second).pixelLocation.x, (iter->second).pixelLocation.y, rx, ry,
					(iter->second).Color,
					0.0, 360.0,
					(iter->second).Angle * 57.295779513, // Convert from radians to angle
					((iter->second).Fill ? -1 : (iter->second).Thickness));

			}
		}
	}


	// Validity
	cv::Point2f midPoint0, midPoint1, midPoint2;
	IplImage* cvimg = bgimage->IplImageRef(VideoCh);

	CvFont Font;
	cvInitFont(&Font, CV_FONT_HERSHEY_PLAIN, 1.2, 1.2, 0, 1, 4);
	if(valMidpoint0 && valMidpoint1 && valMidpoint2)
	{
		unsigned char R0, G0, B0;
		unsigned char R1, G1, B1;
		unsigned char R2, G2, B2;

		midPoint0 = computeMidpoint((int)bx,(int)by,(int)gx,(int)gy);
		midPoint1 = computeMidpoint((int)gx,(int)gy,(int)yx,(int)yy);
		midPoint2 = computeMidpoint((int)bx,(int)by,(int)yx,(int)yy);

		//getPixelRGB(bgimage,midPoint0.x, midPoint0.y, R0, G0, B0);
		//getPixelRGB(bgimage,midPoint1.x, midPoint1.y, R1, G1, B1);
		//getPixelRGB(bgimage,midPoint2.x, midPoint2.y, R2, G2, B2);

		valMidpoint0 = isColorRGB(10, R0, G0, B0, GREEN);
		valMidpoint1 = isColorRGB(10, R1, G1, B1, GREEN);
		valMidpoint2 = isColorRGB(10, R2, G2, B2, GREEN);

		//if(draw)
		//{
			//if(!valMidpoint0)
			//{
			//	char buff0[100];
			//	sprintf(buff0, "%d,%d,%d", R0, B0, G0);
			//	cvPutText(cvimg, buff0 ,midPoint0, &Font, cvScalar(0, 0, 0));
			//}
			//if(!valMidpoint1)
			//{
			//	char buff1[100];
			//	sprintf(buff1, "%d,%d,%d", R1, B1, G1);
			//	cvPutText(cvimg, buff1 ,midPoint1, &Font, cvScalar(0, 0, 0));
			//}
			//if(!valMidpoint2)
			//{
			//	char buff2[100];
			//	sprintf(buff2, "%d,%d,%d", R2, B2, G2);
			//	cvPutText(cvimg, buff2 ,midPoint2, &Font, cvScalar(0, 0, 0));
			//}

			//svlDraw::Ellipse(bgimage, VideoCh, midPoint0.x, midPoint0.y, 5, 5, svlRGB(64, 0, 0),0.0, 360.0);
			//svlDraw::Ellipse(bgimage, VideoCh, midPoint1.x, midPoint1.y, 5, 5, svlRGB(64, 0, 0),0.0, 360.0);
			//svlDraw::Ellipse(bgimage, VideoCh, midPoint2.x, midPoint2.y, 5, 5, svlRGB(64, 0, 0),0.0, 360.0);

			if(Valid)
			{
				svlDraw::Line(bgimage,VideoCh, bx, by, gx, gy, 64, 0, 0);
				svlDraw::Line(bgimage,VideoCh, bx, by, yx, yy, 0, 64, 64);
			}
		//}
	}
}

bool compareContour(svlFilterImageRegistrationGUI::ContourInternal first, svlFilterImageRegistrationGUI::ContourInternal second)
{
	bool checkCircle = false;
	bool checkEllipse = false;
	bool checkSize = false;
	bool firstIsBetter = true;
	//color
	if(first.fiducialColor == svlFilterImageRegistrationGUI::GREEN)
	{
		if(second.fiducialColor == svlFilterImageRegistrationGUI::GREEN)
		{
			checkCircle = true;
		}else
		{
			return !firstIsBetter;
		}
	}else
	{
		if(second.fiducialColor == svlFilterImageRegistrationGUI::GREEN)
		{
			return firstIsBetter;
		}else
		{
			checkCircle = true;
		}
	}

	//circle
	if(checkCircle)
	{
		if(first.matchToCircle < 0.003)
		{
			if(second.matchToCircle < 0.003)
			{
				checkEllipse = true;
			}else
			{
				return firstIsBetter;
			}
		}else
		{
			if(second.matchToCircle < 0.003)
			{
				return !firstIsBetter;
			}else
			{
				checkEllipse = true;
			}
		}
	}

	//ellipse
	if(checkEllipse)
	{
		if(first.matchToEllipse < 0.003)
		{
			if(second.matchToEllipse < 0.003)
			{
				checkSize = true;
			}else
			{
				return firstIsBetter;
			}
		}else
		{
			if(second.matchToEllipse < 0.003)
			{
				return !firstIsBetter;
			}else
			{
				checkSize = true;
			}
		}
	}

	//size
	if(checkSize)
	{
		first.area > second.area;
	}
}

void svlFilterImageRegistrationGUI::getContourColor(svlFilterImageRegistrationGUI::ContourInternal* contour,
													double R,double G,double B,double H, double S, double V)
{
	contour->fiducialColor = RED;
	contour->color = cvScalar(0,0,255);

	if(isColorRGB(m_tolerance[VideoCh][BLACK],R,G,B,BLACK))
	{
		contour->fiducialColor = BLACK;
		contour->color = cvScalar(0,0,0);
	}
	else if(isColorRGB(m_tolerance[VideoCh][WHITE],R,G,B,WHITE))
	{
		contour->fiducialColor = WHITE;
		contour->color = cvScalar(255,255,255);
	}else if(isColorHSV(m_tolerance[VideoCh][YELLOW],H,S,V,YELLOW))
	{
		contour->fiducialColor = YELLOW;
		contour->color = cvScalar(0,255,255);
		//}else if(isColorHSV(10,H,S,V,BLUE))
		//{
		//	contour->fiducialColor = BLUE;
		//	contour->color = cvScalar(0,0,255);
	}else if(isColorHSV(m_tolerance[VideoCh][GREEN],H,S,V,GREEN))
	{
		contour->fiducialColor = GREEN;
		contour->color = cvScalar(0,255,0);
	}
}

IplImage* svlFilterImageRegistrationGUI::getContourMask(svlSampleImage* bgimage, CvSeq* contour)
{
	IplImage  *contourMask;
	//IplImage  *contourROI;
	contourMask = cvCreateImage(cvGetSize(bgimage->IplImageRef(VideoCh)), 8, 1);
	//contourROI = cvCreateImage(cvGetSize(bgimage->IplImageRef(VideoCh)), 8, 3);
	cvZero(contourMask);
	//cvZero(contourROI);
	cvDrawContours(contourMask, contour, CV_RGB(255,255,255), CV_RGB(255,255,255), -1, -1 /*CV_FILLED*/, CV_AA ); 

	return contourMask;
}

void svlFilterImageRegistrationGUI::processContour(svlSampleImage* bgimage, CvSeq* approximateContour,svlFilterImageRegistrationGUI::ContourInternal* contour)
{
	//ID
	contour->pixelMatchID = -1;
	contour->contour = approximateContour;
	contour->pixelLocation = findMoment(approximateContour);
	contour->area = std::abs((double)cvContourArea(contour->contour,cvSlice(0, CV_WHOLE_SEQ_END_INDEX),0));
	contour->arcLength = std::abs((double)cvContourPerimeter(contour->contour));
	IplImage  *imgHSV, *contourMask,*ellipseMask, *ellipseROI, *circleMask, *circleROI;// *contourROI, 
	imgHSV = cvCreateImage(cvGetSize(bgimage->IplImageRef(VideoCh)), 8, 3);
	ellipseMask = cvCreateImage(cvGetSize(bgimage->IplImageRef(VideoCh)), 8, 1);
	ellipseROI = cvCreateImage(cvGetSize(bgimage->IplImageRef(VideoCh)), 8, 3);
	circleMask = cvCreateImage(cvGetSize(bgimage->IplImageRef(VideoCh)), 8, 1);
	circleROI = cvCreateImage(cvGetSize(bgimage->IplImageRef(VideoCh)), 8, 3);
	cvZero(ellipseMask);
	cvZero(ellipseROI);
	cvZero(circleMask);
	cvZero(circleROI);

	contourMask = getContourMask(bgimage, contour->contour);
	contour->ellipse = cvFitEllipse2(contour->contour);
	CvPoint2D32f center; 
	float radius; 
	int circleResult = cvMinEnclosingCircle(contour->contour,&center,&radius); 
	cvEllipseBox(ellipseMask,contour->ellipse,CV_RGB(255,255,255),-1);
	cvCircle(circleMask,cvPoint(center.x,center.y),(int)radius,CV_RGB(255,255,255),-1);
	contour->pixelLocation = findMoment(contour->contour);
	if(circleResult != 0)
	{
		contour->matchToCircle = cvMatchShapes(contourMask,circleMask,CV_CONTOURS_MATCH_I1);
		//contour->image = cvCreateImage(cvGetSize(bgimage->IplImageRef(VideoCh)), 8, 3);
		//cvAnd(bgimage->IplImageRef(VideoCh), bgimage->IplImageRef(VideoCh), contour->image, circleMask);

	}else
	{
		contour->matchToCircle = 0;
	}
	contour->matchToEllipse = cvMatchShapes(contourMask,ellipseMask,CV_CONTOURS_MATCH_I1);

	cvCvtColor(bgimage->IplImageRef(VideoCh), imgHSV, CV_BGR2HSV);
	contour->meanColorHSV = cvAvg(imgHSV,contourMask);
	contour->meanColorRGB = cvAvg(bgimage->IplImageRef(VideoCh),contourMask);
	getContourColor(contour, contour->meanColorRGB.val[0],contour->meanColorRGB.val[1],contour->meanColorRGB.val[2],contour->meanColorHSV.val[0],contour->meanColorHSV.val[1],contour->meanColorHSV.val[2]);

	contour->Valid = true;

	if(imgHSV)
		cvReleaseImage(&imgHSV);
	if(contourMask)
		cvReleaseImage(&contourMask);
	//if(contourROI)
	//	cvReleaseImage(&contourROI);
	if(ellipseMask)
		cvReleaseImage(&ellipseMask);
	if(ellipseROI)
		cvReleaseImage(&ellipseROI);
	if(circleMask)
		cvReleaseImage(&circleMask);
	if(circleROI)
		cvReleaseImage(&circleROI);
}

void svlFilterImageRegistrationGUI::filterConcentricContours(std::list<svlFilterImageRegistrationGUI::ContourInternal>* contourList)
{
	std::list<ContourInternal>::iterator currentContourIter, contourIter;

	for (currentContourIter=contourList->begin(); currentContourIter!=contourList->end(); ++currentContourIter)
	{
		for (contourIter=contourList->begin(); (contourIter!=contourList->end() && contourIter != currentContourIter); ++contourIter)
		{
			double inside = cvPointPolygonTest(contourIter->contour,currentContourIter->pixelLocation,1);
			if(inside > 0.0)
			{
				currentContourIter->fiducialColor = GREEN;
				break;
			}
		}
	}
	
	contourList->sort(compareContour);
}


std::list<svlFilterImageRegistrationGUI::ContourInternal> svlFilterImageRegistrationGUI::categorizeContours(svlSampleImage* bgimage, IplImage *thresholdImage, svlFilterImageRegistrationGUI::FiducialColor fiducialColor)
{
	bool debugLocal = true;
	CvSeq* contours;
	std::list<ContourInternal> contourList, concentricFilteredContourList;
	IplImage *thresholdImageErode = cvCreateImage(cvGetSize(thresholdImage), 8, 1);

	// decrease noise? cvErode, cvDilate, cvCopy
	IplConvKernel* element  = cvCreateStructuringElementEx(3+abs(m_dilate), 3+abs(m_dilate), abs(m_dilate), abs(m_dilate), CV_SHAPE_RECT, NULL);
	if(m_dilate == 0)
	{
		cvCopy(thresholdImage,thresholdImageErode);
	}else if(m_dilate < 0)
	{	
		cvErode(thresholdImage,thresholdImageErode, element);
	}else
	{
		cvDilate(thresholdImage,thresholdImageErode,element);
	}
	
	if(m_debug)
	{
		if(VideoCh == 0)
			cvShowImage("Green_L", thresholdImageErode);
		else
			cvShowImage("Green_R", thresholdImageErode);
	}

	//finding all contours in the image
	int numContours = cvFindContours(thresholdImageErode, m_storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
	if(numContours > 0)
	{
		//iterating through each contour
		while(contours)
		{
			if(contours->total > 10)
			{
				ContourInternal currentContour;
				processContour(bgimage, cvApproxPoly( contours, sizeof(CvContour), m_storageApprox, CV_POLY_APPROX_DP, 0),&currentContour);
				contourList.push_back(currentContour);
			}

			contours = contours->h_next; 
		}
		contourList.sort(compareContour);

		//filter concentric contours
		filterConcentricContours(&contourList);
		
	}

	if(debugLocal)
	{
		std::list<ContourInternal>::iterator sortedContourIter;
		int count = 0;
		//std::cout << "-------------------------------- " << VideoCh << " --------------------------------" << std::endl;

		for (sortedContourIter=contourList.begin(); sortedContourIter!=contourList.end(); ++sortedContourIter)
		{
			if(count < 4)
			{

				CvScalar color = CV_RGB(0,0,0);
				char buff[100];
				sprintf(buff, "%d (%.3f,%.3f,%.2f)", count,(*sortedContourIter).meanColorRGB.val[0],(*sortedContourIter).meanColorRGB.val[1],(*sortedContourIter).meanColorRGB.val[2]);
				//(*sortedContourIter).matchToCircle,(*sortedContourIter).matchToEllipse,(*sortedContourIter).area);
				//(*sortedContourIter).meanColorHSV.val[0],(*sortedContourIter).meanColorHSV.val[1],(*sortedContourIter).meanColorHSV.val[2]);
				//if(VideoCh)
				//std::cout << " # " << count << " circle: " << (*sortedContourIter).matchToCircle << " ellipse: " << (*sortedContourIter).matchToEllipse << " area: " <<(*sortedContourIter).area << std::endl;
				//std::cout << " # " << count << " R: " << (*sortedContourIter).meanColorRGB.val[0] << " G: " << (*sortedContourIter).meanColorRGB.val[1] << " B: " <<(*sortedContourIter).meanColorRGB.val[2] << std::endl;
				//std::cout << " # " << count << " H: " << (*sortedContourIter).meanColorHSV.val[0] << " S: " << (*sortedContourIter).meanColorHSV.val[1] << " V: " <<(*sortedContourIter).meanColorHSV.val[2] << std::endl;
				CvFont Font;
				cvInitFont(&Font, CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 1, 4);
				//cvPutText(bgimage->IplImageRef(VideoCh), buff ,(*sortedContourIter).pixelLocation, &Font, color);
				color = (*sortedContourIter).color;
				cvDrawContours( bgimage->IplImageRef(VideoCh), (*sortedContourIter).contour, color, color, -1, 1 /*CV_FILLED*/, CV_AA );

				count++;
			}else
			{
				//std::cout << std::endl;
				break;
			}
		}
	}

	cvReleaseImage(&thresholdImageErode);
	return contourList;

}

CvSeq* svlFilterImageRegistrationGUI::findLargestContour(svlSampleImage* bgimage, IplImage *thresholdImage, svlFilterImageRegistrationGUI::FiducialColor fiducialColor)
{
	int numContours;
	CvRect boundingRect;
	CvSeq* largestContour = NULL;
	CvSeq* contours;  //hold the pointer to a contour in the memory block
	IplImage *thresholdImageErode = cvCreateImage(cvGetSize(thresholdImage), 8, 1);

	// decrease noise
	//cvErode(thresholdImage,thresholdImageErode);

	cvCopy(thresholdImage,thresholdImageErode);

	//finding all contours in the image
	numContours = cvFindContours(thresholdImageErode, m_storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
	if(numContours > 0)
	{
		//iterating through each contour
		largestContour = contours;
		double maxArea = 0.0;
		while(contours)
		{
			double area = cvContourArea(contours);
			if (area > maxArea) {
				maxArea = area;
				largestContour = contours;
			}
			contours = contours->h_next; 
		}
	}

	cvReleaseImage(&thresholdImageErode);
	return largestContour;

}

void svlFilterImageRegistrationGUI::Update(svlSampleImage* bgimage)
{
	// Get ROI from green
	CvRect ROI = cvRect(0,0,bgimage->GetWidth(),bgimage->GetHeight());
	_PointCacheMap::iterator foundPoint = getPointByColor(GREEN);
	if(foundPoint== m_Points.end()) 
	{
		return;
	}
	updateFiducial(bgimage, GREEN, &ROI);

	updateFiducial(bgimage, WHITE, &ROI, m_resetPixelHSV);
	updateFiducial(bgimage, BLACK, &ROI, m_resetPixelHSV);
	updateFiducial(bgimage, YELLOW, &ROI, m_resetPixelHSV);
	this->m_resetPixelHSV = false;
}

void svlFilterImageRegistrationGUI::updateFiducial(svlSampleImage* bgimage, svlFilterImageRegistrationGUI::FiducialColor color, CvRect* boundingBox, bool reset)
{

	IplImage *thresholdImage, *cropImageMask, *cropImageROI;
	_PointCacheMap::iterator foundPoint = getPointByColor(color);
	_PointCacheMap::iterator foundGreen = getPointByColor(GREEN);
	ContourInternal	closestContour;
	float distanceToClosestContour = FLT_MAX;
	int count = 0;

	if (foundPoint != m_Points.end() && foundGreen != m_Points.end()) {
		(foundPoint->second).isColor = false;
		//CvSeq*	contour = findLargestContour(bgimage, cropImageROI, (foundPoint->second).fiducialColor);
		if((foundPoint->second).fiducialColor == GREEN)
		{
			thresholdImage = getThresholdedImage(bgimage, (foundPoint->second).fiducialColor);
			cropImageMask = cvCreateImage(cvGetSize(thresholdImage), 8, 1);
			cropImageROI = cvCreateImage(cvGetSize(thresholdImage), 8, 1);
			cvZero(cropImageMask);
			cvZero(cropImageROI);
			cvRectangle(cropImageMask,cv::Point(boundingBox->x,boundingBox->y),cv::Point(boundingBox->x+boundingBox->width,boundingBox->y+boundingBox->height),CV_RGB(255, 255, 255), -1, 8, 0);
			cvAnd(thresholdImage, thresholdImage, cropImageROI, cropImageMask);
			(foundPoint->second).sortedContourList = categorizeContours(bgimage, cropImageROI, (foundPoint->second).fiducialColor);
			(foundPoint->second).pixelLocation = findMoment(cropImageROI);
			//(foundPoint->second).isColor = true;
			cvReleaseImage(&cropImageMask);
			cvReleaseImage(&cropImageROI);
			cvReleaseImage(&thresholdImage);
		}else
		{
			std::list<ContourInternal>::iterator sortedContourIter;
			for (sortedContourIter=(foundGreen->second).sortedContourList.begin(); sortedContourIter!=(foundGreen->second).sortedContourList.end(); ++sortedContourIter)
			{
				if(count < 10)
				{
					if(reset)
					{
						double inside = cvPointPolygonTest(sortedContourIter->contour,(foundPoint->second).pixelLocation,1);
						if(inside > 0.0)
						{
							(foundPoint->second).fiducialContour = (*sortedContourIter);
							samplePixelHSV(bgimage,(foundPoint->second));
							(foundPoint->second).isColor = true;
							break;
						}
					}
					else
					{
						if(sortedContourIter->pixelMatchID == -1)
						{
							if(sortedContourIter->fiducialColor == color)
							{
								(foundPoint->second).fiducialContour = (*sortedContourIter);
								(foundPoint->second).isColor = true;
								sortedContourIter->pixelMatchID = (foundPoint->second).ID;
								break;
							}
							float distance = distanceBetweenTwoPoints((foundPoint->second).projectedPixelLocation.x,(foundPoint->second).projectedPixelLocation.y,sortedContourIter->pixelLocation.x,sortedContourIter->pixelLocation.y);
							if(distance < distanceToClosestContour)
							{
								closestContour = (*sortedContourIter);
								distanceToClosestContour = distance;
							}
						}
					}
					count++;
				}else
				{
					break;
				}
			}
			if((foundPoint->second).isColor == false)
			{
				if(distanceToClosestContour < FLT_MAX)
				{
					(foundPoint->second).fiducialContour = closestContour;
					(foundPoint->second).isColor = true;
				}
			}
			if((foundPoint->second).isColor == true)
			{
				(foundPoint->second).pixelLocation = findMoment((foundPoint->second).fiducialContour.contour);
				(foundPoint->second).boundingBox = cvBoundingRect((foundPoint->second).fiducialContour.contour);
			}
		}
	}
}
