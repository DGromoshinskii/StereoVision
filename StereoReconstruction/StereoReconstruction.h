#ifndef STEREO_RECONSTRUCTION_H
#define STEREO_RECONSTRUCTION_H

/// OpenCV
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
/// PCL
#ifdef _WIN32
#pragma warning (disable : 4521)
#pragma warning (disable : 4503)
#pragma warning (disable : 4996)
#include <time.h>
#endif

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/bilateral.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/passthrough.h>

#include "boost/math/special_functions/round.hpp"

/// local headers
#include "ErrorListSR.h"
#include "ExportDll.h"
#include "PCLWrapper.h"

using namespace std;

struct CameraPairMat
{
  CameraPairMat()
  {
    ClearData();
  }
  void ClearData()
  {
    m_rotMat.release();
    m_transMat.release();
    m_R1.release();
    m_R2.release();
    m_P1.release();
    m_P2.release();
    m_Q.release();
    m_baseCamGripperToManBase.release();
    m_auxCamGripperToManBase.release();
  }
  cv::Mat m_rotMat;
  cv::Mat m_transMat;
  cv::Mat m_R1;
  cv::Mat m_R2;
  cv::Mat m_P1;
  cv::Mat m_P2;
  cv::Mat m_Q;

  cv::Mat m_baseCamGripperToManBase;
  cv::Mat m_auxCamGripperToManBase;
};

struct InternalCalibMat
{
  InternalCalibMat()
  {
    m_calibMat.release();
    m_distortMat.release();
  }
  cv::Mat m_calibMat;
  cv::Mat m_distortMat;
};

struct CameraPairParams
{
  CameraPairParams()
  {
    m_roiLeft = cv::Rect(0, 0, 0, 0);
    m_roiRight = cv::Rect(0, 0, 0, 0);
  }
  void ClearData()
  {
    m_roiLeft = cv::Rect(0, 0, 0, 0);
    m_roiRight = cv::Rect(0, 0, 0, 0);
    m_cameraPairMat.ClearData();
  }
  CameraPairMat m_cameraPairMat;
  cv::Rect m_roiLeft;
  cv::Rect m_roiRight;
};

struct StereoParams
{
  StereoParams()
  {
    m_preFilterCap = 0;
    m_SADWinSize = 5;
    m_minDisparity = 0;
    m_uniquenessRatio = 1;
    m_speckleWinSize = 0;
    m_speckleRange = 1;
    disp12MaxDiff = -1;
    m_fullDP = false;
    m_numberOfDisparities = 640; // must be divisible by 16
    m_P1 = 600;
    m_P2 = 2400;
  }
  int m_preFilterCap;
  int m_SADWinSize;
  int m_P1;
  int m_P2;
  int m_minDisparity;
  int m_numberOfDisparities;
  int m_uniquenessRatio;
  int m_speckleWinSize;
  int m_speckleRange;
  int disp12MaxDiff;
  bool m_fullDP;
};

struct CameraPos
{
  CameraPos()
  {
    m_x = DBL_MAX;
    m_y = DBL_MAX;
    m_z = DBL_MAX;
    m_alpha = DBL_MAX;
    m_beta = DBL_MAX;
    m_gamma = DBL_MAX;
  }
  double m_x;
  double m_y;
  double m_z;
  double m_alpha;
  double m_beta;
  double m_gamma;
};

struct ImagePair
{
  ImagePair()
  {
    m_leftImage.release();
    m_rightImage.release();
    m_leftRectImage.release();
    m_rightRectImage.release();
    m_dispImage.release();
  }
  void ClearData()
  {
    m_leftImage.release();
    m_rightImage.release();
    m_leftRectImage.release();
    m_rightRectImage.release();
    m_dispImage.release();
    m_source3dPointCloud->clear();
    m_filtered3dPointCloud->clear();
    m_leftImToSrcCloud.clear();
    m_srcCloudToLeftIm.clear();
  }
  cv::Mat m_leftImage;
  cv::Mat m_rightImage;
  cv::Mat m_leftRectImage;
  cv::Mat m_rightRectImage;
  cv::Mat m_dispImage;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_source3dPointCloud;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_filtered3dPointCloud;
  pcl::IndicesConstPtr m_removedIndices;
  map<int, int> m_leftImToSrcCloud;
	map<int, int> m_srcCloudToLeftIm;
};

struct PointIndex
{
	PointIndex()
	{
		m_index = UINT_MAX;
		m_existance = true;
	}
	unsigned int m_index;
	bool m_existance;
};

class StereoReconstruction
{
public:
	STEREO_RECONSTRUCTION_DLL StereoReconstruction();
	STEREO_RECONSTRUCTION_DLL ~StereoReconstruction();
	STEREO_RECONSTRUCTION_DLL int ImagePairHandler(int pairID);
  STEREO_RECONSTRUCTION_DLL bool LoadInternalCalibMatrices(const string &filePath);
  STEREO_RECONSTRUCTION_DLL bool LoadExternalCalibMatrices(const string &filePath, int pairID);
  STEREO_RECONSTRUCTION_DLL bool SetImagePair(int pairID, const cv::Mat &leftImage, const cv::Mat &rightImage);
  
	STEREO_RECONSTRUCTION_DLL const pcl::PointCloud<pcl::PointXYZRGB>::Ptr *const GetSource3dPointCloud(int pairID) const;
	STEREO_RECONSTRUCTION_DLL const pcl::PointCloud<pcl::PointXYZRGB>::Ptr *const GetFiltered3dPointCloud(int pairID) const;

	STEREO_RECONSTRUCTION_DLL void SetStereoAlgParams(const StereoParams &stereoAlgParams);
  STEREO_RECONSTRUCTION_DLL void GetStereoAlgParams(StereoParams &stereoAlgParams) const;

  STEREO_RECONSTRUCTION_DLL void SetUseOneCameraFlag(bool enable);
  STEREO_RECONSTRUCTION_DLL bool GetUseOneCameraFlag() const;

  STEREO_RECONSTRUCTION_DLL void SetStatFilterParams(const PCL_WRAPPER::StatFilterParams &statFilterP);
  STEREO_RECONSTRUCTION_DLL const PCL_WRAPPER::StatFilterParams &GetStatFilterParams() const;

  STEREO_RECONSTRUCTION_DLL const ImagePair *const GetImagePair(int pairID) const;
  STEREO_RECONSTRUCTION_DLL const CameraPairParams *const GetCamaraPairParams(int pairID) const;

  STEREO_RECONSTRUCTION_DLL void SetUseValidROIFiltrationFlag(bool enable);
  STEREO_RECONSTRUCTION_DLL bool GetUseValidROIFiltrationFlag() const;

  STEREO_RECONSTRUCTION_DLL void SetUseDistZThreshold(bool enable);
  STEREO_RECONSTRUCTION_DLL bool GetUseDistZThreshold() const;

  STEREO_RECONSTRUCTION_DLL void SetLowerDistZThreshold(float lowerDistZThreshold);
  STEREO_RECONSTRUCTION_DLL float GetLowerDistZThreshold() const;

  STEREO_RECONSTRUCTION_DLL void SetUpperDistZThreshold(float upperDistZThreshold);
  STEREO_RECONSTRUCTION_DLL float GetUpperDistZThreshold() const;

  STEREO_RECONSTRUCTION_DLL void SetInputImageScale(double imageScale);
  STEREO_RECONSTRUCTION_DLL double GetInputImageScale() const;

  STEREO_RECONSTRUCTION_DLL int Visualize3dPointCloud(int pairID);

  STEREO_RECONSTRUCTION_DLL int SetInternalCalibMat(const cv::Mat &calibMat, const cv::Mat &cameraDistortion, bool isFirstCam);
  //////////////////////////////////////////////////////////////////////////
  STEREO_RECONSTRUCTION_DLL int StereoImageRectify(int pairID);

  
  STEREO_RECONSTRUCTION_DLL void FormRotMat(double rotAngleXRad, double rotAngleYRad, double rotAngleZRad, cv::Mat &rotMat);
  STEREO_RECONSTRUCTION_DLL static float Calc3dPointsDist(const cv::Point3f &firstPoint, const cv::Point3f &secondPoint);
  STEREO_RECONSTRUCTION_DLL void FormTransitionMat(double rotAngleXRad, double rotAngleYRad, double rotAngleZRad, 
                                                   double distX, double distY, double distZ, cv::Mat &transMat);
  STEREO_RECONSTRUCTION_DLL int ClearData(int pairID);
  
  STEREO_RECONSTRUCTION_DLL void SetUsePreRectifyImages(bool enable);
  STEREO_RECONSTRUCTION_DLL void SetUseScaledImages(bool enable);
  STEREO_RECONSTRUCTION_DLL void SetUseBilateralFilter(bool enable);

protected:
  int Calc3dCoord(const cv::Point& dispMapCoord, double dispVal, const cv::Mat &dispTo3dMat, cv::Point3f &pointCoord3d); 
  bool CalculateDisparityImage(int pairID);

  int Calculate3dPointCloud(int pairID);
  int PostFilteration(int pairID);
  int StatisticalFilteration(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source3dPointCloud, const PCL_WRAPPER::StatFilterParams &filterParams, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &filtered3dPointCloud, pcl::IndicesConstPtr &removedInds);
  double MatrixInversion(const cv::Mat &srcMat, cv::Mat &inverseMat, int &usedMethod);
  //////////////////////////////////////////////////////////////////////////
  void FormRotMat(int rotAxis, double rotAngleRad, cv::Mat &rotMat);
  bool DecomposTransMat(const cv::Mat &transitionMat, cv::Mat &rotMat, cv::Mat &translationVector);

protected:
  cv::StereoSGBM m_stereoSGBM;
  map<int, ImagePair> m_imagePairs;
  map<int, CameraPairParams> m_cameraPairParams;
	bool m_useValidROIFiltration;
  bool m_useDistZThreshold;
  float m_lowerDistZThreshold;
  float m_upperDistZThreshold;
  PCL_WRAPPER::StatFilterParams m_statFilterParams;
  bool m_useOneCamera;
  InternalCalibMat m_internalCalibMat;
  InternalCalibMat m_internalCalibMatSecond;
  double m_inputImageScale;
  cv::Mat m_camToManip;
  cv::Mat m_manipToCam;

  bool m_usePreRectifyImages;
  bool m_useScaledImages;
  bool m_useBilateralFilter;
}; 

#endif
