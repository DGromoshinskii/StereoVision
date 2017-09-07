#ifndef PCL_WRAPPER_H
#define PCL_WRAPPER_H

#ifdef _WIN32
#pragma warning (disable : 4521)
#pragma warning (disable : 4503)
#pragma warning (disable : 4996)
#endif

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/bilateral.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/passthrough.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

#include <boost/thread/thread.hpp>

#include <stdio.h>
#include <math.h>

#include "ExportDll.h"
#include "ErrorListSR.h"

using namespace std;

struct DistHist
{
  float m_binDist;
  int m_binSize;
  DistHist()
  {
    m_binDist = -1.f;
    m_binSize = -1;
  }
  DistHist(float binDist, int binSize)
  {
    m_binDist = binDist;
    m_binSize = binSize;
  }
};

namespace PCL_WRAPPER
{
  struct StatFilterParams
  {
    int m_meanK;
    double m_stdDevMulThresh;
    bool m_isKeepOrganize;
    StatFilterParams()
    {
      m_meanK = 0;
      m_stdDevMulThresh = 0.;
      m_isKeepOrganize = false;
    }
    StatFilterParams(int meanK, double stdDevMulThresh, bool isKeepOrganize)
    {
      m_meanK = meanK;
      m_stdDevMulThresh = stdDevMulThresh;
      m_isKeepOrganize = isKeepOrganize;
    }
  };

  struct PassThroughFilterParams
  {
    string m_fieldName;
    float m_lowerLimit;
    float m_upperLimit;
    PassThroughFilterParams()
    {
      m_fieldName = "";
      m_lowerLimit = 0.f;
      m_upperLimit = 0.f;
    }
    PassThroughFilterParams(const string &fieldName, float lowerLimit, float upperLimit)
    {
      m_fieldName = fieldName;
      m_lowerLimit = lowerLimit;
      m_upperLimit = upperLimit;
    }
  };

  STEREO_RECONSTRUCTION_DLL void BuildDistanceHist(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &inputCloud, vector<DistHist> &distanceHistogram, int &maxBinHeight, float histStep);
  STEREO_RECONSTRUCTION_DLL void BuildDistanceHist(vector<float> &distancesVector, vector<DistHist> &distanceHistogram, int &maxBinHeight, float histStep);
  STEREO_RECONSTRUCTION_DLL void ShowDistanceHist(const vector<DistHist> &distanceHistogram, int maxBinNumber, cv::Mat &histImage);
  STEREO_RECONSTRUCTION_DLL void PCLStatisticalFilter(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &inputCloud, const PCL_WRAPPER::StatFilterParams &filterParams, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &filteredCloud, pcl::IndicesConstPtr &removedInds);
  STEREO_RECONSTRUCTION_DLL void PassThroughFilter(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &inputCloud, const PCL_WRAPPER::PassThroughFilterParams &PTFilterParams,
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr &filteredCloud);
  STEREO_RECONSTRUCTION_DLL int GetPointCloudArea(const cv::Point3f &areaCenter, double searchRadius, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &sourceCloud,
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloudPart);
  STEREO_RECONSTRUCTION_DLL void PlanarSegmentation(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &inputCloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &planarCloud, pcl::ModelCoefficients::Ptr &planeCoeffs);
  STEREO_RECONSTRUCTION_DLL void CalculatePlanarDeviation(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &inputCloud, float &planarDev, float &maxDev, bool buildHist = false);
  STEREO_RECONSTRUCTION_DLL float CalcPointToPlaneDist(const pcl::PointXYZRGB &cloudPoint, const pcl::ModelCoefficients::Ptr &planeCoefficients);
  STEREO_RECONSTRUCTION_DLL boost::shared_ptr<pcl::visualization::PCLVisualizer> createVisualizer (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud);
  STEREO_RECONSTRUCTION_DLL int CalculateTransformMatrix(const vector<cv::Point3f> &srcOrigin, const vector<cv::Point3f> &dstOrigin, cv::Mat &transformMat,
                                cv::Mat &inliersMat, double ransacThreshold = 3.0, double confidenceLevel = 0.99);
  STEREO_RECONSTRUCTION_DLL void GenerateDstOriginPoints(vector<cv::Point3f> &dstOrigin);
  STEREO_RECONSTRUCTION_DLL bool SaveTransformData(const string &fileName, const cv::Mat &transformMat, const vector<cv::Point3f> &srcOrigin, const vector<cv::Point3f> &dstOrigin,
                         const cv::Mat &inliersMat, double ransacThreshold, double confidenceLevel);
  STEREO_RECONSTRUCTION_DLL bool LoadTransformData(const string &fileName, cv::Mat &transformMat, vector<cv::Point3f> &srcOrigin, vector<cv::Point3f> &dstOrigin,
                         cv::Mat &inliersMat, double &ransacThreshold, double &confidenceLevel);
  STEREO_RECONSTRUCTION_DLL int TransformPoint(const cv::Point3f &queryPoint, cv::Point3f &transformPoint, const cv::Mat &transformMat);
}

#endif
