#include "StereoReconstruction.h"

StereoReconstruction::StereoReconstruction()
{
  m_useValidROIFiltration = true;
  m_useDistZThreshold = true;
  m_lowerDistZThreshold = 0.f;
  m_upperDistZThreshold = 55.5f;
  m_statFilterParams.m_isKeepOrganize = true;
  m_statFilterParams.m_meanK = 3;
  m_statFilterParams.m_stdDevMulThresh = 0.2;
	m_useOneCamera = false;
  m_inputImageScale = 1.;

  m_usePreRectifyImages = false;
  m_useScaledImages = false;
  m_useBilateralFilter = false;
}

StereoReconstruction::~StereoReconstruction()
{
  //
}

bool StereoReconstruction::CalculateDisparityImage(int pairID)
{
  map<int, ImagePair>::iterator curImagePairIter = m_imagePairs.find(pairID);
  if (curImagePairIter == m_imagePairs.end())
    return false;

#ifdef _DEBUG
  cv::imshow("FlipL", curImagePairIter->second.m_leftRectImage);
  cv::imshow("FlipR", curImagePairIter->second.m_rightRectImage);
  cvWaitKey();
#endif

  cv::Mat dispImage;
  m_stereoSGBM(curImagePairIter->second.m_leftRectImage, curImagePairIter->second.m_rightRectImage, dispImage);

  curImagePairIter->second.m_dispImage.release();

	dispImage.convertTo(curImagePairIter->second.m_dispImage, CV_32F, 1. / 16);

  if (m_useBilateralFilter)
  {
    cv::Mat bilFilter;
    float sizeNeigh = 7.f;
    cv::bilateralFilter(curImagePairIter->second.m_dispImage, bilFilter, sizeNeigh, sizeNeigh * 2, sizeNeigh / 2);
    bilFilter.copyTo(curImagePairIter->second.m_dispImage);
  }

  return true;
}

int StereoReconstruction::Calc3dCoord(const cv::Point& dispMapCoord, double dispVal, const cv::Mat &dispTo3dMat, cv::Point3f &pointCoord3d)
{
  if ( dispVal == 0 ) 
    return SR_ERROR_ZERO_DISPARITY_ERROR; //Discard bad pixels
  double coef = 1.;
  double pw = /*-1.0 **/ (dispVal * dispTo3dMat.at<double>(3, 2)) / coef + dispTo3dMat.at<double>(3, 3); 

  double xVal = static_cast<double>(dispMapCoord.x) + dispTo3dMat.at<double>(0, 3);
  double yVal = static_cast<double>(dispMapCoord.y) + dispTo3dMat.at<double>(1, 3);
  double zVal = dispTo3dMat.at<double>(2, 3);

  if (fabs(pw) <= DBL_EPSILON)
    return SR_ERROR_DIVISION_BY_ZERO_ERROR;

  zVal = zVal / pw;
  xVal = xVal / pw;
  yVal = yVal / pw;

  pointCoord3d = cv::Point3f((float)xVal, (float)yVal, (float)zVal);

  return SR_ERROR_OK;
}

int StereoReconstruction::Calculate3dPointCloud(int pairID)
{
  map<int, ImagePair>::iterator curImagePairIter = m_imagePairs.find(pairID);
  if (curImagePairIter == m_imagePairs.end())
    return SR_ERROR_IMAGE_PAIR_NOT_EXIST_ERROR;

  map<int, CameraPairParams>::iterator curCameraPairParamsIter = m_cameraPairParams.find(pairID);
  if (curCameraPairParamsIter == m_cameraPairParams.end())
    return SR_ERROR_CAMERA_PAIR_PARAMS_NOT_EXIST_ERROR;

  if (curImagePairIter->second.m_source3dPointCloud == NULL)
  {
    curImagePairIter->second.m_source3dPointCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
  }
  else if (curImagePairIter->second.m_source3dPointCloud->size() != 0)
  {
    curImagePairIter->second.m_source3dPointCloud->clear();
  }

  if (!curImagePairIter->second.m_leftImToSrcCloud.empty())
    curImagePairIter->second.m_leftImToSrcCloud.clear();

	if (!curImagePairIter->second.m_srcCloudToLeftIm.empty())
		curImagePairIter->second.m_srcCloudToLeftIm.clear();

  uchar pr, pg, pb;

	const cv::Rect &roiLeft = curCameraPairParamsIter->second.m_roiLeft;
	bool useROIFiltration = true;
	if (!m_useValidROIFiltration || roiLeft.area() == 0)
		useROIFiltration = false;

  int curPixelIndex = -1;
  int curCloudPointInd = -1;
  for (int i = 0; i < curImagePairIter->second.m_leftRectImage.rows; i++)
  {
    uchar* leftImRowPtr = curImagePairIter->second.m_leftRectImage.ptr<uchar>(i);
    float* dispImRowPtr = curImagePairIter->second.m_dispImage.ptr<float>(i);
    
    for (int j = 0; j < curImagePairIter->second.m_leftRectImage.cols; j++)
    {
      curPixelIndex++;
      if (useROIFiltration)
			{
				if (!roiLeft.contains(cv::Point(j, i)))
				{
          dispImRowPtr[j] = 0.f;
					continue;
				}
			}

      //////////////////////////////////////////////////////////////////////////
      if (dispImRowPtr[j] < (float)m_stereoSGBM.minDisparity)
        dispImRowPtr[j] = 0.f;
      //////////////////////////////////////////////////////////////////////////


      cv::Point3f pointCoord3d;
      if (Calc3dCoord(cv::Point(j, i), dispImRowPtr[j], curCameraPairParamsIter->second.m_cameraPairMat.m_Q, pointCoord3d) != SR_ERROR_OK)
        continue;

      if (m_useDistZThreshold                    && 
         (pointCoord3d.z > m_upperDistZThreshold || 
          pointCoord3d.z < m_lowerDistZThreshold))
          continue;

      curCloudPointInd++;
      curImagePairIter->second.m_leftImToSrcCloud.insert(pair<int, int> (curPixelIndex, curCloudPointInd));
			curImagePairIter->second.m_srcCloudToLeftIm.insert(pair<int, int> (curCloudPointInd, curPixelIndex));

      //Get RGB info
      pb = leftImRowPtr[3*j];
      pg = leftImRowPtr[3*j+1];
      pr = leftImRowPtr[3*j+2];

      //Insert info into point cloud structure
      pcl::PointXYZRGB point;
      point.x = pointCoord3d.x;
      point.y = pointCoord3d.y;
      point.z = pointCoord3d.z;
      uint32_t rgb = (static_cast<uint32_t>(pr) << 16 |
        static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));
      point.rgb = *reinterpret_cast<float*>(&rgb);

      curImagePairIter->second.m_source3dPointCloud->points.push_back (point);
    }
  }

  curImagePairIter->second.m_source3dPointCloud->width = curImagePairIter->second.m_source3dPointCloud->size();
  curImagePairIter->second.m_source3dPointCloud->height = 1;

  return SR_ERROR_OK;
}

void StereoReconstruction::SetStereoAlgParams(const StereoParams &stereoAlgParams)
{
	m_stereoSGBM.disp12MaxDiff = stereoAlgParams.disp12MaxDiff;
	m_stereoSGBM.fullDP = stereoAlgParams.m_fullDP;
	m_stereoSGBM.minDisparity = stereoAlgParams.m_minDisparity;
	m_stereoSGBM.numberOfDisparities = stereoAlgParams.m_numberOfDisparities;
	m_stereoSGBM.P1 = stereoAlgParams.m_P1;
	m_stereoSGBM.P2 = stereoAlgParams.m_P2;
	m_stereoSGBM.preFilterCap = stereoAlgParams.m_preFilterCap;
	m_stereoSGBM.SADWindowSize = stereoAlgParams.m_SADWinSize;
	m_stereoSGBM.speckleRange = stereoAlgParams.m_speckleRange;
	m_stereoSGBM.speckleWindowSize = stereoAlgParams.m_speckleWinSize;
	m_stereoSGBM.uniquenessRatio = stereoAlgParams.m_uniquenessRatio;
}

void StereoReconstruction::GetStereoAlgParams(StereoParams &stereoAlgParams) const
{
  stereoAlgParams.disp12MaxDiff = m_stereoSGBM.disp12MaxDiff;
  stereoAlgParams.m_fullDP = m_stereoSGBM.fullDP;
  stereoAlgParams.m_minDisparity = m_stereoSGBM.minDisparity;
  stereoAlgParams.m_numberOfDisparities = m_stereoSGBM.numberOfDisparities;
  stereoAlgParams.m_P1 = m_stereoSGBM.P1;
  stereoAlgParams.m_P2 = m_stereoSGBM.P2;
  stereoAlgParams.m_preFilterCap = m_stereoSGBM.preFilterCap;
  stereoAlgParams.m_SADWinSize = m_stereoSGBM.SADWindowSize;
  stereoAlgParams.m_speckleRange = m_stereoSGBM.speckleRange;
  stereoAlgParams.m_speckleWinSize = m_stereoSGBM.speckleWindowSize;
  stereoAlgParams.m_uniquenessRatio = m_stereoSGBM.uniquenessRatio;
}

bool StereoReconstruction::SetImagePair(int pairID, const cv::Mat &leftImage, const cv::Mat &rightImage)
{
	if (leftImage.empty() || rightImage.empty())
		return false;

  map<int, ImagePair>::iterator curImagePairIter = m_imagePairs.find(pairID);
  if (curImagePairIter == m_imagePairs.end())
  {
    ImagePair tempImPair;
    tempImPair.m_leftImage = leftImage;
    tempImPair.m_rightImage = rightImage;
    m_imagePairs.insert(pair<int, ImagePair> (pairID, tempImPair));
  }
  else
  {
    curImagePairIter->second.ClearData();
    curImagePairIter->second.m_leftImage = leftImage;
    curImagePairIter->second.m_rightImage = rightImage;
  }

  return true;
}

int StereoReconstruction::PostFilteration(int pairID)
{
  map<int, ImagePair>::iterator curImagePairIter = m_imagePairs.find(pairID);
  if (curImagePairIter == m_imagePairs.end())
    return SR_ERROR_IMAGE_PAIR_NOT_EXIST_ERROR;

  if (curImagePairIter->second.m_filtered3dPointCloud == NULL)
  {
    curImagePairIter->second.m_filtered3dPointCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
  }
  else if (curImagePairIter->second.m_filtered3dPointCloud->size() != 0)
  {
    curImagePairIter->second.m_filtered3dPointCloud->clear();
  }

  int statFilterError = StatisticalFilteration(curImagePairIter->second.m_source3dPointCloud, m_statFilterParams, curImagePairIter->second.m_filtered3dPointCloud, curImagePairIter->second.m_removedIndices);
  if (statFilterError != SR_ERROR_OK)
    return statFilterError;


  //curImagePairIter->second.m_filtered3dPointCloud = curImagePairIter->second.m_source3dPointCloud;
  // Set to -1 index of the filtered points
  for (vector<int>::const_iterator remPointsIndIter = curImagePairIter->second.m_removedIndices->begin(); remPointsIndIter != curImagePairIter->second.m_removedIndices->end(); ++remPointsIndIter)
  {
    map<int, int>::const_iterator cloudToLeftIter = curImagePairIter->second.m_srcCloudToLeftIm.find(*remPointsIndIter);
		if (cloudToLeftIter != curImagePairIter->second.m_srcCloudToLeftIm.end())
		{
			map<int, int>::iterator leftToCloudIter = curImagePairIter->second.m_leftImToSrcCloud.find(cloudToLeftIter->second);
			if (leftToCloudIter != curImagePairIter->second.m_leftImToSrcCloud.end())
			{
				leftToCloudIter->second = -1;
			}
			else
			{
				//cout << "Filtering Error!" << endl;
			}
		}
		else
		{
			//cout << "Filtering Error!" << endl;
		}
  }

#ifdef _DEBUG
  cout << "Source Points size = " <<curImagePairIter->second.m_source3dPointCloud->size() << endl;
  cout << "Filter Points size = " <<curImagePairIter->second.m_filtered3dPointCloud->size() << endl;
  cout << "Removed Ind = " << curImagePairIter->second.m_removedIndices->size() << endl;
#endif

  return SR_ERROR_OK;
}

int StereoReconstruction::StatisticalFilteration(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &source3dPointCloud, const PCL_WRAPPER::StatFilterParams &filterParams, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &filtered3dPointCloud, 
                                                 pcl::IndicesConstPtr &removedInds)
{
  if (source3dPointCloud == NULL)
    return SR_ERROR_INPUT_POINT_CLOUD_IS_EMPTY_ERROR;
  else if (source3dPointCloud->empty())
    return SR_ERROR_INPUT_POINT_CLOUD_IS_EMPTY_ERROR;

  PCL_WRAPPER::PCLStatisticalFilter(source3dPointCloud, filterParams, filtered3dPointCloud, removedInds);

  return SR_ERROR_OK;
}

int StereoReconstruction::StereoImageRectify(int pairID)
{
  map<int, ImagePair>::iterator curImagePairIter = m_imagePairs.find(pairID);
  if (curImagePairIter == m_imagePairs.end())
    return SR_ERROR_IMAGE_PAIR_NOT_EXIST_ERROR;

  map<int, CameraPairParams>::iterator curCameraPairParamsIter = m_cameraPairParams.find(pairID);
  if (curCameraPairParamsIter == m_cameraPairParams.end())
    return SR_ERROR_CAMERA_PAIR_PARAMS_NOT_EXIST_ERROR;

  CameraPairMat &tempCamPairParams = curCameraPairParamsIter->second.m_cameraPairMat;
  ImagePair &tempImagePair = curImagePairIter->second;


  /// Scale image
  cv::Mat scaledLeftImage;
  cv::Mat scaledRightImage;
  cv::Mat intCalibMatFirst;
  cv::Mat intCalibMatSecond;
  if (m_inputImageScale != 1.)
  {
    //////////////////////////////////////////////////////////////////////////
    if (m_useScaledImages)
    {
      scaledLeftImage = tempImagePair.m_leftImage;
      scaledRightImage = tempImagePair.m_rightImage;
    }
    else
    {
      int method = m_inputImageScale < 1. ? cv::INTER_AREA : cv::INTER_CUBIC;
      cv::resize(tempImagePair.m_leftImage, scaledLeftImage, cv::Size(), m_inputImageScale, m_inputImageScale, method);
      cv::resize(tempImagePair.m_rightImage, scaledRightImage, cv::Size(), m_inputImageScale, m_inputImageScale, method);
    }
    //////////////////////////////////////////////////////////////////////////
    intCalibMatFirst = m_internalCalibMat.m_calibMat * m_inputImageScale;
    intCalibMatSecond = m_internalCalibMatSecond.m_calibMat * m_inputImageScale;
  }
  else
  {
    scaledLeftImage = tempImagePair.m_leftImage;
    scaledRightImage = tempImagePair.m_rightImage;

    intCalibMatFirst = m_internalCalibMat.m_calibMat;
    intCalibMatSecond = m_internalCalibMatSecond.m_calibMat;
  }
  
  intCalibMatFirst.at<double> (2,2) = 1.;
  intCalibMatSecond.at<double> (2,2) = 1.;
  cout << "calib mat = " << intCalibMatFirst << endl;

  cv::Size imSize(scaledLeftImage.cols, scaledLeftImage.rows);

  cv::Size addImSize(2560, 1440);
  addImSize = imSize;
  
	///!!
	cv::Mat map11, map12, map21, map22;
	if (m_useOneCamera)
	{
    cv::stereoRectify(intCalibMatFirst, m_internalCalibMat.m_distortMat, intCalibMatSecond, m_internalCalibMatSecond.m_distortMat,
      imSize, tempCamPairParams.m_rotMat, tempCamPairParams.m_transMat, tempCamPairParams.m_R1, tempCamPairParams.m_R2, tempCamPairParams.m_P1,
      tempCamPairParams.m_P2, tempCamPairParams.m_Q, cv::CALIB_ZERO_DISPARITY/*0*/, -1, /*imSize*/addImSize, &curCameraPairParamsIter->second.m_roiLeft, &curCameraPairParamsIter->second.m_roiRight);

#ifdef _DEBUG
    cout << "P1" << tempCamPairParams.m_P1 << endl << endl;
    cout << "P2" << tempCamPairParams.m_P2 << endl << endl;
    cout << "Q" << tempCamPairParams.m_Q << endl << endl;
#endif
    
    initUndistortRectifyMap(intCalibMatFirst, m_internalCalibMat.m_distortMat, tempCamPairParams.m_R1, tempCamPairParams.m_P1, /*imSize*/addImSize, CV_16SC2, map11, map12);
    initUndistortRectifyMap(intCalibMatSecond, m_internalCalibMatSecond.m_distortMat, tempCamPairParams.m_R2, tempCamPairParams.m_P2, /*imSize*/addImSize, CV_16SC2, map21, map22);
	}
	else
	{
		cv::stereoRectify(intCalibMatFirst, m_internalCalibMat.m_distortMat, intCalibMatSecond, m_internalCalibMatSecond.m_distortMat,
			                imSize, tempCamPairParams.m_rotMat, tempCamPairParams.m_transMat, tempCamPairParams.m_R1, tempCamPairParams.m_R2, tempCamPairParams.m_P1,
			                tempCamPairParams.m_P2, tempCamPairParams.m_Q, cv::CALIB_ZERO_DISPARITY, 0, imSize, &curCameraPairParamsIter->second.m_roiLeft, &curCameraPairParamsIter->second.m_roiRight);

		initUndistortRectifyMap(intCalibMatFirst, m_internalCalibMat.m_distortMat, tempCamPairParams.m_R1, tempCamPairParams.m_P1, imSize, CV_16SC2, map11, map12);
		initUndistortRectifyMap(intCalibMatSecond, m_internalCalibMatSecond.m_distortMat, tempCamPairParams.m_R2, tempCamPairParams.m_P2, imSize, CV_16SC2, map21, map22);
	}

  remap(scaledLeftImage, tempImagePair.m_leftRectImage, map11, map12, cv::INTER_LINEAR);
  remap(scaledRightImage, tempImagePair.m_rightRectImage, map21, map22, cv::INTER_LINEAR);

  if (m_usePreRectifyImages)
  {
    tempImagePair.m_rightRectImage = scaledRightImage;
    tempImagePair.m_leftRectImage = scaledLeftImage;
  }

  return SR_ERROR_OK;
}

int StereoReconstruction::ImagePairHandler(int pairID)
{
  int stereoRectError = StereoImageRectify(pairID);
  if (stereoRectError != SR_ERROR_OK)
    return stereoRectError;
  
  bool calcDispMapError = CalculateDisparityImage(pairID);
  if (!calcDispMapError)
    return SR_ERROR_IMAGE_PAIR_NOT_EXIST_ERROR;

  int calc3dPointCloudError = Calculate3dPointCloud(pairID);
  if (calc3dPointCloudError != SR_ERROR_OK)
    return calc3dPointCloudError;

  int postFiltrationError = PostFilteration(pairID);
  if (postFiltrationError != SR_ERROR_OK)
    return postFiltrationError;

  return SR_ERROR_OK;
}

bool StereoReconstruction::LoadInternalCalibMatrices(const string &filePath)
{
  cv::FileStorage fs(filePath, CV_STORAGE_READ);
  if (!fs.isOpened())
    return false;

  if (!m_internalCalibMat.m_calibMat.empty())
    m_internalCalibMat.m_calibMat.release();
  if (!m_internalCalibMat.m_distortMat.empty())
    m_internalCalibMat.m_distortMat.release();
	if (!m_internalCalibMatSecond.m_calibMat.empty())
		m_internalCalibMatSecond.m_calibMat.release();
	if (!m_internalCalibMatSecond.m_distortMat.empty())
		m_internalCalibMatSecond.m_distortMat.release();

	if (m_useOneCamera)
	{
		fs["M"] >> m_internalCalibMat.m_calibMat;
		fs["D"] >> m_internalCalibMat.m_distortMat;
	}
	else
	{
		fs["M1"] >> m_internalCalibMat.m_calibMat;
		fs["D1"] >> m_internalCalibMat.m_distortMat;

		fs["M2"] >> m_internalCalibMatSecond.m_calibMat;
		fs["D2"] >> m_internalCalibMatSecond.m_distortMat;

		if (m_internalCalibMatSecond.m_calibMat.empty() || m_internalCalibMatSecond.m_distortMat.empty())
			return false;
	}
  if (m_internalCalibMat.m_calibMat.empty() || m_internalCalibMat.m_distortMat.empty())
    return false;

  return true;
}

bool StereoReconstruction::LoadExternalCalibMatrices(const string &filePath, int pairID)
{
	m_cameraPairParams[pairID];
  map<int, CameraPairParams>::iterator curCameraPairParamsIter = m_cameraPairParams.find(pairID);
  if (curCameraPairParamsIter == m_cameraPairParams.end())
    return SR_ERROR_CAMERA_PAIR_PARAMS_NOT_EXIST_ERROR;

  cv::FileStorage fs(filePath, CV_STORAGE_READ);
  if (!fs.isOpened())
    return false;

  if (!curCameraPairParamsIter->second.m_cameraPairMat.m_rotMat.empty())
    curCameraPairParamsIter->second.m_cameraPairMat.m_rotMat.release();
  if (!curCameraPairParamsIter->second.m_cameraPairMat.m_transMat.empty())
    curCameraPairParamsIter->second.m_cameraPairMat.m_transMat.release();

  fs["R"] >> curCameraPairParamsIter->second.m_cameraPairMat.m_rotMat;
  fs["T"] >> curCameraPairParamsIter->second.m_cameraPairMat.m_transMat;

  if (curCameraPairParamsIter->second.m_cameraPairMat.m_rotMat.empty() ||
      curCameraPairParamsIter->second.m_cameraPairMat.m_transMat.empty())
      return false;
  
  return true;
}

const pcl::PointCloud<pcl::PointXYZRGB>::Ptr *const StereoReconstruction::GetSource3dPointCloud(int pairID) const
{
	map<int, ImagePair>::const_iterator curImagePairIter = m_imagePairs.find(pairID);
	if (curImagePairIter == m_imagePairs.end())
    return NULL;

  return &curImagePairIter->second.m_source3dPointCloud;
}

const pcl::PointCloud<pcl::PointXYZRGB>::Ptr *const StereoReconstruction::GetFiltered3dPointCloud(int pairID) const
{
  map<int, ImagePair>::const_iterator curImagePairIter = m_imagePairs.find(pairID);
  if (curImagePairIter == m_imagePairs.end())
    return NULL;

  return &curImagePairIter->second.m_filtered3dPointCloud;
}

int StereoReconstruction::Visualize3dPointCloud(int pairID)
{
  map<int, ImagePair>::iterator curImagePairIter = m_imagePairs.find(pairID);
  if (curImagePairIter == m_imagePairs.end())
    return SR_ERROR_IMAGE_PAIR_NOT_EXIST_ERROR;

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = PCL_WRAPPER::createVisualizer(curImagePairIter->second.m_filtered3dPointCloud);

  viewer->removeCoordinateSystem();

  //Main loop
  while ( !viewer->wasStopped())
  {
    viewer->spinOnce(100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }


  return SR_ERROR_OK;
}

void StereoReconstruction::SetUseOneCameraFlag(bool enable)
{
  m_useOneCamera = enable;
}

bool StereoReconstruction::GetUseOneCameraFlag() const
{
  return m_useOneCamera;
}

void StereoReconstruction::SetStatFilterParams(const PCL_WRAPPER::StatFilterParams &statFilterP)
{
  m_statFilterParams = statFilterP;
}

const PCL_WRAPPER::StatFilterParams &StereoReconstruction::GetStatFilterParams() const
{
  return m_statFilterParams;
}

const ImagePair *const StereoReconstruction::GetImagePair(int pairID) const
{
  map<int, ImagePair>::const_iterator curImagePairIter = m_imagePairs.find(pairID);
  if (curImagePairIter == m_imagePairs.end())
    return NULL;

  return &curImagePairIter->second;
}

const CameraPairParams *const StereoReconstruction::GetCamaraPairParams(int pairID) const
{
  map<int, CameraPairParams>::const_iterator curCameraPairParamsIter = m_cameraPairParams.find(pairID);
  if (curCameraPairParamsIter == m_cameraPairParams.end())
    return NULL;

  return &curCameraPairParamsIter->second;
}

void StereoReconstruction::SetUseValidROIFiltrationFlag(bool enable)
{
  m_useValidROIFiltration = enable;
}

bool StereoReconstruction::GetUseValidROIFiltrationFlag() const
{
  return m_useValidROIFiltration;
}

void StereoReconstruction::SetUseDistZThreshold(bool enable)
{
  m_useDistZThreshold = enable;
}

bool StereoReconstruction::GetUseDistZThreshold() const
{
  return m_useDistZThreshold;
}

void StereoReconstruction::SetLowerDistZThreshold(float lowerDistZThreshold)
{
  m_lowerDistZThreshold = lowerDistZThreshold;
}

float StereoReconstruction::GetLowerDistZThreshold() const
{
  return m_lowerDistZThreshold;
}

void StereoReconstruction::SetUpperDistZThreshold(float upperDistZThreshold)
{
  m_upperDistZThreshold = upperDistZThreshold;
}

float StereoReconstruction::GetUpperDistZThreshold() const
{
  return m_upperDistZThreshold;
}

void StereoReconstruction::SetInputImageScale(double imageScale)
{
  m_inputImageScale = imageScale;
}

double StereoReconstruction::GetInputImageScale() const
{
  return m_inputImageScale;
}

double StereoReconstruction::MatrixInversion(const cv::Mat &srcMat, cv::Mat &inverseMat, int &usedMethod)
{
  usedMethod = cv::DECOMP_LU;
  double invRes = DBL_MAX;
  invRes = cv::invert(srcMat, inverseMat, usedMethod);

  if (invRes == 0)
  {
    usedMethod = cv::DECOMP_SVD;
    if (!inverseMat.empty())
      inverseMat.release();
    invRes = cv::invert(srcMat, inverseMat, usedMethod);
  }

  return invRes;
}

int StereoReconstruction::SetInternalCalibMat(const cv::Mat &calibMat, const cv::Mat &cameraDistortion, bool isFirstCam)
{
  if (calibMat.rows != 3 || calibMat.cols != 3 || cameraDistortion.rows != 1 || cameraDistortion.cols != 5)
    return SR_ERROR_WRONG_INPUT_MAT_SIZE_ERROR;

  if (isFirstCam)
  {
    calibMat.copyTo(m_internalCalibMat.m_calibMat);
    cameraDistortion.copyTo(m_internalCalibMat.m_distortMat);
  }
  else
  {
    calibMat.copyTo(m_internalCalibMatSecond.m_calibMat);
    cameraDistortion.copyTo(m_internalCalibMatSecond.m_distortMat);
  }


  return SR_ERROR_OK;
}

void StereoReconstruction::FormRotMat(int rotAxis, double rotAngleRad, cv::Mat &rotMat)
{
  switch (rotAxis)
  {
  case 0:
    {
      rotMat = (cv::Mat_<double>(4,4) <<  1., 0., 0., 0.,
                                          0., cos(rotAngleRad), sin(rotAngleRad), 0.,
                                          0., -sin(rotAngleRad), cos(rotAngleRad), 0.,
                                          0., 0., 0., 1.);
      break;
    }
  case 1:
    {
      rotMat = (cv::Mat_<double>(4,4) <<  cos(rotAngleRad), 0., -sin(rotAngleRad), 0.,
                                          0., 1., 0., 0.,
                                          sin(rotAngleRad), 0., cos(rotAngleRad), 0.,
                                          0., 0., 0., 1.);
      break;
    }
  case 2:
    {
      rotMat = (cv::Mat_<double>(4,4) <<  cos(rotAngleRad), sin(rotAngleRad), 0., 0.,
                                          -sin(rotAngleRad), cos(rotAngleRad), 0., 0.,
                                          0., 0., 1., 0.,
                                          0., 0., 0., 1.);
      break;
    }
  }
}

void StereoReconstruction::FormRotMat(double rotAngleXRad, double rotAngleYRad, double rotAngleZRad, cv::Mat &rotMat)
{
	cv::Mat rotMatX;
	FormRotMat(0, rotAngleXRad, rotMatX);
	cv::Mat rotMatY;
	FormRotMat(1, rotAngleYRad, rotMatY);
	cv::Mat rotMatZ;
	FormRotMat(2, rotAngleZRad, rotMatZ);

	rotMat = rotMatX * rotMatY;
	rotMat = rotMat * rotMatZ;
}

void StereoReconstruction::FormTransitionMat(double rotAngleXRad, double rotAngleYRad, double rotAngleZRad, 
  double distX, double distY, double distZ, cv::Mat &transMat)
{
  FormRotMat(rotAngleXRad, rotAngleYRad, rotAngleZRad, transMat);
  transMat.at<double>(0, 3) = distX;
  transMat.at<double>(1, 3) = distY;
  transMat.at<double>(2, 3) = distZ;
}

float StereoReconstruction::Calc3dPointsDist(const cv::Point3f &firstPoint, const cv::Point3f &secondPoint)
{
  float pointsDist = sqrt(pow((firstPoint.x - secondPoint.x), 2) + pow((firstPoint.y - secondPoint.y), 2) + pow((firstPoint.z - secondPoint.z), 2));

  return pointsDist;
}

bool StereoReconstruction::DecomposTransMat(const cv::Mat &transitionMat, cv::Mat &rotMat, cv::Mat &translationVector)
{
  if (transitionMat.rows != 4 && transitionMat.cols != 4)
    return false;

  cv::Rect rotMatRect(0, 0, 3, 3);
  cv::Rect transVecRect(3, 0, 1, 3);

  transitionMat(rotMatRect).copyTo(rotMat);
  transitionMat(transVecRect).copyTo(translationVector);

  return true;
}

int StereoReconstruction::ClearData(int pairID)
{
  if (pairID == -1)
  {
    for (map<int, ImagePair>::iterator iter = m_imagePairs.begin(); iter != m_imagePairs.end(); ++iter)
    {
      iter->second.ClearData();
    }
    for (map<int, CameraPairParams>::iterator iter = m_cameraPairParams.begin(); iter != m_cameraPairParams.end(); ++iter)
    {
      iter->second.ClearData();
    }

    m_imagePairs.clear();
    m_cameraPairParams.clear();
  }
  else
  {
    map<int, ImagePair>::iterator curImagePairIter = m_imagePairs.find(pairID);
    if (curImagePairIter == m_imagePairs.end())
      return SR_ERROR_IMAGE_PAIR_NOT_EXIST_ERROR;

    curImagePairIter->second.ClearData();
    m_imagePairs.erase(curImagePairIter);

    map<int, CameraPairParams>::iterator curCameraPairParamsIter = m_cameraPairParams.find(pairID);
    if (curCameraPairParamsIter == m_cameraPairParams.end())
      return SR_ERROR_CAMERA_PAIR_PARAMS_NOT_EXIST_ERROR;

    curCameraPairParamsIter->second.ClearData();
    m_cameraPairParams.erase(curCameraPairParamsIter);
  }

  return SR_ERROR_OK;
}

void StereoReconstruction::SetUsePreRectifyImages(bool enable)
{
  m_usePreRectifyImages = enable;
}

void StereoReconstruction::SetUseScaledImages(bool enable)
{
  m_useScaledImages = enable;
}

void StereoReconstruction::SetUseBilateralFilter(bool enable)
{
  m_useBilateralFilter = enable;
}