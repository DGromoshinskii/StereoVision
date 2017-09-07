#include "PCLWrapper.h"

void PCL_WRAPPER::BuildDistanceHist(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &inputCloud, vector<DistHist> &distanceHistogram, int &maxBinHeight, float histStep)
{
  vector<float> distsVector;
  for (pcl::PointCloud<pcl::PointXYZRGB>::const_iterator cloudIter = inputCloud->begin(); cloudIter != inputCloud->end(); ++cloudIter)
  {
     double dist = sqrt(/*pow(cloudIter->x, 2) + pow(cloudIter->y, 2) +*/ pow(cloudIter->z, 2));

#ifdef _WIN32
     if (_isnan(dist))
       continue;
#elif __GNUC__ >= 4
     if (isnan(dist))
       continue;
#endif

     distsVector.push_back((float)dist);
  }


  BuildDistanceHist(distsVector, distanceHistogram, maxBinHeight, histStep);
}

void PCL_WRAPPER::BuildDistanceHist(vector<float> &distancesVector, vector<DistHist> &distanceHistogram, int &maxBinHeight, float histStep)
{
  if (!distanceHistogram.empty())
    distanceHistogram.clear();

  sort(distancesVector.begin(), distancesVector.end());
  //float histStep = /*0.004*/0.001;
  vector<float>::const_iterator distVIter = distancesVector.begin();
  vector<float>::const_iterator distBinIter = distancesVector.begin();

  vector<float>::const_iterator lastDVIter = distancesVector.end() - 1;

  float meanBinValue = 0.f;
  int numberBinDist = 0;
  maxBinHeight = INT_MIN;
  for (; distVIter != distancesVector.end(); ++distVIter)
  {
    if (*distVIter - *distBinIter < histStep)
    {
      meanBinValue += *distVIter;
      numberBinDist++;

      if (distVIter == lastDVIter)
      {
        if (maxBinHeight < numberBinDist)
          maxBinHeight = numberBinDist;

        meanBinValue /= numberBinDist;
        DistHist tempDistHist(meanBinValue, numberBinDist);
        distanceHistogram.push_back(tempDistHist);
      }
    }
    else
    {
      if (maxBinHeight < numberBinDist)
        maxBinHeight = numberBinDist;

      meanBinValue /= numberBinDist;
      DistHist tempDistHist(meanBinValue, numberBinDist);
      distanceHistogram.push_back(tempDistHist);
      meanBinValue = *distVIter;//0.f;
      numberBinDist = 1;
      distBinIter = distVIter;

      if (distVIter == lastDVIter)
      {
        DistHist tempDistHist1(meanBinValue, numberBinDist);
        distanceHistogram.push_back(tempDistHist1);
      }
    }
  }
}

void PCL_WRAPPER::ShowDistanceHist(const vector<DistHist> &distanceHistogram, int maxBinNumber, cv::Mat &histImage)
{
  cv::Size histImSize = cv::Size(640, 480);
	int beginSpace = 5;
  int binStep = histImSize.width / distanceHistogram.size();
	if (binStep == 0)
		binStep = 1;
	histImSize.width = binStep * (int)distanceHistogram.size() + beginSpace * 2;
	histImage = cv::Mat::zeros(histImSize, CV_8UC3);
	int binInd = 0;
  for (vector<DistHist>::const_iterator iter = distanceHistogram.begin(); iter != distanceHistogram.end(); ++iter, binInd++)
  {
#ifdef _DEBUG
    std::cout << " " << iter->m_binSize << " " << iter->m_binDist << endl;
#endif
		int binBeginImageInd = binInd * binStep + beginSpace;
    for(int i = binBeginImageInd; i < binBeginImageInd + binStep; i++)
		{
			int binHeight = iter->m_binSize * histImSize.height / maxBinNumber;
			cv::Point binBottom(i, histImSize.height - 1);
			cv::Point binTop(i, histImSize.height - 1 - binHeight);
			line(histImage, binBottom, binTop, cv::Scalar(255, 255, 255));
      if (i == binBeginImageInd + binStep - 1)
      {

        transpose(histImage, histImage);
        cv::flip(histImage, histImage, 1);
        char buf[10];
        sprintf(buf, "%f", iter->m_binDist);
        putText(histImage, buf, cv::Point(histImSize.height - binBottom.y - 1, binBottom.x), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
        transpose(histImage, histImage);
        cv::flip(histImage, histImage, 0);
      }
		}
  }
}

void PCL_WRAPPER::PCLStatisticalFilter(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &inputCloud, const PCL_WRAPPER::StatFilterParams &filterParams, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &filteredCloud, pcl::IndicesConstPtr &removedInds)
{
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  sor.setInputCloud (inputCloud);
  sor.setMeanK (filterParams.m_meanK);
  sor.setStddevMulThresh (filterParams.m_stdDevMulThresh);
  sor.setKeepOrganized(filterParams.m_isKeepOrganize);
  sor.filter (*filteredCloud);

  removedInds = sor.getRemovedIndices();  
}

void PCL_WRAPPER::PassThroughFilter(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &inputCloud, const PCL_WRAPPER::PassThroughFilterParams &PTFilterParams,
                                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &filteredCloud)
{
  pcl::PassThrough<pcl::PointXYZRGB> passFilter;
  passFilter.setInputCloud(inputCloud);
  passFilter.setFilterFieldName(PTFilterParams.m_fieldName);
  passFilter.setFilterLimits(PTFilterParams.m_lowerLimit, PTFilterParams.m_upperLimit);
  passFilter.filter(*filteredCloud);
}

int PCL_WRAPPER::GetPointCloudArea(const cv::Point3f &areaCenter, double searchRadius, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &sourceCloud,
                                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloudPart)
{
  if (sourceCloud == NULL)
    return SR_ERROR_INPUT_POINT_CLOUD_IS_EMPTY_ERROR;
  else if (sourceCloud->empty())
    return SR_ERROR_INPUT_POINT_CLOUD_IS_EMPTY_ERROR;

  vector<float> area2D(4);
  area2D[0] = areaCenter.x - searchRadius;
  area2D[1] = areaCenter.y - searchRadius;
  area2D[2] = area2D[3] = searchRadius * 2;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr xCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  PassThroughFilterParams PTFilterParamsX("x", area2D[0], area2D[0] + area2D[2]);
  PassThroughFilter(sourceCloud, PTFilterParamsX, xCloud);

  PassThroughFilterParams PTFilterParamsY("y", area2D[1], area2D[1] + area2D[2]);
  PassThroughFilter(xCloud, PTFilterParamsY, cloudPart);

  return SR_ERROR_OK;
}

void PCL_WRAPPER::PlanarSegmentation(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &inputCloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &planarCloud, pcl::ModelCoefficients::Ptr &planeCoeffs)
{
  /*pcl::PCLPointCloud2::Ptr cloud_blob (new pcl::PCLPointCloud2), cloud_filtered_blob (new pcl::PCLPointCloud2);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>), cloud_p (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);

  // Fill in the cloud data
  pcl::PCDReader reader;
  reader.read ("table_scene_lms400.pcd", *cloud_blob);

  std::cerr << "PointCloud before filtering: " << cloud_blob->width * cloud_blob->height << " data points." << std::endl;

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloud_blob);
  sor.setLeafSize (0.01f, 0.01f, 0.01f);
  sor.filter (*cloud_filtered_blob);

  // Convert to the templated PointCloud
  pcl::fromPCLPointCloud2 (*cloud_filtered_blob, *cloud_filtered);*/

  //std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height << " data points." << std::endl;

  // Write the downsampled version to disk
  //pcl::PCDWriter writer;
  //writer.write<pcl::PointXYZ> ("table_scene_lms400_downsampled.pcd", *cloud_filtered, false);

   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
   cloud_filtered = inputCloud;
   //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_p (new pcl::PointCloud<pcl::PointXYZRGB>);
   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZRGB>);

  //pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZRGB> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (1000);
  seg.setDistanceThreshold (0.1);

  // Create the filtering object
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;

  int i = 0, nr_points = (int) cloud_filtered->points.size ();
  // While 30% of the original cloud is still there
  while (cloud_filtered->points.size () > 0.3 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *planeCoeffs);
    if (inliers->indices.size () == 0)
    {
      std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Extract the inliers
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);
    extract.filter (*planarCloud);
    std::cerr << "PointCloud representing the planar component: " << planarCloud->width * planarCloud->height << " data points." << std::endl;

    /*std::stringstream ss;
    ss << "table_scene_lms400_plane_" << i << ".pcd";
    writer.write<pcl::PointXYZ> (ss.str (), *cloud_p, false);*/

    // Create the filtering object
    extract.setNegative (true);
    extract.filter (*cloud_f);
    cloud_filtered.swap (cloud_f);
    i++;
  }

}

float PCL_WRAPPER::CalcPointToPlaneDist(const pcl::PointXYZRGB &cloudPoint, const pcl::ModelCoefficients::Ptr &planeCoefficients)
{
  float resDist = -1.f;

  resDist = abs(planeCoefficients->values[0] * cloudPoint.x  + planeCoefficients->values[1] * cloudPoint.y + planeCoefficients->values[2] * cloudPoint.z + planeCoefficients->values[3]);
  resDist /= sqrt(pow(planeCoefficients->values[0], 2) + pow(planeCoefficients->values[1], 2) + pow(planeCoefficients->values[2], 2));

  return resDist;
}

void PCL_WRAPPER::CalculatePlanarDeviation(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &inputCloud, float &planarDev, float &maxDev, bool buildHist)
{
  pcl::ModelCoefficients::Ptr planeCoefficients (new pcl::ModelCoefficients ());
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr planarCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  PlanarSegmentation(inputCloud, planarCloud, planeCoefficients);

  planarDev = 0;
  maxDev = FLT_MIN;
  vector<float> devVector;
  for (pcl::PointCloud<pcl::PointXYZRGB>::const_iterator cloudIter = planarCloud->begin(); cloudIter != planarCloud->end(); ++cloudIter)
  {
    float distToPlane = CalcPointToPlaneDist(*cloudIter, planeCoefficients);
    planarDev += distToPlane;

    if (maxDev < distToPlane)
      maxDev = distToPlane;

    if (buildHist)
    {
      devVector.push_back(distToPlane);
    }
  }
  planarDev /= (float)planarCloud->size();

  if (buildHist)
  {
    vector<DistHist> histBins;
    int maxBinHeight = 0;
    float histStep = 0.0005;
    BuildDistanceHist(devVector, histBins, maxBinHeight, histStep);
    cv::Mat histImage;
    ShowDistanceHist(histBins, maxBinHeight, histImage);
    string winNameHist = "DeviationHist";
    cv::namedWindow(winNameHist, 0);
    cv::imshow(winNameHist, histImage);
    cvWaitKey(30);
  }
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> PCL_WRAPPER::createVisualizer (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "reconstruction");
  //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "reconstruction");
  viewer->addCoordinateSystem ( 1.0 );
  viewer->initCameraParameters ();
  return (viewer);
}

int PCL_WRAPPER::CalculateTransformMatrix(const vector<cv::Point3f> &srcOrigin, const vector<cv::Point3f> &dstOrigin,
                                           cv::Mat &transformMat, cv::Mat &inliersMat, double ransacThreshold, double confidenceLevel)
{
  if (srcOrigin.size() == 0 || dstOrigin.size() == 0)
    return 1;
  if (srcOrigin.size() != dstOrigin.size())
    return 2;

  cv::estimateAffine3D(srcOrigin, dstOrigin, transformMat, inliersMat, ransacThreshold, confidenceLevel);

  return 0;
}

void PCL_WRAPPER::GenerateDstOriginPoints(vector<cv::Point3f> &dstOrigin)
{
  if (!dstOrigin.empty())
    dstOrigin.clear();

  cv::Point3f dstOriginPoint;
  dstOriginPoint = cv::Point3f(1, 1, 1);
  dstOrigin.push_back(dstOriginPoint);
  //
}

bool PCL_WRAPPER::SaveTransformData(const string &fileName, const cv::Mat &transformMat, const vector<cv::Point3f> &srcOrigin, const vector<cv::Point3f> &dstOrigin, 
                                    const cv::Mat &inliersMat, double ransacThreshold, double confidenceLevel)
{
  cv::FileStorage fs(fileName, cv::FileStorage::WRITE);

  if (!fs.isOpened())
    return false;

  fs << "TransformMat" << transformMat;
  fs << "SrcOrigin" << srcOrigin;
  fs << "DstOrigin" << dstOrigin;
  fs << "InliersMat" << inliersMat;
  fs << "RansacThreshold" << ransacThreshold;
  fs << "ConfidenceLevel" << confidenceLevel;
  fs.release();

  return true;
}

bool PCL_WRAPPER::LoadTransformData(const string &fileName, cv::Mat &transformMat, vector<cv::Point3f> &srcOrigin, vector<cv::Point3f> &dstOrigin, 
                                    cv::Mat &inliersMat, double &ransacThreshold, double &confidenceLevel)
{
  cv::FileStorage fs(fileName, cv::FileStorage::READ);

  if (!fs.isOpened())
    return false;

  fs["TransformMat"] >> transformMat;
  fs["SrcOrigin"] >> srcOrigin;
  fs["DstOrigin"] >> dstOrigin;
  fs["InliersMat"] >> inliersMat;
  fs["RansacThreshold"] >> ransacThreshold;
  fs["ConfidenceLevel"] >> confidenceLevel;
  fs.release();

  return true;
}

int PCL_WRAPPER::TransformPoint(const cv::Point3f &queryPoint, cv::Point3f &transformPoint, const cv::Mat &transformMat)
{
  if (transformMat.rows != 3 || transformMat.cols != 4)
    return 1;

  cv::Mat queryPointMat(4, 1, CV_32FC1);
  
  queryPointMat.at<float> (0, 0) = queryPoint.x;
  queryPointMat.at<float> (1, 0) = queryPoint.y;
  queryPointMat.at<float> (2, 0) = queryPoint.z;
  queryPointMat.at<float> (3, 0) = 1.f;

  cv::Mat transformPointMat = transformMat * queryPointMat;

  transformPoint.x = transformPointMat.at<float> (0, 0);
  transformPoint.y = transformPointMat.at<float> (1, 0);
  transformPoint.z = transformPointMat.at<float> (2, 0);

  return 0;
}
