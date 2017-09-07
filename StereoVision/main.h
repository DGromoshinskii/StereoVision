#ifndef MAIN_H
#define MAIN_H

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core.hpp"

#ifdef _WIN32
#pragma warning (disable : 4521)
#pragma warning (disable : 4503)
#pragma warning (disable : 4996)
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

#include <boost/thread/thread.hpp>

#include <stdio.h>
#include "../StereoReconstruction/PCLWrapper.h"

#include "../StereoReconstruction/StereoReconstruction.h"

//using namespace cv;
//using namespace pcl;
using namespace std;

#endif
