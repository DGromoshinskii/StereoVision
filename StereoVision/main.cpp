#include "main.h"

cv::Mat img1;
cv::Mat img2;
cv::Mat leftIm;
cv::Mat Q;
cv::Mat dispCopy;

//////////////////////////////////////////////////////////////////////////
int Calc3dCoord(const cv::Point& dispMapCoord, double dispVal, const cv::Mat &dispTo3dMat, cv::Point3f &pointCoord3d)
{
  if ( dispVal == 0 ) 
    return -1; //Discard bad pixels

  double pw = /*-1.0 **/ (dispVal * dispTo3dMat.at<double>(3, 2))  + dispTo3dMat.at<double>(3, 3);//!!! 

  double xVal = static_cast<double>(dispMapCoord.x) + dispTo3dMat.at<double>(0, 3);
  double yVal = static_cast<double>(dispMapCoord.y) + dispTo3dMat.at<double>(1, 3);
  double zVal = dispTo3dMat.at<double>(2, 3);

  zVal = zVal / pw;
  xVal = xVal / pw;
  yVal = yVal / pw;

  pointCoord3d = cv::Point3f((float)xVal, (float)yVal, (float)zVal);

  return 0;
}



int mouse_X = -1;
int mouse_Y = -1;


void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
  if  ( event == cv::EVENT_RBUTTONDOWN )
  {
    cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    mouse_X = x;
    mouse_Y = y;

    cv::Point3f coord3d;

    if (Calc3dCoord(cv::Point(x, y), (double)dispCopy.at<float>(y,x), Q, coord3d) == -1)
      return;

    cout << "X = " << x << endl;
    cout << "Y = " << y << endl;
    cout << "Disparity = " << (double)dispCopy.at<float>(y,x) << endl;
    cout << "Depth val disp = " << coord3d.z << endl;
    cout << "X val disp = " << coord3d.x   << endl;
    cout << "Y val disp = " << coord3d.y   << endl;


    cout << "Distance = " << sqrt(pow(coord3d.y, 2) + pow(coord3d.x, 2) + pow(coord3d.z, 2)) << endl;
  }
}


int main(int argc, char** argv)
{
  if (argc != 8)
  {
    cout << "usage: " << "<LeftImage> <RightImage> <IntrinsicsCamerasParams> <ExtrinsicsCamerasParams> <image scale> <distThresholdMax> <distThresholdMin>" << endl;
    return 0;
  }

#ifdef _WIN32
  const char* img1_filename = argv[1];
  const char* img2_filename = argv[2];
#elif __GNUC__ >= 4
  const char* img1_filename = /*argv[1];*/"/home/dmitry/StereoCalib/Left_11.jpg";
  const char* img2_filename = /*argv[2];*/"/home/dmitry/StereoCalib/Right_11.jpg";
#endif


#ifdef _WIN32
  const char* intrinsic_filename = argv[3];
  const char* extrinsic_filename = argv[4];
#elif __GNUC__ >= 4
  const char* intrinsic_filename = "/home/dmitry/StereoCalib/intrinsics.yml";
  const char* extrinsic_filename = "/home/dmitry/StereoCalib/extrinsics.yml";
#endif

  const char* disparity_filename = "dispIm.jpg";
  const char* point_cloud_filename =  "cloud.xml";


  double scale = atof(argv[5]);

  if( !img1_filename || !img2_filename )
  {
    printf("Command-line parameter error: both left and right images must be specified\n");
    return -1;
  }

  if( (intrinsic_filename != 0) ^ (extrinsic_filename != 0) )
  {
    printf("Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)\n");
    return -1;
  }

  if( extrinsic_filename == 0 && point_cloud_filename )
  {
    printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified to compute the point cloud\n");
    return -1;
  }

  int color_mode = -1;
  img1 = cv::imread(img1_filename, color_mode);
  img2 = cv::imread(img2_filename, color_mode);

  StereoReconstruction stereoReconst;

  stereoReconst.SetInputImageScale(scale);

  StereoParams stereoAlgParams;
  stereoAlgParams.m_numberOfDisparities = 64;
  stereoAlgParams.m_minDisparity = -15;
  stereoAlgParams.m_uniquenessRatio = 15;
  stereoAlgParams.m_speckleWinSize = 150;
  stereoAlgParams.m_speckleRange = 2;
  //stereoAlgParams.m_preFilterCap = 10;
  stereoAlgParams.m_fullDP = false;
  stereoAlgParams.m_SADWinSize = 3;
  stereoAlgParams.m_P1 = 600;
  stereoAlgParams.m_P2 = 2400;
  //stereoAlgParams.disp12MaxDiff = 5;
  /// Set stereo algorithm params
  stereoReconst.SetStereoAlgParams(stereoAlgParams);



  int pairID = 1;
  if (!stereoReconst.SetImagePair(pairID, img1, img2))
  {
    cout << "SetImagePairError" << endl;
    return 1;
  }

  if (!stereoReconst.LoadInternalCalibMatrices(intrinsic_filename))
  {
    cout << "LoadInternalCalibMatricesError" << endl;
    return 2;
  }

  if (!stereoReconst.LoadExternalCalibMatrices(extrinsic_filename, pairID))
  {
    cout << "LoadExternalCalibMatricesError" << endl;
    return 3;
  }

  stereoReconst.SetUpperDistZThreshold(atof(argv[6]));
  stereoReconst.SetLowerDistZThreshold(atof(argv[7]));

  PCL_WRAPPER::StatFilterParams statFiltParams;
  statFiltParams.m_isKeepOrganize = true;
  statFiltParams.m_meanK = 300;
  statFiltParams.m_stdDevMulThresh = 0.2;

  stereoReconst.SetStatFilterParams(statFiltParams);


  if (stereoReconst.ImagePairHandler(pairID) != SR_ERROR_OK)
  {
    cout << "ImagePairHandlerError" << endl;
    return 4;
  }

  if (stereoReconst.Visualize3dPointCloud(pairID) != SR_ERROR_OK)
  {
    cout << "Visualize3dPointCloudError" << endl;
    return 8;
  }

  const CameraPairParams *const qwe = stereoReconst.GetCamaraPairParams(pairID);
  Q = qwe->m_cameraPairMat.m_Q;

  string rectifyLeftImageWin = "RectifyLeftImage";
  string rectifyRightImageWin = "RectifyRightImage";
  string dispImageWin = "DispImage";
  cv::namedWindow(rectifyLeftImageWin, 0);
  cv::namedWindow(rectifyRightImageWin, 0);
  cv::namedWindow(dispImageWin, 0);
  cv::setMouseCallback("RectifyLeftImage", CallBackFunc, NULL);
  const ImagePair *tempImPair = stereoReconst.GetImagePair(pairID);
  if (tempImPair == NULL)
    return 5;
  dispCopy = tempImPair->m_dispImage;

  cv::Mat resizeLeftIm;
  resize(tempImPair->m_leftImage, resizeLeftIm, cv::Size(800, 600));
  cv::imshow(rectifyLeftImageWin, tempImPair->m_leftRectImage);
  cv::imshow(rectifyRightImageWin, tempImPair->m_rightRectImage);

  cv::Mat disp8;
  tempImPair->m_dispImage.convertTo(disp8, CV_8U, 255./( StereoParams().m_numberOfDisparities));
  cv::imshow(dispImageWin, disp8);
  cvWaitKey(0);

  cout << "OK" << endl;
  
  return 0;
}
