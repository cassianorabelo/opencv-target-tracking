//
//  deck.cpp
//  TP
//
//  Created by Cassiano Rabelo on oct/16.
//  Copyright © 2016 Cassiano Rabelo. All rights reserved.
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "deck.hpp"

using namespace std;
using namespace cv;

unsigned int gIdx = 0;                          // current index position
unsigned long long int gCurFrame = 0;           // keep track of current frame
unsigned long long int gLastDetectedFrame = 0;  // keep track of current frame
int gNumFramesUntilConsideredLost = 10;         // # of frames until target is considered lost
bool gHasBeenDetected = false;

vector<Point2f> gDeckPosition; // history of deck's positions
bool gDebug = false;

void drawTrajectory(InputOutputArray image) {
  Scalar crossColor = Scalar(160, 80, 80);
  Scalar pathColor = Scalar(0, 242, 255, .5);
  int lineWidth = 3;
  
  if (gHasBeenDetected) {
    for (size_t i = 1; i <= gIdx; i++) {
      line(image, gDeckPosition[i - 1], gDeckPosition[i], pathColor, lineWidth, LINE_AA);
    }
    
    for (size_t i = gIdx + 2; i < NUM_POSITIONS; i++) {
      line(image, gDeckPosition[i - 1], gDeckPosition[i], pathColor, lineWidth, LINE_AA);
    }
    
    if (gIdx != NUM_POSITIONS - 1) {  // connect first to last if that is the case
      line(image, gDeckPosition[0], gDeckPosition[NUM_POSITIONS - 1], pathColor, lineWidth, LINE_AA);
    }
    
    Point center = Point(gDeckPosition[gIdx].x, gDeckPosition[gIdx].y);
    line(image,	Point(center.x - 10, center.y - 10),
         Point(center.x + 10, center.y + 10), crossColor, lineWidth*.8, LINE_AA);
    
    line(image,	Point(center.x + 10, center.y - 10),
         Point(center.x - 10, center.y + 10), crossColor, lineWidth*.8, LINE_AA);
  }
}


void storeDeckPosition(Point2f pos) {
  gDeckPosition[gIdx] = pos;
}


void drawDetectedDecks(InputOutputArray image,
                       InputArrayOfArrays corners) {
  
  Scalar squareColor = Scalar(35, 90, 240);
  Scalar circleColor = Scalar(100, 35, 180);
  Scalar linesColor = Scalar(80, 165, 0);
  int lineWidth = 3;
  
  // keep track of the position
  gCurFrame++;
  int prevIdx = gIdx;
  gIdx = gCurFrame % NUM_POSITIONS;
  
  int nMarkers = (int)corners.total();
  
  // certify we only get the first detected set of corners
  if (nMarkers > 1) {
    gHasBeenDetected = true;
    gLastDetectedFrame = gCurFrame;
    nMarkers = 1;
  } else {
    if (gCurFrame - gLastDetectedFrame > gNumFramesUntilConsideredLost) {
      gHasBeenDetected = false;
    }
    
    // no marker found, store last found position again
    storeDeckPosition(gDeckPosition[prevIdx]);
  }
  
  for(int i = 0; i < (int)corners.total(); i++) {
    
    Mat currentMarker = corners.getMat(i);
    
    Point2f p0 = currentMarker.ptr< Point2f >(0)[0];
    Point2f p1 = currentMarker.ptr< Point2f >(0)[1];
    Point2f p2 = currentMarker.ptr< Point2f >(0)[2];
    Point2f p3 = currentMarker.ptr< Point2f >(0)[3];
    Point2f center;
    
    // draw lines
    line(image, p0, p2, linesColor, lineWidth, LINE_AA);
    line(image, p1, p3, linesColor, lineWidth, LINE_AA);
    
    // draw square
    for(size_t j = 0; j < 4; j++) {
      Point2f p0, p1;
      p0 = currentMarker.ptr< Point2f >(0)[j];
      p1 = currentMarker.ptr< Point2f >(0)[(j + 1) % 4];
      line(image, p0, p1, squareColor, lineWidth, LINE_AA);
    }
    
    double area = contourArea(currentMarker);
    double sideLen = sqrt(area);
    double radius = sideLen * 0.1; // center circle is aprox. 5% the len of a side
    
    if ( intersection(p0, p2, p1, p3, center) ) {
      circle(image, center, radius, circleColor, -1, LINE_AA);
      storeDeckPosition(center);
    }
  }
  
  drawTrajectory(image);
}

Mat removePerspective(InputArray image,
                             InputArray corners) {
  
  Mat resultImg;
  int resultImgSize = 128;
  Mat resultImgCorners(4, 1, CV_32FC2);
  resultImgCorners.ptr< Point2f >(0)[0] = Point2f(0, 0);
  resultImgCorners.ptr< Point2f >(0)[1] = Point2f((float)resultImgSize - 1, 0);
  resultImgCorners.ptr< Point2f >(0)[2] =
  Point2f((float)resultImgSize - 1, (float)resultImgSize - 1);
  resultImgCorners.ptr< Point2f >(0)[3] = Point2f(0, (float)resultImgSize - 1);
  
  Mat transformation = getPerspectiveTransform(corners, resultImgCorners);
  warpPerspective(image, resultImg, transformation, Size(resultImgSize, resultImgSize), INTER_NEAREST);
  
  return resultImg;
}

void threshold(InputArray _in, OutputArray _out, int winSize) {
  if(winSize % 2 == 0) winSize++; // win size must be odd
  int adaptiveThreshConstant = 7;
  adaptiveThreshold(_in, _out, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, winSize, adaptiveThreshConstant);
}


bool validateInside(InputArray image,
                    InputOutputArray corners,
                    InputOutputArray &frame2
                    ) {
  
  Mat frame = frame2.getMat();
  Mat flatCandidate =  removePerspective(image, corners);
  
  int cannyThreshold = 90;
  int accumulatorThreshold = 25;
  int minRadius = (flatCandidate.rows * 0.55)/2; // 55%
  int maxRadius = (flatCandidate.rows * 0.75)/2; // 75%
  int thresholdVal = 30;
  
  std::vector<Vec3f> circles; // will hold the results of the detection
  
  HoughCircles( flatCandidate, circles, HOUGH_GRADIENT, 1, flatCandidate.rows/2, cannyThreshold, accumulatorThreshold, minRadius, maxRadius );
  
  if (gDebug) { // draw the perspective corrected image with the detected circle
    Mat houghCirclesDisplay;
    flatCandidate.copyTo(houghCirclesDisplay);
    cvtColor(houghCirclesDisplay, houghCirclesDisplay, CV_GRAY2BGR);
    
    for( size_t i = 0; i < circles.size(); i++ ) {
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      circle( houghCirclesDisplay, center, 3, Scalar(0,255,0), -1, 8, 0 );        // circle center
      circle( houghCirclesDisplay, center, radius, Scalar(0,0,255), 3, 8, 0 );    // circle outline
      int top = (frame.rows * 0.25) + 10 +10; // top position of the PIP
      int left = 10;
      houghCirclesDisplay.copyTo(frame(Rect(left, top, houghCirclesDisplay.cols, houghCirclesDisplay.rows)));
    }
  }
  
  if (circles.size() == 1) {
    Mat canny, houghLinesDsp;
    Canny(flatCandidate, canny, 50, 200, 3);
    
    int marginClip = 8;
    Mat srcHoughLines(canny, Rect(marginClip, marginClip, canny.rows - marginClip*2, canny.cols - marginClip*2) );
    
    if (gDebug)
      cvtColor(srcHoughLines, houghLinesDsp, CV_GRAY2BGR);
    
    vector<Vec2f> lines;
    HoughLines(srcHoughLines, lines, 1, 10 * CV_PI/180, thresholdVal, 0, 0 );
    
    int xLines = 0;
    for( size_t i = 0; i < lines.size(); i++ ) {
      float rho = lines[i][0], theta = lines[i][1];
      if( (theta > CV_PI/180 * 125 && theta < CV_PI / 180 * 145) || (theta > CV_PI/180 * 35 && theta < CV_PI / 180 * 55) ) {
        xLines++;
      }
      
      if (gDebug) {
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( houghLinesDsp, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
      }
    }
    
    float linesAtExpectedAngle = (float)xLines/lines.size();
    if ( linesAtExpectedAngle >= 0.95 ) {
      if (gDebug) {
        int top = (frame.rows * 0.25) + 10 +10; // top position of the PIP
        int left = flatCandidate.cols + 10 + 10; // left position of the PIP
        houghLinesDsp.copyTo(frame(Rect(left, top, houghLinesDsp.cols, houghLinesDsp.rows)));
      }
      return true;
    }
  }
  
  return false;
}


/**
 * ParallelLoopBody class for the parallelization of the deck identification step
 * Called from function identifyCandidates()
 */
class _identifyCandidatesParallel : public ParallelLoopBody {
public:
  _identifyCandidatesParallel(const Mat *_grey, InputArrayOfArrays _candidates,
                              InputArrayOfArrays _contours,
                              vector< char > *_validCandidates)
  : grey(_grey), candidates(_candidates), contours(_contours), validCandidates(_validCandidates) {}
  
  void operator()(const Range &range) const {
    const int begin = range.start;
    const int end = range.end;
    
    for(int i = begin; i < end; i++) {
      Mat currentCandidate = candidates.getMat(i);
      if(validateInside(*grey, currentCandidate)) {
        (*validCandidates)[i] = 1;
      }
    }
  }
  
private:
  _identifyCandidatesParallel &operator=(const _identifyCandidatesParallel &); // to quiet MSVC
  
  const Mat *grey;
  InputArrayOfArrays candidates, contours;
  vector< char > *validCandidates;
};


void findSquareContours(InputArray _in, vector< vector< Point2f > > &candidates,
                               vector< vector< Point > > &contoursOut) {
  
  double minPerimeterRate = 0.03;
  double maxPerimeterRate = 4.0;
  
  // calculate maximum and minimum sizes in pixels
  unsigned int minPointsPerimeter =
  (unsigned int)(minPerimeterRate * max(_in.getMat().cols, _in.getMat().rows));
  unsigned int maxPointsPerimeter =
  (unsigned int)(maxPerimeterRate * max(_in.getMat().cols, _in.getMat().rows));
  
  Mat contoursImg;
  _in.getMat().copyTo(contoursImg);
  vector< vector< Point > > contours;
  
  findContours(contoursImg, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
  
  // now filter list of contours
  for(size_t i = 0; i < contours.size(); i++) {
    // check perimeter
    if(contours[i].size() < minPointsPerimeter || contours[i].size() > maxPointsPerimeter)
      continue;
    
    // check is square and is convex
    vector< Point > approxCurve;
    approxPolyDP(contours[i], approxCurve, double(contours[i].size()) * 0.03, true);
    if(approxCurve.size() != 4 || !isContourConvex(approxCurve)) continue;
    
    // if it passes all the test, add to candidates vector
    vector< Point2f > currentCandidate;
    currentCandidate.resize(4);
    for(size_t j = 0; j < 4; j++) {
      currentCandidate[j] = Point2f((float)approxCurve[j].x, (float)approxCurve[j].y);
    }
    candidates.push_back(currentCandidate);
    contoursOut.push_back(contours[i]);
  }
}


/**
 * ParallelLoopBody class for the parallelization of the basic candidate detections using
 * different threhold window sizes. Called from function detectInitialCandidates()
 */
class _findSquareContoursParallel : public ParallelLoopBody {
public:
  _findSquareContoursParallel(const Mat *grey,
                              vector< vector< vector< Point2f > > > *candidatesArrays,
                              vector< vector< vector< Point > > > *contoursArrays)
  : grey(grey), candidatesArrays(candidatesArrays), contoursArrays(contoursArrays) {}
  
  void operator()(const Range &range) const {
    const int begin = range.start;
    const int end = range.end;
    
    int adaptiveThreshWinSizeMin = 3;
    int adaptiveThreshWinSizeStep = 10;
    
    for(int i = begin; i < end; i++) {
      int currScale = adaptiveThreshWinSizeMin + i * adaptiveThreshWinSizeStep;
      // threshold
      Mat thresh;
      threshold(*grey, thresh, currScale);
      
      // detect rectangles
      findSquareContours(thresh, (*candidatesArrays)[i], (*contoursArrays)[i]);
    }
  }
  
private:
  _findSquareContoursParallel &operator=(const _findSquareContoursParallel &);
  
  const Mat *grey;
  vector< vector< vector< Point2f > > > *candidatesArrays;
  vector< vector< vector< Point > > > *contoursArrays;
};


void findSquareCandidates(const Mat &grey,
                                 vector< vector< Point2f > > &candidates,
                                 vector< vector< Point > > &contours) {
  
  int adaptiveThreshWinSizeMin = 3;
  int adaptiveThreshWinSizeMax = 23;
  int adaptiveThreshWinSizeStep = 10;
  
  // number of window sizes (scales) to apply adaptive thresholding
  int nScales = (adaptiveThreshWinSizeMax - adaptiveThreshWinSizeMin) / adaptiveThreshWinSizeStep + 1;
  
  vector< vector< vector< Point2f > > > candidatesArrays((size_t) nScales);
  vector< vector< vector< Point > > > contoursArrays((size_t) nScales);
  
  if (gDebug) { // parallelize only if not in debug mode
    //for each value in the interval of thresholding window sizes
    for(int i = 0; i < nScales; i++) {
      int currScale = adaptiveThreshWinSizeMin + i * adaptiveThreshWinSizeStep;
      // treshold
      Mat thresh;
      threshold(grey, thresh, currScale);
      
      // detect rectangles
      findSquareContours(thresh, candidatesArrays[i], contoursArrays[i]);
    }
    
  } else {
    parallel_for_(Range(0, nScales), _findSquareContoursParallel(&grey, &candidatesArrays, &contoursArrays));
  }
  
  // join candidates
  for(size_t i = 0; i < nScales; i++) {
    for(unsigned int j = 0; j < candidatesArrays[i].size(); j++) {
      candidates.push_back(candidatesArrays[i][j]);
      contours.push_back(contoursArrays[i][j]);
    }
  }
}


void identifyCandidates(Mat &frame,
                        InputArray image,
                        InputArrayOfArrays _candidates,
                        InputArrayOfArrays _contours,
                        OutputArrayOfArrays _accepted,
                        OutputArrayOfArrays _rejected) {
  
  int ncandidates = (int)_candidates.total();
  
  vector<Mat> accepted;
  vector<Mat> rejected;
  
  Mat grey = image.getMat();
  
  vector<char> validCandidates(ncandidates, 0);
  
  // Analyze each of the candidates
  if (gDebug) { // parallelize only if not in debug mode
    for (int i = 0; i < ncandidates; i++) {
      Mat currentCandidate = _candidates.getMat(i);
      if (validateInside(grey, currentCandidate, frame)) {
        validCandidates[i] = 1;
      }
    }
  } else {
    parallel_for_(Range(0, ncandidates), _identifyCandidatesParallel(&grey, _candidates, _contours, &validCandidates));
  }
  
  for(int i = 0; i < ncandidates; i++) {
    if(validCandidates[i] == 1) {
      accepted.push_back(_candidates.getMat(i));
    } else {
      rejected.push_back(_candidates.getMat(i));
    }
  }
  
  // parse output
  _accepted.create((int)accepted.size(), 1, CV_32FC2);
  
  for (unsigned int i = 0; i < accepted.size(); i++) {
    _accepted.create(4, 1, CV_32FC2, i, true);
    Mat m = _accepted.getMat(i);
    accepted[i].copyTo(m);
  }
}


bool intersection(Point2f l1p1, Point2f l1p2, Point2f l2p1, Point2f l2p2, Point2f &r) {
  Point2f x = l2p1 - l1p1;
  Point2f d1 = l1p2 - l1p1;
  Point2f d2 = l2p2 - l2p1;
  
  float cross = d1.x*d2.y - d1.y*d2.x;
  if (abs(cross) < /*EPS*/1e-8)
    return false;
  
  double t1 = (x.x * d2.y - x.y * d2.x)/cross;
  r = l1p1 + d1 * t1;
  return true;
}


void detectDecks(Mat &frame,
                 InputArray image,
                 OutputArrayOfArrays corners,
                 OutputArrayOfArrays _rejectedImgPoints
                 ) {
  
  /// STEP 1: Detect deck candidates
  vector< vector< Point2f > > candidates;
  vector< vector< Point > > contours;
  
  detectCandidates(image, candidates, contours);
  
  if (gDebug) { // show image with possible candidates
    Mat candidatesDisplay;
    cvtColor(image, candidatesDisplay, COLOR_GRAY2BGR);
    drawContours(candidatesDisplay, contours, -1, Scalar(0,0,255), 3, LINE_AA);
    resize(candidatesDisplay, candidatesDisplay, Size(), 0.25,0.25);
    candidatesDisplay.copyTo(frame(Rect(10,10,candidatesDisplay.cols, candidatesDisplay.rows)));
  }
  
  /// STEP 2: Check candidate
  identifyCandidates(frame, image, candidates, contours, corners, _rejectedImgPoints);
}


void detectCandidates(InputArray image,
                             OutputArrayOfArrays _candidates,
                             OutputArrayOfArrays _contours) {
  
  Mat grey;
  image.copyTo(grey);
  
  vector< vector< Point2f > > candidates;
  vector< vector< Point > > contoursOut;
  
  /// 1. DETECT FIRST SET OF SQUARE CANDIDATES
  findSquareCandidates(grey, candidates, contoursOut);
  
  /// 2. PARSE OUTPUT
  _candidates.create((int)candidates.size(), 1, CV_32FC2);
  _contours.create((int)contoursOut.size(), 1, CV_32SC2);
  for(int i = 0; i < (int)candidates.size(); i++) {
    _candidates.create(4, 1, CV_32FC2, i, true);
    Mat m = _candidates.getMat(i);
    for(int j = 0; j < 4; j++)
      m.ptr< Vec2f >(0)[j] = candidates[i][j];
    
    _contours.create((int)contoursOut[i].size(), 1, CV_32SC2, i, true);
    Mat c = _contours.getMat(i);
    for(unsigned int j = 0; j < contoursOut[i].size(); j++)
      c.ptr< Point2i >()[j] = contoursOut[i][j];
  }
}

void display(Mat &img, Point pos, Scalar fontColor, const string &ss) {
  int fontFace = FONT_HERSHEY_DUPLEX;
  double fontScale = 0.5;
  int fontThickness = 1;
  Size fontSize = cv::getTextSize("T[]", fontFace, fontScale, fontThickness, 0);
  
  Point org;
  org.x = pos.x - fontSize.width;
  org.y = pos.y - fontSize.height/ 2;
  putText(img, ss, org, fontFace, fontScale, Scalar(255, 255, 255), fontThickness, LINE_AA);
}
