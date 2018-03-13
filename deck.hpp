//
//  deck.hpp
//  TP
//
//  Created by Cassiano Rabelo on oct/16.
//  Copyright © 2016 Cassiano Rabelo. All rights reserved.
//

#ifndef deck_h
#define deck_h


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

#define NUM_POSITIONS 50 //# of positions to store

extern vector<Point2f> gDeckPosition; // history of deck's positions
extern bool gDebug; // debug mode ON/OFF

void threshold(InputArray _in, OutputArray _out, int winSize);

// Initiates deck detection
void detectDecks(Mat &frame, InputArray image, OutputArrayOfArrays corners, OutputArrayOfArrays _rejectedImgPoints);

// Given a tresholded image, find the contours, and select those that pass certain tests
void findSquareContours(InputArray _in, vector< vector< Point2f > > &candidates, vector< vector< Point > > &contoursOut);

// Initial steps on finding square candidates
void findSquareCandidates(const Mat &grey, vector< vector< Point2f > > &candidates, vector< vector< Point > > &contours);

// Detect square candidates in the input image
void detectCandidates(InputArray image, OutputArrayOfArrays _candidates, OutputArrayOfArrays _contours);

// Identify possible decks
void identifyCandidates(InputArray image,
                        InputArrayOfArrays _candidates,
                        InputArrayOfArrays _contours,
                        OutputArrayOfArrays _accepted,
                        OutputArrayOfArrays _rejected = noArray());

// Flatten the contents (removes perspective) of the provided polygon
Mat removePerspective(InputArray image, InputArray corners);

// Checks if the provided contour is actually a deck by analysing its inside content
bool validateInside(InputArray image, InputOutputArray corners, InputOutputArray &frame = noArray());

// stores the provided position
void storeDeckPosition(Point2f pos);

// Draws the detected decks on screen
void drawDetectedDecks(InputOutputArray image, InputArrayOfArrays corners);

// Draws the stored trajectory
void drawTrajectory(InputOutputArray image);


// Finds the intersection of two lines. Returns true if there is an intersection. False otherwise.
// The lines are defined by (l1p1, l1p2) and (l2p1, p2).
bool intersection(Point2f l1p1, Point2f l1p2, Point2f l2p1, Point2f l2p2, Point2f &r);

// Adds textual info to the provided image
void display(Mat &img, Point pos, Scalar fontColor, const string &ss);

#endif /* deck_h */