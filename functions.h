#pragma once

#ifndef functions_INCLUDE
#define functions_INCLUDE
#include "stdafx.h"
#include "opencv2\opencv.hpp"
#include "iostream"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include "functions.h"

using namespace cv;
using namespace std;


void drawing_box(Mat dst, vector<Point> points);
Mat binarization(Mat src, int threshold);
vector<Vec4i> rotation_lines(vector<Vec4i> r_lines, float angle, Mat src);
Mat rotation_image(Mat src, float angle_rotation);
tuple <vector<Vec4i>, float> barcode_orientation(Mat src);
int counter_tickness_bars(Mat img, vector<float> px);
vector <Vec4i> gap(vector<Vec4i> r_lines, int max_gap);
vector<float> corners_detector(vector<Vec4i> r_lines);
Mat clahe(Mat bgr_image);

int clahe_detector(Mat src);
void plot_histogram(Mat Hist, int histSize);

#endif
