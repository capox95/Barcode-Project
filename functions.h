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
Mat rotation_image(Mat src, float angle_rotation);
tuple <vector<Vec4i>, float> barcode_orientation(Mat src, bool *flag);
int counter_thickness_bars(Mat img, vector<float> px);
vector <Vec4i> gap(vector<Vec4i> r_lines, int max_gap);
vector<float> FirstLastDetector(vector<Vec4i> r_lines);
Mat clahe(Mat bgr_image);
vector <Vec4i> vertical_gap(vector<Vec4i> r2_lines, Mat src);
int clahe_detector(Mat src);
void plot_histogram(Mat Hist, int histSize);
vector<float> Harris(Mat src, vector<Point> roi, int *height);

vector<float> scan_images(Mat src);
float edges_counter(Mat src);

vector <float> scan_parameters(Mat working);
vector <float> scan_images_average(Mat src, vector<Point> harris_points);




#endif
#pragma once

