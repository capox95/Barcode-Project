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
tuple <vector<Vec4i>, float> barcode_orientation(Mat src);
int counter_tickness_bars(Mat img, vector<float> px);
vector <Vec4i> gap(vector<Vec4i> r_lines, int max_gap);
vector<float> FirstLastDetector(vector<Vec4i> r_lines);
Mat clahe(Mat bgr_image);
vector <Vec4i> vertical_gap(vector<Vec4i> r2_lines, int max_vertical_gap);
int clahe_detector(Mat src);
void plot_histogram(Mat Hist, int histSize);
vector<float> Harris(Mat src, vector<Point> roi, int *height);

#endif
