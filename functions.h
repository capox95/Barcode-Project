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


struct barcode_result {
	string name;
	string grade;
	vector < vector <float>> parameters;
	int x_dimension;
	int height;
	float orientation;
	vector <Point> bounding_box;
	Point center;
	vector <int> number_edges;
	vector < vector <float>> sequence;
};




//USEFUL FUNCTIONS
void drawing_box(Mat dst, vector<Point> points);
int writeFile(struct barcode_result barcode);



//BARCODE IDENTIFICATION FUNCTIONS
Mat rotation_image(Mat src, float angle_rotation);
tuple <vector<Vec4i>, float> barcode_orientation(Mat src, bool *flag);
int counter_thickness_bars(Mat img, vector<float> px);
vector <Vec4i> gap(vector<Vec4i> r_lines, int max_gap);
vector<float> FirstLastDetector(vector<Vec4i> r_lines);
Mat clahe(Mat bgr_image);
vector <Vec4i> vertical_gap(vector<Vec4i> r2_lines, Mat src);
int clahe_detector(Mat src);
void plot_histogram(Mat Hist, int histSize);
vector<int> Harris(Mat src, vector<Point> roi);
vector <int> broken_lines_removal(Mat src, vector<Point> roi, vector <Vec4i> hough);



//QUALITY PARAMETERS CALCULATION FUNCTIONS
vector <float> scan_parameters(Mat scan, vector < vector <int> >& crosses, vector<bool>& flag, int p);
vector <vector <float>> scan_images_average(Mat src, vector<Point> harris_points, vector <int>& grade, vector< vector <int> >& crosses, vector<bool>& flag);
float ecmin_calculation(vector<int> cross, Mat scan, float threshold, vector <int>& spaces, vector <int>& bars);
float ernmax_calculation(Mat scan, vector <int> cross, float threshold, vector<int> spaces, vector<int> bars, vector<int>& defects_space, vector<int>& defects_bar);
int minimum_grade(vector <float> param);
string overall_grade(vector <int> grade);
vector < vector <float>> sequence_spaces_bars(vector< vector <int> > crosses, int X, vector<bool> flag);
vector <int> number_edges(vector < vector <int> > crosses);

#endif
#pragma once
