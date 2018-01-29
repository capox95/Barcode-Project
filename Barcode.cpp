#include "stdafx.h"
#include "opencv2\opencv.hpp"
#include "iostream"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include "functions.h"

// for the selection of the good lines among the set provided by Hough
#define delta 0.05 

using namespace cv;
using namespace std;

//EAN-UPC-DECODABILITY IMGB.bmp
//UPC#11.bmp
//UPC#01.bmp >>>> ONLY ONE WORKING!!!!

int main(int argc, char** argv)
{
	Mat src = imread("data/UPC#01.bmp", 1);
	imshow("src", src);

	//src = clahe(src);

	// EDGE DETECTION
	Mat dst, cdst, dst2;
	Canny(src, dst, 50, 180, 3, true);
	//cvtColor(dst, cdst, CV_GRAY2BGR);
	GaussianBlur(dst, dst, Size(3, 3), 0, 0, BORDER_DEFAULT);


	// BARCODE ORIENTATION VIA HOUGH TRANSFORMS
	vector<Vec4i> r_lines;
	float angle_rotation;
	tie(r_lines, angle_rotation) = barcode_orientation(dst);
	cout << "angle rotation" << angle_rotation << endl;


	
	// ROTATION OF THE IMAGE
	Mat rotated_barcode = rotation_image(src, angle_rotation);
	imshow("rotated_barcode", rotated_barcode);

	
	//HOUGH TRANSFORM ON ROTATED IMAGE
	vector<Vec4i> r_lines2;
	Mat edges = rotated_barcode.clone();
	Canny(rotated_barcode, edges, 50, 180, 3, true);
	GaussianBlur(edges, edges, Size(3, 3), 0, 0, BORDER_DEFAULT);
	tie(r_lines2, angle_rotation) = barcode_orientation(edges);

	
	vector <Vec4i> barcode_lines = gap(r_lines2, 50);
	vector <float> px = corners_detector(barcode_lines);

	// BINARIZATION OF THE IMAGE
	Mat binary = binarization(rotated_barcode, 100);


	
	//EXTRACTION THICKNESS SMALLEST BAR, BOUNDING BOX UPDATED!
	int X = counter_tickness_bars(binary, px);
	cout << "size smaller bar " << X << endl;

	
	// BOUNDING BOX DRAWING
	// vector containing the four corners for the bounding box, in order: top_left, bottom_left, bottom_right, top_right
	vector <Point> points = { Point(px[0], px[2]), Point(px[0], px[3]), Point(px[1], px[3]), Point(px[1], px[2]) };
	drawing_box(binary, points);
	cvtColor(binary, binary, CV_GRAY2RGB);
	vector <Point> points_updated = { Point(px[0] - 10 * X, px[2] - X), Point(px[0] - 10 * X, px[3] + X), Point(px[1] + 10 * X, px[3] + X), Point(px[1] + 10 * X, px[2] - X) };
	drawing_box(binary, points_updated);
	imshow("binarized image with bounding box", binary);
	
	waitKey();
	return 0;
}





