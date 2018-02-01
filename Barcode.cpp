#include "stdafx.h"
#include "opencv2\opencv.hpp"
#include "iostream"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include "functions.h"

// for the selection of the good lines among the set provided by Hough
#define delta 0.02
#define h 10

using namespace cv;
using namespace std;

//EAN-UPC-DECODABILITY IMGB.bmp
//UPC#11.bmp
//UPC#01.bmp >>>> ONLY ONE WORKING!!!!

//to check the filter
//EAN128-CONTRAST IMGB ---> NO sencodo gauss barcode ruotato  ---> uguale risultato con angolo rotazione nevativo/positivo!
//EAN128-LOW DECODABILITY IMGB ---> rotazione senso sbagliato ---> Angolo rotazione negativo migliora!
//EAN128-MASTER IMGB ---> hough lines gap verticale errore ---> risolto! forse con angolo rotazione nevativo migliora!
//I25-DEFECTS IMGB ---> vector out of range --> risolto: crop_image.rows invece di scr.rows (harris function, top_y_min) Angolo rotazione negativo migliora!
//I25-MASTER GRADE IMGB ---> exeption ROI ---> risolto: threshold gap 80 (prima 100). Angolo rotazione negativo migliora!
//C128_7.5LOW ---> bounding box non perfetta ---> OK!
//UPC#07 ---> prima e ultima barra non prese ---> risolto: threshold2 canny modificata da 180 a 150


//C39_4.4LOW ---> non funziona con gap verticali attivo!
// DA RISOLVERE: ROTAZIONE IMMAGINI LUNGHE


int main(int argc, char** argv)
{
	Mat src = imread("data/I25-DEFECTS IMGB.bmp", 1);
	//imshow("src", src);


	// CLAHE DETECTION
	int use_clahe = clahe_detector(src);
	cout << "use clahe " << use_clahe << endl;
	if (use_clahe) src = clahe(src);


	// EDGE DETECTION
	Mat dst = src.clone();
	Mat	cdst, dst2;
	cvtColor(dst, dst, CV_RGB2GRAY);

	GaussianBlur(dst, dst, Size(3, 3), 0, 0, BORDER_DEFAULT);
	Canny(dst, dst, 50, 180, 3, true);
	//fastNlMeansDenoising(dst, dst, h, 21, 7);
	GaussianBlur(dst, dst, Size(3, 3), 0, 0, BORDER_DEFAULT);


	// BARCODE ORIENTATION VIA HOUGH TRANSFORMS
	vector<Vec4i> r_lines;
	float angle_rotation;
	tie(r_lines, angle_rotation) = barcode_orientation(dst);
	cout << "angle rotation" << angle_rotation << endl;



	// ROTATION OF THE IMAGE
	Mat rotated_barcode = rotation_image(src, angle_rotation);
	//imshow("rotated_barcode", rotated_barcode);


	//HOUGH TRANSFORM ON ROTATED IMAGE - STEP 2
	vector<Vec4i> r_lines2;
	Mat edges = rotated_barcode.clone();
	cvtColor(edges, edges, CV_RGB2GRAY);

	GaussianBlur(edges, edges, Size(3, 3), 0, 0, BORDER_DEFAULT);
	Canny(edges, edges, 50, 150, 3, true);
	imshow("canny", edges);
	//fastNlMeansDenoising(edges, edges, h, 21, 7);
	GaussianBlur(edges, edges, Size(3, 3), 0, 0, BORDER_DEFAULT);
	tie(r_lines2, angle_rotation) = barcode_orientation(edges);
	

	//CANCEL VERTICAL LINES AWAY FROM THE BARCODE
	vector <Vec4i> barcode_lines = gap(r_lines2, 80);
	//vector<Vec4i> barcode_lines2 = vertical_gap(barcode_lines, 30);
	vector <float> px = FirstLastDetector(barcode_lines); //obtain initial and final bar


	// BINARIZATION OF THE IMAGE
	Mat binary = rotated_barcode.clone();
	cvtColor(binary, binary, CV_RGB2GRAY);
	double th = threshold(binary, binary, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	cout << " threshold OTSU " << th << endl;


	//EXTRACTION THICKNESS SMALLEST BAR, BOUNDING BOX UPDATED!
	int X = counter_tickness_bars(binary, px);
	cout << "size smaller bar " << X << endl;


	// BOUNDING BOX DRAWING
	//vector containing the four corners for the bounding box, in order: top_left, bottom_left, bottom_right, top_right
	vector <Point> points = { Point(px[0], px[2]), Point(px[0], px[3]), Point(px[1], px[3]), Point(px[1], px[2]) };
	drawing_box(binary, points);
	cvtColor(binary, binary, CV_GRAY2RGB);
	vector <Point> points_updated = { Point(px[0] - 10 * X, px[2] - X), Point(px[0] - 10 * X, px[3] + X), Point(px[1] + 10 * X, px[3] + X), Point(px[1] + 10 * X, px[2] - X) };
	drawing_box(binary, points_updated);
	imshow("binarized image with bounding box", binary);

	
	//HARRIS vector<Point> Harris(Mat src, vector<Point> roi);
	cvtColor(rotated_barcode, rotated_barcode, CV_RGB2GRAY);
	int height;
	vector<float> y_coord = Harris(rotated_barcode, points_updated, &height);
	cvtColor(rotated_barcode, rotated_barcode, CV_GRAY2RGB);
	vector <Point> harris_points = { Point(px[0], y_coord[0]), Point(px[0], y_coord[1]), Point(px[1], y_coord[1]), Point(px[1], y_coord[0]) };
	drawing_box(rotated_barcode, harris_points);
	


	imshow("BOUNDING BOX FINAL", rotated_barcode);

	waitKey();
	return 0;
}





