#include "stdafx.h"
#include "opencv2\opencv.hpp"
#include "iostream"
#include <stdlib.h>
#include <stdio.h>
#include <vector>


// for the selection of the good lines among the set provided by Hough
#define delta 0.05 

using namespace cv;
using namespace std;


void drawing_box(Mat dst, vector<Point> points);
MatND histogram(Mat src);
Mat binarization(Mat src, int threshold);
vector<float> rotation_lines(vector<Vec4i> r_lines, float rotation[]);
Mat rotation_image(Mat src, float angle_rotation);
tuple <vector<Vec4i>, float> barcode_orientation(Mat src);
int counter_tickness_bars(Mat img, vector<float> px);

//EAN-UPC-DECODABILITY IMGB.bmp
//UPC#11.bmp
//UPC#01.bmp >>>> ONLY ONE WORKING!!!!


int main(int argc, char** argv)
{

	Mat src = imread("data/UPC#11.bmp", 0);

	// EDGE DETECTION
	Mat dst, cdst, dst2;
	Canny(src, dst, 220, 255, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);
	GaussianBlur(dst, dst, Size(3, 3), 0, 0, BORDER_DEFAULT);
	
	// BARCODE ORIENTATION VIA HOUGH TRANSFORMS
	vector<Vec4i> r_lines;
	float angle_rotation;
	tie (r_lines, angle_rotation) = barcode_orientation(dst);
	cout << "angle rotation" << angle_rotation << endl;


	
	// ROTATION OF THE IMAGE + SELCTED_LINES
	float rotation[2];
	rotation[0] = (cos(angle_rotation));
	rotation[1] = (sin(angle_rotation));
	cout << "rotation cosine" << rotation[0] << endl;
	cout << "rotation sine" << rotation[1] << endl;
	Mat rotated_barcode = rotation_image(src, angle_rotation);
	vector <float> px = rotation_lines(r_lines, rotation);


	// BINARIZATION OF THE IMAGE
	Mat binary = binarization(rotated_barcode, 128);

	//EXTRACTION THICKNESS SMALLEST BAR, BOUNDING BOX UPDATED!
	int X = counter_tickness_bars(binary, px);
	cout << "size bar " << X << endl;

	/*

	// BOUNDING BOX DRAWING
	// vector containing the four corners for the bounding box, in order: top_left, bottom_left, bottom_right, top_right
	vector <Point> points = { Point(px[0], px[2]), Point(px[0], px[3]), Point(px[1], px[3]), Point(px[1], px[2]) };
	drawing_box(binary, points);


	*/

	cvtColor(binary, binary, CV_GRAY2RGB);
	vector <Point> points_updated = { Point(px[0]-10*X, px[2]-X), Point(px[0]-10*X, px[3]+X), Point(px[1]+10*X, px[3]+X), Point(px[1]+10*X, px[2]-X) };
	drawing_box(binary, points_updated);


	imshow("binarized image with bounding box", binary);

	waitKey();

	return 0;
}







void drawing_box(Mat dst, vector<Point> points) {
	
	line(dst, points[0], points[1], Scalar(7, 254, 47), 3, CV_AA);
	line(dst, points[1], points[2], Scalar(7, 254, 47), 3, CV_AA);
	line(dst, points[2], points[3], Scalar(7, 254, 47), 3, CV_AA);
	line(dst, points[3], points[0], Scalar(7, 254, 47), 3, CV_AA);
}


MatND histogram(Mat src) {

	// Initialize parameters
	int histSize = 256;    // bin size
	float range[] = { 0, 255 };
	const float *ranges[] = { range };

	// Calculate histogram
	MatND hist;
	calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

	/*
	// Show the calculated histogram in command window
	double total;
	total = src.rows * src.cols;
	for (int h = 0; h < histSize; h++)
	{
		float binVal = hist.at<float>(h);
		//cout << " " << binVal;
	}

	// Plot the histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}
	*/
	return hist;
}


Mat binarization(Mat src, int threshold) {

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) <= threshold)  src.at<uchar>(i, j) = 0;
			else src.at<uchar>(i, j) = 255;
		}
	}
	return src;
}


vector<float> rotation_lines(vector<Vec4i> r_lines, float rotation[]) {

	float corners_x[2] = { 100000, 0 }; //primo valore minimo secondo valore massimo
	float corners_y[2] = { 100000, 0 }; //primo valore minimo secondo valore massimo

	int i_max = 0, i_min = 0, j_max = 0, j_min = 0;
	for (size_t i = 0; i < r_lines.size(); i++) {
		r_lines[i][0] = (r_lines[i][0] * rotation[0] + r_lines[i][1] * rotation[1]);
		r_lines[i][1] = (-r_lines[i][0] * rotation[1] + r_lines[i][1] * rotation[0]);
		r_lines[i][2] = (r_lines[i][2] * rotation[0] + r_lines[i][3] * rotation[1]);
		r_lines[i][3] = (-r_lines[i][2] * rotation[1] + r_lines[i][3] * rotation[0]);
		//i_min e i_max ci definiscono la riga della barra iniziale e finale
		//calcolo x minore
		if (r_lines[i][0] < corners_x[0]) {
			corners_x[0] = r_lines[i][0];
			i_min = i;
		}
		//calcolo x maggiore
		if (r_lines[i][0] > corners_x[1]) {
			corners_x[1] = r_lines[i][0];
			i_max = i;
		}
		//line(destination, Point(r_lines[i][0], r_lines[i][1]), Point(r_lines[i][2], r_lines[i][3]), Scalar(0, 0, 255), 1, CV_AA);
	}
	for (size_t i = 0; i < r_lines.size(); i++) {
		//calcolo y minore
		if (r_lines[i][3] < corners_y[0]) {
			corners_y[0] = r_lines[i][3];
			j_min = i;
		}
		//calcolo y maggiore
		if (r_lines[i][1] > corners_y[1]) {
			corners_y[1] = r_lines[i][1];
			j_max = i;
		}
	}

	vector<float> vec = { corners_x[0], corners_x[1], corners_y[0], corners_y[1] };
	return vec;
}


Mat rotation_image(Mat src, float angle_rotation) {

	angle_rotation = (angle_rotation * 180) / (CV_PI);
	//rotation vector
	//Point2f src_center(src.cols / 2.0F, src.rows / 2.0F);
	Point2f src_center(0, 0);
	Mat rot_mat = getRotationMatrix2D(src_center, angle_rotation, 1.0);
	Mat dst;
	warpAffine(src, dst, rot_mat, src.size());
	return dst;
}


tuple <vector<Vec4i>, float> barcode_orientation(Mat src) {
	
	float media = 0, sum = 0, median = 0, angle_rotation = 0;
	int count = 0;
	bool flag_median = false;

	vector<Vec4i> lines;
	vector<Vec4i> r_lines;
	vector<float> theta, theta_backup;
	HoughLinesP(src, lines, 1, CV_PI / 180, 50, 200, 10);
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		if (atan2((l[3] - l[1]), (l[2] - l[0])) > 1.5) flag_median = true; // cambio nella direzione della rotazione PORCATA!
		theta.push_back(abs(atan2((l[3] - l[1]), (l[2] - l[0]))));
		sum = sum + abs(theta[i]);
	}


	theta_backup = theta;
	sort(theta.begin(), theta.end());
	median = theta[theta.size() / 2];
	media = sum / count;
	//ciclo dopo aver calcolato la mediana per selezionare le righe giuste
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		if ((theta_backup[i] < median + delta) && (theta_backup[i] > (median - delta)))
		{
			//cout << theta_backup[i] << endl;
			r_lines.push_back(Vec4i(l[0], l[1], l[2], l[3]));
			//line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, CV_AA);
		}
	}

	//rotation of the image
	if (flag_median) angle_rotation = median - CV_PI/2;
	else angle_rotation = CV_PI/2 - median;

	return make_tuple(r_lines, angle_rotation);
}


int counter_tickness_bars(Mat img, vector<float> px) {

	int distance_x = px[1] - px[0];
	int distance_y = px[3] - px[2];
	int start_x = px[0];
	int start_y = int(px[2] + distance_y/2);

	int count_min = 0, result = 1000, temp = 0;

	/*
	cout << "distance x " << distance_x << endl;
	cout << "distance y " << distance_y << endl;
	cout << "start x " << start_x << endl;
	cout << "start y " << start_y << endl;
	*/
	
	for (int i = start_x; i < (start_x + distance_x); i++)
	{
		int x = (int)img.at<uchar>(start_y, i);
		int x0 = (int)img.at<uchar>(start_y, i-1);

		if (x == 0 && x0 == 0) {
			count_min++;
			}
		else {
			temp = count_min;
			count_min = 0;
			if (result > temp && temp != 0) {
				result = temp+1;
			}
		}
	}

	return result;
}



