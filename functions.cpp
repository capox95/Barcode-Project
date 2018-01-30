#include "stdafx.h"
#include "opencv2\opencv.hpp"
#include "iostream"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include "functions.h"

using namespace cv;
using namespace std;

// for the selection of the good lines among the set provided by Hough
#define delta 0.05 
#define range2 20

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

	cvtColor(src, src, CV_BGR2GRAY);
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


vector<Vec4i> rotation_lines(vector<Vec4i> r_lines, float angle, Mat src) {

	cvtColor(src, src, CV_GRAY2RGB);

	float rotation[2];
	rotation[0] = cos(angle);
	rotation[1] = sin(angle);



	int x = (int)src.cols / 2, y = (int)src.rows / 2;

	for (size_t i = 0; i < r_lines.size(); i++) {
		r_lines[i][0] -= x;
		r_lines[i][1] -= y;
		r_lines[i][2] -= x;
		r_lines[i][3] -= y;
		r_lines[i][0] = (r_lines[i][0] * rotation[0] - r_lines[i][1] * rotation[1]);
		r_lines[i][1] = (r_lines[i][0] * rotation[1] + r_lines[i][1] * rotation[0]);
		r_lines[i][2] = (r_lines[i][2] * rotation[0] - r_lines[i][3] * rotation[1]);
		r_lines[i][3] = (r_lines[i][2] * rotation[1] + r_lines[i][3] * rotation[0]);
		r_lines[i][0] += x;
		r_lines[i][1] += y;
		r_lines[i][2] += x;
		r_lines[i][3] += y;


	}

	/*
	for (int i = 0; i < r_lines.size(); i++) {
		Vec4i l = r_lines[i];
		line(src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 1, CV_AA);
		}

	imshow("roated_lines", src);

	*/

	return r_lines;


}

vector<float> corners_detector(vector<Vec4i> r_lines) {

	float corners_x[2] = { 100000, 0 }; //primo valore minimo secondo valore massimo
	float corners_y[2] = { 100000, 0 }; //primo valore minimo secondo valore massimo
	int i_max = 0, i_min = 0, j_max = 0, j_min = 0;



	for (size_t i = 0; i < r_lines.size(); i++) {
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

	angle_rotation = ((angle_rotation * 180) / (CV_PI));
	//rotation vector
	Point2f src_center(src.cols / 2.0F, src.rows / 2.0F);
	//Point2f src_center(0, 0);
	Mat rot_mat = getRotationMatrix2D(src_center, angle_rotation, 1.0);
	Mat dst;
	warpAffine(src, dst, rot_mat, src.size());
	return dst;
}


tuple <vector<Vec4i>, float> barcode_orientation(Mat src) {
	Mat cdst = src.clone();
	cvtColor(cdst, cdst, CV_GRAY2RGB);
	float media = 0, sum = 0, median = 0, angle_rotation = 0, valore, val;
	int count = 0, counter = 0, counter2, differenza = 0, orientation = 0;
	bool flag_median = false;
	vector<Vec4i> lines;
	vector<Vec4i> r_lines;
	vector<float> theta, theta_backup;
	int minLenght = min((src.rows / 10), (src.cols / 10));
	HoughLinesP(src, lines, 1, CV_PI / 180, 50, minLenght, 5);


	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		theta.push_back(abs(atan2((l[3] - l[1]), (l[2] - l[0]))));
		if (valore = (atan2((l[3] - l[1]), (l[2] - l[0]))) > 0) counter++;
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 1, CV_AA);

	}

	// verso rotazione
	counter2 = lines.size() - counter;
	differenza = counter - counter2;
	if (differenza > range2) orientation = -1; // senso orario
	else if (differenza < -range2) orientation = 1;
	else if ((-range2 <= differenza) && (differenza <= range2)) orientation = 0;
	cout << "counter " << counter << endl;
	cout << "counter2 " << counter2 << endl;

	theta_backup = theta;

	// calcolo mediana
	sort(theta.begin(), theta.end());

	median = theta[theta.size() / 2];
	cout << "mediana " << median << endl;

	angle_rotation = (CV_PI / 2 - abs(median))*(orientation);





	//ciclo dopo aver calcolato la mediana per selezionare le righe giuste
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		if ((theta_backup[i] < median + delta) && (theta_backup[i] > (median - delta)))
		{
			//cout << theta_backup[i] << endl;
			r_lines.push_back(Vec4i(l[0], l[1], l[2], l[3]));
			//line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 1, CV_AA);
		}
	}



	return make_tuple(r_lines, angle_rotation);

}

vector <Vec4i> gap(vector<Vec4i> r_lines, int max_gap){
	
	// bubble sort
	bool swap = true;
	while (swap) {
		swap = false;
		for (int i = 0; i < r_lines.size() - 1; i++) {
			if (r_lines[i][0] > r_lines[i + 1][0]) {
				//cout << r_lines[i] << endl;
				//cout << "i+1 vecchio" << r_lines[i + 1] << endl;
				Vec4i j = r_lines[i];
				r_lines[i] = r_lines[i + 1];
				r_lines[i + 1] = j;

				//cout << r_lines[i] << endl;
				//cout << "i+1" << r_lines[i + 1] << endl;
				swap = true;
			}
		}
	}


	vector<Vec4i> barcode_lines;
	bool gap = true;
	int result[2] = { 0, 0 };
	int k = 0;
	while (gap) {

		gap = false;
		for (int i = 0; i < r_lines.size() - 1; i++) {
			Vec4i l = r_lines[i];
			Vec4i l0 = r_lines[i + 1];

			if ((l0[0] - l[0]) > max_gap) {
				result[k] = i;
				k++;
			}
		}
	}


	switch (k)
	{
	case (0): {
		for (int i = 0; i < r_lines.size(); i++) {
			barcode_lines.push_back(r_lines[i]);

		}
		break;
	}

	case (1): {
		if (r_lines.size() - result[0] > result[0])
		{ //gap sinistra
			for (int i = result[0]; i <= (r_lines.size() - result[0]); i++) {
				barcode_lines.push_back(r_lines[i]);
			}
		}
		else { //gap destra
			for (int i = 0; i <= result[0]; i++) {
				barcode_lines.push_back(r_lines[i]);
			}
		}
		break;
	}
	case (2): {
		for (int i = result[0] + 1; i <= (result[1]); i++) {
			barcode_lines.push_back(r_lines[i]);
		}
		break;
	}

	}

		
	return barcode_lines;	
}


int counter_tickness_bars(Mat img, vector<float> px) {

	int distance_x = px[1] - px[0];
	int distance_y = px[3] - px[2];
	int start_x = px[0];
	int start_y = int(px[2] + distance_y / 2);

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
		//int x0 = (int)img.at<uchar>(start_y, i-1);

		if (x == 0) {
			count_min++;
		}
		else {
			temp = count_min;
			count_min = 0;
			if (result > temp && temp != 0) {
				result = temp;
			}
		}
	}

	return result;
}



Mat clahe(Mat bgr_image) {

	Mat lab_image;
	cvtColor(bgr_image, lab_image, CV_BGR2Lab);

	// Extract the L channel
	vector<Mat> lab_planes(3);
	split(lab_image, lab_planes);  // now we have the L image in lab_planes[0
								   // apply the CLAHE algorithm to the L channel
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(4);
	Mat dst;
	clahe->apply(lab_planes[0], dst);

	// Merge the the color planes back into an Lab image
	dst.copyTo(lab_planes[0]);
	merge(lab_planes, lab_image);

	// convert back to RGB
	Mat image_clahe;
	cvtColor(lab_image, image_clahe, CV_Lab2BGR);

	// display the results  (you might also want to see lab_planes[0] before and after).
	return image_clahe;

}

int clahe_detector(Mat src) {

	cvtColor(src, src, CV_RGB2GRAY);

	int threshold = 0;

	int histSize = 256;    // bin size
	float range[] = { 0, 255 };
	const float *ranges[] = { range };


	Mat Hist = Mat::zeros(1, 256, CV_32F); // size=256
	
	calcHist(&src, 1, 0, Mat(), Hist, 1, &histSize, ranges, true, false);
	
	//GaussianBlur(Hist, Hist, Size(1, 25), 0, 8);



	double prob[256];
	double cdf[256];

	for (int i = 0; i < 256; i++) {
		cdf[i] = 0;
	}



	for (int i = 0; i < 256; i++) {
		prob[i] = Hist.at<float>(i)/ double(src.rows * src.cols);
		cdf[i] = cdf[i - 1] + prob[i];

		cout <<  i << " " << prob[i] << endl;
		
	}
	cdf[0] = prob[0];

	for (int i = 1; i < 256; i++) {
		cdf[i] = cdf[i - 1] + prob[i];
		cout << i << " " << cdf[i] << endl;


	}


	if (cdf[128] > 0.9) return 1;
	else return 0;

}


void plot_histogram(Mat Hist, int histSize) {

	// Plot the histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(Hist, Hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(Hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(Hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	namedWindow("Result", 1);    imshow("Result", histImage);
}


