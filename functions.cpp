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
#define delta 0.02 // barcode_orientation function, median calculation
#define range2 20 // barcode_orientation function, counter number positive/negative results




int writeFile(struct barcode_result barcode)
{
	ofstream myfile;
	myfile.open("data/result.txt", ios_base::app);
	
	myfile << "Image: " << barcode.name << " Grade: " << barcode.grade << endl;
	myfile << endl;
	myfile << "Quality Parameters for each scan line: Rmin, Rmax, Symbol Contrast, ECmin, Modulation, Defects " << endl;
	for (int i = 0; i < barcode.parameters.size(); i++) {
		for (int j = 0; j < barcode.parameters[i].size(); j++) {
			myfile << barcode.parameters[i][j] * 100 << "\t" << "\t" << "\t" ;
		}
		myfile << endl;
	}
	myfile << endl;

	myfile << "X Dimension: " << barcode.x_dimension << endl;
	myfile << "Height: " << barcode.height << endl;
	myfile << "Orientation: " << barcode.orientation << endl;
	myfile << "Bounding Box Vertexes: ";
	for (int i = 0; i < barcode.bounding_box.size(); i++) {
		myfile << barcode.bounding_box[i] << "\t";
	}
	myfile << "Center Bounding Box: " << barcode.center << endl;
	myfile << endl;
	myfile << "Number Edges for each scan line: ";
	for (int i = 0; i < barcode.number_edges.size(); i++) {
		myfile << barcode.number_edges[i] << "\t";
	}
	myfile << endl;
	myfile << "Sequence bars and spaces " << endl;
	for (int i = 0; i < barcode.sequence.size(); i++) {
		for (int j = 0; j < barcode.sequence[i].size(); j++) {
			myfile << barcode.sequence[i][j] << "\t";
		}
		myfile << endl;
	}
	myfile << endl;
	myfile << endl;
	myfile << endl;


	return 0;
}

void drawing_box(Mat dst, vector<Point> points) {

	line(dst, points[0], points[1], Scalar(7, 254, 47), 3, CV_AA);
	line(dst, points[1], points[2], Scalar(7, 254, 47), 3, CV_AA);
	line(dst, points[2], points[3], Scalar(7, 254, 47), 3, CV_AA);
	line(dst, points[3], points[0], Scalar(7, 254, 47), 3, CV_AA);
}


//ROTATION OF THE IMAGE
Mat rotation_image(Mat src, float angle_rotation) {

	angle_rotation = ((angle_rotation * 180) / (CV_PI));
	Point2f src_center(src.cols / 2.0F, src.rows / 2.0F);
	Mat rot_mat = getRotationMatrix2D(src_center, angle_rotation, 1.0);
	Mat dst;
	warpAffine(src, dst, rot_mat, src.size());
	return dst;
}


// HOUGH LINES: output vector lines + angle of rotation
tuple <vector<Vec4i>, float> barcode_orientation(Mat src, bool *flag) {
	Mat cdst = src.clone();
	cvtColor(cdst, cdst, CV_GRAY2RGB);
	float media = 0, sum = 0, median = 0, angle_rotation = 0, valore, val;
	int count = 0, counter = 0, counter2 = 0, counter3 = 0, differenza = 0, orientation = 0;
	bool flag_median = false;
	vector<Vec4i> lines;
	vector<Vec4i> r_lines;
	vector<float> theta, theta_backup, theta_r;
	int minLenght = min((src.rows / 10), (src.cols / 10));
	HoughLinesP(src, lines, 1, CV_PI / 180, 30, minLenght, 5);


	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		theta.push_back(abs(atan2((l[3] - l[1]), (l[2] - l[0]))));
		//if (valore = (atan2((l[3] - l[1]), (l[2] - l[0]))) > 0) counter++;
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 1, CV_AA);

	}

	
	theta_backup = theta;

	// MEDIAN
	sort(theta.begin(), theta.end());
	median = theta[theta.size() / 2];
	//cout << "MEDIAN " << median << endl;



	//LOOP AFTER THE CALCULATION OF THE MEDIAN TO SELECT THE CORRECT LINES
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		if ((theta_backup[i] < median + delta) && (theta_backup[i] > (median - delta)))
		{
			r_lines.push_back(Vec4i(l[0], l[1], l[2], l[3]));
			sum = sum + theta_backup[i]; //Sum to obtain the mean value
			line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 1, CV_AA);
		}
	}


	//LOOP TO INCREMENT THE TWO COUNTERS NEEDED FOR THE IDENTIFICATION OF THE SENSE OF THE ROTATION (DO NOT CONSIDER THE LINES ALMOST VERTICAL)
	for (size_t i = 0; i < r_lines.size(); i++)
	{
		Vec4i l = r_lines[i];
		theta_r.push_back(atan2((l[3] - l[1]), (l[2] - l[0])));
		if ((theta_r[i] >= 0) && !(theta_r[i] <= 1.5710 && theta_r[i] >= 1.5706)) counter++;
		if ((theta_r[i] < 0) && !(theta_r[i] >= -1.5710 && theta_r[i] <= -1.5706)) counter2++;

	}

	//CLOCKWISE / COUNTERCLOCKWISE ROTATION
	differenza = counter - counter2;
	if (differenza > range2) orientation = -1; // CLOCKWISE
	else if (differenza < -range2) orientation = 1; // COUNTER-CLOCKWISE
	else if ((-range2 <= differenza) && (differenza <= range2)) orientation = 0;


	//MEAN
	media = sum / r_lines.size();
	angle_rotation = (CV_PI / 2 - abs(media))*(orientation);


	//FLAG TO ADDRESS THE PROBLEM WITH THE SECOND APPLICATION OF THIS FUNCTION WITH OR WITHOUT THE GAUSSIANBLUR
	*flag = false;
	if ((media < CV_PI / 2 - 0.01) || (media > CV_PI / 2 + 0.01)) *flag = true;

	//imshow("HOUGH LINES DETECTED", cdst);
	return make_tuple(r_lines, angle_rotation);

}


//HORIZONTAL, VERTICAL GAPS AND REMOVAL OF THE BROKEN LINES.
vector <Vec4i> gap(vector<Vec4i> r_lines, int max_gap) {

	// bubble sort
	bool swap = true;
	while (swap) {
		swap = false;
		for (int i = 0; i < r_lines.size() - 1; i++) {
			if (r_lines[i][0] > r_lines[i + 1][0]) {
				Vec4i j = r_lines[i];
				r_lines[i] = r_lines[i + 1];
				r_lines[i + 1] = j;
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
		{ //gap LEFT
			for (int i = result[0] + 1; i < (r_lines.size() - result[0]); i++) {
				barcode_lines.push_back(r_lines[i]);
			}
		}
		else { //gap RIGHT
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

vector <Vec4i> vertical_gap(vector<Vec4i> r2_lines, Mat src) {

	const int offset = 10;

	vector<Vec4i> barcode_lines;
	vector <int> hist;
	for (int i = 0; i < src.rows; i++) {
		hist.push_back(0);
	}
	const int size = hist.size();


	for (int i = 0; i < r2_lines.size(); i++) {
		for (int j = r2_lines[i][3]; j < r2_lines[i][1]; j++) {
			hist[j]++;
		}
	}

	int middle = src.rows / 2;
	int i_min, i_max;
	bool found = false;
	//TOP 
	for (int i = middle; i > 0; i--) {
		if (hist[i] == 0 && found == false) {
			i_min = i;
			found = true;
		}
	}

	found = false;
	//BOTTOM
	for (int i = middle; i < size; i++) {
		if (hist[i] == 0 && found == false) {
			i_max = i;
			found = true;
		}
	}

	//SELECTION OF THE GOOD HOUGH LINES
	for (int i = 0; i < r2_lines.size(); i++) {
		if ((r2_lines[i][1] > i_min - offset) && (r2_lines[i][3] < i_max + offset)) {
			barcode_lines.push_back(r2_lines[i]);
		}
	}

	return barcode_lines;
}

vector <int> broken_lines_removal(Mat src, vector<Point> roi, vector <Vec4i> hough) {

	Point corner = roi[0];
	int h = roi[1].y - roi[3].y; // rows - height
	int x = roi[2].x - roi[0].x; // columns - width

	//ROI E CROP DELL'IMMAGINE
	Rect myroi(corner.x, corner.y, x, h);
	Mat crop_image = src(myroi);


	//CREATION OF TWO ROW MAT AND THEN FILLING
	int line_top = crop_image.rows / 3;
	int line_bottom = 2 * crop_image.rows / 3;
	Mat scan_top = Mat::zeros(1, crop_image.cols, CV_8U);
	Mat scan_bottom = Mat::zeros(1, crop_image.cols, CV_8U);
	for (int i = 0; i < scan_top.cols; i++) {
		scan_top.at<uchar>(i) = crop_image.at<uchar>(line_top, i);
		scan_bottom.at<uchar>(i) = crop_image.at<uchar>(line_bottom, i);
	}


	//INDEX VECTOR WILL CONTAIN ONLY THE COLUMNS HAVING A (DARK) BAR OVER THE WHOLE LENGTH
	vector <int> index;
	for (int i = 0; i < scan_top.cols - 2; i++) {
		int vt = scan_top.at<uchar>(i);
		int vb = scan_bottom.at<uchar>(i);
		int vt2 = scan_top.at<uchar>(i+2);
		int vb2 = scan_bottom.at<uchar>(i+2);
		if ((vt < 127) || (vb < 127)) {
			if ((abs(vt - vb) <= 5) && (abs(vt2-vb2) <= 5)) index.push_back(i);

		}
	}


	// X VECTOR WILL CONTAIN THE VALUE XMIN AND XMAX ALREADY TRANSFORMED ACCORDING THE CROPPING OF THE ORIGINAL IMAGE
	vector <int> X;
	X.push_back(index[0] + corner.x);
	X.push_back(index[index.size() - 1] + corner.x);
	bool flag0 = false, flag1 = false;
	int t = 0, s = 0;


	//DOUBLE CHECK WITH THE HOUGH LINES IN ORDER TO SELECT THE CORRECT FIRST AND LAST BAR (MORE ROBUST WITH RESPECT TO NOISE, IMAGE UPC#07)
	for (int i = 0; i < hough.size(); i++) {
		if (abs(hough[i][0] - X[0]) <= 2) flag0 = true;
		if (abs(hough[i][0] - X[1]) <= 2) flag1 = true;
	}

	while (!flag0 || !flag1) {
		if (!flag0) {
			X[0] = index[t + 1] + corner.x;
			for (int i = 0; i < hough.size(); i++) {
				if (abs(hough[i][0] - X[0]) <= 2) flag0 = true;
			}
		}
		else if (!flag1) {
			X[1] = index[index.size() - 1 - s] + corner.x;
			for (int i = 0; i < hough.size(); i++) {
				if (abs(hough[i][0] - X[1]) <= 2) flag0 = true;
			}
		}
		t++;
		s++;

	}

	//ONLY FOR DEBUGGING 
	/*
	cvtColor(crop_image, crop_image, CV_GRAY2RGB);
	for (int i = 0; i < index.size(); i++) {
		circle(crop_image, Point(index[i], crop_image.rows / 2), 1, Scalar(0, 255, 0), 2, 1, 0);
	}
	line(crop_image, Point(X[0] - corner.x, 0), Point(X[0] - corner.x, h), Scalar(0, 0, 255), 5, CV_AA);
	line(crop_image, Point(X[1] - corner.x, 0), Point(X[1] - corner.x, h), Scalar(0, 0, 255), 5, CV_AA);
	//imshow("crop image broken lines removal", crop_image);
	*/
	return X;
}


//OUTPUT THE FIRST AND LAST HOUGH LINES IN ORDER TO DRAW A FIRST APPROXIMATED BOUNDING BOX
vector<float> FirstLastDetector(vector<Vec4i> r_lines) {

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


// COUNTER THICKNESS BARS
int counter_thickness_bars(Mat img, vector<float> px) {

	int distance_x = px[1] - px[0];
	int distance_y = px[3] - px[2];
	int start_x = px[0];
	int start_y = int(px[2] + distance_y / 2);

	int count_min = 0, result = 1000, temp = 0;

	for (int i = start_x; i < (start_x + distance_x); i++)
	{
		int x = (int)img.at<uchar>(start_y, i);

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



// CLAHE EQUALIZATION
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
	int histSize = 256;    // bin size
	float range[] = { 0, 255 };
	const float *ranges[] = { range };
	Mat Hist = Mat::zeros(1, 256, CV_32F); // size=256
	calcHist(&src, 1, 0, Mat(), Hist, 1, &histSize, ranges, true, false);


	double prob[256];
	double cdf[256] = { 0 };

	for (int i = 0; i < 256; i++) {
		prob[i] = Hist.at<float>(i) / double(src.rows * src.cols);
		cdf[i] = cdf[i - 1] + prob[i];
	}

	cdf[0] = prob[0];
	for (int i = 1; i < 256; i++) {
		cdf[i] = cdf[i - 1] + prob[i];
	}

	if (cdf[128] > 0.9) return 1;
	else return 0;

}

void plot_histogram(Mat Hist, int histSize) {

	// Plot the histogram
	int hist_w = histSize; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(Hist, Hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(Hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(Hist.at<float>(i))),
			Scalar(255, 0, 0), 1, 1, 0);
	}

	namedWindow("Result", 1);    imshow("Result", histImage);
}



//HARRIS CORNER DETECTOR
vector<int> Harris(Mat src, vector<Point> roi) {

	Point corner = roi[0];
	int h = roi[1].y - roi[3].y; // rows - height
	int x = roi[2].x - roi[0].x; // columns - width


	//ROI AND CROP OF THE IMAGE
	Rect myroi(corner.x, corner.y, x, h);
	Mat crop_image = src(myroi);
	Mat crop_bk = crop_image.clone();
	cornerHarris(crop_image, crop_image, 4, 5, 0.05, BORDER_DEFAULT);
	Mat dst_norm, dst_norm_scaled;
	normalize(crop_image, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);



	//FILLING OF THE VECTOR CORNERS ACCORDING TO A THRESHOLD
	vector<Point> corners;
	for (int j = 0; j < dst_norm.rows; j++)
	{
		for (int i = 0; i < dst_norm.cols; i++)
		{
			if ((int)dst_norm.at<float>(j, i) > 127)
			{
				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
				corners.push_back(Point(i, j));
			}
		}
	}


	vector<Point> top, bottom, middle;
	cvtColor(crop_bk, crop_bk, CV_GRAY2RGB);


	//HARRIS CORNERS SPLIT BETWEEN TOP AND BOTTOM ACCORDING TO A WINDOWS OF ROWS OF 1/5 OF THE TOTAL NUMBER OF ROWS
	for (int i = 0; i < corners.size(); i++) {
		if (corners[i].y < crop_bk.rows / 5) {
			top.push_back(corners[i]);
			//circle(crop_bk, Point(corners[i]), 2, Scalar(0, 255, 0), 1);
		}
		else if (corners[i].y > 4 * crop_bk.rows / 5) {
			bottom.push_back(corners[i]);
			//circle(crop_bk, Point(corners[i]), 2, Scalar(0, 255, 0), 1);
		}
		else {
			middle.push_back(corners[i]);
			//circle(crop_bk, Point(corners[i]), 2, Scalar(255, 0, 0), 1);
		}
	}


	//CORRESPONDENCES BETWEEN TOP AND BOTTOM
	vector<Point> result_top, result_bot;
	for (int i = 0; i < top.size(); i++) {
		for (int j = 0; j < bottom.size(); j++) {
			if ((bottom[j].x == top[i].x)) {
				result_top.push_back(top[i]);
				result_bot.push_back(bottom[j]);
				circle(crop_bk, Point(top[i]), 2, Scalar(0, 0, 255), 1);
				circle(crop_bk, Point(bottom[j]), 2, Scalar(0, 0, 255), 1);
			}
		}
	}


	//MINIMUM HEIGHT BARCODE
	int min = crop_image.rows;
	int riga = 0;
	for (int i = 0; i < result_top.size(); i++) {
		int diff = result_bot[i].y - result_top[i].y;
		if (diff < min) {
			min = diff;
			riga = i;
		}
	}

	//MINIMUM HEIGHT LINE
	line(crop_bk, result_top[riga], result_bot[riga], Scalar(0, 255, 0), 3, CV_AA);


	//YMIN AND YMAX FOR THE BOUNDING BOX DRAWING
	int bot_y_min = crop_image.rows;
	int top_y_min = 0;
	for (int i = 0; i < result_bot.size(); i++) {
		if (result_bot[i].y < bot_y_min) bot_y_min = result_bot[i].y;
		if (result_top[i].y > top_y_min) top_y_min = result_top[i].y;
	}



	//TRANSFORMATION FROM THE CROPPED REFERENCE FRAME TO THE ORIGINAL IMAGE REFERENCE FRAME
	vector<int> result;
	result.push_back(top_y_min + corner.y);
	result.push_back(bot_y_min + corner.y);
	imshow("HARRIS CORNERS AND MINIMUM HEIGHT", crop_bk);

	return result;
}


//PARAMETERS CALCULATION (REFLECTANCE, CONTRAST, MODULATION, DEFECTS)
vector <vector <float>> scan_images_average(Mat src, vector<Point> harris_points, vector <int>& grade, vector< vector <int> >& crosses, vector <bool>& flag) {

	int h = harris_points[1].y - harris_points[0].y;
	int space = (h / 11);
	int w = harris_points[2].x - harris_points[0].x;

	float Number_Edges = 0, Rmin = 0, Rmax = 0, ECmin = 0, SC = 0, Mod = 0, Def = 0, TH = 0;

	// VETTORE CON LE COORDINATE DELLE SCAN PROFILES (Y)
	int update = harris_points[0].y + space;
	vector <int> coord;
	for (int i = 0; i < 10; i++) {
		coord.push_back(update);
		update += space;
	}



	//IMMAGINE DOVE OGNI RIGA è UNA SCAN PROFILE
	Mat scan_profiles = Mat::zeros(10, w, CV_32F); // size=width
	for (int k = 0; k < 10; k++) {
		for (int i = harris_points[0].x; i < harris_points[2].x; i++) {
			int value = src.at<uchar>(coord[k], i);
			scan_profiles.at<float>(k, i - harris_points[0].x) = value;
		}
	}

	vector<Mat> scan;
	int temp = 0;
	for (int i = 0; i < 10; i++) {
		Rect myroi(0, i, w, 1);
		scan.push_back(scan_profiles(myroi));
	}



	vector <vector <float>> parameters;
	for (int i = 0; i < 10; i++) {
		parameters.push_back(scan_parameters(scan[i], crosses, flag, i));
		grade.push_back(minimum_grade(parameters[i]));
	}



	

	return parameters;
}


//COMPUTATION OF THE QUALITY PARAMETERS FOR EACH SCAN REFLECTANCE PROFILE
vector <float> scan_parameters(Mat scan, vector < vector <int> >& crosses,vector <bool>& flag, int p) {

	vector <float> parameters;
	vector <int> cross;


	//REFLECTANCE
	float Rmax = 0, Rmin = 255;
	for (int i = 0; i < scan.cols; i++) {
		float temp = scan.at<float>(i);
		if (temp > Rmax) Rmax = temp;
		if (temp < Rmin) Rmin = temp;
	}

	Rmax = Rmax / 256;
	Rmin = Rmin / 256;
	parameters.push_back(Rmin);
	parameters.push_back(Rmax);
	//cout << "Max Reflectance: " << Rmax << endl;
	//cout << "Min Reflectance: " << Rmin << endl;
	
	
	//SYMBOL CONTRAST
	float symbol_contrast = Rmax - Rmin;
	parameters.push_back(symbol_contrast);
	//cout << "Symbol Contast: " << symbol_contrast << endl;


	//GLOBAL THRESHOLD
	float gt = Rmin + symbol_contrast / 2;
	//cout << "Global Threshold: " << gt << endl;



	float threshold = gt * 255;
	//cout << "Threshold: " << threshold << endl;


	// CROSS CONTAINS X COORDINATES OF TRANSITION PIXELS ACCORDING TO THE GLOBAL THRESHOLD LINE
	for (int i = 1; i < scan.cols; i++) {
		float temp = scan.at<float>(i);
		float temp0 = scan.at<float>(i - 1);
		if (((temp >= threshold) && (temp0 < threshold)) || ((temp <= threshold) && (temp0 > threshold))) {
			cross.push_back(i);

		}
	}
	//NUMBER EDGES
	float number_edges = cross.size();
	//cout << "number edges: " << number_edges << endl;



	//MIN EDGE CONTRAST - ECMIN
	vector <int> spaces, bars;
	float ECmin = ecmin_calculation(cross, scan, threshold, spaces, bars);

	ECmin = ECmin / 256;
	parameters.push_back(ECmin);
	//cout << "ECmin " << ECmin << endl;


	//MODULATION
	float Modulation = (ECmin) / symbol_contrast;	
	parameters.push_back(Modulation);
	//cout << "Modulation " << Modulation * 100 << "%" << endl;



	//ERN
	vector<int> defects_space, defects_bar;
	float ERNmax = ernmax_calculation(scan, cross, threshold, spaces, bars, defects_space, defects_bar);	
	//cout << "ERN Max " << ERNmax << endl;


	//DEFECTS
	float Defects = (ERNmax / 256) / symbol_contrast;
	parameters.push_back(Defects);
	//cout << "Defects " << Defects * 100 << "%" << endl;


	//FLAG USED TO ADDRESS THE CASE IN WHICH SOME SCAN REFLECTANCE PROFILES STARTS WITH A SPACE AND OTHERS WITH A BAR (ON THE SAME BARCODE, example UPC#01)
	flag[p] = false;
	if (scan.at<float>(cross[0]) < threshold) flag[p] = true;

	crosses.push_back(cross);

	//cout << endl;

	/*
	// CODICE DI TEST SU UNA SOLA SCAN PROFILE LINE
	int hist_w = scan.cols; int hist_h = 255;
	int bin_w = cvRound((double)hist_w / scan.cols);
	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	for (int i = 1; i < scan.cols; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(scan.at<float>(i - 1))), Point(bin_w*(i), hist_h - cvRound(scan.at<float>(i))), Scalar(255, 0, 0), 1, 1, 0);
	}

	imshow("Threshold = RED, Rmax = BLUE, Rmin = GREEN", histImage);
	*/

	return parameters;
}


//ECMIN COMPUTATION
float ecmin_calculation(vector<int> cross, Mat scan, float threshold, vector <int>& spaces, vector <int>& bars) {
	
	//CHECKING IF THE BEGINNING OF THE BOUNDING BOX IS A SPACE OR A BAR
	bool check;
	if (scan.at<float>(cross[0] / 2) <= threshold) check = true; //BAR
	else check = false; // SPACE
	

	int temp_max = 0, temp_min = 255;
	for (int i = 0; i < cross[0]; i++) {
		int temp = scan.at<float>(i);
		if (temp > temp_max && !check) temp_max = temp; //BAR
		if (temp < temp_min && check) temp_min = temp; //SPACE
	}
	if (!check) spaces.push_back(temp_max);
	else bars.push_back(temp_min);

	check = !check;

	temp_min = 255;
	temp_max = 0;
	for (int k = 0; k < cross.size() - 1; k++) {
		for (int i = cross[k]; i < cross[k + 1]; i++) {
			int temp = scan.at<float>(i);
			if (check) {
				if (temp < temp_min) temp_min = temp; // BAR
			}
			else {
				if (temp > temp_max) temp_max = temp; // SPACE
			}
		}
		if (check) bars.push_back(temp_min);
		else spaces.push_back(temp_max);
		temp_min = 255;
		temp_max = 0;
		check = !check;
	}

	temp_max = 0;
	temp_min = 255;
	for (int i = cross[cross.size() - 1]; i < scan.cols; i++) {
		int temp = scan.at<float>(i);
		if (temp > temp_max && !check) temp_max = temp; //SPACE
		if (temp < temp_min && check) temp_min = temp; //BAR
	}
	if (!check) spaces.push_back(temp_max);
	else bars.push_back(temp_min);



	vector <int> ec;
	int dim, diff, diff2;
	bool same_size = false;
	if (spaces.size() > bars.size()) {
		dim = bars.size();
		check = true;
	}
	else if(spaces.size() < bars.size()) {
		dim = spaces.size();
		check = false;
	}
	else {
		dim = spaces.size();
		same_size = true;
	}

	for (int i = 0; i < dim; i++) {
		diff = spaces[i] - bars[i];
		ec.push_back(diff);

		if (same_size && (i = dim - 1)) break;
		if (check) diff2 = spaces[i + 1] - bars[i];
		else diff2 = spaces[i] - bars[i + 1];
		ec.push_back(diff2);
		}

	float ECmin = 255, ECmin_index = 0;
	for (int i = 0; i < ec.size(); i++) {
		int temp = ec[i];
		if (temp < ECmin) {
			ECmin = temp;
			ECmin_index = i;
		}

	}

	return ECmin;
}


//ERNMAX COMPUTATION
float ernmax_calculation(Mat scan, vector <int> cross, float threshold, vector<int> spaces, vector<int> bars, vector<int>& defects_space, vector<int>& defects_bar) {
	//CHECKING IF THE BEGINNING OF THE BOUNDING BOX IS A SPACE OR A BAR
	bool check;
	if (scan.at<float>(cross[0] / 2) <= threshold) check = true; //BAR
	else check = false; // SPACE
	

	// FIRST INTERVAL
	int t_space = 255, t_bar = 0;
	for (int j = 1; j < cross[0] - 1; j++) {
		if (!check) {
			if ((scan.at<float>(j) <= scan.at<float>(j - 1)) && (scan.at<float>(j) <= scan.at<float>(j + 1)) && (scan.at<float>(j) <= t_space)) {
				t_space = scan.at<float>(j);
			}
		}
		else {
			if ((scan.at<float>(j) >= scan.at<float>(j - 1)) && (scan.at<float>(j) >= scan.at<float>(j + 1)) && (scan.at<float>(j) >= t_bar)) {
				t_bar = scan.at<float>(j);
			}
		}
		
	}
	if(!check) defects_space.push_back(t_space);
	else defects_bar.push_back(t_bar);
	check = !check;
	
	//INTERVALS IN BETWEEN TWO CROSSES
	for (int i = 0; i < cross.size() - 1; i++) {
		t_bar = 0;
		t_space = 255;
		for (int j = cross[i]; j < cross[i + 1]; j++) {
			if (check) { 
				if ((scan.at<float>(j) >= scan.at<float>(j - 1)) && (scan.at<float>(j) >= scan.at<float>(j + 1)) && (scan.at<float>(j) >= t_bar)) {
					t_bar = scan.at<float>(j);
				}
			}
			else { 
				if ((scan.at<float>(j) <= scan.at<float>(j - 1)) && (scan.at<float>(j) <= scan.at<float>(j + 1)) && (scan.at<float>(j) <= t_space)) {
					t_space = scan.at<float>(j);
				}
			}
		}
		if (check) defects_bar.push_back(t_bar);
		else defects_space.push_back(t_space);
		check = !check;
	}

	// LAST INTERVAL
	t_space = 255;
	t_bar = 0;
	for (int j = cross[cross.size()-1]+1; j < scan.cols-1; j++) {
		if (!check) {
			if ((scan.at<float>(j) <= scan.at<float>(j - 1)) && (scan.at<float>(j) <= scan.at<float>(j + 1)) && (scan.at<float>(j) <= t_space)) {
				t_space = scan.at<float>(j);
			}
		}
		else {
			if ((scan.at<float>(j) >= scan.at<float>(j - 1)) && (scan.at<float>(j) >= scan.at<float>(j + 1)) && (scan.at<float>(j) >= t_bar)) {
				t_bar = scan.at<float>(j);
			}
		}

	}
	if (!check) defects_space.push_back(t_space);
	else defects_bar.push_back(t_bar);



	// LOOKING FOR THE MAXIMUM
	float ERNmax = 0;
	for (int i = 0; i < defects_space.size() - 1; i++) {
		if (defects_space[i] != 255) {
			int value = spaces[i] - defects_space[i];
			if (value > ERNmax) {
				ERNmax = value;
			}
		}
	}
	for (int i = 0; i < defects_bar.size() - 1; i++) {
		if (defects_bar[i] != 0) {
			int value = defects_bar[i] - bars[i];
			if (value > ERNmax) {
				ERNmax = value;
			}
		}
	}

	return ERNmax;
}


//GRADE ASSIGNMENT
int minimum_grade(vector <float> param) {
	vector<int> result;
	
	//Reflectance
	if (param[0] <= 0.5*param[1]) result.push_back(4);
	else result.push_back(0);

	//Symbol Contrast
	if (param[2] >= 0.7) result.push_back(4);
	else if (param[2] >= 0.55) result.push_back(3);
	else if (param[2] >= 0.40) result.push_back(2);
	else if (param[2] >= 0.20) result.push_back(1);
	else result.push_back(0);

	//ECMIN
	if (param[3] >= 0.15) result.push_back(4);
	else result.push_back(0);

	//MODULATION
	if (param[4] >= 0.7) result.push_back(4);
	else if (param[4] >= 0.6) result.push_back(3);
	else if (param[4] >= 0.5) result.push_back(2);
	else if (param[4] >= 0.4) result.push_back(1);
	else result.push_back(0);

	//DEFECTS
	if (param[5] > 0.3) result.push_back(0);
	else if (param[5] > 0.25 && param[5] <= 0.3) result.push_back(1);
	else if (param[5] > 0.2 && param[5] <= 0.25) result.push_back(2);
	else if (param[5] > 0.15 && param[5] <= 0.2) result.push_back(3);
	else result.push_back(4);

	//SCAN GRADE
	int temp = result[0];
	for (int i = 0; i < result.size(); i++) {
		if (temp >= result[i]) temp = result[i];
	}

	result.push_back(temp);

	return temp;
}

string overall_grade(vector <int> grade) {

	float sum = 0;
	for (int i = 0; i < 10; i++) {
		sum = sum + grade[i];
	}
	float mean_grade = sum / 10;


	//SYMBOL AVERAGE
	string symbol_grade;
	if (mean_grade <= 4 && mean_grade >= 3.5) symbol_grade = "A";
	else if (mean_grade < 3.5 && mean_grade >= 2.5) symbol_grade = "B";
	else if (mean_grade < 2.5 && mean_grade >= 1.5) symbol_grade = "C";
	else if (mean_grade < 1.5 && mean_grade >= 0.5) symbol_grade = "D";
	else symbol_grade = "F";

	return symbol_grade;
}


//MULTIDIMENTIONAL VECTOR CONTAINING THE SEQUENCE OF SPACES/BARS FOR EACH SCAN REFLECTANCE PROFILE
vector < vector <float>> sequence_spaces_bars(vector< vector <int> > crosses, int X, vector <bool> flag) {


	vector < vector <float> > sizes;
	vector <float> temp;

	for (int i = 0; i < crosses.size(); i++) {
		if (flag[i]) {
			for (int j = 1; j < crosses[i].size(); j++) {
				int size = crosses[i][j] - crosses[i][j - 1];
				temp.push_back(size/X);
			}
		}
		else {
			for (int j = 2; j < crosses[i].size()-1; j++) {
				int size = crosses[i][j] - crosses[i][j - 1];
				temp.push_back(size);
			}
		}
		sizes.push_back(temp);
		temp.clear();
	}


	return sizes;
}


//COUNTER OF THE NUMBER OF EDGES
vector <int> number_edges(vector < vector <int> > crosses) {

	vector <int> edges;

	for (int i = 0; i < crosses.size(); i++) {
		edges.push_back(crosses[i].size());
	}

	return edges;
}
