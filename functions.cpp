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

void drawing_box(Mat dst, vector<Point> points) {

	line(dst, points[0], points[1], Scalar(7, 254, 47), 3, CV_AA);
	line(dst, points[1], points[2], Scalar(7, 254, 47), 3, CV_AA);
	line(dst, points[2], points[3], Scalar(7, 254, 47), 3, CV_AA);
	line(dst, points[3], points[0], Scalar(7, 254, 47), 3, CV_AA);
}


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


Mat rotation_image(Mat src, float angle_rotation) {

	angle_rotation = ((angle_rotation * 180) / (CV_PI));
	Point2f src_center(src.cols / 2.0F, src.rows / 2.0F);
	Mat rot_mat = getRotationMatrix2D(src_center, angle_rotation, 1.0);
	Mat dst;
	warpAffine(src, dst, rot_mat, src.size());
	return dst;
}


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
	//cout << "minimum lenght " << minLenght << endl;
	HoughLinesP(src, lines, 1, CV_PI / 180, 30, minLenght, 5);


	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		theta.push_back(abs(atan2((l[3] - l[1]), (l[2] - l[0]))));
		//if (valore = (atan2((l[3] - l[1]), (l[2] - l[0]))) > 0) counter++;
		//line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 1, CV_AA);

	}



	theta_backup = theta;

	// calcolo mediana
	sort(theta.begin(), theta.end());
	median = theta[theta.size() / 2];
	//cout << "mediana " << median << endl;



	//ciclo dopo aver calcolato la mediana per selezionare le righe giuste
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		if ((theta_backup[i] < median + delta) && (theta_backup[i] > (median - delta)))
		{
			//cout << theta_backup[i] << endl;
			r_lines.push_back(Vec4i(l[0], l[1], l[2], l[3]));
			sum = sum + theta_backup[i];
			line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 1, CV_AA);
		}
	}


	for (size_t i = 0; i < r_lines.size(); i++)
	{
		Vec4i l = r_lines[i];
		theta_r.push_back(atan2((l[3] - l[1]), (l[2] - l[0])));
		if ((theta_r[i] >= 0) && !(theta_r[i] <= 1.5710 && theta_r[i] >= 1.5706)) counter++;
		if ((theta_r[i] < 0) && !(theta_r[i] >= -1.5710 && theta_r[i] <= -1.5706)) counter2++;
		//cout << theta_r[i] << endl;

	}


	// verso rotazione
	//cout << "counter positivi " << counter << endl;
	//cout << "counter negativi " << counter2 << endl;

	differenza = counter - counter2;
	if (differenza > range2) orientation = -1; // senso orario
	else if (differenza < -range2) orientation = 1; // più negativi che positivi
	else if ((-range2 <= differenza) && (differenza <= range2)) orientation = 0;
	//cout << "verso di rotazione:  " << orientation << endl;


	media = sum / r_lines.size();
	//cout << "media theta " << media << endl;
	angle_rotation = (CV_PI / 2 - abs(media))*(orientation);


	*flag = false;
	if ((media < CV_PI / 2 - 0.01) || (media > CV_PI / 2 + 0.01)) *flag = true;

	//imshow("cdst", cdst);


	return make_tuple(r_lines, angle_rotation);

}


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
	//cout << " numero di gap:  " << k << endl;


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
			for (int i = result[0] + 1; i < (r_lines.size() - result[0]); i++) {
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


vector <Vec4i> vertical_gap(vector<Vec4i> r2_lines, Mat src) {

	const int offset = 10;

	vector<Vec4i> barcode_lines;
	//distribuzione tipo "istogramma" del barcode verticalmente
	const int size = 1280;
	int hist[size] = { 0 };
	for (int i = 0; i < r2_lines.size(); i++) {
		for (int j = r2_lines[i][3]; j < r2_lines[i][1]; j++) {
			hist[j]++;
		}
	}

	int middle = src.rows / 2;
	int i_min, i_max;
	bool found = false;
	//parte superiore
	for (int i = middle; i > 0; i--) {
		if (hist[i] == 0 && found == false) {
			i_min = i;
			found = true;
		}
	}

	found = false;
	//parte inferire
	for (int i = middle; i < size; i++) {
		if (hist[i] == 0 && found == false) {
			i_max = i;
			found = true;
		}
	}

	//selezione delle hough lines buone dentro barcode_lines
	for (int i = 0; i < r2_lines.size(); i++) {
		if ((r2_lines[i][1] > i_min - offset) && (r2_lines[i][3] < i_max + offset)) {
			barcode_lines.push_back(r2_lines[i]);
		}
	}

	return barcode_lines;
}


vector<float> Harris(Mat src, vector<Point> roi, int *height) {

	Point corner = roi[0];
	int h = roi[1].y - roi[3].y; // rows - height
	int x = roi[2].x - roi[0].x; // columns - width


								 //ROI E CROP DELL'IMMAGINE
	Rect myroi(corner.x, corner.y, x, h);
	Mat crop_image = src(myroi);
	Mat crop_bk = crop_image.clone();
	cornerHarris(crop_image, crop_image, 4, 5, 0.05, BORDER_DEFAULT);
	Mat dst_norm, dst_norm_scaled;
	normalize(crop_image, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);


	//RIEMPIO VECTOR CORNERS IN BASE AD UNA THRESHOLD
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


	//imshow("crop", dst_norm_scaled);
	vector<Point> top, bottom;
	cvtColor(crop_bk, crop_bk, CV_GRAY2RGB);

	//HARRIS CORNERS DIVISI TRA BOTTOM E TOP CERCANDO IN UNA FASCIA DI PIXELS UGUALE AD 1/5 DEL NUMERO TOTALE DI COLONNE
	for (int i = 0; i < corners.size(); i++) {
		if (corners[i].y < crop_bk.rows / 5) {
			top.push_back(corners[i]);
			circle(crop_bk, Point(corners[i]), 2, Scalar(0, 255, 0), 1);
		}
		else if (corners[i].y > 4 * crop_bk.rows / 5) {
			bottom.push_back(corners[i]);
			circle(crop_bk, Point(corners[i]), 2, Scalar(0, 255, 0), 1);
		}
	}

	//CORRISPONDENZE TRA TOP E BOTTOM PER OTTENERE UNA PRIMA SELEZIONE
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


	// ALTEZZA MINIMA BARCODE
	int min = crop_image.rows;
	int riga = 0;
	for (int i = 0; i < result_top.size(); i++) {
		int diff = result_bot[i].y - result_top[i].y;
		if (diff < min) {
			min = diff;
			riga = i;
		}
	}
	//DISPLAY DELLA LINEA DOVE E' STATA CALCOLATA L'ALEZZA MINIMA
	//line(crop_bk, result_top[riga], result_bot[riga], Scalar(0, 255, 0), 3, CV_AA);


	//CALCOLO VALORI Y (MIN E MAX) DA USARE PER IL BOUNDING BOX
	int bot_y_min = crop_image.rows;
	int top_y_min = 0;
	for (int i = 0; i < result_bot.size(); i++) {
		if (result_bot[i].y < bot_y_min) bot_y_min = result_bot[i].y;
		if (result_top[i].y > top_y_min) top_y_min = result_top[i].y;
	}


	//TRASLAZIONE PER MUOVERSI DAL CROP ALL'IMMAGINE ORIGINALE
	*height = min;
	vector<float> result;
	result.push_back(top_y_min + corner.y);
	result.push_back(bot_y_min + corner.y);


	//imshow("HARRIS CORNERS AND MINIMUM HEIGHT", crop_bk);


	return result;
}



float edges_counter(Mat src) {

	//cvtColor(src, src, CV_RGB2GRAY);
	Mat edges;
	Canny(src, edges, 50, 150, 3, true);
	float count_edges = 0;
	for (int i = 0; i < edges.cols; i++) {
		int value = edges.at<uchar>(edges.rows / 2, i);
		if (value == 255) count_edges++;
	}
	cout << "Number Edges: " << count_edges << endl;

	return count_edges;
}




void scan_images_average(Mat src, vector<Point> harris_points) {

	int h = harris_points[1].y - harris_points[0].y;
	int space = (h / 10);
	int w = harris_points[2].x - harris_points[0].x;
	Rect myroi(harris_points[0].x, harris_points[0].y, w, h);
	Mat crop_image = src(myroi);

	//CROP IN 10 IMMAGINI SEPARATE
	vector<Mat> crop_images;
	int temp = 0;
	for (int i = 0; i < 10; i++) {
		temp = temp + space;
		Rect myroi(harris_points[0].x, harris_points[1].y - temp, w, space);
		crop_images.push_back(src(myroi));
	}


	float Number_Edges = 0, Rmin = 0, Rmax = 0, ECmin = 0, SC = 0, Mod = 0, Def = 0;

	for (int i = 0; i < 10; i++) {
		vector<float> result = scan_parameters(crop_images[i]);
		Rmax += result[0];
		Rmin += result[1];
		SC += result[2];
		ECmin += result[3];
		Mod += result[4];

	}


	// media di ogni valore;
	cout << endl;
	cout << "MEAN VALUES " << endl;
	cout << "Rmin " << Rmin / 10 << endl;
	cout << "Rmax " << Rmax / 10 << endl;
	cout << "SC " << SC / 10 << endl;
	cout << "ECmin " << ECmin / 10 << endl;
	cout << "Mod " << Mod / 10 << endl;




	

}









vector<float> scan_images(Mat working) {


	float number_edges = edges_counter(working);


	//Scan Reflectance Profile
	Mat scan_profile = Mat::zeros(1, working.cols, CV_32F); // size=width
	for (int i = 0; i < working.cols; i++) {
		int value = working.at<uchar>(working.rows / 2, i);
		scan_profile.at<float>(i) = value;
	}


	//CALCOLO PARAMETRI
	float ECmin = 255, symbol_contrast = 0, modulation = 0, ERNman = 0;


	//REFLECTANCE
	float Rmax = 0, Rmin = 255;
	for (int i = 0; i < scan_profile.cols; i++) {
		float temp = scan_profile.at<float>(i);
		if (temp > Rmax) Rmax = temp;
		if (temp < Rmin) Rmin = temp;
	}
	cout << "Max Reflectance: " << Rmax << endl;
	cout << "Min Reflectance: " << Rmin << endl;


	//MIN EDGE CONTRAST - ECMIN
	int threshold = (Rmax - Rmin) / 2;

	vector<int> cross;
	for (int i = 1; i < scan_profile.cols; i++) {
		float temp = scan_profile.at<float>(i);
		float temp0 = scan_profile.at<float>(i - 1);
		if (((temp >= threshold) && (temp0 < threshold)) || ((temp <= threshold) && (temp0 > threshold))) {
			cross.push_back(i);
			//circle(working, Point(i,working.rows/2), 1, Scalar(0, 0, 255), 1);
		}
	}

	vector <float> estremi, local_estremi;

	// intervallo: inizio barcode - cross[0] 
	bool flag; // true se il primo è un massimo
	if ((scan_profile.at<float>(cross[0])) < threshold) { // cerchiamo un massimo
		int tp = 0;
		flag = true;
		for (int i = 0; i < cross[0]; i++) {
			if (scan_profile.at<float>(i) > tp) {
				tp = scan_profile.at<float>(i);
			}
		}
		estremi.push_back(tp);
	}
	else {	 // cerchiamo un minimo
		int tp = 255;
		flag = false;
		for (int i = 0; i < cross[0]; i++) {
			if (scan_profile.at<float>(i) < tp) {
				tp = scan_profile.at<float>(i);
			}
		}
		estremi.push_back(tp);
	}


	// intervallo: tra due cross
	for (int i = 0; i < cross.size() - 1; i++) {
		int t_max = 0, t_min = 255;
		for (int j = cross[i]; j < cross[i + 1]; j++) {
			if (flag) {
				if (scan_profile.at<float>(j) < t_min) {
					t_min = scan_profile.at<float>(j);
				}
			}
			else {
				if (scan_profile.at<float>(j) > t_max) {
					t_max = scan_profile.at<float>(j);
				}
			}
		}
		if (flag) estremi.push_back(t_min);
		else estremi.push_back(t_max);
		flag = !flag;
	}

	//ultimo intervallo: cross[ultimo] - fine barcode
	if (flag) {
		int tp = 0;
		for (int i = cross[cross.size() - 1]; i < working.cols; i++) {
			if (scan_profile.at<float>(i) > tp) {
				tp = scan_profile.at<float>(i);
				//cout << scan_profile.at<float>(i) << endl;
			}
		}
		estremi.push_back(tp);

	}
	else {	 // cerchiamo un minimo
		int tp = 255;
		for (int i = cross[cross.size() - 1]; i < working.cols; i++) {
			if (scan_profile.at<float>(i) < tp) {
				tp = scan_profile.at<float>(i);
			}
		}
		estremi.push_back(tp);

	}


	vector<float> defects;
	//primo intervallo

	if ((scan_profile.at<float>(cross[0])) < threshold) { // cerchiamo un picco di minimo nel massimo
		int t_min = 0;
		flag = true;
		for (int j = 1; j < cross[0] - 1; j++) {
			if ((scan_profile.at<float>(j) >= scan_profile.at<float>(j - 1)) && (scan_profile.at<float>(j) >= scan_profile.at<float>(j + 1)) && (scan_profile.at<float>(j) > t_min)) {
				t_min = scan_profile.at<float>(j);
			}
		}
		defects.push_back(t_min);
	}
	else {																 // cerchiamo un picco di massimo nel minimo
		int t_max = 255;
		flag = false;
		for (int j = 1; j < cross[0] - 1; j++) {
			if ((scan_profile.at<float>(j) <= scan_profile.at<float>(j - 1)) && (scan_profile.at<float>(j) <= scan_profile.at<float>(j + 1)) && (scan_profile.at<float>(j) < t_max)) {
				t_max = scan_profile.at<float>(j);
			}
		}
		defects.push_back(t_max);
	}
	//intervalli successivi
	for (int i = 0; i < cross.size() - 1; i++) {
		int t_max = 255, t_min = 0;
		for (int j = cross[i]; j < cross[i + 1]; j++) {
			if (flag) { // cerchiamo un massimo nel minimo
				if ((scan_profile.at<float>(j) >= scan_profile.at<float>(j - 1)) && (scan_profile.at<float>(j) >= scan_profile.at<float>(j + 1)) && (scan_profile.at<float>(j) > t_min)) {
					t_min = scan_profile.at<float>(j);
				}
			}
			else { //cerchiamo un minimo nel massimo
				if ((scan_profile.at<float>(j) <= scan_profile.at<float>(j - 1)) && (scan_profile.at<float>(j) <= scan_profile.at<float>(j + 1)) && (scan_profile.at<float>(j) < t_max)) {
					t_max = scan_profile.at<float>(j);
				}
			}
		}
		if (flag) defects.push_back(t_min);
		else defects.push_back(t_max);
		flag = !flag;
	}
	//ultimo intervall
	if ((scan_profile.at<float>(cross[cross.size() - 1])) > threshold) { // cerchiamo un picco di minimo nel massimo
		int t_min = 0;
		flag = true;
		for (int j = cross[cross.size() - 1] + 1; j < working.cols - 1; j++) {
			if ((scan_profile.at<float>(j) <= scan_profile.at<float>(j - 1)) && (scan_profile.at<float>(j) <= scan_profile.at<float>(j + 1)) && (scan_profile.at<float>(j) > t_min)) {
				t_min = scan_profile.at<float>(j);
			}
		}
		defects.push_back(t_min);
	}
	else {																 // cerchiamo un picco di massimo nel minimo
		int t_max = 255;
		flag = false;
		for (int j = cross[cross.size() - 1] + 1; j < working.cols - 1; j++) {

			if ((scan_profile.at<float>(j) >= scan_profile.at<float>(j - 1)) && (scan_profile.at<float>(j) >= scan_profile.at<float>(j + 1)) && (scan_profile.at<float>(j) < t_max)) {
				t_max = scan_profile.at<float>(j);
			}
		}
		defects.push_back(t_max);
	}



	for (int i = 0; i < estremi.size() - 1; i++) {
		int x = abs(estremi[i] - estremi[i + 1]);
		if (x < ECmin) ECmin = x;

	}
	cout << "Min Edge Contrast (ECmin): " << ECmin << endl;


	//SYMBOL CONTRAST
	symbol_contrast = Rmax - Rmin;
	float symbol_contrast_perc = ((Rmax - Rmin) / 255) * 100;
	cout << "Symbol Constrast (SC): " << symbol_contrast << endl;
	cout << "Symbol Constrast (%): " << symbol_contrast_perc << endl;

	//MODULATION
	modulation = ECmin / symbol_contrast;
	cout << "Modulation: " << modulation << endl;



	//DEFECT CALCULATION - ERNmax
	vector<float> ern;
	for (int i = 0; i < estremi.size(); i++) {
		if (defects[i] == 0) ern.push_back(0);
		else {
			int value = abs(estremi[i] - defects[i]);
			ern.push_back(value);
		}
	}

	int t = ern[0];
	for (int i = 1; i < ern.size(); i++) {
		if (ern[i] > t) t = ern[i];
	}
	int max_ern = t;
	cout << "Max Ern: " << max_ern << endl;

	float defect = max_ern / symbol_contrast;
	cout << "Defect (ERN): " << defect << endl;

	//imshow("crop", working);



	vector<float> vec = { number_edges, Rmin, Rmax, ECmin, symbol_contrast, modulation, defect };
	return vec;
}


vector <float> scan_parameters(Mat working) {

	//float number_edges = edges_counter(working);

	//Scan Reflectance Profile
	Mat scan_profile = Mat::zeros(1, working.cols, CV_8U); // size=width
	for (int i = 0; i < working.cols; i++) {
		int value = working.at<uchar>(working.rows / 2, i);
		scan_profile.at<uchar>(i) = value;
	}



	//REFLECTANCE
	float Rmax = 0, Rmin = 255;
	for (int i = 0; i < scan_profile.cols; i++) {
		float temp = scan_profile.at<uchar>(i);
		if (temp > Rmax) Rmax = temp;
		if (temp < Rmin) Rmin = temp;
	}

	cout << "Max Reflectance: " << Rmax << endl;
	cout << "Min Reflectance: " << Rmin << endl;


	Rmax = Rmax / 255;
	Rmin = Rmin / 255;

	cout << "Max Reflectance: " << Rmax << endl;
	cout << "Min Reflectance: " << Rmin << endl;

	float symbol_contrast = Rmax - Rmin;
	cout << "Symbol Contast: " << symbol_contrast << endl;

	float gt = Rmin + symbol_contrast / 2;
	cout << "Global Threshold: " << gt << endl;


	//MIN EDGE CONTRAST - ECMIN
	float threshold = gt*255;

	vector<int> cross;
	for (int i = 1; i < scan_profile.cols; i++) {
		float temp = scan_profile.at<uchar>(i);
		float temp0 = scan_profile.at<uchar>(i - 1);
		//cout << "scan profile " << i << " values "<< temp << endl;
		if (((temp >= threshold) && (temp0 < threshold)) || ((temp <= threshold) && (temp0 > threshold))) {
			cross.push_back(i);
			circle(working, Point(i,working.rows/2), 1, Scalar(0, 0, 255), 1);
		}
	}

	
	vector <float> estremi, local_estremi;
	
	// intervallo: inizio barcode - cross[0] 
	bool flag; // true se il primo è un massimo
	if ((scan_profile.at<uchar>(cross[0])) < threshold) { // cerchiamo un massimo
		int tp = 0;
		flag = true;
	}
	else flag = false;
	/*	for (int i = 0; i < cross[0]; i++) {
			if (scan_profile.at<uchar>(i) > tp) {
				tp = scan_profile.at<uchar>(i);
			}
		}
		estremi.push_back(tp);
	}
	else {	 // cerchiamo un minimo
		int tp = 255;
		flag = false;
		for (int i = 0; i < cross[0]; i++) {
			if (scan_profile.at<uchar>(i) < tp) {
				tp = scan_profile.at<uchar>(i);
			}
		}
		estremi.push_back(tp);
	}


	*/



	// intervallo: tra due cross
	for (int i = 0; i < cross.size() - 1; i++) {
		int t_max = 0, t_min = 255;
		for (int j = cross[i]; j < cross[i + 1]; j++) {
			if (flag) {
				if (scan_profile.at<uchar>(j) < t_min) {
					t_min = scan_profile.at<uchar>(j);
				}
			}
			else {
				if (scan_profile.at<uchar>(j) > t_max) {
					t_max = scan_profile.at<uchar>(j);
				}
			}
		}

		if (flag) {
			estremi.push_back(t_min);
			//cout << "t min " << t_min << endl;
		}
		else {
			estremi.push_back(t_max);
			//cout << "t max " << t_max << endl;
		}
		flag = !flag;
	}
	/*
	//ultimo intervallo: cross[ultimo] - fine barcode
	if (flag) {
		int tp = 0;
		for (int i = cross[cross.size() - 1]; i < working.cols; i++) {
			if (scan_profile.at<uchar>(i) > tp) {
				tp = scan_profile.at<uchar>(i);
				//cout << scan_profile.at<float>(i) << endl;
			}
		}
		estremi.push_back(tp);

	}
	else {	 // cerchiamo un minimo
		int tp = 255;
		for (int i = cross[cross.size() - 1]; i < working.cols; i++) {
			if (scan_profile.at<uchar>(i) < tp) {
				tp = scan_profile.at<uchar>(i);
			}
		}
		estremi.push_back(tp);

	}
	//cout << "estremi ultimo " << estremi[estremi.size() -1] << endl;
	*/
	


	float ECmin = 255;

	for (int i = 0; i < estremi.size() - 1; i++) {
		int x = abs(estremi[i] - estremi[i + 1]);
		cout << "differenza " << x << endl;
		if (x < ECmin) ECmin = x;

	}
	ECmin = ECmin / 255;

	cout << "Min Edge Contrast (ECmin): " << ECmin << endl;



	//MODULATION
	float modulation = ECmin / symbol_contrast;
	cout << "Modulation: " << modulation << endl;
	

	vector <float> result = { Rmax, Rmin, symbol_contrast, ECmin, modulation };
	imshow("working", working);

	int hist_w = scan_profile.cols; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / scan_profile.cols);
	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	for (int i = 1; i < scan_profile.cols; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(working.at<uchar>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(working.at<uchar>(i))),
			Scalar(255, 0, 0), 1, 1, 0);
	}

  
	imshow("Result", histImage);



	return result;
}
