#include "stdafx.h"
#include "opencv2\opencv.hpp"
#include "iostream"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include "functions.h"
#include <fstream>
#include <string>

// for the selection of the good lines among the set provided by Hough
#define delta 0.01
#define h 10

using namespace cv;
using namespace std;


//gap verticale da rivedere
//EAN128 - MASTER IMGB





int main(int argc, char** argv)
{

	string line;
	ifstream myfile("data/data.txt");
	if (myfile.is_open())
	{
		while (myfile.good())
		{
			getline(myfile, line);



			Mat src = imread("data/" + line + ".bmp", 1);
			//Mat src = imread("data/EAN128-MASTER IMGB.bmp", 1);

			cout << endl;
			cout << "IMAGE: " << line << endl;
			
			//imshow("src", src);

			
			// CLAHE DETECTION
			int use_clahe = clahe_detector(src);
			//cout << "use clahe " << use_clahe << endl;
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
			bool flag_gauss;
			tie(r_lines, angle_rotation) = barcode_orientation(dst, &flag_gauss);
			//cout << "angle rotation" << angle_rotation << endl;



			// ROTATION OF THE IMAGE
			Mat rotated_barcode = rotation_image(src, angle_rotation);
			Mat scan_image = rotated_barcode.clone();
			//imshow("rotated_barcode", rotated_barcode);

			//HOUGH TRANSFORM ON ROTATED IMAGE - STEP 2
			vector<Vec4i> r_lines2;
			Mat edges = rotated_barcode.clone();
			Mat blur;
			cvtColor(edges, edges, CV_RGB2GRAY);
			GaussianBlur(edges, edges, Size(3, 3), 0, 0, BORDER_DEFAULT);
			Canny(edges, edges, 50, 150, 3, true);
			//imshow("canny", edges);

			GaussianBlur(edges, blur, Size(3, 3), 0, 0, BORDER_DEFAULT);
			flag_gauss = false;
			tie(r_lines2, angle_rotation) = barcode_orientation(blur, &flag_gauss);

			if ((flag_gauss)) {
				//cout << "Hough Again Without GaussianBlur" << endl;
				tie(r_lines2, angle_rotation) = barcode_orientation(edges, &flag_gauss);
			}



			//CANCEL VERTICAL LINES AWAY FROM THE BARCODE
			vector <Vec4i> barcode_lines = gap(r_lines2, 80);
			vector<Vec4i> barcode_lines2 = vertical_gap(barcode_lines, rotated_barcode);
			vector <float> px = FirstLastDetector(barcode_lines2); //obtain initial and final bar


																  // BINARIZATION OF THE IMAGE
			Mat binary = rotated_barcode.clone();
			cvtColor(binary, binary, CV_RGB2GRAY);
			double th = threshold(binary, binary, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			//cout << " threshold OTSU: " << th << endl;


			//EXTRACTION THICKNESS SMALLEST BAR, BOUNDING BOX UPDATED!
			int X = counter_tickness_bars(binary, px);
			cout << "Size Smaller Bar X: " << X << endl;


			// BOUNDING BOX DRAWING
			//vector containing the four corners for the bounding box, in order: top_left, bottom_left, bottom_right, top_right
			vector <Point> points = { Point(px[0], px[2]), Point(px[0], px[3]), Point(px[1], px[3]), Point(px[1], px[2]) };
			drawing_box(binary, points);
			cvtColor(binary, binary, CV_GRAY2RGB);
			vector <Point> points_updated = { Point(px[0] - 10 * X, px[2] - X), Point(px[0] - 10 * X, px[3] + X), Point(px[1] + 10 * X, px[3] + X), Point(px[1] + 10 * X, px[2] - X) };
			drawing_box(binary, points_updated);
			//imshow("binarized image with bounding box", binary);



			//HARRIS vector<Point> Harris(Mat src, vector<Point> roi);
			cvtColor(rotated_barcode, rotated_barcode, CV_RGB2GRAY);
			int height;
			vector<float> y_coord = Harris(rotated_barcode, points_updated, &height);
			cvtColor(rotated_barcode, rotated_barcode, CV_GRAY2RGB);
			vector <Point> harris_points = { Point(px[0], y_coord[0]), Point(px[0], y_coord[1]), Point(px[1], y_coord[1]), Point(px[1], y_coord[0]) };
			drawing_box(rotated_barcode, harris_points);

			//scan_images_average(scan_image, harris_points);

			resize(rotated_barcode, rotated_barcode, Size(rotated_barcode.cols / 1.5, rotated_barcode.rows / 1.5));
			namedWindow("BOUNDING BOX FINAL", CV_WINDOW_AUTOSIZE);
			imshow("BOUNDING BOX FINAL", rotated_barcode);

			waitKey(-30);

		}
		myfile.close();

	}
	else cout << "Unable to open file";
	

return 0;
}
