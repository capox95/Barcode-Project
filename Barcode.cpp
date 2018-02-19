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




barcode_result barcode;



int main(int argc, char** argv)
{


	string line;
	ifstream myfile("data/data.txt");
	if (myfile.is_open())
	{
		while (myfile.good())
		{
			getline(myfile, line);



			//line = "C39_4.4LOW" ;
			Mat src = imread("data/" + line + ".bmp", 1);
			Mat scan_image = src.clone();
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

			barcode.orientation = angle_rotation;
			//cout << "angle rotation" << angle_rotation << endl;



			// ROTATION OF THE IMAGE
			Mat rotated_barcode = rotation_image(src, angle_rotation);
			scan_image = rotation_image(scan_image, angle_rotation);
			cvtColor(scan_image, scan_image, CV_RGB2GRAY);

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



			//CANCEL LINES AWAY FROM THE BARCODE
			vector <Vec4i> barcode_lines = gap(r_lines2, 80);
			vector<Vec4i> barcode_lines2 = vertical_gap(barcode_lines, rotated_barcode);
			vector <float> px = FirstLastDetector(barcode_lines2); //obtain initial and final bar


																   // BINARIZATION OF THE IMAGE
			Mat binary = rotated_barcode.clone();
			cvtColor(binary, binary, CV_RGB2GRAY);
			double th = threshold(binary, binary, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			//cout << " threshold OTSU: " << th << endl;


			//EXTRACTION THICKNESS SMALLEST BAR, BOUNDING BOX UPDATED!
			int X = counter_thickness_bars(binary, px);
			//cout << "Size Smaller Bar X: " << X << endl;


			// BOUNDING BOX DRAWING
			//vector containing the four corners for the bounding box, in order: top_left, bottom_left, bottom_right, top_right
			vector <Point> points = { Point(px[0], px[2]), Point(px[0], px[3]), Point(px[1], px[3]), Point(px[1], px[2]) };
			vector <Point> points_updated = { Point(px[0] - 10 * X, px[2] - X), Point(px[0] - 10 * X, px[3] + X), Point(px[1] + 10 * X, px[3] + X), Point(px[1] + 10 * X, px[2] - X) };

			// FUNCTION TO REMOVE THE BROKEN LINES
			vector <int> x_bb = broken_lines_removal(binary, points_updated, barcode_lines2);


			drawing_box(binary, points);
			cvtColor(binary, binary, CV_GRAY2RGB);
			drawing_box(binary, points_updated);
			//imshow("binarized image with bounding box", binary);


			//HARRIS vector<Point> Harris(Mat src, vector<Point> roi);
			cvtColor(rotated_barcode, rotated_barcode, CV_RGB2GRAY);
			vector<int> y_coord = Harris(rotated_barcode, points_updated);


			cvtColor(rotated_barcode, rotated_barcode, CV_GRAY2RGB);
			vector <Point> harris_points = { Point(x_bb[0] - 10 * X, y_coord[0] - X), Point(x_bb[0] - 10 * X, y_coord[1] + X), Point(x_bb[1] + 10 * X, y_coord[1] + X), Point(x_bb[1] + 10 * X, y_coord[0] - X) };
			//vector <Point> harris_points = { Point(x_bb[0] -5* X, y_coord[0] - X), Point(x_bb[0] -5* X, y_coord[1] + X), Point(x_bb[1] + 5*X, y_coord[1] + X), Point(x_bb[1] + 5*X, y_coord[0] - X) };
			drawing_box(rotated_barcode, harris_points);
			int height = y_coord[1] - y_coord[0];
			Point center;
			center.x = (x_bb[1] - x_bb[0]) / 2 + x_bb[0];
			center.y = (height / 2) + y_coord[0];



			vector <int> grade, counter_edges;
			vector< vector <int> > crosses;
			vector< vector <float> > parameters;
			vector <bool> flag_black;

			for (int i = 0; i < 10; i++) {
				flag_black.push_back(0);
			}


			parameters = scan_images_average(scan_image, harris_points, grade, crosses, flag_black);
			string barcode_grade = overall_grade(grade);
			vector < vector <float> > sequences = sequence_spaces_bars(crosses, X, flag_black);

			counter_edges = number_edges(crosses);


			
			//cout << "Barcode Grade " << barcode_grade << endl;


			for (int i = 0; i < parameters.size(); i++) {
				for (int j = 0; j < parameters[i].size(); j++) {
					//cout << parameters[i][j] << endl;
				}
				//cout << "__________" << endl;
			}


			barcode.name = line;
			barcode.grade = barcode_grade;
			barcode.parameters = parameters;
			barcode.x_dimension = X;
			barcode.height = height;
			barcode.center = center;
			barcode.bounding_box = harris_points;
			barcode.number_edges = counter_edges;
			barcode.sequence = sequences;


			writeFile(barcode);
			//imshow("scan image", scan_image);

			//cout << "Image: " << line << endl;
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


