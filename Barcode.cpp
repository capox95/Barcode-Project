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



			line = "UPC#01" ;
			Mat src = imread("data/" + line + ".bmp", 1);
			Mat scan_image = src.clone();
			cout << endl;
			cout << "IMAGE: " << line << endl;
			imshow("original image", src);


			/* NOISE ANALYSIS
			Mat noise(src.size(), src.type());
			randn(noise, 0, 50);
			src += noise;
			*/
			

			// CLAHE DETECTION
			int use_clahe = clahe_detector(src);
			if (use_clahe) src = clahe(src);
			

			// EDGE DETECTION
			Mat dst = src.clone();
			Mat	cdst, dst2;
			cvtColor(dst, dst, CV_RGB2GRAY);
			GaussianBlur(dst, dst, Size(3, 3), 0, 0, BORDER_DEFAULT);
			Canny(dst, dst, 50, 180, 3, true);
			GaussianBlur(dst, dst, Size(3, 3), 0, 0, BORDER_DEFAULT);


			// BARCODE ORIENTATION VIA HOUGH TRANSFORMS
			vector<Vec4i> r_lines;
			float angle_rotation;
			bool flag_gauss;
			tie(r_lines, angle_rotation) = barcode_orientation(dst, &flag_gauss);
			barcode.orientation = angle_rotation;

			
			// ROTATION OF THE IMAGE
			Mat rotated_barcode = rotation_image(src, angle_rotation);
			scan_image = rotation_image(scan_image, angle_rotation);
			cvtColor(scan_image, scan_image, CV_RGB2GRAY); // image used for the calculation of the quality parameters


			//HOUGH TRANSFORM ON ROTATED IMAGE - STEP 2
			vector<Vec4i> r_lines2;
			Mat edges = rotated_barcode.clone();
			Mat blur;
			cvtColor(edges, edges, CV_RGB2GRAY);
			GaussianBlur(edges, edges, Size(3, 3), 0, 0, BORDER_DEFAULT);
			Canny(edges, edges, 50, 150, 3, true);
			GaussianBlur(edges, blur, Size(3, 3), 0, 0, BORDER_DEFAULT);
			flag_gauss = false;
			tie(r_lines2, angle_rotation) = barcode_orientation(blur, &flag_gauss);

			//check flag for the gaussian filter action between canny and hough lines
			if ((flag_gauss)) {
				tie(r_lines2, angle_rotation) = barcode_orientation(edges, &flag_gauss); // Hough again without GaussianBlur
			}



			//CANCEL LINES AWAY FROM THE BARCODE
			vector <Vec4i> barcode_lines = gap(r_lines2, 80); //remove horizontal gaps with a threshold of 80pixels value
			vector<Vec4i> barcode_lines2 = vertical_gap(barcode_lines, rotated_barcode); //remove vertical gaps according to the "probability mass function method"
			vector <float> px = FirstLastDetector(barcode_lines2); //obtain initial and final bar for the first approximated bounding box


			// BINARIZATION OF THE IMAGE
			Mat binary = rotated_barcode.clone();
			cvtColor(binary, binary, CV_RGB2GRAY);
			double th = threshold(binary, binary, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);


			//EXTRACTION THICKNESS SMALLEST BAR, 
			int X = counter_thickness_bars(binary, px);

			// BOUNDING BOX DRAWING
			//vector containing the four corners for the bounding box, in order: top_left, bottom_left, bottom_right, top_right
			vector <Point> points = { Point(px[0], px[2]), Point(px[0], px[3]), Point(px[1], px[3]), Point(px[1], px[2]) }; 
			vector <Point> points_updated = { Point(px[0] - 10 * X, px[2] - X), Point(px[0] - 10 * X, px[3] + X), Point(px[1] + 10 * X, px[3] + X), Point(px[1] + 10 * X, px[2] - X) }; //vertexes updated according to the above calculated thickness


			// FUNCTION TO REMOVE THE BROKEN LINES
			vector <int> x_bb = broken_lines_removal(binary, points_updated, barcode_lines2);


			//DRAWING OF THE BOUNDING BOX ON THE BINARIZED IMAGE
			drawing_box(binary, points);
			cvtColor(binary, binary, CV_GRAY2RGB);
			drawing_box(binary, points_updated);
			imshow("binarized image with bounding box", binary);


			//HARRIS CORNER DETECTOR TO OBTAIN THE MINIMUM HEIGHT OF THE BARCODE
			cvtColor(rotated_barcode, rotated_barcode, CV_RGB2GRAY);
			vector<int> y_coord = Harris(rotated_barcode, points_updated); //Y_COORD contains the two value of the y coordinate (min and max)
			vector <Point> harris_points = { Point(x_bb[0] - 10 * X, y_coord[0] - X), Point(x_bb[0] - 10 * X, y_coord[1] + X), Point(x_bb[1] + 10 * X, y_coord[1] + X), Point(x_bb[1] + 10 * X, y_coord[0] - X) }; // updated bounding box with the correct height


			//DRAWING OF THE FINAL BOUNDING BOX
			cvtColor(rotated_barcode, rotated_barcode, CV_GRAY2RGB);
			drawing_box(rotated_barcode, harris_points);
			resize(rotated_barcode, rotated_barcode, Size(rotated_barcode.cols / 1.5, rotated_barcode.rows / 1.5));
			namedWindow("BOUNDING BOX FINAL", CV_WINDOW_AUTOSIZE);
			imshow("BOUNDING BOX FINAL", rotated_barcode);
			


			
			//////////////////////////////////////////////////////////////////////
			//					QUALITY PARAMETERS CALCULATION					//	
			//////////////////////////////////////////////////////////////////////
			
			
			vector <int> grade, counter_edges;
			vector< vector <int> > crosses;
			vector< vector <float> > parameters;
			vector <bool> flag_black;


			for (int i = 0; i < 10; i++) {
				flag_black.push_back(0);
			}


			//PARAMETERS CALCULATION
			parameters = scan_images_average(scan_image, harris_points, grade, crosses, flag_black);
			string barcode_grade = overall_grade(grade);
			vector < vector <float> > sequences = sequence_spaces_bars(crosses, X, flag_black);
			counter_edges = number_edges(crosses);

			


			//DATA STRUCTURE BARCODE UPDATE

			int height = y_coord[1] - y_coord[0];
			Point center;
			center.x = (x_bb[1] - x_bb[0]) / 2 + x_bb[0];
			center.y = (height / 2) + y_coord[0];


			
			barcode.name = line;
			barcode.grade = barcode_grade;
			barcode.parameters = parameters;
			barcode.x_dimension = X;
			barcode.height = height;
			barcode.center = center;
			barcode.bounding_box = harris_points;
			barcode.number_edges = counter_edges;
			barcode.sequence = sequences;



			//WRITE FUNCTION TO OBTAIN THE FINAL TXT FILE
			//writeFile(barcode);


			waitKey(-30);

		}
		myfile.close();

	}
	else cout << "Unable to open file";


	return 0;
}


