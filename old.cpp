/*
#include "stdafx.h"
#include "opencv2\opencv.hpp"
#include "iostream"
#define kernel_dimension 3
#define threshold_value 100
using namespace cv;
using namespace std;
int main()
{
Mat img = imread("EAN-UPC-DECODABILITY IMGB.bmp", CV_LOAD_IMAGE_GRAYSCALE);
//Mat img = imread("barcode_01.jpg", CV_LOAD_IMAGE_GRAYSCALE);
Mat grad_x, grad_y, grad;
Mat abs_grad_x, abs_grad_y;
Mat diff_im, dst;
int ddepth = CV_16S;
int scale = 1;
int delta = 0;
//GaussianBlur(img, img, Size(11, 11), 1.5, 1.5, BORDER_DEFAULT);
//imshow("preprocessing gaussian blur", img);
//gradient on x
Sobel(img, grad_x, ddepth, 1, 0, kernel_dimension, scale, delta, BORDER_DEFAULT);
convertScaleAbs(grad_x, abs_grad_x);
//gradient y
Sobel(img, grad_y, ddepth, 0, 1, kernel_dimension, scale, delta, BORDER_DEFAULT);
convertScaleAbs(grad_y, abs_grad_y);
//total gradient
//imshow("gradient on x", abs_grad_x);
//imshow("gradient on y", abs_grad_y);
//difference of gradients
diff_im = abs_grad_x - abs_grad_y;
//imshow("difference of images", diff_im);
//weighted sum
//addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
//imshow("weighted sum", grad);
GaussianBlur(diff_im, diff_im, Size(19, 19), 3, 3, BORDER_DEFAULT);
//imshow("after blur", diff_im);
threshold(diff_im, dst, threshold_value, 255, 1);
imshow("after threshold", dst);
*/
/*
// Create a structuring element (SE)
int morph_size = 3;
Mat element = getStructuringElement(MORPH_RECT, Size(9, 5), Point(morph_size, morph_size));
cout << element;
Mat destination; // result matrix
// Apply the specified morphology operation
for (int i = 1; i<10; i++)
{
morphologyEx(dst, destination, MORPH_TOPHAT, element, Point(-1, -1), i);
//morphologyEx( src, dst, MORPH_TOPHAT, element ); // here iteration=1
imshow("result", destination);
waitKey(1000);
}
*/
/*
waitKey(0);
return 0;
}
*/
#include "stdafx.h"
#include "opencv2\opencv.hpp"
#include "iostream"
#include <stdlib.h>
#include <stdio.h>
#include <vector>

#define min_range 0.05
#define max_range 0.05

using namespace cv;
using namespace std;

void help()
{
	cout << "\nThis program demonstrates line finding with the Hough transform.\n"
		"Usage:\n"
		"./houghlines <image_name>, Default is pic1.jpg\n" << endl;
}

int main(int argc, char** argv)
{

	Mat src = imread("EAN-UPC-DECODABILITY IMGB.bmp", 0);

	Mat dst, cdst, dst2;
	Canny(src, dst, 220, 255, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);
	GaussianBlur(dst, dst, Size(3, 3), 0, 0, BORDER_DEFAULT);
	//imshow("after blur", dst);
	float media = 0;
	int count = 0;
	float sum = 0;
	float median = 0, angle_rotation = 0;
	/*
	HOUGH LINES WITH THETA
	vector<Vec2f> lines;
	HoughLines(dst, lines, 1, CV_PI / 180, 380, 0, 0);
	for (size_t i = 0; i < lines.size(); i++)
	{
	float rho = lines[i][0], theta = lines[i][1];
	Point pt1, pt2;
	double a = cos(theta), b = sin(theta);
	double x0 = a*rho, y0 = b*rho;
	pt1.x = cvRound(x0 + 1000 * (-b));
	pt1.y = cvRound(y0 + 1000 * (a));
	pt2.x = cvRound(x0 - 1000 * (-b));
	pt2.y = cvRound(y0 - 1000 * (a));
	line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
	}
	*/
	vector<Vec4i> lines;
	vector<Vec4i> r_lines;
	vector<float> theta, theta_backup;
	HoughLinesP(dst, lines, 1, CV_PI / 180, 50, 200, 10);
	//disegno delle righe 
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		theta.push_back(abs(atan2((l[3] - l[1]), (l[2] - l[0]))));
		//line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, CV_AA);
		sum = sum + abs(theta[i]);
	}
	theta_backup = theta;
	sort(theta.begin(), theta.end());
	cout << theta.size() << endl;
	median = theta[theta.size() / 2];
	media = sum / count;
	//ciclo dopo aver calcolato la mediana per selezionare le righe giuste
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		if ((theta_backup[i] < median + max_range) && (theta_backup[i] > (median - min_range)))
		{
			cout << theta_backup[i] << endl;
			r_lines.push_back(Vec4i(l[0], l[1], l[2], l[3]));
			line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, CV_AA);
		}
	}
	//rotation of the image
	angle_rotation = 1.57 - median;
	Point2f src_center(src.cols / 2.0F, src.rows / 2.0F);
	Mat rot_mat = getRotationMatrix2D(src_center, angle_rotation, 1.0);
	Mat destination;
	warpAffine(src, destination, rot_mat, src.size());
	//draw of the lines in the new image
	for (size_t i = 0; i < r_lines.size(); i++) {
		line(destination, Point(r_lines[i][0], r_lines[i][1]), Point(r_lines[i][2], r_lines[i][3]), Scalar(255, 0, 255), 1, CV_AA);
	}
	imshow("rotated image", destination);
	cout << r_lines[100] << endl;
	//cout << median << endl;
	//cout << media << endl;
	//imshow("source", src);
	imshow("detected lines", cdst);

	waitKey();

	return 0;
}