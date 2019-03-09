// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
#include <random>
#include <math.h>
#include<algorithm>
#include <time.h>
using namespace std;
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height);
void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = MAX_PATH - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

										   // the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
										  //VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}
void negative_image() {
	Mat img = imread("Images/cameraman.bmp",
		CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);
		}
	}
	imshow("negative image", img);
	waitKey(0);
}
void aditiveGrayScale() {
	Mat img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat dest = img.clone();
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) + 50 <= 255)
				dest.at<uchar>(i, j) = img.at<uchar>(i, j) + 50;
			else
				dest.at<uchar>(i, j) = 255;
		}
	}
	imshow("first image", img);
	imshow("modified image image", dest);
	waitKey(0);
}
void multiplicativeGrayScale() {
	Mat img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat dest = img.clone();
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<uchar>(i, j) * 2 <= 255)
				dest.at<uchar>(i, j) = img.at<uchar>(i, j) * 2;
			else
				dest.at<uchar>(i, j) = 255;
		}
	}
	imshow("original", img);
	imshow("negative image", dest);
	imwrite("D://Student//30237//BDan//OpenCVApplication-VS2013_OCV2413_basic (3)//OpenCVApplication-VS2013_OCV2413_basic//Images//new//newImage2.jpg", dest);
	waitKey(0);
}
void createImageColor() {

	Mat img(256, 256, CV_8UC3);//8 biti unsigned char 3 canale(RGB)

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			Vec3b BGR = img.at<Vec3b>(i, j);
			if (i < 128 && j < 128) {//[B][G][R]
				BGR[0] = 255;//alb
				BGR[1] = 255;//alb
				BGR[2] = 255;//alb
			}

			if (i < 128 && j >= 128) {
				BGR[0] = 0;//rosu
				BGR[1] = 0;//rosu
				BGR[2] = 255;//rosu
			}

			if (i >= 128 && j < 128) {
				BGR[0] = 0;//green
				BGR[1] = 255;//green
				BGR[2] = 0;//green
			}

			if (i >= 128 && j >= 128) {
				BGR[0] = 0;//yellow
				BGR[1] = 255;//yellow
				BGR[2] = 255;//yellow
			}

			img.at<Vec3b>(i, j) = BGR;
		}
	imshow("Img is :\n", img);
	waitKey(0);
}

float determinantCalcul2x2(Mat M)
{
	float a00 = M.at<float>(0, 0);
	float a01 = M.at<float>(0, 1);
	float a10 = M.at<float>(1, 0);
	float a11 = M.at<float>(1, 1);
	float rez = a00 * a11 - a10 * a01;
	return rez;

}
float determinantCalcul3x3(Mat M)
{
	float a00 = M.at<float>(0, 0);
	float a01 = M.at<float>(0, 1);
	float a02 = M.at<float>(0, 2);
	float a10 = M.at<float>(1, 0);
	float a11 = M.at<float>(1, 1);
	float a12 = M.at<float>(1, 2);
	float a20 = M.at<float>(2, 0);
	float a21 = M.at<float>(2, 1);
	float a22 = M.at<float>(2, 2);
	float rez = a00 * a11 * a22 + a10 * a21 * a02 + a01 * a12 * a20
		- a02 * a11 * a20 - a00 * a21 * a12 - a10 * a01 * a22;
	return rez;

}

void invertMatrix33() {
	float valueMatrix[9] = { 1, -1, 1, 2, 0, 3, 1, 1, -2 };
	Mat matrixInput(3, 3, CV_32FC1, valueMatrix); //constructor cu 4 parametri
	printf("Matrice:\n");
	std::cout << matrixInput << std::endl;
	Mat matrixAux = matrixInput.clone();
	float a11, a12, a13, a21, a22, a23, a31, a32, a33;
	float invDet = 1.00 / determinantCalcul3x3(matrixInput);

	a11 = matrixInput.at<float>(0, 0);
	a12 = matrixInput.at<float>(0, 1);
	a13 = matrixInput.at<float>(0, 2);
	a21 = matrixInput.at<float>(1, 0);
	a22 = matrixInput.at<float>(1, 1);
	a23 = matrixInput.at<float>(1, 2);
	a31 = matrixInput.at<float>(2, 0);
	a32 = matrixInput.at<float>(2, 1);
	a33 = matrixInput.at<float>(2, 2);

	float b11, b12, b13, b21, b22, b23, b31, b32, b33;


	float valueMatrix1[4] = { a22, a23, a32, a33 };
	Mat m1(2, 2, CV_32FC1, valueMatrix1);
	b11 = determinantCalcul2x2(m1);

	float valueMatrix2[4] = { a13, a12, a33, a32 };
	Mat m2(2, 2, CV_32FC1, valueMatrix2);
	b12 = determinantCalcul2x2(m2);

	float valueMatrix3[4] = { a12, a13, a22, a23 };
	Mat m3(2, 2, CV_32FC1, valueMatrix3);
	b13 = determinantCalcul2x2(m3);

	float valueMatrix4[4] = { a23, a21, a33, a31 };
	Mat m4(2, 2, CV_32FC1, valueMatrix4);
	b21 = determinantCalcul2x2(m4);

	float valueMatrix5[4] = { a11, a13, a31, a33 };
	Mat m5(2, 2, CV_32FC1, valueMatrix5);
	b22 = determinantCalcul2x2(m5);

	float valueMatrix6[4] = { a13, a11, a23, a21 };
	Mat m6(2, 2, CV_32FC1, valueMatrix6);
	b23 = determinantCalcul2x2(m6);

	float valueMatrix7[4] = { a21, a22, a31, a32 };
	Mat m7(2, 2, CV_32FC1, valueMatrix7);
	b31 = determinantCalcul2x2(m7);

	float valueMatrix8[4] = { a12, a11, a32, a31 };
	Mat m8(2, 2, CV_32FC1, valueMatrix8);
	b32 = determinantCalcul2x2(m8);

	float valueMatrix9[4] = { a11, a12, a21, a22 };
	Mat m9(2, 2, CV_32FC1, valueMatrix9);
	b33 = determinantCalcul2x2(m9);

	float valueMatrixAll[9] = { b11, b12, b13, b21, b22, b23,b31,b32,b33 };
	Mat mAll(3, 3, CV_32FC1, valueMatrixAll);
	float allDet = determinantCalcul2x2(mAll);


	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++)
		{
			mAll.at<float>(i, j) = mAll.at<float>(i, j) * invDet;
		}
	}
	printf("Matrice inversa:\n");
	std::cout << mAll << std::endl;


	//printf("a11: %f", a11);
	//float determinant = ;
	//waitKey(100000);
	system("pause");
}

void copyCanalsRGB3() {
	Mat imagine = imread("Images/flowers_24bits.bmp", CV_LOAD_IMAGE_COLOR);
	Mat blue, green, red;
	blue = Mat(imagine.rows, imagine.cols, CV_8UC1);
	green = Mat(imagine.rows, imagine.cols, CV_8UC1);
	red = Mat(imagine.rows, imagine.cols, CV_8UC1);

	for (int i = 0; i < imagine.rows; i++)
	{
		for (int j = 0; j < imagine.cols; j++)
		{
			Vec3b BGR = imagine.at<Vec3b>(i, j);//pixelul
			blue.at<uchar>(i, j) = BGR[0];
			green.at<uchar>(i, j) = BGR[1];
			red.at<uchar>(i, j) = BGR[2];
		}
	}
	imshow("Imagine originala", imagine);
	imshow("Blue Monocrom", blue);
	imshow("Green Monocrom", green);
	imshow("Red Monocrom ", red);
	waitKey(0);
	return;
}
void colorToGray() {
	Mat imagine = imread("Images/flowers_24bits.bmp", CV_LOAD_IMAGE_COLOR);
	Mat imagineFinala = Mat(imagine.rows, imagine.cols, CV_8UC1);

	for (int i = 0; i < imagine.rows; i++)
		for (int j = 0; j < imagine.cols; j++) {
			Vec3b BGR = imagine.at<Vec3b>(i, j);
			float blue = BGR[0];
			float green = BGR[1];
			float red = BGR[2];
			imagineFinala.at<uchar>(i, j) = (blue + green + red) / 3;
		}

	imshow("Imagine originala", imagine);
	imshow("Imagine Gray", imagineFinala);
	waitKey(0);
}
void grayToWhiteBlack() {
	Mat imagine = imread("Images/Lena_gray.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat imagineFinala = Mat(imagine.rows, imagine.cols, CV_8UC1);
	int prag = 255 / 2;//thresholding
	printf("\nIntroduceti pragul: ");
	scanf("%d", &prag);

	for (int i = 0; i < imagine.rows; i++)
		for (int j = 0; j < imagine.cols; j++) {
			if (imagine.at<uchar>(i, j) >= prag)
				imagineFinala.at<uchar>(i, j) = 0;//white
			else
				imagineFinala.at<uchar>(i, j) = 255;//black
		}

	imshow("Imagine originala", imagine);
	imshow("Alb-negru", imagineFinala);

	waitKey(0);
	return;
}
void RGB24toHSV() {
	Mat imagine = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);

	Mat imagineh = Mat(imagine.rows, imagine.cols, CV_8UC1);
	Mat imagines = Mat(imagine.rows, imagine.cols, CV_8UC1);
	Mat imaginev = Mat(imagine.rows, imagine.cols, CV_8UC1);

	float red, green, blue, M, m, C, H, S, V;

	for (int i = 0; i < imagine.rows; i++)
		for (int j = 0; j < imagine.cols; j++) {
			Vec3b BGR = imagine.at<Vec3b>(i, j);
			red = (float)BGR[2] / 255;//normalizat
			green = (float)BGR[1] / 255;
			blue = (float)BGR[0] / 255;

			M = max(red, green, blue);
			m = min(red, green, blue);
			C = M - m;
			//value – reprezinta înaltimea culorii curente in piramida/con 			
			V = M;

			//Saturation: reprezinta distanta culorii curente fata de centrul bazei piramidei/conului 			
			if (V != 0)
				S = C / V;
			else
				S = 0;
			//Hue: reprezinta unghiul facut de culoarea curenta cu raza corespunzatoare culorii Rosu 			
			if (C) {
				if (M == red)
					H = 60 * (green - blue) / C;
				if (M == green)
					H = 120 + 60 * (blue - red) / C;
				if (M == blue)
					H = 240 + 60 * (red - green) / C;
			}
			else // grayscale
				H = 0;

			if (H < 0)
				H = H + 360;

			H = H * 255 / 360;//Normalizare
			S = S * 255;//Normalizare
			V = V * 255;//Normalizare

			imagineh.at<uchar>(i, j) = (uchar)H;
			imagines.at<uchar>(i, j) = (uchar)S;
			imaginev.at<uchar>(i, j) = (uchar)V;
		}

	imshow("Imagine originala", imagine);
	imshow("H", imagineh);
	imshow("S", imagines);
	imshow("V", imaginev);
	waitKey(0);
}
int isInside(Mat imagine, int i, int j)
{
	int numberRows = imagine.rows;
	int numberColumn = imagine.cols;

	if (i >= 0 && j >= 0 && i <= numberRows && j <= numberColumn)
	{
		return 0;
	}
	return 1;
}

void testIsInside() {
	Mat imagine = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);

	int linie, coloana;

	printf("\nIntrodu numar linie:");
	scanf("%d", &linie);
	printf("\nIntrodu numar coloana:");
	scanf("%d", &coloana);
	int verif = isInside(imagine, linie, coloana);
	if (verif)
	{
		printf("\ne in afara imaginii\n");
	}
	else
	{
		printf("\ne in imagine\n");
	}
	system("pause");
}


void CalculProprietatiGeometrice(int event, int x, int y, int flags, void* param)
{

	///----------------------------- ARIE  + CENTRU MASA -----------------------------------------------------------
	Mat* imagine = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDBLCLK)
	{
		int R = (int)(*imagine).at<Vec3b>(y, x)[2];
		int G = (int)(*imagine).at<Vec3b>(y, x)[1];
		int B = (int)(*imagine).at<Vec3b>(y, x)[0];
		printf("\nPozitie(x,y): %d,%d  Culoare: R =  %d, G= %d, B = %d", x, y, R, G, B);

		Vec3b culoare = (*imagine).at<Vec3b>(y, x);
		int height = (*imagine).rows;
		int width = (*imagine).cols;
		int area = 0;
		int ri = 0, ci = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b val_culoare = (*imagine).at<Vec3b>(i, j);

				if (val_culoare == culoare)
				{
					area += 1;
					ri += i;
					ci += j;
				}

			}
		}
		printf("\nArie : %d", area);
		ri = ri / area;
		ci = ci / area;
		printf("\nCentrul de masa : (%d, %d)", ri, ci);



		//-----------------------UNGHI ALUNGIRE---------------------
		int numarator = 0;//numarator
		int sum = 0;
		int sum2 = 0;

		for (int i = 0; i < height; i++) //-- randuri
		{
			for (int j = 0; j < width; j++) //--coloane
			{
				Vec3b val_culoare2 = (*imagine).at<Vec3b>(i, j);

				if (val_culoare2 == culoare) {
					numarator += (i - ri) * (j - ci);
					sum += (j - ci) * (j - ci);
					sum2 += (i - ri) * (i - ri);
				}

			}
		}
		int numitor = sum - sum2;
		double tetaGrade = atan2(2 * numarator, numitor) / 2;
		if (tetaGrade < 0) {
			tetaGrade += CV_PI;
		}
		double tetaRad = tetaGrade;
		tetaGrade = tetaGrade * 180 / CV_PI; //transformare din radiani in grade
		printf("\nUnghiul de alungire in grade: %lf", tetaGrade);



		//----------------------PERIMETRU-----------------------------
		int numar_pixeli = 0;
		Mat contur = (*imagine).clone();

		for (int i = 1; i < height - 1; i++) //-- randuri
		{
			for (int j = 1; j < width - 1; j++) //--coloane
			{
				contur.at<Vec3b>(i, j).val[0] = 255;//ii fac pe toti albi la inceput
				contur.at<Vec3b>(i, j).val[1] = 255;
				contur.at<Vec3b>(i, j).val[2] = 255;
				Vec3b val_culoare3 = (*imagine).at<Vec3b>(i, j);

				Vec3b val1 = (*imagine).at<Vec3b>(i, j + 1);//dr
				Vec3b val2 = (*imagine).at<Vec3b>(i, j - 1);//st
				Vec3b val3 = (*imagine).at<Vec3b>(i + 1, j);//jos
				Vec3b val4 = (*imagine).at<Vec3b>(i - 1, j);//sus
				Vec3b val5 = (*imagine).at<Vec3b>(i + 1, j + 1);//jos dr
				Vec3b val6 = (*imagine).at<Vec3b>(i + 1, j - 1);//jos st
				Vec3b val7 = (*imagine).at<Vec3b>(i - 1, j - 1);//sus st
				Vec3b val8 = (*imagine).at<Vec3b>(i - 1, j + 1);//sus dr

				if (val_culoare3 == culoare)
				{
					if (val1 != culoare || val2 != culoare || val3 != culoare || val4 != culoare ||
						val5 != culoare || val6 != culoare || val7 != culoare || val8 != culoare)
					{
						numar_pixeli += 1;
						contur.at<Vec3b>(i, j).val[0] = 0;//negru
						contur.at<Vec3b>(i, j).val[1] = 0;
						contur.at<Vec3b>(i, j).val[2] = 0;
					}
				}
				else
				{
					contur.at<Vec3b>(i, j).val[0] = 255;//alb
					contur.at<Vec3b>(i, j).val[1] = 255;
					contur.at<Vec3b>(i, j).val[2] = 255;
				}



			}
		}
		double P = numar_pixeli * CV_PI / 4;
		printf("\nPerimetrul : %lf", P);
		imshow("Contur", contur);

		//--------------------  Factorul de subtiere ------------------------

		double subtiere = 4 * CV_PI * area / P / P;
		printf("\nFactorul de subtiere : %lf", subtiere);

		//-------------------------FACTOR DE ASPECT---ELONGATIA--------------------------
		int colMin = width;//coloana maxim
		int colMax = 0;
		int randMin = height;//rand maxim
		int randMax = 0;

		for (int i = 0; i < height; i++) //-- randuri
		{
			for (int j = 0; j < width; j++) //--coloane
			{
				Vec3b val = (*imagine).at<Vec3b>(i, j);

				if (val == culoare)
				{
					if (i < randMin)
						randMin = i;
					if (i > randMax)
						randMax = i;
					if (j < colMin)
						colMin = j;
					if (j > colMax)
						colMax = j;
				}

			}
		}
		double aspect = (double)(colMax - colMin + 1) / (double)(randMax - randMin + 1);
		printf("\nFactorul de aspect (Elongatia) : %lf", aspect);


		//--------------------------  AXA   ----------------------------
		double ra = ri + tan(tetaRad) * (colMin - ci);
		int ca = colMin;
		double rb = ri + tan(tetaRad) * (colMax - ci);
		int cb = colMax;

		Mat axa = (*imagine).clone();
		line(axa, Point(ca, (int)ra), Point(cb, (int)rb), Scalar(0, 0, 0));
		imshow("Axa", axa);


		//------------------------PROIECTIE---------------------------
		Mat proiectie = Mat(height, width, CV_8UC3);
		int contorJ = 0;
		for (int i = 0; i < height; i++) //-- randuri
		{

			contorJ = 0;
			for (int j = 0; j < width; j++) //--coloane
			{
				Vec3b val_culoare = (*imagine).at<Vec3b>(i, j);

				if (val_culoare == culoare) {//am gasit obj

					proiectie.at<Vec3b>(i, contorJ) = culoare;
					contorJ++;
				}

			}

		}

		imshow("Proiectie", proiectie);
		//--------------------------------------------------
	}

}
void proprietatiGeometrice() {

	Mat imagine;

	char fname[MAX_PATH];
	while (openFileDlg(fname)) // citeste imagine
	{
		imagine = imread(fname, CV_LOAD_IMAGE_COLOR);

		namedWindow("My Window", 1);//Creare imagine
		setMouseCallback("My Window", CalculProprietatiGeometrice, &imagine);
		imshow("My Window", imagine);

		waitKey(0);
	}
}

void displayMat(Mat_<uchar> &labels) {//functia ce se ocupa de afisare

	Mat_<Vec3b> colorLabels(labels.rows, labels.cols, CV_8UC3);
	default_random_engine gen;
	uniform_int_distribution<int> d(0, 255);

	Vec3b colors[256];

	colors[0] = (Vec3b(255, 255, 255));

	for (int i = 1; i < 256; i++) {
		colors[i] = Vec3b(d(gen), d(gen), d(gen));//generez culoare aleatoriu
	}

	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			colorLabels(i, j) = colors[labels(i, j)];//pun culoarea in imagine
		}
	}
	imshow("image", colorLabels);
	waitKey(0);

}

void labelBSF(int is4or8) {


	char fname[MAX_PATH];

	Mat_<uchar> img;
	while (openFileDlg(fname))
	{
		img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		uchar label = 0;
		Mat_<uchar> labels(img.rows, img.cols); //creez matricea
		labels.setTo(0); //umplu matricea cu val de 0
		queue<Point2i> Q; //declar o coada

		for (int i = 0; i < img.rows - 1; i++) {
			for (int j = 0; j < img.cols - 1; j++) {
				if (img(i, j) == 0 && labels(i, j) == 0) {
					label++;
					labels(i, j) = label;
					Q.push({ i, j });//pun perechea

					while (!Q.empty())  //daca coada nu e goala
					{
						Point2i q = Q.front();//salvez primul el si apoi il scot din coada
						Q.pop();
						if (is4or8 == 4)//vecinatate de tip 4
						{
							int di[4] = { -1, 0, 1, 0 };
							int dj[4] = { 0, -1, 0, 1 };
							uchar neighbors[4];
							for (int k = 0; k < 4; k++)
							{
								neighbors[k] = img.at<uchar>(i + di[k], j + dj[k]);
								if (img(q.x, q.y) == 0 && labels(q.x + di[k], q.y + dj[k]) == 0)//daca e neetichetat
								{
									labels(q.x + di[k], q.y + dj[k]) = label;//primeste eticheta noua
									Q.push({ q.x + di[k], q.y + dj[k] });
								}
							}
						}
						else if (is4or8 == 8)//vecinatate de tip 8
						{
							int di[8] = { -1, -1, -1,  0,  0,  1, 1, 1 };
							int dj[8] = { -1,  0,  1, -1,  1, -1, 0, 1 };
							uchar neighbors8[8];
							for (int k = 0; k < 8; k++)
							{
								neighbors8[k] = img(i + di[k], j + dj[k]);
								if (img(q.x, q.y) == 0 && labels(q.x + di[k], q.y + dj[k]) == 0) //daca e neetichetat
								{
									labels(q.x + di[k], q.y + dj[k]) = label; //primeste eticheta noua
									Q.push({ q.x + di[k], q.y + dj[k] });
								}

							}
						}

					}

				}
			}
		}
		displayMat(labels);//afisarea
						   //imshow("image", img);
	}

}

int minim_label(vector<int> a, int m)
{
	int low = a[0];
	for (int i = 0; i < m; ++i)
	{
		if (a[i] < low)
			low = a[i];
	}
	return low;
}

void TwoPassLabelling() {
	char fname[MAX_PATH];

	Mat_<uchar> img;
	while (openFileDlg(fname))
	{
		img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		uchar label = 0;
		Mat_<uchar> labels(img.rows, img.cols);
		labels.setTo(0);

		queue<int> Q;
		vector<vector<int>> edges;



		for (int i = 0; i < img.rows - 1; i++) //prima parcurgere cu etichetele initiale
		{
			for (int j = 0; j < img.cols - 1; j++)
			{
				if (img(i, j) == 0 && labels(i, j) == 0) //np anterior
				{
					vector<int> L;
					int di[4] = { -1, -1, -1, 0 };
					int dj[4] = { -1, 0, 1, -1 };


					for (int k = 0; k < 4; k++)
					{
						if (labels(i + di[k], j + dj[k]) > 0)  //daca avem vecini etichetati
						{
							L.push_back(labels(i + di[k], j + dj[k]));
						}
					}

					if (L.size() == 0)
					{
						label++; //incrementeaza etichetele
						labels(i, j) = label;  //se pune eticheta
					}
					else
					{
						edges.resize(label + 1);
						int x = minim_label(L, L.size()); //selectam minimul etichetelor
						labels(i, j) = x;
						for (int k = 0; k < L.size(); k++)
						{
							if (L[k] != x)
							{
								edges[x].push_back(L[k]);
								edges[L[k]].push_back(x);

							}
						}

					}
				}
			}
		}
		uchar newLabel = 0;


		vector<int> newLabels(label + 1, 0);//lungimea label1 , toate el cu valoarea 0

		for (int i = 0; i <= label; i++)
		{
			if (newLabels[i] == 0)
			{
				newLabel++;
				newLabels[i] = newLabel;
				Q.push(i);

				while (!Q.empty())
				{
					int  x = Q.front();
					Q.pop();

					for (int y = 0; y < edges[x].size(); y++)
					{
						if (newLabels[edges[x][y]] == 0)
						{
							newLabels[edges[x][y]] = newLabel;
							Q.push(edges[x][y]);
						}

					}
				}

			}
		}

		for (int i = 0; i < img.rows - 1; i++)
		{
			for (int j = 0; j < img.cols - 1; j++)
			{
				labels(i, j) = newLabels[labels(i, j)];
			}
		}
		displayMat(labels);
	}
}



typedef struct {
	int i, j;
	byte c;
} my_point;

void apel_contur() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);  //creem imaginea si o afisam
		imshow("img initiala", src);
		Mat dst = Mat::zeros(src.size(), CV_8UC1);  //creem o matrice pentru imaginea destinatie(finala)neagra

		vector <my_point> contur; //vector de points pentru contur

		bool found = false;  //variabila pt gasire
		int i0, j0;

		for (int i = 0; i < src.rows && !found; i++) {  //parcurgem liniile
			for (int j = 0; j < src.cols && !found; j++) {    //parcurgem coloanele
				if (src.at<uchar>(i, j) == 0) {    //daca gasim pixelii,ii atribuim si inseamna ca i-am gasit
					i0 = i;//SALVAM PCT RESPECTIV
					j0 = j;
					found = true;
					//src.at<uchar>(i0, j0) == 255;

				}
			}
		}

		int di[] = { 0, -1, -1, -1, 0, 1, 1, 1 };    //di pentru linii,dj pentru coloane
		int dj[] = { 1, 1, 0, -1, -1, -1, 0, 1 };    //avem nevoie de dj si di pentru a stii pe ce coloana si linie mergem



		contur.push_back(my_point{ i0, j0, 7 });//init  directia cu 7

		bool finished = false;

		int x = i0;  //folosim variabile pentru i0 si j0
		int y = j0;
		int dir = 7;  //dir = 7 : conturul este detectat folosind vecinatate de 8

		while (!finished) {
			if (dir % 2 == 0) {  //vedem daca directia este para
				dir = (dir + 7) % 8; // (b)(dir + 7) mod 8 dacã dir este par
			}
			else {
				dir = (dir + 6) % 8;  //(dir + 6) mod 8 daca dir este impar
			}

			while (src.at<uchar>(x + di[dir], y + dj[dir]) != 0) {   //parcurgem imaginea pixel cu pixel
				dir = dir + 1;                        //atata timp cat exista  si mergem mai departe,pana ajungem la 8
				if (dir == 8) {                    //adica parcurgem vecinatatea de 8
					dir = 0;
				}
			}

			contur.push_back(my_point{ x + di[dir], y + dj[dir], (byte)dir });

			x += di[dir];
			y += dj[dir];

			int n = contur.size();  //n ia dimensiunea vectorului de pixeli care alcatuiesc conturlui
			if (n > 2 && contur.at(0).i == contur.at(n - 2).i &&
				contur.at(0).j == contur.at(n - 2).j &&
				contur.at(1).i == contur.at(n - 1).i &&
				contur.at(1).j == contur.at(n - 1).j)
			{
				finished = true;  //Dacã elementul curent Pn al conturului este egal cu al doilea element P1 din contur
								  //si dacã elementul anterior Pn - 1 este egal cu primul element P0, atunci algoritmul se incheie.
			}
		}


		for (int i = 0; i < contur.size(); i++) {  //parcurgem pixelii din vectorul pt contur
			dst.at<uchar>(contur.at(i).i, contur.at(i).j) = 255;///dau alb pentru valoarea conturului
			int v = 7;   //folosim vecinatatea lui 8(de-aia avem 7)
			if (i > 0) {  //daca mai sunt pixeli
				v = contur.at(i).c - contur.at(i - 1).c;  //pixelului ii atribuim valoarea corespunzatoare
			}
			if (v < 0)  //daca valoarea sa este mai mica de 0
				v += 8;   //reincepem procesul,actualizam valoarea
			printf("index pixel:%i coord pixel: ( %i , %i ) codul: %i , derivata: %i \n", i, contur.at(i).i, contur.at(i).j, contur.at(i).c, v);  //afisam indexul pixelului,coordonatele acestuia intre() ,codul si derivata
		}                                   //printam conturul


		imshow("contur imag init", dst);  //afisam imaginea finala(conturul imaginii initiale)
		waitKey();
	}
}

void reconstruct() {
	FILE *f = fopen("reconstruct.txt", "r");  //deschidem fisierul pt citire
											  //printf("xa");
	int x, y, l, dir;

	fscanf(f, "%d %d", &x, &y);  //citim din fisier valoarea lui x,y,l x,y pozitia pixelului de start
	fscanf(f, "%d", &l);//nr de caractere din fisier

	Mat dst = Mat(1000, 1000, CV_LOAD_IMAGE_GRAYSCALE);  //creem o matrice pt imagine

	int dj[] = { 1, 1, 0, -1, -1, -1, 0, 1 };   //dj si di ca sa stim pe ce linie si coloana mergem
	int di[] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	for (int i = 0; i < l; i++) {
		dst.at<uchar>(x, y) = 0;
		fscanf(f, "%d", &dir);
		x = x + di[dir];  //determinam valoarea lui x si y
		y = y + dj[dir];

	}
	imshow("recon", dst);
	waitKey();
	fclose(f);
}









//dilatare
//Daca originea elementului structural coincide cu un pixel obiect din imagine, atunci toti
//pixelii acoperiti de elementul structural devin pixeli obiect.
Mat dilatareImagine(Mat src) {
	static int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };//vecinatatile
	static int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };//vecinatatile


	Mat dst = src.clone();//clon4m imaginea sursa pentru a putea aploica operatii pe imaginea normala nu cea modificata

	for (int i = 1; i < src.rows - 1; i++)
		for (int j = 1; j < src.cols - 1; j++)
		{
			if (src.at<uchar>(i, j) == 0)//daca e pixel OBIECT => 
			{
				for (int k = 0; k < 7; k++)//coloram toti vecinii lui
					dst.at<uchar>(i + di[k], j + dj[k]) = 0;//cu NEGRU
			}
		}

	return dst;
}

void dilatareImagineTest() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat sursa = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat destinatie = Mat(sursa.size(), CV_8UC1);
		destinatie = dilatareImagine(sursa);
		imshow("Sursa: ", sursa);
		imshow("Destinatie: ", destinatie);
		waitKey();
	}
}
//.Daca originea elementului structural se suprapune peste un pixel obiect din imagine si
//exista cel putin un pixel obiect al elementului structural care se suprapune peste un pixel
//fundal din imagine, atunci pixelul curent din imagine va fi transformat in fundal.
Mat eroziuneImagine(Mat sursa) {
	static int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };//vecinatatile
	static int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };//vecinatatile



	Mat dst = sursa.clone();//clonam imaginea sursa pentru a putea aploica operatii pe imaginea normala nu cea modificata

	for (int i = 1; i < sursa.rows - 1; i++) //parcurgem imaginea
		for (int j = 1; j < sursa.cols - 1; j++)
		{
			if (sursa.at<uchar>(i, j) == 0) //daca e pixel OBIECT ( daca e negru)=> 
			{
				for (int k = 0; k < 7; k++)//coloram toti vecinii lui
				{
					if (sursa.at<uchar>(i + di[k], j + dj[k]) == 255)//daca are vreun vecin alb
						dst.at<uchar>(i, j) = 255;//atunci pixelul copiat in imag dest e alb
				}
			}
		}

	return dst;
}
void eroziuneImagineTest()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat sursa = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat destinatie = eroziuneImagine(sursa);
		imshow("Sursa: ", sursa);
		imshow("Destinatie: ", destinatie);
		waitKey();
	}
}

Mat dilatareImagineN(Mat sursa, int n) {
	Mat destinatie = Mat(sursa.size(), CV_8UC1);
	destinatie = sursa.clone();
	for (int iteration = 0; iteration < n; iteration++)
	{
		destinatie = dilatareImagine(destinatie);
	}
	return destinatie;
}

void dilatareImagineNTest() {
	char fname[MAX_PATH];
	int n;
	printf("De cate ori aplici dilatarea:\n");
	scanf("%d", &n);
	while (openFileDlg(fname))
	{

		Mat sursa = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat destinatie = dilatareImagineN(sursa, n);
		imshow("Sursa: ", sursa);
		imshow("Destinatie: ", destinatie);
		waitKey();
	}
}


Mat eroziuneImagineN(Mat sursa, int n) {
	Mat destinatie = Mat(sursa.size(), CV_8UC1);
	destinatie = sursa.clone();
	for (int iteration = 0; iteration < n; iteration++)
	{
		destinatie = eroziuneImagine(destinatie);
	}
	return destinatie;
}

void eroziuneImagineNTest() {
	char fname[MAX_PATH];
	int n;
	printf("De cate ori applici eroziune n\n");
	scanf("%d", &n);
	while (openFileDlg(fname))
	{

		Mat sursa = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat destinatie = eroziuneImagineN(sursa, n);
		imshow("Sursa: ", sursa);
		imshow("Destinatie: ", destinatie);
		waitKey();
	}
}

void deschidereImagine() { //EROZIUNE + DILATARE
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat sursa = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat aux = Mat(sursa.size(), CV_8UC1);
		Mat destinatie = Mat(sursa.size(), CV_8UC1);
		aux = eroziuneImagine(sursa); //aplic eroziunea
		destinatie = dilatareImagine(aux); //aplic dilatarea pe imag erodata
		imshow("Sursa: ", sursa);
		//imshow("Aux: ", aux);
		imshow("Destinatie: ", destinatie);
		waitKey();
	}
}

void inchidereImagine() {  //DILATARE + EROZIUNE
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat sursa = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat aux = Mat(sursa.size(), CV_8UC1);
		Mat destinatie = Mat(sursa.size(), CV_8UC1);
		aux = dilatareImagine(sursa); //aplic dilatarea pe imaginea originala
		destinatie = eroziuneImagine(aux); //aplic eroziunea pe imaginea dilatata
		imshow("Sursa: ", sursa);
		//imshow("Aux: ", aux);
		imshow("Destinatie: ", destinatie);
		waitKey();
	}
}

void conturImagine() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat sursa = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dilatare = Mat(sursa.size(), CV_8UC1);
		Mat destinatie = Mat(sursa.size(), CV_8UC1);
		dilatare = dilatareImagine(sursa);// cu dilatare
		for (int i = 0; i < sursa.rows; i++) {
			for (int j = 0; j < sursa.cols; j++) {
				if (sursa.at<uchar>(i, j) != 0 && dilatare.at<uchar>(i, j) == 0) {//daca pixel sursa e alb  si  pixel dilatat e negru
					destinatie.at<uchar>(i, j) = 0;
				}
				else {
					destinatie.at<uchar>(i, j) = 255;
				}
			}
		}
		imshow("Sursa: ", sursa);
		imshow("Destinatie: ", destinatie);
		waitKey();
	}
}











void computeHistogram(Mat img)
{
	int histogramValues[256];

	for (int i = 0; i < 256; i++)
		histogramValues[i] = 0;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			histogramValues[img.at<uchar>(i, j)]++;

	showHistogram("Histogram", histogramValues, 255, 200);
	waitKey(0);
}

void media() {
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Imaginea initiala", img);


	//---------------------------media---------------niv de intensitate
	float average = 0;
	int M = img.rows * img.cols;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			average += img.at<uchar>(i, j);

	average /= M;

	std::cout << "Media este: " << average << std::endl;

	//---------------------------------deviatia----------masura a contrastului imaginii, contrast radicat=deviatie mare
	float intensity = average;
	float deviation = 0, x;

	for (int i = 1; i < img.rows; i++)
		for (int j = 1; j < img.cols; j++)
		{
			x = pow(img.at<uchar>(i, j) - intensity, 2);
			deviation += x;
		}

	deviation /= M;
	deviation = sqrt(deviation);

	std::cout << "Deviatia standard este: " << deviation;

	computeHistogram(img);
	//computeHistogram(img);
}

void binarizare_automata() {

	Mat src;
	char fname[MAX_PATH];
	int iMin = 0, iMax = 0;
	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = src.clone();

		int h[256] = { 0 };
		int M = src.rows *  src.cols;

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				uchar g = src.at<uchar>(i, j);
				h[g] ++;
			}
		}

		for (int g = 0; g < 255; g++) {
			if (h[g] != 0) {
				iMin = g; //val intensitate minima
				break;
			}
		}

		for (int g = 255; g > 0; g--) {
			if (h[g] != 0) {
				iMax = g; //val intensitate maxima
				break;
			}
		}

		float T1 = 0.0f, T2 = 0.0f;
		float medieG1 = 0.0f, medieG2 = 0.0f;
		int n1 = 0, n2 = 0;
		int first = 0;


		T1 = (float)(iMin + iMax) / 2;
		T2 = T1 + 1.0f;

		while (T2 - T1 > 0.1) {
			if (first == 0)
				first++;
			else T1 = T2;

			n1 = n2 = 0;

			for (int f = iMin; f <= (int)T1; f++)//calculez media g1
			{
				medieG1 += f * h[f];
				n1 += h[f];
			}

			medieG1 = ((float)1 / n1) * medieG1;

			for (int f = (int)T1 + 1; f <= iMax; f++)//calculez media g2
			{
				medieG2 += f * h[f];
				n2 += h[f];
			}
			medieG2 = ((float)1 / n2) * medieG2;

			T2 = (float)(medieG1 + medieG2) / 2;
		}

		printf("prag = %f\n", T2);
		int T = (int)T2;

		for (int i = 0; i < src.rows; i++)//binarizare folosing pragul T
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (src.at<uchar>(i, j) <= T)
					dst.at<uchar>(i, j) = 0;
				else dst.at<uchar>(i, j) = 255;
			}
		}

		imshow("Sursa", src);
		imshow("Destinatie", dst);

		waitKey(0);
	}

}
void negativul_imaginii()
{
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Imaginea originala", img);

	int histogramValues[256];

	for (int i = 0; i < 256; i++)
		histogramValues[i] = 0;//fac histograma 0 pt a nu avea valori random din memorie

	for (int i = 0; i < img.rows; i++)//calculez histograma
		for (int j = 0; j < img.cols; j++)
			histogramValues[img.at<uchar>(i, j)]++;

	showHistogram("Histograma imaginii initiale", histogramValues, 255, 200);

	Mat rez(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			rez.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);//calculez negativul

	imshow("Imaginea negativa", rez);
	/////----------afisez histograma
	int newHistogramValues[256];

	for (int i = 0; i < 256; i++)//fac histograma 0 ca sa nu aiba valori din memorie
		newHistogramValues[i] = 0;

	for (int i = 0; i < rez.rows; i++)//calculez noua histograma
		for (int j = 0; j < rez.cols; j++)
			newHistogramValues[rez.at<uchar>(i, j)]++;

	showHistogram("Histograma negativului imaginii", newHistogramValues, 255, 200);
	waitKey(0);
}


void modificareContrast()
{
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Imaginea initiala", img);

	int histogramValues[256];

	for (int i = 0; i < 256; i++)
		histogramValues[i] = 0;//fac histograma 0 pt ca sa nu avem zgomot in imagine

							   //calcuylez histograma
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			histogramValues[img.at<uchar>(i, j)]++;

	showHistogram("Histograma imaginii initiale", histogramValues, 255, 200);
	//------------------
	int gOutMin, gOutMax, gInMin = 256, gInMax = 0;

	std::cout << "Introduceti valoarea minima: ";
	std::cin >> gOutMin;
	std::cout << "Introduceti valoarea maxima: ";
	std::cin >> gOutMax;

	for (int i = 0; i < img.rows; i++)//calculez maximul
		for (int j = 0; j < img.cols; j++)
			if (img.at<uchar>(i, j) > gInMax)
				gInMax = img.at<uchar>(i, j);

	for (int i = 0; i < img.rows; i++)//calculez minimul
		for (int j = 0; j < img.cols; j++)
			if (img.at<uchar>(i, j) < gInMin)
				gInMin = img.at<uchar>(i, j);

	int x = (gOutMax - gOutMin) / (gInMax - gInMin);//latire sau ingustarea

	if (x > 1)
		std::cout << "Latire";
	if (x < 1)
		std::cout << "Ingustare";

	Mat rez(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			rez.at<uchar>(i, j) = gOutMin + (img.at<uchar>(i, j) - gInMin)*x;

			if (rez.at<uchar>(i, j) < 0)
				rez.at<uchar>(i, j) = 0;

			if (rez.at<uchar>(i, j) > 255)
				rez.at<uchar>(i, j) = 255;
		}

	imshow("Imaginea dupa modificarea contrastului", rez);
	//--------------calculez histograma noua
	int newHistogramValues[256];

	for (int i = 0; i < 256; i++)
		newHistogramValues[i] = 0;

	for (int i = 0; i < rez.rows; i++)
		for (int j = 0; j < rez.cols; j++)
			newHistogramValues[rez.at<uchar>(i, j)]++;

	showHistogram("Hisrograma dupa modificarea contrastului", newHistogramValues, 255, 200);
	waitKey(0);
}


void corectieGamma()
{
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Imagine initiala", img);

	int histogramValues[256];

	for (int i = 0; i < 256; i++)
		histogramValues[i] = 0;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			histogramValues[img.at<uchar>(i, j)]++;

	showHistogram("Histograma imaginii initiale", histogramValues, 255, 200);

	Mat rez(img.rows, img.cols, CV_8UC1);

	float gamma;
	std::cout << "Introduceti factorul de corectie: ";
	std::cin >> gamma;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			rez.at<uchar>(i, j) = (uchar)(255 * pow(img.at<uchar>(i, j) / (float)255, gamma));//aplic formula

			if (rez.at<uchar>(i, j) < 0)
				rez.at<uchar>(i, j) = 0;

			if (rez.at<uchar>(i, j) > 255)
				rez.at<uchar>(i, j) = 255;
		}
	//--------------calculez noua histograma
	int newHistogramValues[256];

	for (int i = 0; i < 256; i++)
		newHistogramValues[i] = 0;

	for (int i = 0; i < rez.rows; i++)
		for (int j = 0; j < rez.cols; j++)
			newHistogramValues[rez.at<uchar>(i, j)]++;

	showHistogram("Histograma dupa aplicarea corectiei Gamma", newHistogramValues, 255, 200);
	imshow("rez", rez);
	waitKey(0);
}

void egalizare_histograma() {
	Mat src;
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = src.clone();

		int h[256] = { 0 };
		int M = src.rows * src.cols;
		//----calculam histograma
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				uchar g = src.at<uchar>(i, j);
				h[g]++;
			}
		}

		showHistogram("Sursa Histograma", h, 255, 200);
		//----------------------------
		int hc[256] = { 0 };

		hc[0] = h[0];
		for (int g = 1; g < 256; g++)
		{
			hc[g] = hc[g - 1] + h[g];
		}

		showHistogram("Histograma Cumulativa", hc, 255, 200);

		int tab[256] = { 0 };
		for (int g = 0; g < 256; g++)
		{
			tab[g] = 255 * hc[g] / M;
		}

		for (int i = 0; i < src.rows; i++)//calculam intensitatea pixelilor
		{
			for (int j = 0; j < src.cols; j++)
			{
				dst.at<uchar>(i, j) = tab[src.at<uchar>(i, j)];
			}
		}

		int hd[256] = { 0 };

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				uchar g = dst.at<uchar>(i, j);
				hd[g]++;
			}
		}

		showHistogram("Histograma Destinatie", hd, 255, 200);

		imshow("Sursa", src);
		imshow("Destinatie", dst);
		waitKey(0);
	}
}

void filtruGeneral(int w) {
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Imaginea initiala", img);

	Mat dst = img.clone();


	int H[3][3] = { { -1, -1, -1 },
	{ -33, 43, -1 },
	{ -1, 33, -1 } };

	int Splus = 0; //suma nr poz
	int Sminus = 0;//suma nr neg

	int k = w / 2;//ne folosim de rotungirea la int, formula -> w=2k+1
	double fs = 0;

	for (int j = 0; j < w; j++) {
		for (int i = 0; i < w; i++) {
			if (H[j][i] > 0) {//calculez suma tuturor elementelor pozitive
				Splus += H[j][i];
			}
			else if (H[j][i] < 0) {//calculez suma tuturor elementelor negative
				Sminus -= H[j][i];
			}
		}
	}

	fs = 1.0 / (2.0*max(Splus, Sminus));//aplicam formula

	for (int x = k; x < img.rows - k; x++) {
		for (int y = k; y < img.cols - k; y++) {
			int aux = 0;
			for (int i = -k; i <= k; i++) {
				for (int j = -k; j <= k; j++) {
					aux += img.at<uchar>(x + i, y + j)*H[i + k][j + k];
				}
				dst.at<uchar>(x, y) = aux * fs + 127;
			}
		}
	}

	imshow("Imagine rezultat", dst);
	waitKey(0);

}
void filtruTreceSusFrecvential(int w) {
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Imaginea initiala", img);

	Mat dst = img.clone();

	int H[3][3] =
	{ { 0, -1, 0 },
	{ -1, 5, -1 },
	{ 0, -1, 0 } };

	int Splus = 0;
	int Sminus = 0;

	int k = w / 2;
	double fs = 0;

	for (int j = -k; j <= k; j++) {
		for (int i = -k; i <= k; i++) {
			if (H[j + k][i + k] > 0) {
				Splus += H[j + k][i + k];
			}
			else if (H[j + k][i + k] < 0) {
				Sminus -= H[j + k][i + k];
			}
		}
	}

	fs = 1.0 / (2.0*max(Splus, Sminus));

	for (int y = k; y < img.rows - k; y++) {
		for (int x = k; x < img.cols - k; x++) {
			int aux = 0;
			for (int i = -k; i <= k; i++) {
				for (int j = -k; j <= k; j++) {
					aux += img.at<uchar>(x + j, y + i)*H[j + k][i + k];
				}
				dst.at<uchar>(x, y) = aux * fs + 127;
			}
		}
	}

	imshow("Rezultat", dst);
	waitKey(0);
}
void filtruMedieAritmetica(int w) {
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Imaginea initiala", img);

	Mat dst = img.clone();

	int H[3][3] =
	{ { 1, 1, 1 },
	{ 1, 1, 1 },
	{ 1, 1, 1 } };


	int k = (w - 1) / 2;

	int c = 0;
	for (int j = 0; j <= w - 1; j++) {
		for (int i = 0; i <= w - 1; i++) {
			c += H[j][i];
		}
	}

	for (int y = k; y < img.rows - k; y++) {
		for (int x = k; x < img.cols - k; x++) {
			float aux = 0;
			for (int i = -k; i <= k; i++) {
				for (int j = -k; j <= k; j++) {
					aux += img.at<uchar>(x + j, y + i)*H[j + k][i + k];
				}
				dst.at<uchar>(x, y) = aux / c;
			}
		}
	}

	imshow("Imagine Rezultat", dst);
	waitKey(0);


}

void filtruGausian(int w) {
	char fname[MAX_PATH];
	openFileDlg(fname);

	Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Imaginea initiala", img);

	Mat dst = img.clone();

	int H[3][3] = { { 1, 2, 1 },
	{ 2, 4, 2 },
	{ 1, 2, 1 } };


	int k = (w - 1) / 2;

	int c = 0;
	for (int j = 0; j <= w - 1; j++) {
		for (int i = 0; i <= w - 1; i++) {
			c += H[j][i];
		}
	}
	//c = (1.0f / 16.0f);

	for (int y = k; y < img.rows - k; y++) {
		for (int x = k; x < img.cols - k; x++) {
			float aux = 0;
			for (int i = -k; i <= k; i++) {
				for (int j = -k; j <= k; j++) {
					aux += img.at<uchar>(x + j, y + i)*H[j + k][i + k];
				}
				dst.at<uchar>(x, y) = aux / c;
			}
		}
	}

	imshow("Imagine Rezultat", dst);
	waitKey(0);


}



void filtruLaplace()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = src.clone();
		int dim = 3;
		int mat[3][3] = { { 0, -1, 0 },{ -1, 4, -1 },{ 0, -1, 0 } };//// { { 0, 1, 0 },{ 1, -4, 1 },{ 0, 1, 0 } };//{ { 0, -1, 0 },{ -1, 5, -1 },{ 0, -1, 0 } };

		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				int suma = 0;
				for (int k = -dim / 2; k <= dim / 2; k++) {
					for (int m = -dim / 2; m <= dim / 2; m++)
					{
						suma += src.at<uchar>(i + k, j + m)*mat[k + dim / 2][m + dim / 2];
					}

				}
				if (suma >= 255)
					dst.at<uchar>(i, j) = 255;
				else {
					if (suma < 0)
						dst.at<uchar>(i, j) = 0;
					else
						dst.at<uchar>(i, j) = suma;
				}
			}
		}

		imshow("Sursa", src);
		imshow("Destinatie", dst);
		waitKey();
	}
}
void filtruTS()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = src.clone();
		int dim = 3;
		int mat[3][3] = { { 0, -1, 0 },{ -1, 5, -1 },{ 0, -1, 0 } };

		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				int suma = 0;
				for (int k = -dim / 2; k <= dim / 2; k++) {
					for (int m = -dim / 2; m <= dim / 2; m++)
					{
						suma += src.at<uchar>(i + k, j + m)*mat[k + dim / 2][m + dim / 2];
					}

				}
				if (suma >= 255)
					dst.at<uchar>(i, j) = 255;
				else {
					if (suma < 0)
						dst.at<uchar>(i, j) = 0;
					else
						dst.at<uchar>(i, j) = suma;
				}
			}
		}

		imshow("Sursa", src);
		imshow("Destinatie", dst);
		waitKey();
	}
}

void centering_transform(Mat src) {
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			src.at<float>(i, j) = ((i + j) & 1) ? -src.at<float>(i, j) : src.at<float>(i, j);
		}
	}
}

Mat logaritmMagnitudine(Mat src) {
	Mat srcf;
	Mat dst;
	src.convertTo(srcf, CV_32FC1);
	centering_transform(srcf);

	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels);

	Mat mag, phi;
	magnitude(channels[0], channels[1], mag);
	//phase(channels[0], channels[1], phi);

	for (int i = 0; i < mag.rows; i++)
		for (int j = 0; j < mag.cols; j++)
		{
			mag.at<float>(i, j) = log(1 + mag.at<float>(i, j));//logaritmare
		}

	normalize(mag, dst, 0, 255, NORM_MINMAX, CV_8UC1);//normalizare
	return dst;
}

void Fourier()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = logaritmMagnitudine(src);

		imshow("Sursa", src);
		imshow("Destinatie", dst);
		waitKey();
	}
}
Mat trecejos(Mat src) {
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);
	centering_transform(srcf);

	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels);

	int h = fourier.rows;
	int w = fourier.cols;
	int R = 10;
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			if (((h / 2 - i)*(h / 2 - i) + (w / 2 - j)*(w / 2 - j)) > R*R)
			{
				channels[0].at<float>(i, j) = 0;
				channels[1].at<float>(i, j) = 0;
			}

	//aplicarea transformatei Fourier inversa ?i punerea rezultatului în dstf
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

	//transformarea de centrare inversa
	centering_transform(dstf);
	//normalizarea rezultatului în imaginea destina?ie
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}

void FITJ()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = trecejos(src);

		imshow("Sursa", src);
		imshow("Destinatie", dst);
		waitKey();
	}
}
Mat trecesus(Mat src) {
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);
	centering_transform(srcf);

	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels);

	int h = fourier.rows;
	int w = fourier.cols;
	int R = 10;
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			if (((h / 2 - i)*(h / 2 - i) + (w / 2 - j)*(w / 2 - j)) <= R * R)
			{
				channels[0].at<float>(i, j) = 0;
				channels[1].at<float>(i, j) = 0;
			}

	//aplicarea transformatei Fourier inversa ?i punerea rezultatului în dstf

	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

	//transformarea de centrare inversa

	centering_transform(dstf);
	//normalizarea rezultatului în imaginea destina?ie

	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}

void FITS()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = trecesus(src);

		imshow("Sursa", src);
		imshow("Destinatie", dst);
		waitKey();
	}
}
Mat trecejosGauss(Mat src) {
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);
	centering_transform(srcf);

	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels);

	int h = fourier.rows;
	int w = fourier.cols;
	float A = 10;
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
		{
			channels[0].at<float>(i, j) = channels[0].at<float>(i, j)*exp(-(((h / 2 - i)*(h / 2 - i) + (w / 2 - j)*(w / 2 - j)) / (A*A)));
			channels[1].at<float>(i, j) = channels[1].at<float>(i, j)*exp(-(((h / 2 - i)*(h / 2 - i) + (w / 2 - j)*(w / 2 - j)) / (A*A)));
		}

	//aplicarea transformatei Fourier inversa ?i punerea rezultatului în dstf

	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
	//transformarea de centrare inversa

	centering_transform(dstf);
	//normalizarea rezultatului în imaginea destina?ie

	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}

void FGTJ()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = trecejosGauss(src);

		imshow("Sursa", src);
		imshow("Destinatie", dst);
		waitKey();
	}
}


void filtruMedian(int k) {

	clock_t tStart = clock();
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = src.clone();

		int w = 2 * k + 1;

		int d = w / 2;

		vector<uchar> vec;

		for (int i = k; i < src.rows - k; i++) {
			for (int j = k; j < src.cols - k; j++) {

				vec.clear();

				for (int ll = -d; ll <= d; ll++) {
					for (int lc = -d; lc <= d; lc++) {
						vec.push_back(src.at<uchar>(i + ll, j + lc));
					}
				}

				sort(vec.begin(), vec.end());

				dst.at<uchar>(i, j) = vec[w*w / 2];
			}
		}
		printf("\nTime taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
		imshow("src", src);
		imshow("imagine", dst);
		waitKey(0);

	}

}

Mat filtruGauss(Mat img)
{
	Mat dest = Mat(img.rows, img.cols, CV_8UC1, 255);

	int w = 7;
	double t = (double)getTickCount(); // Gase?te timpul curent [ms]

	float G[7][7];
	float sigma = w / 6.0;
	float suma = 0;
	for (int x = 0; x < w; x++)
		for (int y = 0; y < w; y++)
			G[x][y] = 1 / (2 * PI*sigma*sigma) * exp(-(pow(x - w / 2, 2) + pow(y - w / 2, 2)) / (2 * sigma*sigma));

	int k = w / 2;

	for (int x = 0; x < w; x++)
		for (int y = 0; y < w; y++)
			suma += G[x][y];

	printf("suma %f\n", suma);

	for (int y = k; y < img.cols - k; y++)
		for (int x = k; x < img.rows - k; x++)
		{
			int aux = 0;
			for (int i = -k; i <= k; i++)
				for (int j = -k; j <= k; j++)
				{
					aux += img.at<uchar>(x + j, y + i) * G[j + k][i + k];
				}
			dest.at<uchar>(x, y) = aux;
		}

	t = ((double)getTickCount() - t) / getTickFrequency();
	// Afi?area la consola a timpului de procesare [ms]
	printf("Time = %.3f [ms]\n", t * 1000);

	imshow("Output", dest);
	return dest;
}
void gauss()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("src", img);

		filtruGauss(img);
		waitKey(0);
	}
}
/*
Mat gaussian(Mat src, int w){


double t = (double)getTickCount();


int height = src.rows;
int width = src.cols;
Mat dst = Mat(height, width, CV_8UC1);

for (int i = 0; i < height; i++){
for (int j = 0; j < width; j++){
dst.at<uchar>(i, j) = 0;
}
}

float filtru[7][7];
int x0 = w / 2;
float sigma = w / 6.0;
float coef = 1 / (2 * PI*sigma*sigma);
for (int x = 0; x < w; x++){
for (int y = 0; y < w; y++){
float power = -((x - x0)*(x - x0) + (y - x0)*(y - x0)) / (2 * sigma*sigma);
filtru[x][y] = coef*pow(2.71, power);
}

}

for (int i = w / 2; i < height - w / 2; i++){
for (int j = w / 2; j < width - w / 2; j++){
int sum = 0;
for (int k = 0; k < w; k++){
for (int l = 0; l < w; l++){
sum += filtru[k][l] * src.at<uchar>(i - (w / 2) + k, j - (w / 2) + l);

}
}
dst.at<uchar>(i, j) = sum;
}
}
t = (double)getTickCount() - t;

printf("Time = %.3f [ms]\n", t * 1000);

return dst;
}
*/
Mat filtru_gauss_vectorial(int w) {
	char fname[MAX_PATH];
	openFileDlg(fname);
	Mat_<uchar> img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> dst = Mat(img.rows, img.cols, CV_8UC1, Scalar(255));
	Mat_<uchar> temp = Mat(img.rows, img.cols, CV_8UC1, Scalar(255));
	float G[9];
	int x0, y0;
	x0 = w / 2;
	y0 = w / 2;
	int suma = 0;
	float sigma = ((float)w) / 6;
	for (int x = 0; x < w; x++) {
		G[x] = (exp(-((x - x0)*(x - x0)) / (2 * sigma*sigma))) / (sqrt(2 * PI)*sigma);
	}

	double t = (double)getTickCount();
	int d = w / 2;

	for (int i = d; i < img.rows - d; i++) {
		for (int j = d; j < img.cols - d; j++) {
			float val = 0;
			for (int m = -d; m <= d; m++) {

				val += img(i, j + m)*G[m + d];
			}

			temp(i, j) = val;
		}

	}

	for (int i = d; i < img.rows - d; i++) {
		for (int j = d; j < img.cols - d; j++) {
			float val = 0;
			for (int m = -d; m <= d; m++) {

				val += temp(i + m, j)*G[m + d];
			}

			dst(i, j) = val;
		}

	}



	t = ((double)getTickCount() - t) / getTickFrequency();
	printf("time = %3f [ms]\n", t * 1000);
	imshow("SRC", img);
	imshow("DST", dst);
	waitKey(0);

	return dst;

}

Mat dil_img_test(Mat src, int nr) {
	//Mat src = imread("Images/delfin1.jpg", CV_LOAD_IMAGE_COLOR);
	Mat destinatie = src.clone();
	for (int iteration = 0; iteration < nr; iteration++)
	{
		destinatie = dilatareImagine(destinatie);
	}
	//imshow("img", destinatie);
	//waitKey(0);
	return destinatie;
}
Mat conv(Mat src, Mat kernel) {

	Mat dst(src.rows, src.cols, CV_32FC1);

	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			float aux = 0;
			for (int k = 0; k < kernel.rows; k++)
			{
				for (int l = 0; l < kernel.cols; l++)
				{
					aux += kernel.at<float>(k, l) *(float)src.at<uchar>(i + k - 1, j + l - 1);
				}
			}
			dst.at<float>(i, j) = aux;
		}
	}
	return dst;

}
#define WEAK 128
#define STRONG 255
int adaptive_histograma(Mat img)
{
	int his[256];
	float p = 0.1;
	int k = 0.4;

	for (int i = 0; i < 256; i++) {
		his[i] = 0;
	}
	for (int i = 0; i < img.rows - 1; i++) {
		for (int j = 0; j < img.cols - 1; j++) {
			uchar pixel = img.at<uchar>(i, j);
			his[pixel]++;

		}
	}


	float NrNonMuchie = (1 - p) * (img.rows * img.cols - his[0]);

	int suma = 0;
	for (int i = 1; i < 255; i++) {
		suma += his[i];
		if (suma > NrNonMuchie) {
			return i;
		}
	}
	return 255;


}

void gradient_sobel() {


	float sobelY[9] =
	{ 1.0, 2.0, 1.0,
		0.0, 0.0, 0.0,
		-1.0, -2.0, -1.0 };


	float sobelX[9] =
	{ -1.0, 0.0, 1.0,
		-2.0, 0.0, 2.0,
		-1.0, 0.0, 1.0 };

	Mat kernelx = Mat(3, 3, CV_32FC1, sobelX);
	Mat kernely = Mat(3, 3, CV_32FC1, sobelY);


	Mat img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat dst = Mat(img.rows, img.cols, CV_8UC1);
	Mat tetaSobel = Mat(img.rows, img.cols, CV_8UC1);


	Mat dstx = conv(img, kernelx);
	Mat dsty = conv(img, kernely);

	std::queue<Point2i> queue;

	int dir = 0;

	for (int i = 1; i < img.rows - 1; i++)
	{
		for (int j = 1; j < img.cols - 1; j++)
		{
			float squareX = dstx.at<float>(i, j)* dstx.at<float>(i, j);
			float squareY = dsty.at<float>(i, j)* dsty.at<float>(i, j);

			dst.at<uchar>(i, j) = sqrt(squareX + squareY) / (4 * sqrt(2));

			float teta = atan2(dsty.at<float>(i, j), dstx.at<float>(i, j));

			if ((teta > 3 * PI / 8 && teta < 5 * PI / 8) || (teta > -5 * PI / 8 && teta < -3 * PI / 8)) dir = 0;
			if ((teta >   PI / 8 && teta < 3 * PI / 8) || (teta > -7 * PI / 8 && teta < -5 * PI / 8)) dir = 1;
			if ((teta >  -PI / 8 && teta <   PI / 8) || teta > 7 * PI / 8 && teta < -7 * PI / 8) dir = 2;
			if ((teta > 5 * PI / 8 && teta < 7 * PI / 8) || (teta > -3 * PI / 8 && teta <   -PI / 8)) dir = 3;
			tetaSobel.at<uchar>(i, j) = dir;

		}
	}


	imshow("original", img);
	imshow("raw", dst);
	//imshow("Directie", tetaSobel);

	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			switch (tetaSobel.at<uchar>(i, j)) {
			case 0:
				if ((dst.at<uchar>(i, j) < dst.at<uchar>(i - 1, j)) || (dst.at<uchar>(i, j) < dst.at<uchar>(i + 1, j)))
					dst.at<uchar>(i, j) = 0;
				break;
			case 1:
				if ((dst.at<uchar>(i, j) < dst.at<uchar>(i - 1, j + 1)) || (dst.at<uchar>(i, j) < dst.at<uchar>(i + 1, j - 1)))
					dst.at<uchar>(i, j) = 0;
				break;
			case 2:
				if ((dst.at<uchar>(i, j) < dst.at<uchar>(i, j - 1)) || (dst.at<uchar>(i, j) < dst.at<uchar>(i, j + 1)))
					dst.at<uchar>(i, j) = 0;
				break;
			case 3:
				if ((dst.at<uchar>(i, j) < dst.at<uchar>(i - 1, j - 1)) || (dst.at<uchar>(i, j) < dst.at<uchar>(i + 1, j + 1)))
					dst.at<uchar>(i, j) = 0;
				break;
			default:
				break;
			}
		}
	}



	imshow("Model dupa NMS", dst);

	int ph = adaptive_histograma(dst);
	//int ph = ph + 10;
	int pl = 0.4 * ph;

	printf("prag high: %d\n", ph);
	printf("prag low: %d\n", pl);

	//binarizarea cu histeza
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			if (dst.at<uchar>(i, j) < pl) {
				dst.at<uchar>(i, j) = 0;
			}
			if (dst.at<uchar>(i, j) > ph) {
				dst.at<uchar>(i, j) = STRONG;
			}
			if (dst.at<uchar>(i, j) > pl && dst.at<uchar>(i, j) < ph) {
				dst.at<uchar>(i, j) = WEAK;
			}
		}
	}

	imshow("Normalizare cu histereza", dst);

	Mat modul = dst.clone();

	Mat visited = Mat::zeros(dst.size(), CV_8UC1);
	queue = std::queue<Point>();

	int dx[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	int dy[] = { -1, 0, 1, -1, 1, -1, 0, 1 };

	for (int i = 1; i < dst.rows - 1; i++) {
		for (int j = 1; j < dst.cols - 1; j++) {
			if (dst.at<uchar>(i, j) == STRONG && visited.at<uchar>(i, j) == 0) {
				queue.push(Point(j, i));
				while (!queue.empty()) {
					Point oldest = queue.front();
					int jj = oldest.x;
					int ii = oldest.y;
					queue.pop();

					for (int n = 0; n < 8; n++) {
						if (dst.at<uchar>(ii + dx[n], jj + dy[n]) == WEAK) {
							dst.at<uchar>(ii + dx[n], jj + dy[n]) = STRONG;
							visited.at<uchar>(ii + dx[n], jj + dy[n]) = 1;


						}
					}
					visited.at<uchar>(i, j) = 1;
				}
			}


		}
	}

	for (int i = 1; i < dst.rows - 1; i++) {
		for (int j = 1; j < dst.cols - 1; j++) {
			if (dst.at<uchar>(i, j) == WEAK) {
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	//dupa extinderea muchiilor
	imshow("Final", dst);

	waitKey(0);
}
























void gradient_prewitt() {


	float prewittX[9] =
	{ -1.0, 0.0, 1.0,
		-1.0, 0.0, 1.0,
		-1.0, 0.0, 1.0 };


	float prewittY[9] =
	{ 1.0, 1.0, 1.0,
		0.0, 0.0, 0.0,
		-1.0, -1.0, -1.0 };

	Mat kernelx = Mat(3, 3, CV_32FC1, prewittX);
	Mat kernely = Mat(3, 3, CV_32FC1, prewittY);


	Mat img = imread("Images/balloons.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat dst = Mat(img.rows, img.cols, CV_8UC1);
	Mat tetaPrewitt = Mat(img.rows, img.cols, CV_8UC1);


	Mat dstx = conv(img, kernelx);
	Mat dsty = conv(img, kernely);

	int dir = 0;

	for (int i = 1; i < img.rows - 1; i++)
	{
		for (int j = 1; j < img.cols - 1; j++)
		{
			float squareX = dstx.at<float>(i, j)* dstx.at<float>(i, j);
			float squareY = dsty.at<float>(i, j)* dsty.at<float>(i, j);

			dst.at<uchar>(i, j) = sqrt(squareX + squareY) / (4 * sqrt(2));

			float teta = atan2(dsty.at<float>(i, j), dstx.at<float>(i, j));

			if ((teta > 3 * PI / 8 && teta < 5 * PI / 8) || (teta > -5 * PI / 8 && teta < -3 * PI / 8)) dir = 0;
			if ((teta >   PI / 8 && teta < 3 * PI / 8) || (teta > -7 * PI / 8 && teta < -5 * PI / 8)) dir = 1;
			if ((teta >  -PI / 8 && teta <   PI / 8) || teta > 7 * PI / 8 && teta < -7 * PI / 8) dir = 2;
			if ((teta > 5 * PI / 8 && teta < 7 * PI / 8) || (teta > -3 * PI / 8 && teta <   -PI / 8)) dir = 3;
			tetaPrewitt.at<uchar>(i, j) = dir;


		}
	}


	imshow("original", img);
	imshow("dest", dst);
	imshow("Directia", tetaPrewitt);

	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			switch (tetaPrewitt.at<uchar>(i, j)) {
			case 0:
				if ((dst.at<uchar>(i, j) < dst.at<uchar>(i - 1, j)) || (dst.at<uchar>(i, j) < dst.at<uchar>(i + 1, j)))
					dst.at<uchar>(i, j) = 0;
				break;
			case 1:
				if ((dst.at<uchar>(i, j) < dst.at<uchar>(i - 1, j + 1)) || (dst.at<uchar>(i, j) < dst.at<uchar>(i + 1, j - 1)))
					dst.at<uchar>(i, j) = 0;
				break;
			case 2:
				if ((dst.at<uchar>(i, j) < dst.at<uchar>(i, j - 1)) || (dst.at<uchar>(i, j) < dst.at<uchar>(i, j + 1)))
					dst.at<uchar>(i, j) = 0;
				break;
			case 3:
				if ((dst.at<uchar>(i, j) < dst.at<uchar>(i - 1, j - 1)) || (dst.at<uchar>(i, j) < dst.at<uchar>(i + 1, j + 1)))
					dst.at<uchar>(i, j) = 0;
				break;
			default:
				break;
			}
		}
	}


	imshow("Model dupa NMS", dst);

	waitKey(0);
}


void gradient_roberts() {


	float robertsX[4] =
	{ 1.0, 0.0,
		0.0, -1.0
	};


	float robertsY[4] =
	{ 0.0, -1.0,
		1.0, 0.0 };

	Mat kernelx = Mat(2, 2, CV_32FC1, robertsX);
	Mat kernely = Mat(2, 2, CV_32FC1, robertsY);


	Mat img = imread("Images/balloons.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat dst = Mat(img.rows, img.cols, CV_8UC1);
	Mat tetaRoberts = Mat(img.rows, img.cols, CV_8UC1);


	Mat dstx = conv(img, kernelx);
	Mat dsty = conv(img, kernely);

	int dir = 0;

	for (int i = 1; i < img.rows - 1; i++)
	{
		for (int j = 1; j < img.cols - 1; j++)
		{
			float squareX = dstx.at<float>(i, j)* dstx.at<float>(i, j);
			float squareY = dsty.at<float>(i, j)* dsty.at<float>(i, j);

			dst.at<uchar>(i, j) = sqrt(squareX + squareY) / (4 * sqrt(2));

			float teta = atan2(dsty.at<float>(i, j), dstx.at<float>(i, j));

			if ((teta > 3 * PI / 8 && teta < 5 * PI / 8) || (teta > -5 * PI / 8 && teta < -3 * PI / 8)) dir = 0;
			if ((teta >   PI / 8 && teta < 3 * PI / 8) || (teta > -7 * PI / 8 && teta < -5 * PI / 8)) dir = 1;
			if ((teta >  -PI / 8 && teta <   PI / 8) || teta > 7 * PI / 8 && teta < -7 * PI / 8) dir = 2;
			if ((teta > 5 * PI / 8 && teta < 7 * PI / 8) || (teta > -3 * PI / 8 && teta <   -PI / 8)) dir = 3;
			tetaRoberts.at<uchar>(i, j) = dir;


		}
	}


	imshow("original", img);
	imshow("dest", dst);
	imshow("Directia", tetaRoberts);

	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			switch (tetaRoberts.at<uchar>(i, j)) {
			case 0:
				if ((dst.at<uchar>(i, j) < dst.at<uchar>(i - 1, j)) || (dst.at<uchar>(i, j) < dst.at<uchar>(i + 1, j)))
					dst.at<uchar>(i, j) = 0;
				break;
			case 1:
				if ((dst.at<uchar>(i, j) < dst.at<uchar>(i - 1, j + 1)) || (dst.at<uchar>(i, j) < dst.at<uchar>(i + 1, j - 1)))
					dst.at<uchar>(i, j) = 0;
				break;
			case 2:
				if ((dst.at<uchar>(i, j) < dst.at<uchar>(i, j - 1)) || (dst.at<uchar>(i, j) < dst.at<uchar>(i, j + 1)))
					dst.at<uchar>(i, j) = 0;
				break;
			case 3:
				if ((dst.at<uchar>(i, j) < dst.at<uchar>(i - 1, j - 1)) || (dst.at<uchar>(i, j) < dst.at<uchar>(i + 1, j + 1)))
					dst.at<uchar>(i, j) = 0;
				break;
			default:
				break;
			}
		}
	}


	imshow("Model dupa NMS", dst);



	waitKey(0);
}











/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

																		 //computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Negative image\n");
		printf(" 11 - Gri cu un factor aditiv\n");
		printf(" 12 - Gri cu un factor multiplicativ\n");
		printf(" 13 - Imagine COLOR 4 sectoare\n");
		printf(" 14 - Invert MATRIX 3X3\n");
		printf(" 15 - Copiere canale RGB in 3 ferestre\n");
		printf(" 16 - Conversie din color la grayscale\n");
		printf(" 17 - Conversie din grayscale la alb-negru\n");
		printf(" 18 - Conversie din RGB la HSV\n");
		printf(" 19 - Test daca linie si coloana se afla in imaginea Lena_24bits.bmp \n");
		printf(" 20 - Trasaturi geometrice ale obiectelor binare \n");
		printf(" 21 - Label BFS 4\n");
		printf(" 22 - Label BFS 8\n");
		printf(" 23 - Label 2 pass\n");
		printf(" 24 - Contur\n");
		printf(" 25 - Reconstruct\n");
		printf(" 26 - Dilatare imagine\n");
		printf(" 27 - Eroziune imagine\n");
		printf(" 28 - Deschidere\n");
		printf(" 29 - Inchidere\n");
		printf(" 30 - Dilatare de n ori\n");
		printf(" 31 - Eroziune de n ori\n");
		printf(" 32 - Contur imagine\n");
		printf(" 33 - Media, deviatia standard, histograma\n");
		printf(" 34 - Binarizare globala\n");
		printf(" 35 - Negativul imaginii\n");
		printf(" 36 - Modificare contrast\n");
		printf(" 37 - Egalizare histograma\n");
		printf(" 38 - Filtru general\n");
		printf(" 39 - Filtru medie aritmetica\n");
		printf(" 40 - Filtru gauss\n");
		printf(" 41 - Filtru Laplace\n");
		printf(" 42 - Filtru TS Secvential\n");
		printf(" 43 - Fourier - logaritmul magnitudinii transformatei fourier\n");
		printf(" 44 - FITJ\n");
		printf(" 45 - FITS\n");
		printf(" 46 - FGTJ\n");
		printf(" 47 - Median\n");
		printf(" 48 - Gauss\n");
		printf(" 49 - Gauss vectorial\n");
		printf(" 50 -prewitt\n");
		printf(" 51 - roberts\n");
		printf(" 52 - gradient_sobel\n");
	
		//printf(" 51 - prewet\n");
		//printf(" 52 - robert\n");
		//printf(" 47- filtru trece sus secvential\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testParcurgereSimplaDiblookStyle(); //diblook style
			break;
		case 4:
			//testColor2Gray();
			testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;
		case 10:
			negative_image();
			break;
		case 11:
			aditiveGrayScale();
			break;
		case 12:
			multiplicativeGrayScale();
			break;
		case 13:
			createImageColor();
			break;
		case 14:
			invertMatrix33();
			break;
		case 15:
			copyCanalsRGB3();
			break;
		case 16:
			colorToGray();
			break;
		case 17:
			grayToWhiteBlack();
			break;
		case 18:
			RGB24toHSV();
			break;
		case 19:
			testIsInside();
			break;
		case 20:
			proprietatiGeometrice();
			break;
		case 21:
			labelBSF(4);
			break;
		case 22:
			labelBSF(8);
			break;
		case 23:
			TwoPassLabelling();
			break;
		case 24:
			apel_contur();
			break;
		case 25:
			reconstruct();
			break;
		case 26:
			dilatareImagineTest();
			break;
		case 27:
			eroziuneImagineTest();
			break;
		case 28:
			deschidereImagine();
			break;
		case 29:
			inchidereImagine();
			break;
		case 30:
			dilatareImagineNTest();
			break;
		case 31:
			eroziuneImagineNTest();
			break;
		case 32:
			conturImagine();
			break;
		case 33:
			media();
			break;
		case 34:
			binarizare_automata();
			break;
		case 35:
			negativul_imaginii();
			break;
		case 36:
			modificareContrast();
			break;
		case 37:
			egalizare_histograma();
			break;
		case 38:
			int w;
			//printf("w= ");
			//scanf("%d", &w);
			filtruGeneral(3);
			break;
		case 39:
			filtruMedieAritmetica(3);
			break;
		case 40:
			filtruGausian(3);
			break;
		case 41:
			filtruLaplace();
			break;
		case 42:
			filtruTS();
			break;
		case 43:
			Fourier();
			break;
		case 44:
			FITJ();
			break;
		case 45:
			FITS();
			break;
		case 46:
			FGTJ();
			break;
			//case 47:
			//filtruTreceSusFrecvential(3);
			//	break;
		case 47:
			filtruMedian(3);
			break;
		case 48:
			gauss();
			break;
		case 49:
			filtru_gauss_vectorial(7);
			break;
		case 50:
			gradient_prewitt();
			break;
		case 51:
			gradient_roberts();
			break;
		case 52:
			gradient_sobel();//+canny
			break;


		
		case 53:
			char fname[MAX_PATH];
			while (openFileDlg(fname))
			{
				Mat src ;
				src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
				imshow("init", src);

				computeHistogram(src);
				waitKey();
			}
			
			break;
		}
	} while (op != 0);
	return 0;
}