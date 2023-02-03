#include "opencv2/opencv.hpp"

#include<iostream>
#include<stdio.h>

using namespace std;
using namespace cv;

Mat img;
Point pt0ld;

void camera_in_video_out();
void filter_embosiong();
void blurring_mean();
void blurring_gaussian();
void unsharp_mask();
void noise_gaussian();
void filter_bilateral();
void filter_median();

//<8. 영상의 기하학적 변환>
void affine_transform();
void affine_translate();
void affine_scale();
void affine_rotation();
void affine_flip();
void on_mouse(int event, int x, int y, int flags, void*);

//<9. 에지 검출과 응용>
void sobel_edge();
void canny_edge();
void hough_lines();
void hough_line_segments();
void hough_circle();

// <11. 이진화와 모폴로지>
void on_threshold(int pos, void* userdata);
void on_trackbar(int pos, void* userdata);
void erode_dilate();
void open_close();

//<12. 레이블링과 외곽선 검출>
void labeling_basic();
void labeling_stats();

void contours_basic();
void contours_hier();

//<13. 객체 검출>

void template_matching();
void CascadeClassifier();
void detect_face();


void corner_harris();
void corner_fast();
void detect_keypoints();

// <트렉바 사용하기>
// void on_level_change(int pos,void* userdata);

Mat src;
Point2f srcPts[4], dstPts[4];

// void on_mouse(int event, int x, int y, int flags, void* userdata);

// <11.1 영상의 이진화>
// int main(int argc, char* argv[]) {
int main(void) 
{
	labeling_basic();

	//<<8.2 투시 변환>
	// src = imread("card.jpg");

	// if (src.empty()) {
	// 	cerr << "Image load failed!" << endl;
	// 	return -1;
	// }

	// namedWindow("src");
	// setMouseCallback("src", on_mouse);

	// imshow("src", src);
	// waitKey(0);

	// return 0;

	//<11.1(1) 영상의 이진화>
 	// Mat src;
    
    // if (argc < 2)
    //     src = imread("chess.jpg", IMREAD_GRAYSCALE);
    // else
    //     src = imread(argv[1], IMREAD_GRAYSCALE);
     
    // if (src.empty()) {
    //     cerr << "Image load failed!" << endl;
    //     return -1;
    // }
     
    // imshow("src", src);
     
    // namedWindow("dst");
    // createTrackbar("Threshold", "dst", 0, 255, on_threshold, (void*)&src);
    // setTrackbarPos("Threshold", "dst", 128);
     
    // waitKey(0);
    // return 0;
	
	// <11.1(2) 영상의 이진화>
	// Mat src = imread("chess.jpg", IMREAD_GRAYSCALE);
     
    // if (src.empty()) {
    //      cerr << "Image load failed!" << endl;
    //     return -1;
    // }
     
    // imshow("src", src);
     
    // namedWindow("dst");
    // createTrackbar("Block Size", "dst", 0, 200, on_trackbar, (void*)&src);
    // setTrackbarPos("Block Size", "dst", 11);
     
    // waitKey(0);
    // return 0;

	// cout << "Hello OpenCV" << CV_VERSION << std::endl; 
	
	// Mat image1;

	//<마우스 이벤트 처리>
	// img = imread("test.jpg");

	// if (img.empty()) {
	//  	cerr <<"IMage load failed!"<<endl;
	//  	return -1;
	//  };
	
	// namedWindow("img");
	// setMouseCallback("img",on_mouse);
	// imshow("img", img);
		
	// waitKey();
	// return 0;
	
	// <트렉바 사용하기>
	// Mat img = Mat::zeros(400,400,CV_8UC1);
	// namedWindow("image");
	// createTrackbar("level","image",0,16,on_level_change,(void*)&img);
	// imshow("image", img);
	// waitKey();
	
	// return 0;

}

void drawLines()
{
	Mat img(400, 400, CV_8UC3, Scalar(255,255,255));

	line(img, Point(50,50), Point(200,50),Scalar(0,0,255));

	arrowedLine(img, Point(50,100), Point(200,100),Scalar(255,0,255),3);

	drawMarker(img, Point(50,150),Scalar(255,0,255),MARKER_STAR);

	imshow("img",img);
	waitKey();

	destroyAllWindows();
}

//<마우스 이벤트 처리>
// void on_mouse(int event, int x, int y, int flags, void*)
// {
// 	switch (event) {
// 	case EVENT_LBUTTONDOWN:
// 		pt0ld = {Point(x,y)};
// 		cout << "EVENT_LBUTTONDOWN: " << x << ", "<< y << endl;
// 		break;
// 	case EVENT_LBUTTONUP:
// 		cout << "EVENT_LBUTTONUP: " << x << ", "<< y << endl;
// 		break;
// 	case EVENT_MOUSEMOVE:
// 		if (flags & EVENT_FLAG_LBUTTON) {
// 			line(img, pt0ld, Point(x,y), Scalar(0,255.255),2);
// 		imshow("img",img);
// 		pt0ld = Point(x,y);
// 		}
// 		break;
// 	default:
// 		break;
// 	}
// }

// <트렉바 사용하기>
// void on_level_change(int pos, void*userdata)
// {
// 	Mat img = *(Mat*)userdata;

// 	img.setTo(pos * 16);
// 	imshow("image",img);
// }

// <데이터 파일 입출력>
// void writeData()
// {
// 	String name="Jane";
// 	int age = 10;
// 	Point pt1(100,200);
// 	vector<int> scores = {80,90,50};
// 	Mat mat1 = (Mat_<float>(2,2) << 1.0f, 1.5f, 2.0f, 3.2f);

// 	FileStorage fs(filename, FileStorage::WRITE);

// }

// <마스크 연산>

// void mask_setTo()
// {
// 	Mat src = imread("test.jpg",IMREAD_COLOR);
// 	Mat mask = imread("mask_");
// }

void camera_in_video_out()
{
	VideoCapture cap(0);

	if(!cap.isOpened()) {
		cerr << "Camera open failed" << endl;
		return;
	}

	int w = cvRound(cap.get(CAP_PROP_FRAME_WIDTH));
	int h = cvRound(cap.get(CAP_PROP_FRAME_HEIGHT));
	double fps = cap.get(CAP_PROP_FPS);

	// cout << "Frame width:" << cvRound(cap.get(CAP_PROP_FRAME_WIDTH)) << endl;
	// cout << "Frame height:" << cvRound(cap.get(CAP_PROP_FRAME_HEIGHT)) << endl;

	int fourcc = VideoWriter::fourcc('D','I','V','X');
	int delay = cvRound(1000/25);

	VideoWriter outputVideo("output1.avi",fourcc,25,Size(w,h));

	if (!outputVideo.isOpened()) {
		cout << "File open failed" << endl;
		return;
	}

	Mat frame, inversed;
	while (true) {
		cap >> frame;
		if (frame.empty())
			break;

		inversed = ~frame;
		outputVideo << inversed;

		imshow("frame",frame);
		imshow("inversed",inversed);

		if (waitKey(10) == 27)
			break;
	}

	destroyAllWindows();
}

void filter_embosiong()
{
	Mat src = imread("rose.jpg", IMREAD_GRAYSCALE);

	if(src.empty()) {
		cerr << "Image Load failed" << endl;
		return;
	}

	float data[]={-1,-1,0,-1,0,1,0,1,1};
	Mat emboss(3,3,CV_32FC1,data);

	Mat dst;
	filter2D(src, dst, -1, emboss,Point(-1,-1),128);

	imshow("src",src);
	imshow("dts",dst);

	waitKey();
	destroyAllWindows();
}

void blurring_mean()
{
	Mat src = imread("rose.jpg", IMREAD_GRAYSCALE);

	if(src.empty()) {
		cerr << "Image Load failed" << endl;
		return;
	}

	imshow("src",src);

	Mat dst;
	for (int ksize=3; ksize <= 7; ksize+=2) {
		blur(src, dst, Size(ksize,ksize));

		String desc = format("Mean: %dx%d",ksize,ksize);
		putText(dst, desc,Point(10,30),FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255),1,LINE_AA);

		imshow("dst",dst);

		waitKey();
	}
	destroyAllWindows();
}

void blurring_gaussian()
{
	Mat src = imread("rose.jpg", IMREAD_GRAYSCALE);

	if(src.empty()) {
		cerr << "Image Load failed" << endl;
		return;
	}

	imshow("src",src);

	Mat dst;
	for (int sigma=1; sigma <= 5; sigma++) {
		GaussianBlur(src, dst, Size(),(double)sigma);

		String text = format("sigma = %d", sigma);
		putText(dst, text, Point(10,30),FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255),1,LINE_AA);

		imshow("dst",dst);

		waitKey();
	}
	destroyAllWindows();
}

void unsharp_mask()
{
	Mat src = imread("rose.jpg", IMREAD_GRAYSCALE);

	if(src.empty()) {
		cerr << "Image Load failed" << endl;
		return;
	}

	imshow("src",src);

	Mat dst;
	for (int sigma=1; sigma <= 5; sigma++) {
		Mat blurred;
		GaussianBlur(src, blurred, Size(),(double)sigma);

		float alpha = 1.f;
		Mat dst = (1+alpha) * src - alpha * blurred;

		String desc = format("sigma = %d", sigma);
		putText(dst, desc, Point(10,30),FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255),1,LINE_AA);

		imshow("dst",dst);

		waitKey();
	}
	destroyAllWindows();	
}

void noise_gaussian()
{
	Mat src = imread("rose.jpg", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	imshow("src", src);

	for (int stddev = 10; stddev <= 100; stddev += 10) {
		Mat noise(src.size(), CV_32SC1);
		randn(noise, 0, stddev);

		Mat dst;
		add(src, noise, dst, Mat(), CV_8U);

		String desc = format("stddev = %d", stddev);
		putText(dst, desc, Point(10, 100), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255), 1, LINE_AA);
		imshow("dst", dst);
		waitKey();
	}

	destroyAllWindows();
}

void filter_bilateral()
{
	Mat src = imread("rose.jpg", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat noise(src.size(), CV_32SC1);
	randn(noise, 0, 5);
	add(src, noise, src, Mat(), CV_8U);

	Mat dst1;
	GaussianBlur(src, dst1, Size(), 5);

	Mat dst2;
	bilateralFilter(src, dst2, -1, 10, 5);

	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);

	waitKey();
	destroyAllWindows();
}

void filter_median()
{
	Mat src = imread("rose.jpg", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	int num = (int)(src.total() * 0.1);
	for (int i = 0; i < num; i++) {
		int x = rand() % src.cols;
		int y = rand() % src.rows;
		src.at<uchar>(y, x) = (i % 2) * 255;
	}

	Mat dst1;
	GaussianBlur(src, dst1, Size(), 1);

	Mat dst2;
	medianBlur(src, dst2, 3);

	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);

	waitKey();
	destroyAllWindows();
}

//<8.1(1) 어파인 변환>
void affine_transform()
{
	Mat src = imread("rose.jpg");

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Point2f srcPts[3], dstPts[3];
	srcPts[0] = Point2f(0, 0);
	srcPts[1] = Point2f(src.cols - 1, 0);
	srcPts[2] = Point2f(src.cols - 1, src.rows - 1);
	dstPts[0] = Point2f(50, 50);
	dstPts[1] = Point2f(src.cols - 100, 100);
	dstPts[2] = Point2f(src.cols - 50, src.rows - 50);

	Mat M = getAffineTransform(srcPts, dstPts);

	Mat dst;
	warpAffine(src, dst, M, Size());

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
	destroyAllWindows();
}

//<8.1(2) 어파인 변환>
void affine_translate()
{
	Mat src = imread("rose.jpg");

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat M = Mat_<double>({ 2,3 }, { 1,0,150,0,1,100 });

	Mat dst;
	warpAffine(src, dst, M, Size());

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

//<8.1(3) 어파인 변환>
void affine_shear()
{
	Mat src = imread("rose.jpg");

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	double mx = 0.3;
	Mat M = Mat_<double>({ 2,3 }, { 1,mx,0,0,1,0 });

	Mat dst;
	warpAffine(src, dst, M, Size(cvRound(src.cols + src.rows * mx),src.rows));

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

//<8.1(4) 어파인 변환_크기 변환>
void affine_scale()
    {
        Mat src = imread("rose.jpg");
     
        if (src.empty()) {
            cerr << "Image load failed!" << endl;
            return;
        }
     
        Mat dst1, dst2, dst3, dst4;
        resize(src, dst1, Size(), 4, 4, INTER_NEAREST);
        resize(src, dst2, Size(1920, 1280));
        resize(src, dst3, Size(1920, 1280), 0, 0, INTER_CUBIC);
        resize(src, dst4, Size(1920, 1280), 0, 0, INTER_LANCZOS4);
     
        imshow("src", src);
        imshow("dst1", dst1(Rect(400, 500, 400, 400)));
        imshow("dst2", dst2(Rect(400, 500, 400, 400)));
        imshow("dst3", dst3(Rect(400, 500, 400, 400)));
        imshow("dst4", dst4(Rect(400, 500, 400, 400)));
     
        waitKey();
        destroyAllWindows();
    }

//<8.1(5) 어파인 변환_회전 변환>
void affine_rotation()
    {
        Mat src = imread("chess.jpg");
     
        if (src.empty()) {
            cerr << "Image load failed!" << endl;
            return;
        }
     
        Point2f cp(src.cols / 2.f, src.rows / 2.f);
        Mat M = getRotationMatrix2D(cp, 20, 1);
     
        Mat dst;
        warpAffine(src, dst, M, Size());
     
        imshow("src", src);
       imshow("dst", dst);
     
        waitKey();
        destroyAllWindows();
    }

//<8.1(6) 어파인 변환_대칭 변환>
 void affine_flip()
    {
        Mat src = imread("chess.jpg");
     
        if (src.empty()) {
            cerr << "Image load failed!" << endl;
            return;
        }
     
        imshow("src", src);
     
        Mat dst;
        int flipCode[] = { 1, 0, -1 };
        for (int i = 0; i < 3; i++) {
            flip(src, dst, flipCode[i]);
     
            String desc = format("flipCode: %d", flipCode[i]);
            putText(dst, desc, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0,                      Scalar(255, 0, 0), 1, LINE_AA);
     
            imshow("dst", dst);
            waitKey();
        }
     
        destroyAllWindows();
    }

//<8.2 투시 변환>
void on_mouse(int event, int x, int y, int flags, void*)
{

	static int cnt = 0;
	if (event == EVENT_LBUTTONDOWN) {
		if (cnt < 4) {
			srcPts[cnt++] = Point2f(x, y);

			circle(src, Point(x, y), 5, Scalar(0, 0, 255), -1);
			imshow("src", src);

			if (cnt == 4) {
				int w = 200, h = 300;

				dstPts[0] = Point2f(0, 0);
				dstPts[1] = Point2f(w - 1, 0);
				dstPts[2] = Point2f(w - 1, h - 1);
				dstPts[3] = Point2f(0, h - 1);

				Mat pers = getPerspectiveTransform(srcPts, dstPts);

				Mat dst;
				warpPerspective(src, dst, pers, Size(w, h));

				imshow("dst", dst);
			}
		}
	}
}

//<9.1 에지 검출과 응용_소벨 마스크>
void sobel_edge()
{
	Mat src = imread("rose.jpg", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "image load failed!" << endl;
		return;
	}

	Mat dx, dy;
	// Sobel() : 소벨마스크를 이용하여 영상을 미분
	Sobel(src, dx, CV_32FC1, 1, 0);		// dx : x방향 편미분 결과 저장
	Sobel(src, dy, CV_32FC1, 0, 1);		// dy : y방향 편미분 결과 저장

	Mat fmag, mag;
	// magnitude() : 그래디언트 크기 계산
	magnitude(dx, dy, fmag);
	fmag.convertTo(mag, CV_8UC1);

	// 임계값을 150으로 설정하여 에지 판별
	// edge의 원소값은 mag 행렬 원소값이 150보다 크면 255, 작으면 0으로 설정
	Mat edge = mag > 150;

	imshow("src", src);
	imshow("mag", mag);
	imshow("edge", edge);

	waitKey();
	destroyAllWindows();
}

//<9.1 에지 검출과 응용_케니 에지 검출기>
void canny_edge()
{
	Mat src = imread("rose.jpg", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat dst1, dst2;
	
	Canny(src, dst1, 50, 100);		// 낮은 임계값 50, 높은 임계값 100
	Canny(src, dst2, 50, 220);		// 낮은 임계값 50, 높은 임계값 220
	// 임계값을 낮출수록 잡음에 해당하는 픽셀도 에지로 검출될 가능성이 높아질 수 있으므로 주의가 필요

	imshow("src", src);
	imshow("dst1", dst1);
	imshow("dst2", dst2);

	waitKey();
	destroyAllWindows();

}

//<9.2(1) 직선 검출_허프 변환> 
void hough_lines()
{
	Mat src = imread("chess.jpg", IMREAD_GRAYSCALE);

	if (src.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}

	Mat edge;
	Canny(src, edge, 50, 100);

	vector<Vec2f> lines;
	HoughLines(edge, lines, 1, CV_PI / 180, 150);

	Mat dst;
	cvtColor(edge, dst, COLOR_GRAY2BGR);

	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		float cos_t = cos(theta), sin_t = sin(theta);
		float x0 = rho * cos_t, y0 = rho * sin_t;
		float alpha = 1000;

		Point pt1(cvRound(x0 - alpha * sin_t), cvRound(y0 + alpha * cos_t));
		Point pt2(cvRound(x0 + alpha * sin_t), cvRound(y0 - alpha * cos_t));
		line(dst, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
	}

	imshow("src", src);
	imshow("dst", dst);

	waitKey(0);
	destroyAllWindows();
}

//<9.2(2) 직선 검출_확률적 허프 변환> 
void hough_line_segments()
    {
        Mat src = imread("chess.jpg", IMREAD_GRAYSCALE);
     
        if (src.empty()) {
            cerr << "Image load failed!" << endl;
            return;
        }
     
        Mat edge;
        Canny(src, edge, 50, 150);
     
        vector<Vec4i> lines;
        HoughLinesP(edge, lines, 1, CV_PI / 180, 160, 50, 5);
     
        Mat dst;
        cvtColor(edge, dst, COLOR_GRAY2BGR);
     
        for (Vec4i l : lines) {
            line(dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255),
                   2, LINE_AA);
        }
     
        imshow("src", src);
        imshow("dst", dst);
     
        waitKey(0);
        destroyAllWindows();
    }

//<9.2(3) 원 검출_허프 변환>
void hough_circle()
{
    Mat src = imread("coins.jpg", IMREAD_GRAYSCALE);
    
    if(src.empty()){
        cerr<<"Image load failed!"<<endl;
        return;
    }
    
    Mat blurred;
    blur(src, blurred, Size(3, 3));
    
    vector<Vec3f> circles;
    HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, 350, 400, 10);
    
    Mat dst;
    cvtColor(src, dst, COLOR_GRAY2BGR);
    
    for(Vec3f c : circles){
        Point center(cvRound(c[0]), cvRound(c[1]));
        int radius = cvRound(c[2]);
        circle(dst, center, radius, Scalar(0, 0, 255), 2, LINE_AA);
    }
    
    imshow("src", src);
    imshow("dst", dst);
    
    waitKey(0);
    destroyAllWindows();
}

//<11.1(1) 영상의 이진화>
void on_threshold(int pos, void* userdata)
    {
        Mat src = *(Mat*)userdata;
     
        Mat dst;
        threshold(src, dst, pos, 255, THRESH_BINARY);
     
        imshow("dst", dst);
    }

//<11.1(2) 영상의 이진화_적응형 이진화>
void on_trackbar(int pos, void* userdata)
    {
        Mat src = *(Mat*)userdata;
     
        int bsize = pos;
        if (bsize % 2 == 0) bsize--;
        if (bsize < 3) bsize = 3;
     
        Mat dst;
        adaptiveThreshold(src, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,
                          bsize, 5);
     
        imshow("dst", dst);
    }

//<11.2(1) 모폴로지 연산_침식,팽창>
void erode_dilate()
    {
        Mat src = imread("chess.jpg", IMREAD_GRAYSCALE);
     
        if (src.empty()) {
            cerr << "Image load failed!" << endl;
            return;
        }
     
        Mat bin;
        threshold(src, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
     
        Mat dst1, dst2;
        erode(bin, dst1, Mat());
        dilate(bin, dst2, Mat());
     
        imshow("src", src);
        imshow("bin", bin);
        imshow("erode", dst1);
        imshow("dilate", dst2);
     
        waitKey();
        destroyAllWindows();
    }

//<11.2(2) 모폴로지 연산_열기,닫기>
void open_close()
    {
        Mat src = imread("chess.jpg", IMREAD_GRAYSCALE);
     
        if (src.empty()) {
            cerr << "Image load failed!" << endl;
            return;
        }
     
        Mat bin;
        threshold(src, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
     
        Mat dst1, dst2;
        morphologyEx(src, dst1, MORPH_OPEN, Mat());
        morphologyEx(src, dst2, MORPH_CLOSE, Mat());
     
        imshow("src", src);
        imshow("bin", bin);
        imshow("opening", dst1);
        imshow("closing", dst2);
     
        waitKey();
        destroyAllWindows();
    }

//<12.1(1) 레이블링>
void labeling_basic()
    {
        uchar data[] = {
            0, 0, 1, 1, 0, 0, 0, 0,
            1, 1, 1, 1, 0, 0, 1, 0,
            1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 1, 0,
            0, 0, 0, 1, 1, 1, 1, 0,
            0, 0, 0, 1, 0, 0, 1, 0,
            0, 0, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        };
     
        Mat src = Mat(8, 8, CV_8UC1, data) * 255;
     
        Mat labels;
        int cnt = connectedComponents(src, labels);
     
        cout << "src:\n" << src << endl;
        cout << "labels:\n" << labels << endl;
        cout << "number of labels: " << cnt << endl;
	}

//<12.1(2) 레이블링 응용>
void labeling_stats()
    {
        Mat src = imread("keyboard.jpg", IMREAD_GRAYSCALE);
     
        if (src.empty()) {
            cerr << "Image load failed!" << endl;
            return;
        }
     
        Mat bin;
        threshold(src, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
     
        Mat labels, stats, centroids;
        int cnt = connectedComponentsWithStats(bin, labels, stats, centroids);
     
        Mat dst;
        cvtColor(src, dst, COLOR_GRAY2BGR);
     
        for (int i = 1; i < cnt; i++) {
            int* p = stats.ptr<int>(i);
    
            if (p[4] < 20) continue;
     
            rectangle(dst, Rect(p[0], p[1], p[2], p[3]), Scalar(0, 255, 255), 2);
        }
     
     
        imshow("src", src);
        imshow("dst", dst);
     
        waitKey();
        destroyAllWindows();
    }

void contours_basic()
    {
        Mat src = imread("contours.jpg", IMREAD_GRAYSCALE);
     
        if (src.empty()) {
            cerr << "Image load failed!" << endl;
            return;
        }
     
        vector<vector<Point>> contours;
        findContours(src, contours, RETR_LIST, CHAIN_APPROX_NONE);
     
        Mat dst;
        cvtColor(src, dst, COLOR_GRAY2BGR);
     
        for (int i = 0; i < contours.size(); i++) {
            Scalar c(rand() & 255, rand() & 255, rand() & 255);
            drawContours(dst, contours, i, c, 2);
        }
     
        imshow("src", src);
        imshow("dst", dst);
     
        waitKey(0);
        destroyAllWindows();
    }

void contours_hier()
    {
        Mat src = imread("contours.jpg", IMREAD_GRAYSCALE);
     
        if (src.empty()) {
            cerr << "Image load failed!" << endl;
            return;
        }
     
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(src, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
     
        Mat dst;
        cvtColor(src, dst, COLOR_GRAY2BGR);
     
        for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
            Scalar c(rand() & 255, rand() & 255, rand() & 255);
            drawContours(dst, contours, idx, c, -1, LINE_8, hierarchy);
        }
     
        imshow("src", src);
        imshow("dst", dst);
     
        waitKey(0);
        destroyAllWindows();
    }

//<13.객체 검출_템플릿 매칭>
void template_matching()
    {
        Mat img = imread("circuit.jpg", IMREAD_COLOR);
        Mat templ = imread("crystal.jpg", IMREAD_COLOR);
     
        if (img.empty() || templ.empty()) {
            cerr << "Image load failed!" << endl;
            return;
        }
     
        img = img + Scalar(50, 50, 50);
     
        Mat noise(img.size(), CV_32SC3);
        randn(noise, 0, 10);
        add(img, noise, img, Mat(), CV_8UC3);
     
        Mat res, res_norm;
        matchTemplate(img, templ, res, TM_CCOEFF_NORMED);
        normalize(res, res_norm, 0, 255, NORM_MINMAX, CV_8U);
     
        double maxv;
        Point maxloc;
        minMaxLoc(res, 0, &maxv, 0, &maxloc);
        cout << "maxv: " << maxv << endl;
     
        rectangle(img, Rect(maxloc.x, maxloc.y, templ.cols, templ.rows), Scalar(0, 0, 255), 2);
     
        imshow("templ", templ);
        imshow("res_norm", res_norm);
        imshow("img", img);
     
        waitKey(0);
        destroyAllWindows();
    }

//<13.2 케스케이드 분류기와 얼굴 검출_class CascadeClassifier>
// class CascadeClassifier
//     {
//     public:
//         CascadeClassifier();
//         CascadeClassifier(const String& filename);
//         ~CascadeClassifier();
     
//         bool load(const String& filename);
//      bool empty() const;
     
//      void detectMultiScale(InputArray image,
//                               std::vector<Rect>& objects,
//                               double scaleFactor = 1.1,
//                               int minNeighbors = 3, int flags = 0,
//                               Size minSize = Size(),
//                               Size maxSize = Size() );
        
//     };

//<13.2 케스케이드 분류기와 얼굴 검출>
//  void detect_face()
//     {
//         Mat src = imread("kids.png");
     
//         if (src.empty()) {
//             cerr << "Image load failed!" << endl;
//             return;
//         }

//         CascadeClassifier classifier("haarcascade_frontalface_default.xml");
     
//         if (classifier.empty()) {
//             cerr << "XML load failed!" << endl;
//             return;
//         }
     
//         vector<Rect> faces;
//         classifier.detectMultiScale(src, faces);
     
//         for (Rect rc : faces) {
//             rectangle(src, rc, Scalar(255, 0, 255), 2);
//         }
     
//         imshow("src", src);
     
//         waitKey(0);
//         destroyAllWindows();
//     }

void corner_harris()
    {
        Mat src = imread("chess.jpg", IMREAD_GRAYSCALE);
     
        if (src.empty()) {
            cerr << "Image load failed!" << endl;
            return;
        }
     
        Mat harris;
        cornerHarris(src, harris, 3, 3, 0.04);
     
        Mat harris_norm;
        normalize(harris, harris_norm, 0, 255, NORM_MINMAX, CV_8U);
     
        Mat dst;
        cvtColor(src, dst, COLOR_GRAY2BGR);
     
        for (int j = 1; j < harris.rows - 1; j++) {
            for (int i = 1; i < harris.cols - 1; i++) {
                if (harris_norm.at<uchar>(j, i) > 130) {
                    if (harris.at<float>(j, i) > harris.at<float>(j - 1, i) &&
                        harris.at<float>(j, i) > harris.at<float>(j + 1, i) &&
                        harris.at<float>(j, i) > harris.at<float>(j, i - 1) &&
                        harris.at<float>(j, i) > harris.at<float>(j, i + 1) ) {
                        circle(dst, Point(i, j), 5, Scalar(0, 0, 255), 2);
                    }
                }
            }
        }
     
        imshow("src", src);
        imshow("harris_norm", harris_norm);
        imshow("dst", dst);
    
       waitKey(0);
        destroyAllWindows();
    }

void corner_fast()

    {
        Mat src = imread("chess.jpg", IMREAD_GRAYSCALE);
     
        if (src.empty()) {
            cerr << "Image load failed!" << endl;
            return;
        }
     
        vector<KeyPoint> keypoints;
        FAST(src, keypoints, 60, true);
     
        Mat dst;
        cvtColor(src, dst, COLOR_GRAY2BGR);
     
        for (KeyPoint kp : keypoints) {
            Point pt(cvRound(kp.pt.x), cvRound(kp.pt.y));
            circle(dst, pt, 5, Scalar(0, 0, 255), 2);
        }
     
        imshow("src", src);
        imshow("dst", dst);
     
        waitKey(0);
        destroyAllWindows();
    }

void detect_keypoints()
    {
        Mat src = imread("chess.jpg", IMREAD_GRAYSCALE);
     
        if (src.empty()) {
            cerr << "Image load failed!" << endl;
            return;
        }
     
        Ptr<Feature2D> feature = ORB::create();
     
        vector<KeyPoint> keypoints;
        feature->detect(src, keypoints);
     
        Mat desc;
        feature->compute(src, keypoints, desc);
     
        cout << "keypoints.size(): " << keypoints.size() << endl;
        cout << "desc.size(): " << desc.size() << endl;
     
        Mat dst;
        drawKeypoints(src, keypoints, dst, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
     
        imshow("src", src);
        imshow("dst", dst);
     
        waitKey();
        destroyAllWindows();
    }

    