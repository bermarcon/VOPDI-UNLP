#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <math.h>
#include <sstream>
#include <opencv2/video/video.hpp>


using namespace std;
using namespace cv;

// Mat o_src; // imagen de color base
Mat o_gray; // contendrá la imagen convertida en escala de grises
vector<Mat> o_grayP;
Mat o_gray2;
vector<Mat> o_gray2P;
Mat o_keypoints;
Mat o_keypoints2;
Mat o_frame;
Mat oflow;
Mat graficar;
Mat Em;
Mat mask;
Mat R,R2;
Mat T,T2;
Mat Vu = Mat::zeros(3, 1, CV_64F);
Vec<float,3> AngE;
bool AngE1=true;


// optidcal flow constantes
vector<uchar> encontrado;
vector<float> err;
const Size winSize = Size(35,35);

const char* c_original = "Foto original";
const char* c_key = "Keypoint FAST";
// Scalar(0,255,0)
vector<KeyPoint> o_puntosfast;
vector<Point2f> o_puntos;
vector<Point2f> o_puntos2;
vector<char> mMask;

bool b_nonmaxSuppression=true;
int  n_threshold = 10; // umbral
//int  n_slider_max = 255; // para Fast
int  n_slider_max = 3000; //para orb
int  MaxF= 2000;
unsigned  Vwidth=800;
unsigned  Vheight=600;

uint encontradoMin = 200;
TermCriteria termcrit(TermCriteria::EPS,20,0.003); //cuentas y epsilon (TermCriteria::COUNT|TermCriteria::EPS,20,0.003)
Size subpixwin(5,5);

cv::Ptr<cv::ORB> pOrb = cv::ORB::create(MaxF, 1.2f, 8, 31, 0, 4, cv::ORB::HARRIS_SCORE ,31);


// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(Mat &R)
{
    Mat Rt;
    transpose(R, Rt);
    Mat shouldBeIdentity = Rt * R;
    Mat I = Mat::eye(3,3, shouldBeIdentity.type());

    return  norm(I, shouldBeIdentity) < 1e-6;

}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).

Vec3f rotationMatrixToEulerAngles(Mat &R)
{

    assert(isRotationMatrix(R));

    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return Vec3f(x, y, z);

}

void on_trackbar( int, void* )
{

	pOrb->setMaxFeatures(MaxF);
	std::cout << "Maxfeatures " << MaxF << std::endl;

}


int nKeyPoints;
stringstream text;

void fastImage(vector<KeyPoint>& o_puntosfast, Mat& o_gray, Mat& o_keypoints, Mat& o_frame,int&  n_threshold, bool b_nonmaxSuppression, int &nKeyPoints)
{
	cvtColor( o_frame, o_gray, CV_BGR2GRAY );
	pOrb->detect(o_gray,o_puntosfast);
	drawKeypoints(o_gray, o_puntosfast, o_keypoints, Scalar(0,255,0), DrawMatchesFlags::DEFAULT );
	nKeyPoints = o_puntosfast.size();

	text << "      KeyDetect Corriendo";
	putText(graficar,text.str(),Point2f(20,20),FONT_HERSHEY_PLAIN,1.5,Scalar(255,0,0),2,8,false);

}

void textoVentana(Mat &graficar,stringstream &text, int nKeyPoints, double fps)
{
	text << "  # de puntos criticos = ";
	text << nKeyPoints;
	putText(graficar,text.str(),Point2f(20,20),FONT_HERSHEY_PLAIN,1.5,Scalar(0,0,255),2,8,false);
}

int main()
{

	namedWindow( c_key, 0 );

	createTrackbar( "MaxF", c_key, &MaxF, n_slider_max, on_trackbar );
	//createTrackbar( "Threshold", c_key, &n_threshold, n_slider_max, on_trackbar );//para fast

	VideoCapture video(0);

	if( !video.isOpened() ) { printf("Error loading src1 \n"); return -1; }

	video.set(CV_CAP_PROP_FRAME_WIDTH, Vwidth);
	video.set(CV_CAP_PROP_FRAME_HEIGHT, Vheight);

	//bool niter=true;
	bool play=1;
	int i=-1;
	vector<int64> timev(30,0);
	int64 end;
	int64 start;
	cout << o_gray2.empty() << endl;
	double fps = 0;
	double sum;

	video >> o_frame;
	graficar = o_frame.clone();
	fastImage(o_puntosfast, o_gray, o_keypoints, o_frame, n_threshold, b_nonmaxSuppression, nKeyPoints); //puntos iniciales
	KeyPoint::convert(o_puntosfast, o_puntos);
	cornerSubPix(o_gray, o_puntos, subpixwin, Size(-1,-1), termcrit);

	Vu.at<double>(0,0) = 1;

	while(video.isOpened()) {

		start = getTickCount();

		i++;

		if (i==30){
			i=0;
			int64 temp = 0;
			for (uint j=0; j<timev.size(); j++){
				temp = temp + timev[i];
			}
			sum = temp/30;
			fps = getTickFrequency()/sum;
		}

		if(play){


			if (!o_gray2.empty()){

				//o_gray2P = o_gray2.clone();
				//o_grayP = o_gray.clone();
				buildOpticalFlowPyramid(
						o_gray2,
						o_gray2P,
						winSize,
						2,
						true,
						BORDER_REFLECT_101,
						BORDER_CONSTANT,
						false);

				buildOpticalFlowPyramid(
						o_gray,
						o_grayP,
						winSize,
						2,
						true,
						BORDER_REFLECT_101,
						BORDER_CONSTANT,
						false);

				calcOpticalFlowPyrLK(
						o_gray2P, // imagen anterior
						o_grayP,	 // imagen actual
						o_puntos2,  // key points antiguos
						o_puntos,   // nuevos key points
						encontrado,
						err,
						winSize,
						2,
						termcrit,
						OPTFLOW_LK_GET_MIN_EIGENVALS,  //OPTFLOW_LK_GET_MIN_EIGENVALS
						1.0e-3);

				uint n = 0;
				for ( uint i=0; i<= encontrado.size(); i++){
					n = n + encontrado[i];
				}



				if ( n < encontradoMin) {  // no alcance el minimo de puntos en la imagen
					cout << "Corriendo fast de nuevo" << endl;
					o_puntos.clear();
					fastImage(o_puntosfast, o_gray, o_keypoints, o_frame,n_threshold,b_nonmaxSuppression,nKeyPoints);
					KeyPoint::convert(o_puntosfast, o_puntos);

					if (o_puntos.size()>10)
						cornerSubPix(o_gray, o_puntos, subpixwin, Size(-1,-1), termcrit);
				}

				else{
					size_t i, k;
					for(i=0, k=0; i<encontrado.size(); i++){
						if (encontrado[i]){
							circle(graficar,cvPoint(o_puntos[i].x,o_puntos[i].y),3,Scalar(0,255,0),-1,8,0);
							o_puntos[k] = o_puntos[i];
							o_puntos2[k++] = o_puntos2[i];
						}
						else{
							circle(graficar,cvPoint(o_puntos[i].x,o_puntos[i].y),3,Scalar(0,0,255),1,7,0);
						}

					}


				o_puntos.resize(k);
				o_puntos2.resize(k);
				nKeyPoints=k;
				//std::cout<< k << std::endl;

				}

				//std::cout<< o_puntos.size()<< std::endl;
				//std::cout<< o_puntos2.size()<< std::endl;

				//findEssentialMat
				//Calculates an essential matrix from the corresponding points in two images.
				//C++: Mat findEssentialMat(InputArray points1, InputArray points2, double focal=1.0, Point2d pp=Point2d(0, 0), int method=RANSAC, double prob=0.999, double threshold=1.0, OutputArray mask=noArray() )

				double focal = 10;
				cv::Point2d pp(Vwidth/2, Vheight/2);

				Em= findEssentialMat(
						o_puntos2,
						o_puntos,
						focal,
						pp,
						RANSAC,
						0.999,
						1.0,
						mask);

				recoverPose(Em, o_puntos2, o_puntos, R, T, focal, pp, mask);


				AngE=rotationMatrixToEulerAngles(R)+AngE;//3.14*360.0+AngE;

				//Vu=R*Vu;
				if (AngE1){
					AngE=0;
					AngE1=false;
				};

				std::cout<< AngE << std::endl;


			}
			text.str(std::string());
			textoVentana(graficar, text, nKeyPoints, fps);
			imshow( c_key, graficar );
			o_gray2 = o_gray.clone(); // guardo imagen anterior
			std::swap(o_puntos, o_puntos2);
			video >> o_frame;		// nueva imagen
			graficar = o_frame.clone();
			cvtColor( o_frame, o_gray, CV_BGR2GRAY );

		}

		char key = (char)waitKey(1);
		if(key == 'p'){
			play=!play;
		}
		else if (key == 'x'){
			break;
		}
		else if (key == 'c'){
			int n = 0;
			for ( uint i=0; i<= encontrado.size(); i++){
				n = n + encontrado[i];
			}
			cout << "el número de puntos encontrados es " << n <<endl;
		}

		end = getTickCount();

		timev[i]=end-start;


	}
}
