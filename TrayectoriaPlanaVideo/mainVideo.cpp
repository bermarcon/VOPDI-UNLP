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
#include "plot.hpp"

using namespace std;
using namespace cv;

// Imagenes y máscaras
Mat o_gray, o_gray2;
Mat o_keypoints, o_keypoints2;
Mat o_frame;
Mat oflow;
Mat graficar;
Mat Em;
Mat mask;

// Video
unsigned  	Vwidth	=800, Vheight=600;

// Ventanas
const char* c_original = "Foto original", *c_key = "Keypoint";
stringstream text;

// Keypoints
vector<KeyPoint> 	o_puntosfast;
vector<Point2f> 	o_puntos, o_puntos2;
vector<char> 	mMask;
vector<uchar> 	encontrado;
vector<float> 	err;

// Especificaciones para KeyDetect
int  	MaxF=400;
uint 	encontradoMin=35;
bool 	b_nonmaxSuppression=true;
int  	n_threshold = 20; // umbral //int  n_slider_max = 255; // para Fast
int 	n_slider_max = 800; //para orb

// Variables para Odometría y Gráfica
double movx=0;
double movy=0;
Mat display;
Mat displayH;
Mat movxh(1,1, CV_64F);
Mat movyh(1,1, CV_64F);

// Especificaciones para OF
const Size winSize = Size(60,60);
TermCriteria termcrit(TermCriteria::EPS,20,0.003);
Size subpixwin(5,5);
cv::Ptr<cv::ORB> pOrb = cv::ORB::create(MaxF, 1.2f, 8, 31, 0, 4, cv::ORB::HARRIS_SCORE ,31);
Mat masko(Vheight,Vwidth,CV_8U,Scalar(255));
int nKeyPoints;

// Para empezar a correr el programa
bool play=1;
bool init=1;
int cntf=0;
int cntfa=0;

void on_trackbar( int, void* )
{
	pOrb->setMaxFeatures(MaxF);
	std::cout << "Maxfeatures " << MaxF << std::endl;
}

void KeyDetect(vector<KeyPoint>& o_puntosfast, Mat& o_gray, Mat& o_keypoints, Mat& o_frame,
					int&  n_threshold, bool b_nonmaxSuppression, int &nKeyPoints,Mat& maskorb)
{
	cvtColor( o_frame, o_gray, CV_BGR2GRAY );
	pOrb->detect(o_gray,o_puntosfast,maskorb);
	drawKeypoints(o_gray, o_puntosfast, o_keypoints, Scalar(0,255,0), DrawMatchesFlags::DEFAULT );
	nKeyPoints = o_puntosfast.size();

	text << "      KeyDetect Corriendo";
	putText(graficar,text.str(),Point2f(20,20),FONT_HERSHEY_PLAIN,1.5,Scalar(255,0,0),2,8,false);
}

void textoVentana(Mat &graficar,stringstream &text, int nKeyPoints)
{
	text << "  # de puntos criticos = ";
	text << nKeyPoints;
	putText(graficar,text.str(),Point2f(20,20),FONT_HERSHEY_PLAIN,1.5,Scalar(0,0,255),2,8,false);
}

int main()
{

	string videosrc;
	cout<<"Seleccione nombre de video\n"<<endl;
	getline(cin,videosrc);

	namedWindow( c_key, 0 );
	createTrackbar( "MaxF", c_key, &MaxF, n_slider_max, on_trackbar );

	VideoCapture video(videosrc);
	if( !video.isOpened() ) { printf("Error loading src1 \n"); return -1; }

	video >> o_frame;
	graficar = o_frame.clone();


	KeyDetect(o_puntosfast, o_gray, o_keypoints, o_frame, n_threshold, b_nonmaxSuppression, nKeyPoints,masko); //puntos iniciales
	KeyPoint::convert(o_puntosfast, o_puntos);
	cornerSubPix(o_gray, o_puntos, subpixwin, Size(-1,-1), termcrit);


	while(video.isOpened()) {

		if(play){
			if (!init){
				calcOpticalFlowPyrLK(
						o_gray2, // imagen anterior
						o_gray,	 // imagen actual
						o_puntos2,  // key points antiguos
						o_puntos,   // nuevos key points
						encontrado,
						err,
						winSize,
						2,
						termcrit,
						OPTFLOW_LK_GET_MIN_EIGENVALS,
						1.0e-3);

				uint n = 0;
				for ( uint i=0; i<= encontrado.size(); i++){
					n = n + encontrado[i];
				}

				if ( n < encontradoMin) {
					cout << "KeyDetect Corriendo" << endl;
					o_puntos.clear();
					KeyDetect(o_puntosfast, o_gray, o_keypoints, o_frame,n_threshold,b_nonmaxSuppression,nKeyPoints,masko);
					KeyPoint::convert(o_puntosfast, o_puntos);

					if (o_puntos.size()>10)
						cornerSubPix(o_gray, o_puntos, subpixwin, Size(-1,-1), termcrit);
				}else{
					size_t i, k;
					for(i=0, k=0; i<encontrado.size(); i++){
						if (encontrado[i]){
							circle(graficar,cvPoint(o_puntos[i].x,o_puntos[i].y),3,Scalar(0,255,0),-1,8,0);
							o_puntos[k] = o_puntos[i];
							o_puntos2[k++] = o_puntos2[i];
						}
						//else{
						//	circle(graficar,cvPoint(o_puntos[i].x,o_puntos[i].y),3,Scalar(0,0,255),1,7,0);
						//}
						}
						o_puntos.resize(k);
						o_puntos2.resize(k);
						nKeyPoints=k;

						for (uint i=0; i<k ; i++){
							movx=(o_puntos2[i].x-o_puntos[i].x)/k+movx;
							movy=(o_puntos2[i].y-o_puntos[i].y)/k+movy;
						}
						//std::cout<< "x= " << movx << std::endl;
						//std::cout<< "y= " << movy << std::endl;

						Ptr<plot::Plot2d> plot;
						plot = plot::Plot2d::create(movx, movy);

						plot->setPlotSize(500, 250);
						plot->setMaxX(500);
						plot->setMinX(-500);
						plot->setMaxY(500);
						plot->setMinY(-500);
						plot->setNeedPlotLine(false);
						plot->render(display);
						imshow("Plot Actual", display);

						movxh.push_back(movx);
						movyh.push_back(movy);

						Ptr<plot::Plot2d> plotH;
						plotH = plot::Plot2d::create(movxh, movyh);

						plotH->setPlotSize(500, 500);
						plotH->setMaxX(500);
						plotH->setMinX(-500);
						plotH->setMaxY(500);
						plotH->setMinY(-500);
						plotH->render(displayH);
						imshow("Plot History", displayH);

					}

				}else if (init){
					init=0;
				}

			text.str(std::string());
			textoVentana(graficar, text, nKeyPoints);
			imshow( c_key, graficar );
			o_gray2 = o_gray.clone(); // guardo imagen anterior
			std::swap(o_puntos, o_puntos2);
			video >> o_frame;		// nueva imagen
			graficar = o_frame.clone();
			cvtColor( o_frame, o_gray, CV_BGR2GRAY );

			cntfa=cntf;
			cntf=video.get(CAP_PROP_POS_FRAMES);
			if ((cntf==cntfa) && (cntf!=1))	{
				waitKey(0);
				destroyAllWindows();
				break;
			}
			}

		char key = (char)waitKey(1);
		if(key == 'p'){
			play=!play;
		}
		else if (key == 'x'){
			destroyAllWindows();
			break;
		}
		else if (key == 'r'){
			init=1;
			break;
		}

	}//while

}




