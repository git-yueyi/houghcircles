#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <cor.h>
#include <cxcore.h>
#include <math.h>
#include <queue>
#include <ctime>

using namespace std;
using namespace cv;


#define M_PI        3.14159265358979323846

enum Emethod{twostage,phasecode};

enum EobjPolrity{bright,dark};

double RoundEx(const double& dInput);
void imgausshpf(Mat & src,float sigma,Mat & out);
void myguassfilter(Mat & src,Mat & ff,Mat & dst);


void imgradient(Mat & A,Mat & Gx,Mat & Gy,Mat & gradientImg);

void myfind(Mat & img,double & threshold,vector<int> &Ey,vector<int> &Ex);
void getEdgeM_PIxels(Mat & gradientImg,double & edgeThresh,vector<int> &Ey,vector<int> &Ex);

Mat imReconstr(Mat marker, Mat mask);