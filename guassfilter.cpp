
#include "img_head.h"
double RoundEx(const double& dInput)
{
	double dIn = dInput - int(dInput);
	if (dIn >= 0.5)    //???
	{
		return int(dInput + 0.5);
	} 
	else
	{
		return int(dInput);
	}
}
void imgausshpf(Mat & src,float sigma,Mat & out){
	out.create(src.size(),CV_32FC1);
	int M = src.rows;
	int N = src.cols;

	for (int i  = 0;i<M;i++)
	{
		for (int j = 0;j<N;j++)
		{
			out.at<float>(i,j) = 1 - exp(-(pow((float)(i-M/2),2) + pow((float)(j-N/2),2))/(2*pow((float)sigma,2)));
		}
	}

}
void myguassfilter(Mat & src,Mat & ff,Mat & dst){
	/*Mat padded;*/
	////扩展为DFT最佳运算尺寸
	//int opw = getOptimalDFTSize(src.cols);
	//int oph = getOptimalDFTSize(src.rows);
	//copyMakeBorder(src,padded,0,oph-src.rows,0,opw-src.cols,BORDER_CONSTANT,Scalar::all(0));

	//imshow("padded",padded);



	Mat planes[] = {Mat_<float>(src),cv::Mat::zeros(src.size(),CV_32F)};
	Mat complexI;

	merge(planes,2,complexI);


	dft(complexI,complexI);
	//cout<<complexI<<endl;
	//傅里叶变换都的数据中心化
	int cx = complexI.cols/2;
	int cy = complexI.rows/2;

	Mat q0(complexI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant 
	Mat q1(complexI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(complexI, Rect(0, cy, cx,  cy));  // Bottom-Left
	Mat q3(complexI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	//滤波.....
	//...

	Mat  complexIs[] = {Mat::zeros(complexI.size(),CV_32FC1),Mat::zeros(complexI.size(),CV_32FC1)};
	split(complexI,complexIs);



	complexIs[0] = complexIs[0].mul(ff);
	complexIs[1] = complexIs[1].mul(ff);

	//cout<<ff<<endl;
	merge(complexIs,2,complexI);


	//反傅里叶变换

	//中心变换
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	//逆变换
	Mat iff;

	dft(complexI,iff,DFT_INVERSE);
	vector<Mat> iffs;
	split(iff,iffs);
	magnitude(iffs[0],iffs[1],dst);

	//dst也变为了32位float类型
	normalize(dst,dst,0,1,CV_MINMAX);
	

	//cout<<dst<<endl;
	//dst = dst.mul(1/255.0);
	//cout<<dst<<endl;
	//normalize(dst,dst,0,1,CV_MINMAX);
	//ifft = ifft(Rect(0,0,src.cols ,src.rows ));

	//得到赋值图像，取了对数
	//split(complexI,planes);

	//magnitude(planes[0],planes[1],planes[0]);////I1,I2都是1维向量,dst=sqrt(x(I)^2+y(I)^2)

	//Mat log_img = planes[0];

	////傅里叶变换的幅度值范围大到不适合在屏幕上显示。高值在屏幕上显示为白点
	////而低值显示为黑点，高值的变化无法有效分变，为了在屏幕上凸显出高低变化的连续性，我们可以用对数尺度来代替线性尺度
	//log_img += Scalar::all(1);

	//log(log_img,log_img);

	//log_img = log_img(Rect(0,0,log_img.cols & -2,log_img.rows & -2);//不大于cols的最大偶数

	////rearrange the quadrants of fft img
	//int cx = log_img.cols/2;
	//int cy = log_img.rows/2;

	//Mat q0(log_img, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant 
	//Mat q1(log_img, Rect(cx, 0, cx, cy));  // Top-Right
	//Mat q2(log_img, Rect(0, cy, cx, cy));  // Bottom-Left
	//Mat q3(log_img, Rect(cx, cy, cx, cy)); // Bottom-Right
	//
	//Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	//q0.copyTo(tmp);
	//q3.copyTo(q0);
	//tmp.copyTo(q3);

	//q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	//q2.copyTo(q1);
	//tmp.copyTo(q2);
	//
	//normalize(log_img, log_img, 0, 1, CV_MINMAX);

	//dst = log_img.clone();

}

//bool mySobel(const Mat& image,Mat& result,int TYPE, bool MAX_STOP)
//{
//	if(image.channels()!=1)
//		return false;
//
//	// 系数设置
//	int kx(0);
//	int ky(0);
//	if( TYPE==1 ){
//		kx=0;ky=1;
//	}
//	else if( TYPE==2 ){
//		kx=1;ky=0;
//	}
//	else if( TYPE==3 ){
//		kx=1;ky=1;
//	}
//	else
//		return false;
//
//	// 设置mask
//	float mask[3][3]={{1,2,1},{0,0,0},{-1,-2,-1}};
//	Mat y_mask=Mat(3,3,CV_32F,mask)/8;
//	Mat x_mask=y_mask.t(); // 转置
//
//	// 计算x方向和y方向上的滤波
//	Mat sobelX,sobelY;
//	filter2D(image,sobelX,CV_32F,x_mask);
//	filter2D(image,sobelY,CV_32F,y_mask);
//	sobelX=abs(sobelX);
//	sobelY=abs(sobelY);
//	// 梯度图
//	Mat gradient=kx*sobelX.mul(sobelX)+ky*sobelY.mul(sobelY);
//
//	// 计算阈值
//	int scale=4;
//	double cutoff=scale*mean(gradient)[0];
//
//	result.create(image.size(),image.type());
//	result.setTo(0);
//	for(int i=1;i<image.rows-1;i++)
//	{
//		float* sbxPtr=sobelX.ptr<float>(i);
//		float* sbyPtr=sobelY.ptr<float>(i);
//		float* prePtr=gradient.ptr<float>(i-1);
//		float* curPtr=gradient.ptr<float>(i);
//		float* lstPtr=gradient.ptr<float>(i+1);
//		uchar* rstPtr=result.ptr<uchar>(i);
//		// 阈值化
//		for(int j=1;j<image.cols-1;j++)
//		{
//			// 极大值抑制
//			if (MAX_STOP){
//				if( curPtr[j]>cutoff && (
//					(sbxPtr[j]>kx*sbyPtr[j] && curPtr[j]>curPtr[j-1] && curPtr[j]>curPtr[j+1]) ||
//					(sbyPtr[j]>ky*sbxPtr[j] && curPtr[j]>prePtr[j] && curPtr[j]>lstPtr[j]) )) 
//					rstPtr[j]=255;
//			}
//			else {
//				if( curPtr[j]>cutoff)
//					rstPtr[j]=255;
//			}
//		}
//	}
//
//	return true;
//}
//int main1(void){
//
//	Mat im = imread("IMG_1255.jpg",0);
//
//	resize(im, im, Size(0,0), 0.2, 0.2);
//
//	if( im.empty() ) 
//	{ 
//		cout << "Can not load image." << endl; 
//		return -1; 
//	}
//
//	Mat result; 
//	mySobel(im, result, 3, 1); 
//
//	namedWindow("resultim", CV_WINDOW_AUTOSIZE);
//	imshow("resultim", result); 
//	waitKey(0); 
//
//	return 0;
//
//}