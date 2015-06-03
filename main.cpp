#include "img_head.h"

#include <shappmgr.h>
#define M_PI 3.1415926
//
//void mySqrt_1C(Mat & A,Mat & B){
//	
//	gradientImg.create(A.size(),A.type());
//
//	for(int i= 0;i<A.rows;i++){
//		for(int j=0;j<A.cols;j++){
//			double B.data
//		}
//	}
//}


//
Mat accumarray( const Mat& _subs, const Mat& _val )
{
	Mat subs = _subs.clone();
	Mat val = _val.clone();

	if(subs.rows != 1)
		subs = subs.reshape(1, 1); // row array
	if(val.rows != 1)
		val = val.reshape(1, 1); //row array
	double minVal, maxVal;
	minMaxLoc(subs, &minVal, &maxVal);
	int maxIdx = (int)maxVal;
	CV_Assert(subs.size() == val.size() && maxIdx > 0);

	if(val.depth() != CV_32F)
		val.convertTo(val, CV_32FC1);

	Mat accum = Mat::zeros(1, maxIdx+1, CV_32FC1); //row array

	Mat mask; //comparing mask
	Mat ValsAtEachIdx;

	int idx;
	for(int i=0; i < val.cols; i++){
		idx = subs.at<int>(0, i);
		accum.at<float>(0, idx) += val.at<float>(0, i);
	}

	return accum;

} 
//

//
//
//
////这个有什么用？
void mysub2ind(cv::Size size,vector<int>&Ey,vector<int>&Ex,vector<int>&ind){
	for(int i=0;i<Ey.size();i++){
		int tmp = Ey.at(i) + (Ex.at(i) /*- 1*/)*size.height;
		ind.push_back(tmp);
	}
}
//

//
////get edge gradient

//
////calculate gradient 

template<class T>
void myaccumarray(Mat & subs,Mat & val,Mat_<T> & accumM,cv::Size size){
	
	accumM.create(size);
	accumM.setTo(Scalar::all(0));
	cout<<"channels: "<<val.channels()<<endl;
	for (int i=0;i<subs.rows;i++)
	{
		for (int c=0;c<val.channels();c++)
		{
			//cout<<(subs.at<int>(i,0))<<","<<(subs.at<int>(i,1))<<" "<<endl;
			//cout<<val.at<T>(i,0)[c]<<endl;
			accumM.at<T>((subs.at<int>(i,0)),(subs.at<int>(i,1)))[c] += val.at<T>(i,0)[c];
			//cout<<(subs.at<int>(i,0))<<","<<(subs.at<int>(i,1))<<" "<<accumM.at<T>((subs.at<int>(i,0)),(subs.at<int>(i,1)))[c]<<endl;
		}
	}
}


void myChaccum(cv::Mat & A,vector<int>& radiusRange,int method,int objPolarity,double edgeThresh,cv::Mat & accumMatrix,cv::Mat &  gradientImg){
	double maxNumElemNHoodMat = 1e6;

	//Mat accumMatrix_2d(accumMatrix.size(),CV_64FC2);

	Mat Gx,Gy;

	//A gradientImg 都是32float
	imgradient(A,Gx,Gy,gradientImg);
	
	//cout<<gradientImg<<endl;

	vector<int> Ey,Ex,idxE;

	getEdgeM_PIxels( gradientImg,edgeThresh,Ey,Ex);

	mysub2ind(gradientImg.size(),Ey,Ex,idxE);

	
	vector<float> radiusfRange;

	float radius_tmp = radiusRange[0];
	float steplength = 0.5;
	while(radius_tmp<=radiusRange[1]){
		radiusfRange.push_back(radius_tmp);
		radius_tmp += steplength;
	}
	vector<float> RR;
	switch(objPolarity){

	case bright:
		RR = radiusfRange;
		break;
	case dark:
		RR = radiusfRange ;
		for(int i=0;i<RR.size();i++){
			RR.at(i) *= -1.0;
		}
		break;
	default:
		break;
	}
	//
	//

	vector<float> lnR;
	vector<float> phi;
	Mat Opca(cv::Size(radiusfRange.size(),1),CV_32FC2);
	Mat w0(cv::Size(radiusfRange.size(),1),CV_32FC2);

	float phi_t= 0.0;
	float opca_R = 0.0;
	float opca_I= 0.0;
	float w0_R = 0.0;
	float w0_I = 0.0;

	int i = 0;

	switch(method){
	case twostage:
		
		for(i=0;i<radiusfRange.size();i++){
			w0.at<cv::Vec2f>(0,i)[0] = 1/(2*M_PI*radiusfRange.at(i));
			w0.at<cv::Vec2f>(0,i)[1]  =0.0;
		}
		break;
	case phasecode:

		for(i=0;i<radiusfRange.size();i++){
			float  w_t = log(radiusfRange.at(i));
			lnR.push_back(w_t);
		}
		
		for(int i=0;i<radiusfRange.size();i++){
			phi_t = ((lnR.at(i) - lnR.at(0)) / ( lnR.at(radiusfRange.size()-1) - lnR.at(0))*2*M_PI)-M_PI;
			phi.push_back(phi_t);
			opca_R = cos(phi_t);//exp(sqrt(-1)*phi_t);//虚数
			opca_I = sin(phi_t);

			Opca.at<cv::Vec2f>(0,i)[0] = opca_R;
			Opca.at<cv::Vec2f>(0,i)[1] = opca_I;

			w0_R  = opca_R / (2*M_PI*radiusfRange.at(i));
			w0_I  = opca_I / (2*M_PI*radiusfRange.at(i));
			
			w0.at<cv::Vec2f>(0,i)[0] = w0_R;
			w0.at<cv::Vec2f>(0,i)[1] = w0_I;

		}
		break;

	default:
			break;	
	}
	//cout<<"radiusfRange : "<<radiusfRange.size()<<endl;
	int xcStep = floor(maxNumElemNHoodMat/RR.size());
	//
	int lenE = Ex.size();

	accumMatrix.setTo(Scalar::all(0));
	//cout<<accumMatrix<<endl;
	vector<int> Ex_chunk;
	vector<int> Ey_chunk;
	vector<int> idxE_chunk;
	vector<bool> rows_to_keep;
	Mat xc;
	Mat yc;
	Mat w;
	Mat inside;
	
	Mat xc_new;
	Mat yc_new;
	Mat w_new;
	Mat inside_new;
	Mat m_yxc;
	Mat m_wval;
	vector<int> xc_vec,yc_vec;
	vector<float> w_vecR,w_vecI;
	for(int i =0;i<lenE;i += xcStep){
		
		Ex_chunk.clear();
		Ey_chunk.clear();
		idxE_chunk.clear();
		for(int j=i;j<min((i+xcStep-1),(lenE));j++){
			Ex_chunk.push_back(Ex.at(j)); 
			Ey_chunk.push_back(Ey.at(j)); 
			idxE_chunk.push_back(idxE.at(j));
		}//j
		
		xc.release();
		yc.release();
		w.release();
		inside.release();

		xc.create(cv::Size(RR.size(),Ex_chunk.size()),CV_32SC1);
		yc.create(cv::Size(RR.size(),Ex_chunk.size()),CV_32SC1);
		w.create(cv::Size(RR.size(),Ex_chunk.size()),CV_32FC2);
		inside.create(cv::Size(RR.size(),Ex_chunk.size()),CV_8UC1);


		int M = A.rows;
		int N = A.cols;
	

		for (int t=0;t<idxE_chunk.size();t++)
		{
			int x = idxE_chunk.at(t) / gradientImg.rows ;
			
			int y = idxE_chunk.at(t)%(gradientImg.rows );
			//cout<<t<<": "<<Ex_chunk.at(t)<<endl;
			for (int j=0;j<xc.cols;j++)
			{
	
					//============================---------------------------------------------------===========
					//============================================================
				double fxc = (-1)*RR.at(j) * Gx.at<float>(y,x)/gradientImg.at<float>(y,x) + Ex_chunk.at(t);
					int txc = /*min*/(RoundEx(fxc)/*,479*/);
					//cout<<gradientImg<<endl;
					xc.at<int>(t,j) = txc;
					
					int tyc = /*min*/(RoundEx((-1)*RR.at(j) * Gy.at<float>(y,x)/gradientImg.at<float>(y,x) + Ey_chunk.at(t))/*,639*/);
					
					yc.at<int>(t,j) = tyc;
					
					w.at<cv::Vec2f>(t,j)[0] = w0.at<cv::Vec2f>(0,j)[0];
					w.at<cv::Vec2f>(t,j)[1] = w0.at<cv::Vec2f>(0,j)[1];

					//Determine which edge M_PIxel votes are wirhin the image domain
					bool inside_t = (xc.at<int>(t,j) >= 0) && (xc.at<int>(t,j)<N) && (yc.at<int>(t,j) >= 0) && (yc.at<int>(t,j) < M);

					inside.at<uchar>(t,j) = inside_t?1:0;//
			}//j
			
		}//t
		//cout<<xc<<endl;

	//	//Keep rows that have at least one candidate position inside the domain
		int sum_ture = 0;
		rows_to_keep.clear();
		for(int t=0;t<inside.rows;t++){
			bool rows_to_keep_tmp = false;
			for (int j=0;j<inside.cols;j++){
				
				if(inside.at<uchar>(t,j)>0){
					rows_to_keep_tmp = true;
					sum_ture++;
					break;
				}
			}//j
			rows_to_keep.push_back(rows_to_keep_tmp);
		
		}//t
		xc_new.release();
		yc_new.release();
		w_new.release();
		inside_new.release();

		xc_new.create(cv::Size(xc.cols,sum_ture),CV_32SC1);
		yc_new.create(cv::Size(yc.cols,sum_ture),CV_32SC1);
		w_new.create(cv::Size(w.cols,sum_ture),w.type());
		inside_new.create(cv::Size(inside.cols,sum_ture),inside.type());

		for (int t=0,f=0;t<xc.rows;t++)
		{
			if(rows_to_keep.at(t)){
				for (int j=0;j<xc.cols;j++)
				{
					xc_new.at<int>(f,j) = xc.at<int>(t,j);
					yc_new.at<int>(f,j) = yc.at<int>(t,j);
					w_new.at<cv::Vec2f>(f,j)[0] = w.at<cv::Vec2f>(t,j)[0];
					w_new.at<cv::Vec2f>(f,j)[1] = w.at<cv::Vec2f>(t,j)[1];
					inside_new.at<uchar>(f,j) = inside.at<uchar>(t,j);

				}
				f++;
			}
			
		}//t

		//accumulate the votes in the parameter plane
		//1.向量化
		xc_vec.clear();
		yc_vec.clear();
		w_vecR.clear();
		w_vecI.clear();
		m_yxc.release();
		m_wval.release();
		for (int j=0;j<inside_new.cols;j++)
		{
			for (int t =0;t<inside_new.rows;t++)
			{
				if((inside_new.at<uchar>(t,j))==1){
					xc_vec.push_back(xc_new.at<int>(t,j));
					yc_vec.push_back(yc_new.at<int>(t,j));
					w_vecR.push_back(w_new.at<cv::Vec2f>(t,j)[0]);
					w_vecI.push_back(w_new.at<cv::Vec2f>(t,j)[1]);
				}
			}//t
			
		}//j
		
		m_yxc.create(cv::Size(2,xc_vec.size()),CV_32SC1);
		m_wval.create(cv::Size(1,xc_vec.size()),CV_32FC2);
		for(int j=0;j<yc_vec.size();j++){
			m_yxc.at<int>(j,0) = yc_vec.at(j);
			m_yxc.at<int>(j,1) = xc_vec.at(j);
			m_wval.at<cv::Vec2f>(j,0)[0] = w_vecR.at(j);
			m_wval.at<cv::Vec2f>(j,0)[1] = w_vecI.at(j);
			//cout<<w_vecR.at(j)<<","<<w_vecI.at(j)<<endl;
		}
		Mat_<cv::Vec2f> accumR;
		myaccumarray(m_yxc,m_wval,accumR,A.size());
		//cout<<"accumR"<<endl;
		accumMatrix += accumR;
		//cout<<accumMatrix<<endl;

	}
}

//Mat morphReconstruct(Mat marker, Mat mask)
//{
//	Mat dst;
//	cv::min(marker,mask, dst);
//	dilate(dst, dst, Mat());
//	cv::min(dst, mask, dst);
//	Mat temp1 = Mat(marker.size(), CV_8UC1);
//	Mat temp2 = Mat(marker.size(), CV_8UC1);
//	do
//	{
//		dst.copyTo(temp1);
//		dilate(dst, dst, Mat());
//		cv::min(dst, mask, dst);
//		compare(temp1, dst, temp2, CV_CMP_NE);
//	}
//	while (sum(temp2).val[0] != 0);
//	return dst;
//}
//
//

int vdist(Point p1,Point p2){
	return RoundEx(sqrt(pow((double)p1.x - p2.x,2) + pow((double)p1.y - p2.y, 2)));
}

//int imregionalmax(Mat input, int nLocMax, float threshold, float minDistBtwLocMax, Mat locations)
//	 {
//		     Mat scratch = input.clone();
//		     int nFoundLocMax = 0;
//		     for (int i = 0; i < nLocMax; i++) {
//			         Point location;
//			         double maxVal;
//			         minMaxLoc(scratch, NULL, &maxVal, NULL, &location);
//			         if (maxVal > threshold) {
//				             nFoundLocMax += 1;
//				             int row = location.y;
//				             int col = location.x;
//				             locations.at<int>(i,0) = row;
//				             locations.at<int>(i,1) = col;
//				             int r0 = (row-minDistBtwLocMax > -1 ? row-minDistBtwLocMax : 0);
//				             int r1 = (row+minDistBtwLocMax < scratch.rows ? row+minDistBtwLocMax : scratch.rows-1);
//				             int c0 = (col-minDistBtwLocMax > -1 ? col-minDistBtwLocMax : 0);
//				             int c1 = (col+minDistBtwLocMax < scratch.cols ? col+minDistBtwLocMax : scratch.cols-1);
//				             for (int r = r0; r <= r1; r++) {
//					                 for (int c = c0; c <= c1; c++) {
//						                     if (vdist(Point(r, c),Point(row, col)) <= minDistBtwLocMax) {
//							                         scratch.at<float>(r,c) = 0.0;
//							                     }
//						                 }
//					             }
//				         } else {
//					             break;
//				         }
//				   }
//		    return nFoundLocMax;
//		 }

//
//
//
////s = regionprops(bw,accumMatrix,'weightedcentroid'); % in opencv, it is the centers of the contours. 
//
//void regionprops(int bw,Mat & accumMatrix,vector<Point> & centers){
//	vector<vector<Point>> contours;
//
//	findContours(accumMatrix,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
//	for (int i=0;i<contours.size();i++)
//	{
//		Rect r0 = boundingRect(Mat(contours[i]));
//		centers.push_back(Point(((r0.X + r0.width)/2),((r0.Y + r0.height)/2)));
//	}
//	
//
//}
//
//
void myChcenters(Mat & accumMatrix, double suppThreshold,vector<Point> & centers,vector<float> & metric){

	int medFiltSize = 5;

	//cout<<accumMatrix<<endl;
	Mat Hd(accumMatrix.rows,accumMatrix.cols,CV_32FC1);
	//abs(accumMatrix)
	for (int i=0;i<accumMatrix.rows;i++)
	{
		for (int j=0;j<accumMatrix.cols;j++)
		{
			Hd.at<float>(i,j) = sqrt(pow(accumMatrix.at<cv::Vec2f>(i,j)[0],2) + pow(accumMatrix.at<cv::Vec2f>(i,j)[1],2));
		}
	}
	
	
	imshow("Hd0",Hd);
	waitKey(1);
	//cout<<Hd<<endl;
	//pre-process the accumulator array
	if(Hd.rows > medFiltSize&&Hd.cols>medFiltSize){
		 medianBlur(Hd,Hd,3);
	}
	imshow("Hd1",Hd);
	//Hd = morphReconstruct(Hd-suppThreshold,Hd);
	
	Hd = imReconstr(Hd-suppThreshold,Hd);
	//Hd = Hd.mul(255);
	imshow("Hd2",Hd);
	Mat bw  = Hd - imReconstr(Hd-1,Hd);
	bw =  bw.mul(255);
	//cout<<bw<<endl;
	imshow("bww",bw);
	waitKey(0);
	//Mat location
	//int bw = imregionalmax(Hd,Hd.rows * Ｈd.cols,0.001,10,);

	////regionprops
	//vector<Point> centers;
	//regionprops(bw,accumMatrix,centers);

	//vector<int> Centers_y,Centers_x;
	//for (int i=0;i<centers.size();i++)
	//{
	//	Centers_x.push_back(centers.at(i).X);
	//	Centers_y.push_back(centers.at(i).Y);
	//}

	//vector<int> Hd_idx;
	//mysub2ind(cv::Size(Hd.cols,Hd.rows),Centers_y,Centers_x,Hd_idx);

	//vector<double> metric;
	//for(int i=0;i<Hd_idx.size();i++){
	//	int y = Hd_idx.at(i)/(Hd.cols -1);
	//	int x = Hd_idx.at(i)%(Hd.cols -1);

	//	metric.push_back(Hd.at<double>(y,x));
	//}


	//double tmp;
	//Point p_tmp;
	//for (int j=0;j<metric.size()-1;j++)
	//{
	//	for (int i=0;i<metric.size()-1-j;i++)
	//	{
	//		if (metric.at(i)<metric.at(i+1))
	//		{
	//			tmp = metric.at(i);
	//			metric.at(i) = metric.at(i+1);
	//			metric.at(i+1) = tmp;

	//			p_tmp = centers.at(i);
	//			centers.at(i) = centers.at(i+1);
	//			centers.at(i+1) = p_tmp;
	//		}
	//	}
	//}
	//metric.sort();
	//centers.sort();

}

//cv::Mat1i
//void myChradiiphcode(vector<Point> centers,Mat & accumMatrix,vector<int> & radiusRange,vector<double>&r_estimated){
//	
//	//判断是不是实数
//	//Decode the phase to get the radius estimate
//	vector<int> Cen_y,Cen_x;
//	for (int i=0;i<centers.size();i++)
//	{
//		Cen_x.push_back(centers.at(i).X);
//		Cen_y.push_back(centers.at(i).Y)
//	}
//	vector<int> accumMatrix_ind;
//	mysub2ind(accumMatrix.size(),Cen_y,Cen_x,accumMatrix_ind);
//
//	vector<double> cenPhase;
//	vector<double> lnR;
//	for (int i=0;i<accumMatrix_ind.size();i++)
//	{
//		int y = accumMatrix_ind.at(i)/(accumMatrix.rows -1);
//		int x =  accumMatrix_ind.at(i)%(accumMatrix.rows );
//		cenPhase.push_back(atan2(accumMatrix.at<cv::Vec2d>(y,x)[1],accumMatrix.at<cv::Vec2d>(y,x)[0]));
//	}
//	for (int i=0;i<radiusRange.size();i++)
//	{
//		lnR.push_back(log((double)(radiusRange.at(i))));
//	}
//	for (int i=0;i<cenPhase.size();i++)
//	{
//		double r = exp(((cenPhase.at(i) + M_M_PI)/(2*M_M_PI)*(lnR.at(1) - lnR.at(0))) + lnR.at(0));
//
//	}
//
//}
//
//void meshgrid(const cv::Mat &xgv, const cv::Mat &ygv,
//               cv::Mat1i &X, cv::Mat1i &Y)
//{
//		  cv::repeat(xgv.reshape(1,1), ygv.total(), 1, X);
//		  cv::repeat(ygv.reshape(1,1).t(), 1, xgv.total(), Y);
//}
//void radial_histogram(Mat & gradientImg,int xc,int yc,int r1,int r2,Mat &h,Mat&bins_){
//	int M = gradientImg.rows;
//	int N = gradientImg.cols;
//
//	cv::Mat1i xx,yy;
//	Mat M_mat(M,1,CV_32S),N_mat(N,1,CV_32S);
//	for (int i = 0;i<M;i++)
//	{
//		M_mat.at<int>(i,0) = i;
//	}
//	for (int i=0;i<N;i++)
//	{
//		N_mat.at<int>(i,0) = i;
//	}
//	Mat1i xx,yy,dx,dy;
//	meshgrid(M_mat,N_mat,xx,yy);
//
//	dx = xx - xc;
//	dy = yy - yc;
//
//	cv::Mat r ;
//	cv::sqrt(cv::abs(dx).mul(cv::abs(dx)) + cv::abs(dy).mul(cv::abs(dy)),r);
//
//	Mat r_int(r.size,CV_32SC1);
//	vector<int> r_vec;
//	for (int i=0;i<r.rows;i++)
//	{
//		for (int j=0;j<r.cols;j++)
//		{
//			r_int.at<int>(i,j) = RoundEx(r.at<double>(i,j));
//		}
//	}
//	//RoundEx()
//	for (int j=0;j<r_int.cols;j++)
//	{
//		for (int i=0;i<r_int.rows;i++)
//		{
//			r_vec.push_back(r_int.at<int>(i,j));
//		}
//	}
//
//	vector<double> gradientImg_vec;
//
//	for (int j=0;j<gradientImg.cols;j++)
//	{
//		for (int i=0;i<gradientImg.rows;i++)
//		{
//			gradientImg_vec.push_back(gradientImg.at<double>.at(i,j));
//		}
//	}
//
//	vector<bool> keep;
//	for(int i=0;i<r_vec.size();i++){
//		if(r.at(i)>=r1||r.at(i)<=r2){
//			keep.push_back(true);
//		}else
//			keep.push_back(false);
//		
//	}
//	vector<int>::iterator it_r = r_vec.begin();
//	vector<double>::iterator it_grad = gradientImg_vec.begin();
//
//	int min_rvec =-1,max_rvec=-1;
//	for(int i=0;i<keep.size();i++){
//		if(!keep.at(i)){
//			r_vec.erase(it_r);
//			gradientImg_vec.erase(it_grad);
//		}else{
//			++it_r;
//			++it_grad;
//			if(min_rvec==-1||min_rvec>it_r){
//				min_rvec = it_r;
//
//			}
//			if(max_rvec==-1||max_rvec<it_r){
//				max_rvec = it_r;
//			}
//		}
//	}
//	vector<int> bins;
//	
//	for (int i = min_rvec;i<max_rvec;i++)
//	{
//		bins.push_back(i);
//	}
//
//	Mat para1(1,r_vec.size(),CV_32FC1);
//
//	for(int i=0;i<r.size();i++){
//		para1.at<int>(0,i) = r_vec.at(i) + 1-bins.at(0);
//	}
//	Mat para2(1,gradientImg_vec.size(),CV_32FC1);
//
//	for (int i=0;i<gradientImg_vec.size();i++)
//	{
//		para2.at<int>(0,i) = gradientImg_vec.at(i);
//	}
//	h = accumarray(para1,para2);
//	bins_(bins);
//	bins_ =bins_.reshape(0,1);
//	cv::divide(h,bins_.mul(2*M_PI),h);
//
//
//}
//
//
//void chardii(vector<Point> centers,Mat & gradientImg,vector<int>& radiusRange){
//	vector<double> r_estimated;
//	int M = gradientImg.rows;
//	int N = gradientImg.cols;
//	
//	for (int i=0;i<centers.size();i++)
//	{
//		int left = std::max(floor(centers.at(i).X - radiusRange.at(1)),1);
//		int right = min(ceil(centers.at(i).X - radiusRange.at(1)),N);
//		int top = max(floor(centers.at(i).Y - radiusRange.at(1)),1);
//		int bottom = min(ceil(centers.at(i).Y - radiusRange.at(1)),M);
//		Rect r(left,top,right-left,bottom-top);
//		Mat h,bins;
//		radial_histogram(gradientImg(r),centers.at(i).Y-left+1,centers.at(i).X-top+1,radiusRange.at(0),radiusRange.at(1),h,bins);
//		int min,max,min_idx,max_idx;
//		cv::minMaxIdx(h,&min,&max,&min_idx,&max_idx);
//		r_estimated.at(i) = bins.at(max_idx);
//
//	}
//}
//
//



void myImfindcircles(cv::Mat & A,vector<int>& radiusRange,
	int objPolarity,double sensitivity,int method){
		

		Mat A_bw;

		//double转换为uchar
		Mat A_uchar;

		A.convertTo(A_uchar,CV_8UC1,255);
		if (A_uchar.type()==CV_8UC1)
		{
			cout<<"A_uchar.type()==CV_8UC1"<<endl;
		}
		imshow("A_uchar",A_uchar);

		double edgeThresh = threshold(A_uchar,A_bw,0,255,CV_THRESH_OTSU)/(double)255.0;

		cout<<"edgeThresh :"<<edgeThresh<<endl;
		namedWindow("bw",0);
		imshow("bw",A_bw);

		

		Mat accumMatrix(A.size(),CV_32FC2),gradientImg;

		myChaccum(A,radiusRange,method,objPolarity,edgeThresh,accumMatrix,gradientImg);

		//如果accumMatrix是全零。。

		//Estimate the centers
		double accumThresh = 1 - sensitivity;
		vector<Point> centers; 
		vector<float> r_estimated;
		vector<float> metric;
		//cout<<accumMatrix<<endl;
		myChcenters(accumMatrix,accumThresh,centers,metric);
		cout<<endl;
		//if(centers.size() == 0){
		//	return;
		//}

		////Retain circles with metric value greater than threshold correspoding to accumulatorThreshold
		//vector<int> idx2Keep;
		//for (int i=0;i<metric.size();i++)
		//{
		//	if(metric.at(i)>accumThresh){
		//		idx2Keep.push_back(metric.at(i));
		//	}
		//}
		//vector<Point> center_need;
		//for (int i =0;i<idx2Keep.size();i++)
		//{
		//	center_need.push_back(centers.at(idx2Keep.at(i)));
		//}
		//if(center_need.size()==0){
		//	return;

		//}

		//Mat r_estimated;
		////estimate radii
		//if(radiusRange.size()==1){
		//	r_estimated.push_back(radiusRange.at(0));
		//}else{
		//	switch(method){
		//	case phasecode:
		//		myChradiiphcode(centers,accumMatrix,radiusRange,r_estimated);
		//		break;
		//	case twostage:

		//		break;
		//	default:
		//		break;
		//	}
		//}


}


int main(void){
	
	Mat img_src,img_resize,img_gray;

	float scale = 0.2;

	float sigma = 50.0;

	img_src = imread("4.jpg",CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);

	
	Size dsize = Size(480/*RoundEx(img_src.cols*scale)*/,640/*RoundEx(img_src.rows*scale)*/);
	cout<<dsize.height<<","<<dsize.width<<endl;
	/*img_resize = Mat(dsize,img_src.type());*/
	if(img_src.type() == CV_8UC3){
		cout<<"img_src.type  CV_8UC3"<<endl;

	}
	


	img_src.convertTo(img_src,CV_32FC3);
	if(img_src.type()==CV_32FC3){
		cout<<"img_src.type CV_32FC3"<<endl;
	}
	img_src = img_src.mul((double)1/255);

	cv::resize(img_src,img_resize,Size(0,0),(double) 640/img_src.rows,(double) 640/img_src.rows,INTER_LINEAR);
	

	cvtColor(img_resize,img_gray,CV_BGR2GRAY);
	if (img_gray.type()==CV_32FC1)
	{
		cout<<"img_gray.type()==CV_32FC1"<<endl;
	}




	//cout<<img_gray.depth()<<endl;
	namedWindow("img_gray",1);
	
	imshow("img_gray",img_gray);
	Mat ff,dst;

	imgausshpf(img_gray,sigma,ff);
	//为计算出的虚数增加一个通道

	imshow("ff",ff);
	
	myguassfilter(img_gray,ff,dst);

	namedWindow("dst",1);
	imshow("dst",dst);
	
	vector<int> radiusRange;
	radiusRange.push_back(10);
	radiusRange.push_back(30);
	//the type of dst if double
	myImfindcircles(dst,radiusRange,bright,0.9,phasecode);

	waitKey(0);
	return 0 ;
}



int main12(void){
	Mat img1 = imread(".\\ct\\ct4.bmp",0);
	Mat img2 = imread(".\\ct\\ct4.bmp",0);
	resize(img2,img2,cv::Size(100,200));
	resize(img1,img1,cv::Size(100,200));
	
	Mat gray_1,gray_2;
	threshold(img1,gray_1,0,255,CV_THRESH_OTSU);
	threshold(img2,gray_2,0,0,CV_THRESH_BINARY);
	//img2.zeros(img1.size(),img2.type());

	//cout<<gray_1<<endl;
	int result_cols = img1.cols - img2.cols + 1;
	int result_rows = img1.rows - img2.cols + 1;
	Mat result(result_rows,result_cols,CV_32FC1);
	clock_t t_start = clock();
	LARGE_INTEGER  large_interger;
	double dff;
	__int64 c1,c2;
	QueryPerformanceFrequency(&large_interger);
	dff = large_interger.QuadPart;
	QueryPerformanceCounter(&large_interger);
	c1 = large_interger.QuadPart;



	//matchTemplate(gray_1,gray_2,result,0);
	//int samilar = 0;
	//for(int i =0;i<gray_1.rows;i++){
	//	for(int j=0;j<gray_1.cols;j++){
	//		if(gray_1.at<uchar>(i,j) & gray_2.at<uchar>(i,j)){
	//			samilar++;
	//		}
	//	}
	//}

	QueryPerformanceCounter(&large_interger);
	c2 = large_interger.QuadPart;
	printf("本机高精度计时器频率%lf\n", dff);    
    printf("第一次计时器值%I64d 第二次计时器值%I64d 计时器差%I64d\n", c1, c2, c2 - c1);    
    printf("计时%lf毫秒\n", (c2 - c1) * 1000 / dff); 
	//cout<<"samilar: "<<samilar<<endl;
	imshow("img1",gray_1);
	imshow("img2",gray_2);
	waitKey(0);
	return 0;


}