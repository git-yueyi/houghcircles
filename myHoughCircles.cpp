#include "img_head.h"

Mat imReconstr(Mat marker, Mat mask)
{
	int width = marker.cols;   // int width = marker.Width;
	int height = marker.rows;  // int height = marker.Height;
	
    queue<cv::Vec2i> myqueue;

	Mat res = marker.clone();

    // scan in raster order
	for (int r = 0; r < height; r++) {

		for (int c = 0; c < width; c++) {
			uchar max = 0;

			//8 connectivity
			int q_y = r - 1;
			int q_x = -1;
			if (q_y > 0)
			{
				max = (res.at<float>(q_y,c) > max) ? res.at<float>(q_y,c) : max;

				q_x = c + 1;
				if (q_x < width)
					max = (res.at<float>(q_y,q_x) > max) ? res.at<float>(q_y,q_x) : max;

				q_x = c - 1;
				if (q_x > 0)
					max = (res.at<float>(q_y,q_x) > max) ? res.at<float>(q_y,q_x) : max;
			}

			q_x = c - 1;
			if (q_x > 0)
				max = (res.at<float>(r,q_x) > max) ? res.at<float>(r,q_x) : max;

			max = (res.at<float>(r,c) > max) ? res.at<float>(r,c) : max;

			// cout << c << "  " << r << endl;
			res.at<float>(r,c) = (max > mask.at<float>(r,c)) ? mask.at<float>(r,c) : max;

		}
    }

    // scan in anti-raster order
    for (int r = height - 1; r >= 0; r--) 
	{
		for (int c = 0; c < width; c++) 
		{
			uchar max = 0;

			//8 connectivity
			int q_y = r + 1;
			int q_x = -1;
			if (q_y < height)
			{
				max = (res.at<float>(q_y,c) > max) ? res.at<float>(q_y,c) : max;

				q_x = c + 1;
				if (q_x < width)
					max = (res.at<float>(q_y,q_x) > max) ? res.at<float>(q_y,q_x) : max;

				q_x = c - 1;
				if (q_x > 0)
					max = (res.at<float>(q_y,q_x) > max) ? res.at<float>(q_y,q_x) : max;
			}

			q_x = c + 1;
			if (q_x < width)
				max = (res.at<float>(r,q_x) > max) ? res.at<float>(r,q_x) : max;

			max = (res.at<float>(r,c) > max) ? res.at<float>(r,c) : max;

			res.at<float>(r,c) = (max > mask.at<float>(r,c)) ? mask.at<float>(r,c) : max;

			q_y = r + 1;
			q_x = -1;
			if (q_y < height)
			{
				if ((res.at<float>(q_y,c) < res.at<float>(r,c)) && res.at<float>(q_y,c) < mask.at<float>(q_y,c))
				{
					Vec2i vec;
					vec[0] = c; vec[1] = r;
					myqueue.push(vec);
					continue;
				}

				q_x = c + 1;
				if (q_x < width)
				{
					if ((res.at<float>(q_y,q_x) < res.at<float>(r,c)) && res.at<float>(q_y,q_x) < mask.at<float>(q_y,q_x))
					{
						Vec2i vec;
						vec[0] = c; vec[1] = r;
						myqueue.push(vec);
						continue;
					}
				}

				q_x = c - 1;
				if (q_x > 0)
				{
					if ((res.at<float>(q_y,q_x) < res.at<float>(r,c)) && res.at<float>(q_y,q_x) < mask.at<float>(q_y,q_x))
					{
						Vec2i vec;
						vec[0] = c; vec[1] = r;
						myqueue.push(vec);
						continue;
					}
				}
			}

			q_x = c + 1;
			if (q_x < width)
			{
				if ((res.at<float>(r,q_x) < res.at<float>(r,c)) && res.at<float>(r,q_x) < mask.at<float>(r,q_x))
				{
					Vec2i vec;
					vec[0] = c; vec[1] = r;
					myqueue.push(vec);
					continue;
				}
			}
		}
    }



	while (myqueue.size() > 0)
    {
		Vec2i p = myqueue.front();
		myqueue.pop();

		int p_x = p[0];
		int p_y = p[1];
		int q_x = 0;
		int q_y = 0;

		for (int i = 0; i < 3; i++) 
		{
			for (int j = 0; j < 3; j++) 
			{
				q_y = p_y - 1;
				
				if (q_y > 0)
				{
					q_x = p_x - 1;
					if (q_x > 0)
					{
						if ((res.at<float>(q_y,q_x) < res.at<float>(p_y,p_x)) && (mask.at<float>(q_y,q_x) != res.at<float>(q_y,q_x)))
						{
							res.at<float>(q_y,q_x) = (mask.at<float>(q_y,q_x) > res.at<float>(p_y,p_x)) ? marker.at<float>(p_y,p_x) : mask.at<float>(q_y,q_x);
							Vec2i vec;
							vec[0] = q_x; vec[1] = q_y;
							myqueue.push(vec);
						}
					}
				}
			}
		}
    }

	return (res);

}

void imgradient(Mat & A,Mat & Gx,Mat & Gy,Mat & gradientImg){
	Mat hy = (Mat_<float>(3,3)<<1.0,2.0,1.0,
		0,0,0,
		-1.0,-2.0,-1.0);
	Mat hx = hy.t();

	Gy = Mat::zeros(A.size(),A.type());
	Gx = Mat::zeros(A.size(),A.type());

	//cout<<A<<endl;
	filter2D(A,Gy,A.depth(),hy,Point(-1,-1),0.0,BORDER_REPLICATE);

	filter2D(A,Gx,A.depth(),hx,Point(-1,-1),0.0,BORDER_REPLICATE);

	gradientImg.create(A.size(),A.type());

	//cout<<Gy<<endl;

	cv::sqrt((Gy.mul(Gy) + Gx.mul(Gx)),gradientImg);

	imshow("gradientImg",gradientImg);

	//waitKey(0);
	//cout<<gradientImg<<endl;
	return;
}

void getEdgeM_PIxels(Mat & gradientImg,double & edgeThresh,vector<int> &Ey,vector<int> &Ex){

	double Gmax = 0.0;//M 2.56  C:2.9

	cv::minMaxIdx(gradientImg,(double*)0,&Gmax);

	if(edgeThresh<0.0){
		cout<<"ãÐÖµ³ö´í"<<endl;
		return;
	}
	double t = Gmax * edgeThresh;//M 0.14  C:0.18

	myfind(gradientImg,t,Ey,Ex);

	cout<<Ex.size()<<" "<<Ey.size()<<endl;
}

void myfind(Mat & img,double & threshold,vector<int> &Ey,vector<int> &Ex){
	int nrow = img.rows;
	int ncol = img.cols;

	float * p;

	for(int i=0;i<nrow;i++){
		p  = img.ptr<float>(i);
		for(int j=0;j<ncol;j++){
			if(p[j]>=threshold){
				Ey.push_back(i);
				Ex.push_back(j);
				//cout<<"v"<<endl;
			}
		}
	}
}