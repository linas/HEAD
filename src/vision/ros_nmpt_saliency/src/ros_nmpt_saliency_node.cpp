#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cv_bridge/cv_bridge.h>

#include <iostream>
#include <sstream>
#include "BlockTimer.h"
#include "FastSalience.h"
#include "LQRPointTracker.h"
#include "NMPTUtils.h"

#include "geometry_msgs/Point.h"
#include "ros_nmpt_saliency/targets.h"

#include <vector>

using namespace std;
using namespace cv;

	Size imSize(320,240);
	BlockTimer bt;
	FastSalience salTracker;
	LQRPointTracker salientSpot(2);
	vector<double> lqrpt(2,.5);
	Mat im, im2, viz, sal ;
	ros::Publisher pub;
	geometry_msgs::Point pt;
	bool debug_mode;

	double degreeof(double x, double y);
    double L,R, T,B;
    double timel =0.0;
    int counter = 0;
	const int turn_around_time =1;
    vector< vector<double> > degreeHandler(10000, vector<double>(5,0));


double degreeof(double x, double y)
{
    timel+=bt.getCurrTime(0);

    int flag=0;
    double delay;
    double turnaround;
    double passDeg;

    if (not counter)
    {
        degreeHandler[counter][0]=x;
        degreeHandler[counter][1]=y;
        degreeHandler[counter][2]=timel;
        passDeg = degreeHandler[counter][3] = 1;

        // Reference for Turn_Around
        degreeHandler[counter][4] = timel;
        ++counter;
    }
    else
    {
        for(int i=0;i<counter;i++)
        {
            L=degreeHandler[i][0]- 0.03; R=degreeHandler[i][0]+ 0.03;
            T=degreeHandler[i][1]- 0.03; B=degreeHandler[i][1]+ 0.03;
            //
            //if(degreeHandler[i][0] == x  && degreeHandler[i][1] == y) //use this for default comparison
            //{
            if ((x>=L && y>=T) && (x<=R && y>=T) && (x>=L && y<=B) && (x<=R && y<=B))
            {
                flag=1;
                turnaround = timel - degreeHandler[i][4];
                // cout<<"Turn Around"<<turnaround<<endl;
                delay = timel - degreeHandler[i][2] ;
                if  (turnaround < turn_around_time)
                {
                    //degree normalization
                    passDeg = degreeHandler[i][3]+= (0.03846/turnaround);//need some boosting mechanism: excessive degree reduction is noticed during turnaroud t.
                }
                else
                {
                if(delay<=0.25){  degreeHandler[i][3]+= (0.03846/turnaround); } //checking for exsitance of salience at this point
                else { degreeHandler[i][3]=1+ (degreeHandler[i][3]/(turnaround/0.03846));}
                passDeg=degreeHandler[i][3];
                degreeHandler[i][4] = timel;
                }

                degreeHandler[i][2] = timel;
                cout<<x<<","<<y<<"  Category "<<degreeHandler[i][0]<<","<<degreeHandler[i][1]<<" Delay: "<<delay<<" Degree: "<<degreeHandler[i][3]<<endl;
                break;
            }
            else { continue;  }
        }

        if (not flag)
        {
            degreeHandler[counter][0]=x;
            degreeHandler[counter][1]=y;
            delay = timel - degreeHandler[counter][2];
            degreeHandler[counter][2]=timel;
            passDeg = degreeHandler[counter][3]=1;
            degreeHandler[counter][4] = timel;

            cout<<"New "<<degreeHandler[counter][0]<<", "<<degreeHandler[counter][1]<<" iTime: "<<degreeHandler[counter][2]<<" Degree "<<degreeHandler[counter][3]<<endl;
            ++counter;
        }
    }

    return passDeg;
}



void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	double saltime, tottime, degree;
	//	capture >> im2;
   	cv_bridge::CvImagePtr cv_ptr;
	//sensor_msgs::Image salmap_;
   //CvBridge bridge;
   try
   {
	 cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
	 im2=cv_ptr->image;
	 double ratio = imSize.width * 1. / im2.cols;
			resize(im2, im, Size(0,0), ratio, ratio, INTER_NEAREST);
     //cvShowImage("view", im);
     //
     	viz.create(im.rows, im.cols*2, CV_32FC3);

		bt.blockRestart(0);
		vector<KeyPoint> pts;
		salTracker.detect(im, pts);
		saltime = bt.getCurrTime(0) ;

		salTracker.getSalImage(sal);


		double min, max;
		Point minloc, maxloc;
		minMaxLoc(sal, &min, &max, &minloc, &maxloc);

		lqrpt[0] = maxloc.x*1.0 / sal.cols;
		lqrpt[1] = maxloc.y*1.0 / sal.rows;

		salientSpot.setTrackerTarget(lqrpt);

		Mat vizRect = viz(Rect(im.cols,0,im.cols, im.rows));
		cvtColor(sal, vizRect, CV_GRAY2BGR);

		vizRect = viz(Rect(0, 0, im.cols, im.rows));
		im.convertTo(vizRect,CV_32F, 1./256.);
		/*
		for (size_t i = 0; i < pts.size(); i++) {
			circle(vizRect, pts[i].pt, 2, CV_RGB(0,255,0));
		}
		*/


		salientSpot.updateTrackerPosition();
		lqrpt = salientSpot.getCurrentPosition();
		pt.x=lqrpt[0];
		pt.y=lqrpt[1];

		degree = degreeof(pt.x, pt.y);
		pt.z=degree; //for testing: add float64 type message in targets.msg and pulsih this one to that msg type

//		pt.z=0;
		ros_nmpt_saliency::targets trg;
		trg.positions.push_back(pt);
		pub.publish(trg);
		/*
		circle(vizRect, Point(lqrpt[0]*sal.cols, lqrpt[1]*sal.rows), 6, CV_RGB(0,0,255));
		circle(vizRect, Point(lqrpt[0]*sal.cols, lqrpt[1]*sal.rows), 5, CV_RGB(0,0,255));
		circle(vizRect, Point(lqrpt[0]*sal.cols, lqrpt[1]*sal.rows), 4, CV_RGB(255,255,0));
		circle(vizRect, Point(lqrpt[0]*sal.cols, lqrpt[1]*sal.rows), 3, CV_RGB(255,255,0));
		*/
		vizRect = viz(Rect(im.cols,0,im.cols, im.rows));
		cvtColor(sal, vizRect, CV_GRAY2BGR);
		////if (usingCamera) flip(viz, viz, 1);

		tottime = bt.getCurrTime(1);
		bt.blockRestart(1);
		/*
		stringstream text;
		text << "FastSUN: " << (int)(saltime*1000) << " ms ; Total: " << (int)(tottime*1000) << " ms.";

		putText(viz, text.str(), Point(20,20), FONT_HERSHEY_SIMPLEX, .33, Scalar(255,0,255));


		imshow("FastSUN Salience", viz);
		*/
	if (debug_mode){
	imshow("view",viz);
	waitKey(1);
	}

   }
   catch (...)
   {
     ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
   }
}

int main(int argc, char **argv)
   {
     ros::init(argc, argv, "image_listener");
     ros::NodeHandle nh;
	nh.getParam("debug",debug_mode);
     if (debug_mode) cvNamedWindow("view");
     cvStartWindowThread();
     image_transport::ImageTransport it(nh);
     image_transport::Subscriber sub = it.subscribe("/camera/image_raw", 1, imageCallback);


     pub = nh.advertise<ros_nmpt_saliency::targets>("/nmpt_saliency_point", 50);

     salientSpot.setTrackerTarget(lqrpt);
     bt.blockRestart(1);

     ros::spin();
     //cvDestroyWindow("view");
   }






