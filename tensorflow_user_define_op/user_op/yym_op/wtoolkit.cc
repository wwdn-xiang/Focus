#include <stdio.h>
#include <cfloat>
#include <list>
#include <cstdlib>
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
#include<opencv2/opencv.hpp>
#include "tensorflow/core/framework/common_shape_fns.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;
using namespace std;
typedef Eigen::ThreadPoolDevice CPUDevice;
/*
 * mask:[H,W] (int8/uint8)
 * output:[X,3] {(x,y,type),....}
 */
REGISTER_OP("SegP63Post")
    .Attr("T: {int8, uint8}")
    .Input("mask: T")
	.Output("output:uint8")
    .SetShapeFn(shape_inference::UnchangedShape);

template <typename Device, typename T>
class SegP63PostOp: public OpKernel {
    public:

     using contours_t = vector<vector<cv::Point>>;
     using contours2i_t = vector<cv::Point2i>;
	public:
		explicit SegP63PostOp(OpKernelConstruction* context) : OpKernel(context) {
		}

		void Compute(OpKernelContext* context) override
        {
        const Tensor &_mask = context->input(0);
        auto          mask = _mask.flat<T>();
        const auto    width       = _mask.dim_size(1);
        const auto    height      = _mask.dim_size(0);

        OP_REQUIRES(context, _mask.dims() == 2, errors::InvalidArgument("mask data must be 2-dimensional"));

        cv::Mat img(height, width, CV_8UC1, (void*)mask.data());

        auto posMask = new T[width*height];
        
        memset(posMask, 0, width * height);

        for (int i = 0; i < width * height; ++i) {
            unsigned char value = mask.data()[i];
            if (value == 1)
                posMask[i] = 0XFF;
        }

        do_dedup_pos(img,posMask);
        
        TensorShape  output_shape  = {height,width};
        Tensor      *output_tensor = nullptr;

        OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));
        auto tensor = output_tensor->template tensor<uint8_t,2>();
        for (auto i=0;i<height;++i) {
            for (auto j=0;j<width;++j) {
                tensor(i,j)=posMask[i*width+j];
                }
        }

        delete[] posMask;
        //cout<<"finish_hp"<<endl;
        }

        
        static void do_dedup_pos(const cv::Mat& img,T* mask)
        {
            const auto        width     = img.cols;
            const auto        height    = img.rows;
            int               iterations = 4;
            contours_t        contours;

            vector<cv::Vec4i> hierarchy;
            cv::Mat srcImg(width, height, CV_8UC1, (void*)mask);
            cv::findContours(srcImg.clone(), contours, hierarchy, CV_RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
            cv::Mat drawImg(srcImg.size(), CV_8UC1, cv::Scalar(0));
            cv::drawContours(drawImg, contours, -1, cv::Scalar(255), cv::FILLED);
            drawImg.copyTo(srcImg);
        }
};




REGISTER_OP("SegDedup")
    .Attr("T: {int8, uint8}")
    .Input("mask: T")
	.Output("output:int32")
    .SetShapeFn([](shape_inference::InferenceContext* c){
        c->set_output(0,c->Matrix(-1,4));
		return Status::OK();
    });

template <typename Device, typename T>
class SegDedupOp: public OpKernel {
    public:
     using result_item_t = tuple<int,int,int>;
     using result_t = vector<result_item_t>;
     using contours_t = vector<vector<cv::Point>>;
     using contours2i_t = vector<cv::Point2i>;
     using contours4i_t = vector<vector<int>>;

	public:
		explicit SegDedupOp(OpKernelConstruction* context) : OpKernel(context) {
		}

		void Compute(OpKernelContext* context) override
        {
        const Tensor &_mask = context->input(0);
        auto          mask = _mask.flat<T>();
        const auto    width       = _mask.dim_size(1);
        const auto    height      = _mask.dim_size(0);

        OP_REQUIRES(context, _mask.dims() == 2, errors::InvalidArgument("mask data must be 2-dimensional"));

        cv::Mat img(height, width, CV_8UC1, (void*)mask.data());

        //auto negMask = new T[width*height];
        auto posMask = new T[width*height];

        //memset(negMask, 0, width * height);
        memset(posMask, 0, width * height);

        for (int i = 0; i < width * height; ++i) {
            unsigned char value = mask.data()[i];
            if (value == 1)
                posMask[i] = 0XFF;
        }

        //const auto neg_points = do_dedup_neg(img,negMask);
        const auto pos_points = do_dedup_pos(img,posMask);

        //delete[] negMask;
        delete[] posMask;

        int len = pos_points.size();
        TensorShape  output_shape  = {len,4};
        Tensor      *output_tensor = nullptr;

        OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));
        auto tensor = output_tensor->template tensor<int,2>();
        Eigen::Tensor<int,1,Eigen::RowMajor> tmp_data(4);

        auto i = 0;
        for(i=0; i<pos_points.size(); ++i) {
            //auto& p = pos_points[i];
            tmp_data(0) = pos_points[i][0];
            tmp_data(1) = pos_points[i][1];
            tmp_data(2) = pos_points[i][2];
            tmp_data(3) = pos_points[i][3];
            tensor.chip(i,0) = tmp_data;
        }
        }





	static contours4i_t do_dedup_pos(const cv::Mat& img,T* mask)
	{
		const auto        width     = img.cols;
		const auto        height    = img.rows;
		contours_t        contours;
		contours2i_t      res;
		vector<cv::Vec4i> hierarchy;
		vector<int> points(4) ;
		vector<vector<int>> output ;
		cv::Mat srcImg(width, height, CV_8UC1, (void*)mask);
		//cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
		//cv::morphologyEx(srcImg, srcImg, cv::MORPH_OPEN, kernel);
		//cv::threshold(srcImg, srcImg, 128, 255, cv::THRESH_BINARY);
		//cv::findContours(srcImg.clone(), contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
                cv::findContours(srcImg.clone(), contours, hierarchy, CV_RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
				

		if (!contours.empty()){
			//轮廓的个数
			for (int i=0;i<contours.size();i++){
				for(int j=0;j<contours[i].size();j++){
					//points.push_back(contours[i][j][0][0]);
					//points.push_back(contours[i][j][0][1]);
					points.push_back(contours[i][j].x);
					points.push_back(contours[i][j].y);
					points.push_back(i);
					//外轮廓类别为1
					points.push_back(1);

					output.push_back(points);
					points.clear();

				}	
						
			}
		}
                //cout<<points.size()<<endl;
		cout<<output.size()<<endl;
		return output;
	}

};





REGISTER_KERNEL_BUILDER(Name("SegP63Post").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), SegP63PostOp<CPUDevice, uint8_t>);
REGISTER_KERNEL_BUILDER(Name("SegP63Post").Device(DEVICE_CPU).TypeConstraint<int8_t>("T"), SegP63PostOp<CPUDevice, int8_t>);
REGISTER_KERNEL_BUILDER(Name("SegDedup").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), SegDedupOp<CPUDevice, uint8_t>);
REGISTER_KERNEL_BUILDER(Name("SegDedup").Device(DEVICE_CPU).TypeConstraint<int8_t>("T"), SegDedupOp<CPUDevice, int8_t>);
//REGISTER_KERNEL_BUILDER(Name("SegDedup").Device(DEVICE_CPU), SegDedupOp);
