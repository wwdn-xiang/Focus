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
REGISTER_OP("HePostprocession")
    .Attr("T: {int8, uint8}")	//数据的类型为int8/uint8
    .Input("mask: T")			//mask的类型为T
	.Output("output:int32")		//output类型为int32
    .SetShapeFn([](shape_inference::InferenceContext* c){
        c->set_output(0,c->Matrix(-1,4));	//设置输出的shape为 [-1,4]
		return Status::OK();
    });

template <typename Device, typename T>
class HePostprocessionOp: public OpKernel {
    public:
	//设置变量的类型
     using contours_t = vector<vector<cv::Point>>;	
     using contours2i_t = vector<cv::Point2i>;
     using contours4i_t = vector<vector<int>>;

	public:
		explicit HePostprocessionOp(OpKernelConstruction* context) : OpKernel(context) {
		}

		void Compute(OpKernelContext* context) override
        {
        const Tensor &_mask = context->input(0);	//获取输入tensor    
        auto          mask = _mask.flat<T>();		//Return the tensor data as an Eigen::Tensor of the data type and a specified shape 【H*W】.
        const auto    width       = _mask.dim_size(1);	//获取tensor width
        const auto    height      = _mask.dim_size(0);	//获取tensor height

        OP_REQUIRES(context, _mask.dims() == 2, errors::InvalidArgument("mask data must be 2-dimensional"));//检查shape
        auto posMask = new T[width*height];	//新建 posMask l类型为T 

        memset(posMask, 0, width * height);	//赋值0

        for (int i = 0; i < width * height; i++) {
            unsigned char value = mask.data()[i];	//将输入的tensor中的值赋给posMask	
            if (value == 1)
            {
                posMask[i] = 0XFF;
            }
        }

        const auto pos_points = do_dedup_pos(height,width,posMask);
        delete[] posMask;

        int len = pos_points.size();
        TensorShape  output_shape  = {len,4};
        Tensor      *output_tensor = nullptr;

        OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));
        auto tensor = output_tensor->template tensor<int,2>();
        Eigen::Tensor<int,1,Eigen::RowMajor> tmp_data(4);

        auto i = 0;
        for(i=0; i<pos_points.size(); ++i) {
            tmp_data(0) = pos_points[i][0];
            tmp_data(1) = pos_points[i][1];
            tmp_data(2) = pos_points[i][2];
            tmp_data(3) = pos_points[i][3];
            tensor.chip(i,0) = tmp_data;
        }
        }
	static contours4i_t do_dedup_pos(int height,int width,T* mask)
	{

		contours_t        contours;
		contours2i_t      res;

		vector<cv::Vec4i> hierarchy;
		vector<int> points(4) ;
		vector<vector<int>> output ;
		cv::Mat srcImg(width, height, CV_8UC1, (void*)mask);
		cv::threshold(srcImg, srcImg, 128, 255, cv::THRESH_BINARY);
		cv::Mat row =srcImg.rowRange(400,401).clone();
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));//20
		cv::erode(srcImg, srcImg, kernel);
		cv::medianBlur(srcImg, srcImg, 11);
        cv::findContours(srcImg.clone(), contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
		if (!contours.empty()){
			//轮廓的个数
			for (int i=0;i<contours.size();i++){
			    if(contours[i].size()>=100)
			    {
			        for(int j=0;j<contours[i].size();j++)
			        {
                        points.push_back(contours[i][j].x);
                        points.push_back(contours[i][j].y);
                        points.push_back(i);
                        if (hierarchy[i][3]==-1)
                        {
                            //外轮廓类别为1
                            points.push_back(1);
                        }
                        else
                        {
                            points.push_back(0);
                        }
					output.push_back(points);
					points.clear();
				    }
			    }
			}
		}
		return output;
	}

};

REGISTER_KERNEL_BUILDER(Name("HePostprocession").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), HePostprocessionOp<CPUDevice, uint8_t>);
REGISTER_KERNEL_BUILDER(Name("HePostprocession").Device(DEVICE_CPU).TypeConstraint<int8_t>("T"), HePostprocessionOp<CPUDevice, int8_t>);
