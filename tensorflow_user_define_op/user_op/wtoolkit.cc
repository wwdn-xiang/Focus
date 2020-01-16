#include <stdio.h>
#include <cfloat>
#include <list>
#include <cstdlib>
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
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
REGISTER_OP("SegDedup")
    .Attr("T: {int8, uint8}")
    .Input("mask: T")
	.Output("output:int32")
    .SetShapeFn([](shape_inference::InferenceContext* c){
        c->set_output(0,c->Matrix(-1,3));
		return Status::OK();
    });

template <typename Device, typename T>
class SegDedupOp: public OpKernel {
    public:
     using result_item_t = tuple<int,int,int>;
     using result_t = vector<result_item_t>;
     using contours_t = vector<vector<cv::Point>>;
     using contours2i_t = vector<cv::Point2i>;
	public:
		explicit SegDedupOp(OpKernelConstruction* context) : OpKernel(context) {
		}

		void Compute(OpKernelContext* context) override
        {
		//获取tensor数据
        const Tensor &_mask = context->input(0);
		//将tensor转为一维
        auto          mask = _mask.flat<T>();
		//获取tensor的w,h
        const auto    width       = _mask.dim_size(1);
        const auto    height      = _mask.dim_size(0);
		//检查mask的shape 必须为2
        OP_REQUIRES(context, _mask.dims() == 2, errors::InvalidArgument("mask data must be 2-dimensional"));
		// 创建数据容器img
        cv::Mat img(height, width, CV_8UC1, (void*)mask.data());
		//T类型，尺寸分别为width*height
        auto negMask = new T[width*height];
        auto posMask = new T[width*height];
		//复制给新建的，negMask，posMask
        memset(negMask, 0, width * height);
        memset(posMask, 0, width * height);
		
		//根据value分别复制给对应索引的negMask，posMask，赋值均为0XFF=255
        for (int i = 0; i < width * height; ++i) {
            unsigned char value = mask.data()[i];
            if (value == 1)
                negMask[i] = 0xFF;
            else if (value == 2)
                posMask[i] = 0XFF;
        }

        const auto neg_points = do_dedup_neg(img,negMask);
        const auto pos_points = do_dedup_pos(img,posMask);

        delete[] negMask;
        delete[] posMask;

        TensorShape  output_shape  = {pos_points.size()+neg_points.size(),3};
        Tensor      *output_tensor = nullptr;

        OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));
        auto tensor = output_tensor->template tensor<int,2>();
        Eigen::Tensor<int,1,Eigen::RowMajor> tmp_data(3);

        auto i = 0;
        for(i=0; i<pos_points.size(); ++i) {
            auto& p = pos_points[i];
            tmp_data(0) = p.x;
            tmp_data(1) = p.y;
            tmp_data(2) = 2;
            tensor.chip(i,0) = tmp_data;
        }
        for(auto j=0; j<neg_points.size(); ++j,++i) {
            auto& p = neg_points[j];
            tmp_data(0) = p.x;
            tmp_data(1) = p.y;
            tmp_data(2) = 1;
            tensor.chip(i,0) = tmp_data;
        }

        }

        static contours2i_t do_dedup_neg(const cv::Mat& img,T* mask)
        {
			//获取图像的w,h
            const auto        width     = img.cols;
            const auto        height    = img.rows;
            contours_t        contours;
            contours2i_t      res;
            vector<cv::Vec4i> hierarchy;
            cv::Mat srcImg(width, height, CV_8UC1, (void*)mask);

            cv::medianBlur(srcImg, srcImg, 5);

            cv::threshold(srcImg, srcImg, 128, 255, cv::THRESH_BINARY);

            cv::findContours(srcImg.clone(), contours, hierarchy, CV_RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

            if (!contours.empty()) {
                std::vector<cv::Moments> mu(contours.size());
                for (int i = 0; i < contours.size(); i++)
                    mu[i] = cv::moments(contours[i], false);

                for (int i = 0; i < contours.size(); i++) {
                    if (mu[i].m00 > 0) {
                        cv::Point2i p(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
                        cv::circle(img, p, 3, cv::Scalar(255, 0, 0), 5);
                        res.push_back(p);
                    }
                }
            }
            return res;
        }
        static contours2i_t do_dedup_pos(const cv::Mat& img,T* mask)
        {
            const auto        width     = img.cols;
            const auto        height    = img.rows;
            contours_t        contours;
            contours2i_t      res;
            vector<cv::Vec4i> hierarchy;
            cv::Mat srcImg(width, height, CV_8UC1, (void*)mask);

            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            cv::morphologyEx(srcImg, srcImg, cv::MORPH_OPEN, kernel);

            cv::threshold(srcImg, srcImg, 128, 255, cv::THRESH_BINARY);
            cv::findContours(srcImg.clone(), contours, hierarchy, CV_RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

            if (!contours.empty()) {
                std::vector<cv::Moments> mu(contours.size());
                for (int i = 0; i < contours.size(); i++)
                    mu[i] = cv::moments(contours[i], false);

                for (int i = 0; i < contours.size(); i++) {
                    if (mu[i].m00 > 0) {
                        cv::Point2i p(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
                        cv::circle(img, p, 3, cv::Scalar(255, 0, 0), 5);
                        res.push_back(p);
                    }
                }
            }
            return res;
        }
};
REGISTER_KERNEL_BUILDER(Name("SegDedup").Device(DEVICE_CPU).TypeConstraint<uint8_t>("T"), SegDedupOp<CPUDevice, uint8_t>);
REGISTER_KERNEL_BUILDER(Name("SegDedup").Device(DEVICE_CPU).TypeConstraint<int8_t>("T"), SegDedupOp<CPUDevice, int8_t>);
