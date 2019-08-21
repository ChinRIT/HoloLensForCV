#include "pch.h"
#include "DepthPvMapper.h"

namespace ComputeOnDevice
{
	cv::Mat DepthPvMapper::createImageToCamMapping(HoloLensForCV::SensorFrame^ depthFrame) {
		cv::Mat imageToCameraMapping = cv::Mat(depthFrame->SoftwareBitmap->PixelHeight, depthFrame->SoftwareBitmap->PixelWidth, CV_32FC2, cv::Scalar::all(0));
		for (int x = 0; x < depthFrame->SoftwareBitmap->PixelWidth; ++x) {
			for (int y = 0; y < depthFrame->SoftwareBitmap->PixelWidth; ++y) {
				Windows::Foundation::Point uv = { float(x), float(y) };
				Windows::Foundation::Point xy(0, 0);
				if (depthFrame->SensorStreamingCameraIntrinsics->MapImagePointToCameraUnitPlane(uv, &xy)) {
					imageToCameraMapping.at<cv::Vec2f>(y, x) = cv::Vec2f(xy.X, xy.Y);
				}
			}
		}
		return imageToCameraMapping;
	}

	DepthPvMapper::DepthPvMapper(HoloLensForCV::SensorFrame^ depthFrame)
	{
		Init(depthFrame);
	}
	
	DepthPvMapper::DepthPvMapper()
	{
	}

	DepthPvMapper::~DepthPvMapper()
	{
	}

	void DepthPvMapper::Init(HoloLensForCV::SensorFrame^ depthFrame) {
		_imageToCameraMapping = createImageToCamMapping(depthFrame);
	}

	static cv::Mat floatMToCvMat(Windows::Foundation::Numerics::float4x4 in) {
		cv::Mat res = cv::Mat(4, 4, CV_32F);
		res.at<float>(0, 0) = in.m11;
		res.at<float>(0, 1) = in.m12;
		res.at<float>(0, 2) = in.m13;
		res.at<float>(0, 3) = in.m14;

		res.at<float>(1, 0) = in.m21;
		res.at<float>(1, 1) = in.m22;
		res.at<float>(1, 2) = in.m23;
		res.at<float>(1, 3) = in.m24;

		res.at<float>(2, 0) = in.m31;
		res.at<float>(2, 1) = in.m32;
		res.at<float>(2, 2) = in.m33;
		res.at<float>(2, 3) = in.m34;

		res.at<float>(3, 0) = in.m41;
		res.at<float>(3, 1) = in.m42;
		res.at<float>(3, 2) = in.m43;
		res.at<float>(3, 3) = in.m44;
		return res;
	}
	
	static cv::Vec4f vecDotM(cv::Vec4f vec, Windows::Foundation::Numerics::float4x4 m) {
		cv::Vec4f res;
		res.val[0] = vec.val[0] * m.m11 + vec.val[1] * m.m21 + vec.val[2] * m.m31 + vec.val[3] * m.m41;
		res.val[1] = vec.val[0] * m.m12 + vec.val[1] * m.m22 + vec.val[2] * m.m32 + vec.val[3] * m.m42;
		res.val[2] = vec.val[0] * m.m13 + vec.val[1] * m.m23 + vec.val[2] * m.m33 + vec.val[3] * m.m43;
		res.val[3] = vec.val[0] * m.m14 + vec.val[1] * m.m24 + vec.val[2] * m.m34 + vec.val[3] * m.m44;
		return res;
	}

	// Projects depth sensor data to PV frame and returns Mat with measured distances in mm in PV frame coordinates
	cv::Mat DepthPvMapper::MapDepthToPV(HoloLensForCV::SensorFrame^ pvFrame, HoloLensForCV::SensorFrame^ depthFrame,
		int depthRangeFrom, int depthRangeTo) {
		int pvWidth = pvFrame->SoftwareBitmap->PixelWidth;
		int pvHeight = pvFrame->SoftwareBitmap->PixelHeight;
		cv::Mat res(pvHeight, pvWidth, CV_16UC1, cv::Scalar::all(0));
		cv::Mat pointCloud = get4DPointCloudFromDepth(depthFrame, depthRangeFrom, depthRangeTo);
		cv::Mat depthImage;
		rmcv::WrapHoloLensSensorFrameWithCvMat(depthFrame, depthImage);
		auto depthFrameToOrigin = depthFrame->FrameToOrigin;
		auto depthCamViewTransform = depthFrame->CameraViewTransform;
		auto pvFrameToOrigin = pvFrame->FrameToOrigin;
		auto pvCamViewTransform = pvFrame->CameraViewTransform;
		auto pvCamProjTransform = pvFrame->CameraProjectionTransform;
		Windows::Foundation::Numerics::float4x4 depthCamViewTransformInv;
		Windows::Foundation::Numerics::float4x4 pvFrameToOriginInv;
		if (!Windows::Foundation::Numerics::invert(depthCamViewTransform, &depthCamViewTransformInv) ||
			!Windows::Foundation::Numerics::invert(pvFrameToOrigin, &pvFrameToOriginInv))
		{
			dbg::trace(L"Can't map depth to pv, invalid transform matrices");
			return res;
		}
		// build point cloud -> pv view transform matrix
		auto depthPointToWorld = depthCamViewTransformInv * depthFrameToOrigin;
		auto depthPointToPvFrame = depthPointToWorld * pvFrameToOriginInv;
		auto depthPointToCamView = depthPointToPvFrame * pvCamViewTransform;
		auto depthPointToImage = depthPointToCamView * pvCamProjTransform;

		// loop through point cloud and estimate coordinates
		for (int x = 0; x < pointCloud.cols; ++x) {
			for (int y = 0; y < pointCloud.rows; ++y) {
				cv::Vec4f point = pointCloud.at<cv::Vec4f>(y, x);
				if (point.val[0] == 0 && point.val[1] == 0 && point.val[2] == 0)
					continue;
				// project point
				cv::Vec4f projPoint = vecDotM(point, depthPointToImage);
				cv::Vec3f normProjPoint = cv::Vec3f(projPoint.val[0] / projPoint.val[3], projPoint.val[1] / projPoint.val[3], projPoint.val[2] / projPoint.val[3]);
				// convert point with central origin and y axis up to pv image coordinates
				if (normProjPoint.val[0] > -1 && normProjPoint.val[0] < 1 && normProjPoint.val[1] > -1 && normProjPoint.val[1] < 1)
				{
					int imgX = (int)(pvWidth * (normProjPoint.val[0] + 1) / 2.0);
					int imgY = (int)(pvHeight * (1 - (normProjPoint.val[1] + 1) / 2.0));
					res.at<ushort>(imgY, imgX) = (ushort)depthImage.at<ushort>(y, x);
				}
			}
		}
		return res;
	}

	cv::Mat DepthPvMapper::get4DPointCloudFromDepth(HoloLensForCV::SensorFrame^ depthFrame, int depthRangeFrom, int depthRangeTo) {
		cv::Mat depthImage;
		rmcv::WrapHoloLensSensorFrameWithCvMat(depthFrame, depthImage);
		cv::Mat pointCloud(depthImage.rows, depthImage.cols, CV_32FC4, cv::Scalar::all(0));
		for (int x = 0; x < depthImage.cols; ++x) {
			for (int y = 0; y < depthImage.rows; ++y) {
				if (depthImage.at<unsigned short>(y, x) < depthRangeFrom || depthImage.at<unsigned short>(y, x) > depthRangeTo) {
					continue;
				}
				auto camPoint = _imageToCameraMapping.at<cv::Vec2f>(y, x);
				Windows::Foundation::Point uv = { float(x), float(y) };
				Windows::Foundation::Point xy(camPoint.val[0], camPoint.val[1]);
				cv::Point3f d(xy.X, xy.Y, 1);
				d *= -(depthImage.at<unsigned short>(y, x) / 1000.0) * (1 / sqrt(d.x*d.x + d.y*d.y + 1));
				pointCloud.at<cv::Vec4f>(y, x) = cv::Vec4f(d.x, d.y, d.z, 1);
			}
		}
		return pointCloud;
	}
		
}