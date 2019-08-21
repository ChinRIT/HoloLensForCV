#pragma once
namespace ComputeOnDevice
{
	class DepthPvMapper
	{
	public:
		// create and initialize an instance
		DepthPvMapper(HoloLensForCV::SensorFrame ^ depthFrame);
		DepthPvMapper();
		~DepthPvMapper();
		// initialize depth image space coordinates to unit plane mapping (reverse depth cam space projection transform, 2D->3D)
		// this transform doesn't depend on actual depth values and can be done once per sensor stream activation
		void Init(HoloLensForCV::SensorFrame ^ depthFrame);
		cv::Mat MapDepthToPV(HoloLensForCV::SensorFrame ^ pvFrame, HoloLensForCV::SensorFrame ^ depthFrame, int depthRangeFrom, int depthRangeTo);
	private:
		cv::Mat _imageToCameraMapping;
		cv::Mat createImageToCamMapping(HoloLensForCV::SensorFrame^ depthFrame);
		cv::Mat get4DPointCloudFromDepth(HoloLensForCV::SensorFrame ^ depthFrame, int depthRangeFrom, int depthRangeTo);
	};
}
