//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#include "pch.h"

#include "AppMain.h"

namespace ComputeOnDevice
{
    AppMain::AppMain(
        const std::shared_ptr<Graphics::DeviceResources>& deviceResources)
        : Holographic::AppMainBase(deviceResources)
        , _selectedHoloLensMediaFrameSourceGroupType(
            HoloLensForCV::MediaFrameSourceGroupType::HoloLensResearchModeSensors)
        , _holoLensMediaFrameSourceGroupStarted(false)
        , _undistortMapsInitialized(false)
        , _isActiveRenderer(false)
    {
		_depthMapper = nullptr;
    }

    void AppMain::OnHolographicSpaceChanged(
        Windows::Graphics::Holographic::HolographicSpace^ holographicSpace)
    {
        //
        // Initialize the HoloLens media frame readers
        //
        StartHoloLensMediaFrameSourceGroup();
    }

    void AppMain::OnSpatialInput(
        _In_ Windows::UI::Input::Spatial::SpatialInteractionSourceState^ pointerState)
    {
        Windows::Perception::Spatial::SpatialCoordinateSystem^ currentCoordinateSystem =
            _spatialPerception->GetOriginFrameOfReference()->CoordinateSystem;

        if (!_isActiveRenderer)
        {
            _currentSlateRenderer =
                std::make_shared<Rendering::SlateRenderer>(
                    _deviceResources);
            _slateRendererList.push_back(_currentSlateRenderer);

            // When a Pressed gesture is detected, the sample hologram will be repositioned
            // two meters in front of the user.
            _currentSlateRenderer->PositionHologram(
                pointerState->TryGetPointerPose(currentCoordinateSystem));

            _isActiveRenderer = true;
        }
        else
        {
            //// Freeze frame
            //_visualizationTextureList.push_back(_currentVisualizationTexture);
            //_currentVisualizationTexture = nullptr;
            //_isActiveRenderer = false;
        }
    }

	void AppMain::OnUpdate(
        _In_ Windows::Graphics::Holographic::HolographicFrame^ holographicFrame,
        _In_ const Graphics::StepTimer& stepTimer)
    {
        UNREFERENCED_PARAMETER(holographicFrame);

        dbg::TimerGuard timerGuard(
            L"AppMain::OnUpdate",
            30.0 /* minimum_time_elapsed_in_milliseconds */);


		Windows::Perception::Spatial::SpatialCoordinateSystem^ currentCoordinateSystem =
			_spatialPerception->GetOriginFrameOfReference()->CoordinateSystem;

		if (!_isActiveRenderer)
		{
			_currentSlateRenderer =
				std::make_shared<Rendering::SlateRenderer>(
					_deviceResources);
			_slateRendererList.push_back(_currentSlateRenderer);
			_isActiveRenderer = true;
		}

		auto pointer = 
		Windows::UI::Input::Spatial::SpatialPointerPose::TryGetAtTimestamp(currentCoordinateSystem, holographicFrame->CurrentPrediction->Timestamp);



		// When a Pressed gesture is detected, the sample hologram will be repositioned
		// two meters in front of the user.
		_currentSlateRenderer->PositionHologram(pointer);


        //
        // Update scene objects.
        //
        // Put time-based updates here. By default this code will run once per frame,
        // but if you change the StepTimer to use a fixed time step this code will
        // run as many times as needed to get to the current step.
        //
        
        for (auto& r : _slateRendererList)
        {
            r->Update(
                stepTimer);
        }
        
        
        //
        // Process sensor data received through the HoloLensForCV component.
        //
        if (!_holoLensMediaFrameSourceGroupStarted)
        {
            return;
        }

        HoloLensForCV::SensorFrame^ latestFrame;
		HoloLensForCV::SensorFrame^ latestDepthFrame;

        latestFrame = _holoLensMediaFrameSourceGroup->GetLatestSensorFrame(HoloLensForCV::SensorType::PhotoVideo);
		latestDepthFrame = _depthMediaFrameSourceGroup->GetLatestSensorFrame(HoloLensForCV::SensorType::ShortThrowToFDepth);

		if (nullptr == latestFrame || nullptr == latestDepthFrame)
		{
			return;
		}

		if (nullptr == _depthMapper)
		{
			_depthMapper = new DepthPvMapper(latestDepthFrame);
		}

        if (_latestSelectedCameraTimestamp.UniversalTime == latestFrame->Timestamp.UniversalTime)
        {
            return;
        }

        _latestSelectedCameraTimestamp = latestFrame->Timestamp;

        cv::Mat wrappedImage;
		cv::Mat wrappedDepthImage;
		cv::Mat pvDepth;

        rmcv::WrapHoloLensSensorFrameWithCvMat(latestFrame, wrappedImage);
		rmcv::WrapHoloLensSensorFrameWithCvMat(latestDepthFrame, wrappedDepthImage);

		pvDepth = _depthMapper->MapDepthToPV(latestFrame, latestDepthFrame, 20, 3000, 5);
		
		auto depthProjRgb = cv::Mat(wrappedImage.rows, wrappedImage.cols, CV_8UC4);
		// map to shorter range than sensor to make sparse dots more visible
		pvDepth *= 255.0 / 1000;
		cv::cvtColor(pvDepth, depthProjRgb, CV_GRAY2BGRA);
		depthProjRgb.convertTo(depthProjRgb, CV_8UC4);

		depthProjRgb = 0.8*depthProjRgb + 0.2*wrappedImage;

        OpenCVHelpers::CreateOrUpdateTexture2D(
            _deviceResources,
			depthProjRgb,
            _currentVisualizationTexture);
    }

    void AppMain::OnPreRender()
    {
    }

    // Renders the current frame to each holographic camera, according to the
    // current application and spatial positioning state.
    void AppMain::OnRender()
    {
        // Draw the sample hologram.
        for (size_t i = 0; i < _visualizationTextureList.size(); ++i)
        {
            _slateRendererList[i]->Render(
                _visualizationTextureList[i]);
        }
        
        if (_isActiveRenderer)
        {
            _currentSlateRenderer->Render(_currentVisualizationTexture);
        }
    }

    // Notifies classes that use Direct3D device resources that the device resources
    // need to be released before this method returns.
    void AppMain::OnDeviceLost()
    {
        
        for (auto& r : _slateRendererList)
        {
            r->ReleaseDeviceDependentResources();
        }

        _holoLensMediaFrameSourceGroup = nullptr;
        _holoLensMediaFrameSourceGroupStarted = false;

        for (auto& v : _visualizationTextureList)
        {
            v.reset();
        }
        _currentVisualizationTexture.reset();
    }

    // Notifies classes that use Direct3D device resources that the device resources
    // may now be recreated.
    void AppMain::OnDeviceRestored()
    {
        for (auto& r : _slateRendererList)
        {
            r->CreateDeviceDependentResources();
        }

        StartHoloLensMediaFrameSourceGroup();
    }

    void AppMain::StartHoloLensMediaFrameSourceGroup()
    {
        _sensorFrameStreamer =
            ref new HoloLensForCV::SensorFrameStreamer();

        _sensorFrameStreamer->EnableAll();

		_depthMediaFrameSourceGroup =
			ref new HoloLensForCV::MediaFrameSourceGroup(
				HoloLensForCV::MediaFrameSourceGroupType::HoloLensResearchModeSensors,
				_spatialPerception,
				_sensorFrameStreamer);

		_depthMediaFrameSourceGroup->Enable(
			HoloLensForCV::SensorType::ShortThrowToFDepth);


        _holoLensMediaFrameSourceGroup =
            ref new HoloLensForCV::MediaFrameSourceGroup(
				HoloLensForCV::MediaFrameSourceGroupType::PhotoVideoCamera,
                _spatialPerception,
                _sensorFrameStreamer);

        _holoLensMediaFrameSourceGroup->Enable(
            HoloLensForCV::SensorType::PhotoVideo);

        concurrency::create_task(_holoLensMediaFrameSourceGroup->StartAsync()).then(
            [&]()
        {
            _holoLensMediaFrameSourceGroupStarted = true;
        });

		concurrency::create_task(_depthMediaFrameSourceGroup->StartAsync()).then(
			[&]()
		{
			_depthMediaFrameSourceGroupStarted = true;
		});
    }
}
