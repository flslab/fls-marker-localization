const rows = 4;
const cols = 4;
const spacing = 0.1;
const gridOrigin = [(rows - 1) * spacing / 2, (cols - 1) * spacing / 2, 0];

function markerSet(frameIndex) {
  const wobbleX = Math.sin(frameIndex * 0.11) * 0.7;
  const wobbleY = Math.cos(frameIndex * 0.09) * 0.55;
  return Array.from({ length: rows * cols }, (_, index) => {
    const row = Math.floor(index / cols);
    const col = index % cols;
    const imageX = 248 + col * 47.5 + wobbleX;
    const imageY = 128 + row * 47.2 + wobbleY;
    return { id: index, row, col, imageX, imageY };
  });
}

export function makeDemoLog() {
  const frames = Array.from({ length: 240 }, (_, frameIndex) => {
    const time = frameIndex / 120;
    const decoded = frameIndex >= 30;
    const poseValid = decoded && frameIndex % 37 !== 0;
    const markers = markerSet(frameIndex).filter((_, index) => index !== ((frameIndex * 7) % 53 === 0 ? 5 : -1));
    const cameraPosition = [
      Math.sin(time * 1.6) * 0.034,
      Math.cos(time * 1.25) * 0.027,
      1.01 + Math.sin(time * 2.1) * 0.012,
    ];
    const cameraOrientation = [
      Math.PI + Math.sin(time * 1.4) * 0.011,
      Math.cos(time * 1.1) * 0.015,
      -Math.PI / 2 + Math.sin(time * 0.8) * 0.008,
    ];
    const error = 0.42 + Math.abs(Math.sin(time * 3.2)) * 0.24;

    const blobs = markers.map((marker) => ({ id: decoded ? marker.id : -1, x: marker.imageX, y: marker.imageY }));
    const decodedTracks = decoded ? markers.map((marker) => ({
      id: marker.id,
      image_x: marker.imageX,
      image_y: marker.imageY,
      visible: true,
      last_seen_age: 0,
      eligible_for_localization: true,
    })) : [];
    const relativeMarkers = decoded ? markers.map((marker) => ({
      id: marker.id,
      image_x: marker.imageX,
      image_y: marker.imageY,
      relative_row: marker.row,
      relative_col: marker.col,
      row_coordinate: marker.row - 1.48 + Math.sin(frameIndex * 0.07 + marker.col) * 0.018,
      col_coordinate: marker.col - 1.51 + Math.cos(frameIndex * 0.08 + marker.row) * 0.018,
      row_rounding_error: Math.abs(Math.sin(frameIndex * 0.07 + marker.col)) * 0.018,
      col_rounding_error: Math.abs(Math.cos(frameIndex * 0.08 + marker.row)) * 0.018,
      accepted: true,
      visible: true,
      last_seen_age: 0,
    })) : [];
    const matchedMarkers = poseValid ? markers.map((marker) => ({
      id: marker.id,
      image_x: marker.imageX,
      image_y: marker.imageY,
      relative_row: marker.row,
      relative_col: marker.col,
      map_row: marker.row,
      map_col: marker.col,
      global_position: [gridOrigin[0] - marker.row * spacing, gridOrigin[1] - marker.col * spacing, gridOrigin[2]],
    })) : [];

    return {
      time,
      frame_id: frameIndex,
      poses: poseValid ? [{
        camera_pose: true,
        source: 'blob_grid',
        camera_position: cameraPosition,
        camera_orientation: cameraOrientation,
        drone_position: cameraPosition,
        drone_position_filtered: cameraPosition.map((value, axis) => value + Math.sin(frameIndex * 0.03 + axis) * 0.0007),
        drone_orientation: [0, 0, 0],
        camera_to_plane_distance: cameraPosition[2],
        markers_used: matchedMarkers.length,
        used_marker_ids: matchedMarkers.map((marker) => marker.id),
        used_map_cells: matchedMarkers.map((marker) => ({ row: marker.map_row, col: marker.map_col })),
        reprojection_error: error,
      }] : [],
      blobs,
      blob_grid_localization: {
        status: !decoded ? 'no_detections' : (poseValid ? 'success' : 'pnp_failed'),
        message: !decoded ? 'no decoded marker detections' : (poseValid ? 'global camera and drone positions solved from marker grid' : 'PnP pose solve failed for this sample'),
        tracked_decoded_marker_count: decodedTracks.length,
        decoded_marker_count: decodedTracks.length,
        decoded_tracks: decodedTracks,
        max_marker_age: 0,
        required_marker_count: 4,
        accepted_marker_count: relativeMarkers.length,
        complete_window_count: decoded ? 1 : 0,
        candidate_count: decoded ? 1 : 0,
        best_match_count: decoded ? relativeMarkers.length : 0,
        lookup_attempted: decoded,
        lookup_status: decoded ? 'unique' : 'not_attempted',
        pose_valid: poseValid,
        distance_used: 1,
        relative_markers: relativeMarkers,
        matched_markers: matchedMarkers,
        ...(decoded ? {
          window_match: {
            window_size: 2,
            relative_origin: [0, 0],
            map_origin: [0, 0],
            signature: [0, 1, 4, 5],
          },
        } : {}),
        ...(poseValid ? { reprojection_error: error, camera_to_plane_distance: cameraPosition[2] } : {}),
      },
    };
  });

  return {
    args: {
      print_logs: true,
      preview: false,
      initial_distance: 1,
      execution_time: 0,
      save_rate: 1,
      save_frames: false,
      save_frames_path: '',
      save_video: true,
      video_fps: 30,
      video_path: '',
      config_file: 'camera_config.json',
      json_path: '',
      raw_preview: false,
      raw_stream: false,
      raw_save_frame: false,
      raw_save_video: false,
      aruco_mode: false,
      enable_streaming: false,
      stream_port: 8080,
      stream_type: 'http',
      stream_rate: 10,
      contrast: -2,
      brightness: -2,
      exposure_time: -2,
      frame_rate: 120,
      encoder_frame_rate: 50,
      cam_width: 640,
      cam_height: 400,
      blob_area_threshold: 0.5,
      payload_size: 4,
      target_id: -1,
      tracking_threshold: 30,
      sync_threshold: 4.5,
      static_markers_mode: false,
      validate_mode: false,
      grid_map_file: 'marker_grid_4x4.json',
      grid_window_size: 2,
      grid_rounding_tolerance: 0.3,
      grid_max_marker_age: 0,
      max_attitude_age: 0.1,
      camera_offset_drone: [0, 0, 0],
      enable_kalman_filter: true,
      kf_process_noise: 0.5,
      kf_measurement_noise: 0.02,
      video_input_path: 'render_LightBender_Base.mp4',
    },
    config: {
      initial_distance: 1,
      git_version: 'demo-4bfd662',
      aruco_mode: false,
      blob_grid_localization_enabled: true,
      video_start_time: 0,
      kalman_filter_enabled: true,
      kf_process_noise: 0.5,
      kf_measurement_noise: 0.02,
      marker_grid: {
        map_file: 'marker_grid_4x4.json',
        rows,
        cols,
        num_ids: 16,
        min_k: 3,
        cell_spacing: spacing,
        grid_origin: gridOrigin,
        window_size: 2,
        total_windows: 9,
        unique_windows: 9,
        image_resolution_pixels: [640, 400],
        focal_length_pixels: [478.11017984, 478.29786406],
        principal_point_pixels: [322.59805209, 195.78709198],
        initial_distance: 1,
        latest_distance: 1,
        attitude_source: 'shared_memory',
        camera_offset_drone: [0, 0, 0],
        shared_memory_position: 'drone_position_world',
        orientation_convention: 'R_c_g=R_d_c^T*R_w_d(q)^T from shared attitude',
        rounding_tolerance_cells: 0.3,
        max_marker_age_seconds: 0,
      },
    },
    viewer_demo_note: 'Synthetic data is shown until a local output log is opened. Uploaded logs never leave the browser.',
    frames,
  };
}
