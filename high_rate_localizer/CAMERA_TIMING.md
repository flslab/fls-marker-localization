# Native camera controls and capture timing (schema 2)

The native `fls_localizer` uses measured sensor exposure timing for localization,
shared memory, attitude queries and pose solving. The runtime time is:

```
capture_monotonic_ns = SensorTimestamp - measured(BOOTTIME - MONOTONIC)
                       + actual ExposureTime_us * 500
                       + camera.timestamp_offset_us * 1000
```

The deployed camera uses a **global shutter**, so all rows share the exposure
interval. The exposure term selects the whole image's exposure midpoint; no
row-delay correction is needed. Keep `camera.timestamp_offset_us` at its default
zero unless a separate measurement establishes an additional fixed capture
phase correction. Unknown fixed FC sensor/estimator/transport phase is not
identified by this conversion. Libcamera describes SensorTimestamp in terms of
the first active row; for this global-shutter camera that is also the start of
the whole image's exposure.
`frames[].time` retains microsecond JSON rounding; `capture_monotonic_ns` records
the integer value. Original framebuffer/sensor timestamps remain available.

The top-level `camera_timing_schema` describes the clock and exposure phases.
Frames without native acquisition metadata, including video-input frames,
have `camera_timing: null`.

## Per-frame fields

| Field | Meaning |
| --- | --- |
| `buffer_timestamp_ns` | Original unsigned framebuffer timestamp, stored as an integer with no conversion/rounding |
| `sensor_timestamp_ns` | Request metadata `SensorTimestamp`, or null when absent |
| `exposure_time_us` | Actual request metadata `ExposureTime`, or null |
| `frame_duration_us` | Actual request metadata `FrameDuration`, or null |
| `analogue_gain` | Actual request metadata `AnalogueGain`, or null |
| `manual_controls_match` | Actual exposure/gain match requested values within sensor quantization tolerance |
| `boottime_minus_monotonic_ns` | BOOTTIME minus the midpoint of two bracketing MONOTONIC reads |
| `clock_pair_uncertainty_ns` | Half-width of that bracket; pairs above 100 us are rejected |
| `capture_monotonic_ns` | Runtime exposure midpoint in the FC attitude's host-monotonic time base; whole-image midpoint for the deployed global-shutter camera |
| `timestamp_offset_us` | Configured additional calibrated capture phase offset |
| `unusable_frames_skipped_total` | Frames skipped for bad buffers, controls, capture metadata or clock checks |
| `buffer_sequence` | Framebuffer source sequence counter |
| `request_sequence` | Request sequence counter captured before request reuse |
| `buffer_status` | Numeric `FrameMetadata::Status` from the linked libcamera SDK |
| `host_completed_monotonic_ns` | Host `CLOCK_MONOTONIC` sampled on entry to the completed-request callback |
| `host_read_started_monotonic_ns` | Host `CLOCK_MONOTONIC` sampled before waiting for an available frame |
| `host_read_finished_monotonic_ns` | Host `CLOCK_MONOTONIC` sampled after copying the selected luma image |
| `stale_requests_skipped` | Completed older requests discarded by this read's existing newest-frame selection |
| `stale_requests_skipped_total` | Cumulative count of those deliberately discarded requests |
| `buffer_sequence_delta`, `request_sequence_delta` | Modular forward difference from the previously delivered frame; null on the first frame or a reset/out-of-order jump |
| `*_sequence_wrapped` | Small forward modular difference across the 32-bit rollover boundary |
| `*_sequence_reset_or_out_of_order` | A jump incompatible with a forward difference smaller than half the counter range |
| `*_sequence_duplicate` | Same counter as the previously delivered frame |
| `buffer_timestamp_nonmonotonic` | Buffer timestamp is no later than the previously delivered buffer timestamp |
| `sensor_timestamp_nonmonotonic` | Available sensor timestamp is no later than the previous available sensor timestamp |

Unavailable controls remain null. They are never replaced with requested camera
settings. Host clock reads also remain null if the clock call fails. A read can
start before the request completes because `read()` waits for a frame; its start
time is not a capture timestamp.

Sequence gaps include frames deliberately skipped by newest-frame selection.
For example, a delivered buffer-sequence delta of three with two stale requests
skipped accounts for the two missing delivered frames. Additional gaps warrant
checking driver drops and frame status. Counters cannot uniquely distinguish a
reset near rollover from a genuine rollover; the flags describe the observed
modular transition and do not invent a count of billions of dropped frames.

## Clock and phase interpretation

Libcamera specifies `SensorTimestamp` in nanoseconds on Linux `CLOCK_BOOTTIME`,
referenced to the exposure of the sensor's first active row. This is recorded
as `sensor_timestamp_phase: first_row_exposure_start` in the schema.
[Libcamera's SensorTimestamp definition](https://lists.libcamera.org/pipermail/libcamera-devel/2021-April/019448.html)
documents that contract. Actual exposure duration is recorded separately; a
midpoint calculation requires both reported fields and an appropriate shutter
model. Rolling-shutter rows have different exposure times.

The framebuffer API specifies a system-relative nanosecond timestamp without
fixing its exact epoch or capture event. The schema therefore labels its clock
and phase as unspecified and does not claim it is the same event as
`SensorTimestamp`. See the official
[FrameMetadata timestamp and sequence documentation](https://docs.libcamera.org/master/public-api/structlibcamera_1_1FrameMetadata.html).
The two original timestamps let offline analysis measure their relationship
on the actual camera and driver instead of assuming one. Version-1 recordings
used the framebuffer timestamp as runtime time; do not reinterpret those old
logs or their fitted timing offsets as schema-2 exposure times. The clock pair
is measured for each candidate frame, including after suspend/resume; captures
in the future relative to callback completion, older than one second, or not
representable on the host clock are rejected.

## Manual controls and attitude alignment

`AeEnable=false`, `ExposureTime` and `AnalogueGain` are submitted together at
camera start, on each initial request, and after every request reuse. This
follows libcamera's [manual exposure/gain control contract](https://github.com/libcamera-org/libcamera/blob/master/Documentation/design/ae.rst).
Exposure tolerance is max(100 us, 2% of requested); gain tolerance is max(0.05,
5% of requested). Actual metadata is mandatory. Startup/transient frames that
do not pass are discarded, not localized using the requested values as a
substitute. Two seconds without usable frames raises a clear acquisition error;
the first failure prints requested and reported controls. Frame delivery also
times out after two seconds without a completed request.

With `tracking.attitude_prediction_enabled=true` and
`tracking.attitude_alignment_wait_s=0.006` (the default wait), native capture
polls the existing FC attitude history for at most 6 ms to obtain a bracket.
Same-generation quaternions are SLERPed to the capture time; an exact-time
sample is also usable. Missing brackets produce `attitude_alignment_unavailable`
and no accepted position. There is no forward extrapolation. Existing bracket
gap and real-source age limits remain in force; EKF reset generations are never
combined. Set zero wait to restore optional interpolation with nearest-sample
fallback, or explicitly disable prediction for legacy nearest-sample operation.

`shared_attitude` diagnostics include `alignment_required`,
`alignment_available`, `alignment_wait_s` and `alignment_wait_timed_out`, along
with the existing effective timestamp, source timestamps, sequence pair and
SLERP fraction. Wait time is separate from vision `processing_ms`.

Request/buffer controls and the callback timestamp are copied before requests
are recycled. The callback performs no JSON formatting or disk logging; the
existing output worker serializes the attached small metadata structure.
The old two-argument camera `read()` overload remains available, and production
uses the new overload to attach timing to `FrameResult`.

Build and test on the Raspberry Pi with its deployed libcamera headers and
camera after installing the change. The portable core tests cover JSON integer
precision, missing controls, exposure/clock conversion, manual-control
quantization/regressions, counter gaps, delayed bracketing samples, timeouts,
duplicates, resets, and rollovers. Full shared-memory tests need OS shared-memory
access. A host without a camera cannot validate actual control delivery.

On the Pi, rebuild `./high_rate_localizer/build.sh`, start a ground recording
with the existing config, and verify actual exposure is near 3000 us, gain near
1, and `manual_controls_match=true`. Confirm schema 2, small clock-pair
uncertainty, positive `capture_monotonic_ns`, `alignment_available=true` on
accepted tracking frames, and effective attitude time equal to capture time.
Inspect timeout/skipped-frame counts and marker visibility before repeating
the hover. Existing hashed experiment configs inherit the new zero phase offset
and 6 ms wait defaults; their assets do not need to be edited or rehashed.
