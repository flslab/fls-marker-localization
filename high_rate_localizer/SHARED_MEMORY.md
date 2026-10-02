# Shared-memory ABI

The POSIX shared-memory name defaults to `/fls_localizer_v3`. The ABI is the
1280-byte `flsloc::shared::Layout` declared in
`include/fls_localizer/shared_memory.hpp`.

The first cache line is an immutable header:

- magic: `0x334c5346`
- ABI version: `3`
- layout size: `1280`

The second cache line is controller metadata. The next 16 cache lines are the
controller's attitude history, and the final two belong to the localizer.
Keeping ownership separate avoids both processes writing the same fields.

## Controller input

The controller writes a 16-entry ring of individually committed samples. Each
sample contains:

- Crazyflie sample timestamp mapped from its millisecond-since-power-on clock
  into host `CLOCK_MONOTONIC` camera-compatible seconds using the minimum
  callback delay observed during startup calibration;
- normalized drone-to-world quaternion in `x,y,z,w` order;
- `attitude_valid`;
- EKF reset generation being acknowledged;
- its monotonically increasing sample sequence.

The controller metadata contains the newest attitude sequence, landing request,
and known landing tile `(i,j)`. At a 10 ms Crazyflie log period the ring covers
approximately 150 ms before the newest sample.

The one-way mapping is causal but cannot distinguish the host/FC clock offset
from the minimum radio and callback transport delay. Consequently, its residual
uncertainty is that minimum delay plus the FC clock's 1 ms quantization. Exact
removal of that residual requires an independent two-way clock calibration.

For every camera frame, the localizer reads the stable samples and chooses the
one minimizing `abs(camera_capture_timestamp - attitude_timestamp)`. An
exactly equidistant tie selects the later sample. With
`tracking.attitude_prediction_enabled` disabled, that closest quaternion is
used unchanged. With it enabled, the localizer uses shortest-arc quaternion
interpolation between samples that bracket the capture time, or bounded
constant-angular-velocity extrapolation after the newest sample. It never
combines samples from different EKF reset generations, never extrapolates
backward before the oldest sample, and falls back to the closest quaternion
when the samples or configured time bound are unsuitable.

`tracking.maximum_attitude_prediction_s` bounds both the sample interval used
for alignment and forward extrapolation from the newest sample. The existing
maximum-attitude-age check is still applied to the closest source sample, not
to the synthesized capture-time attitude. Each frame log records that source
sequence and timestamp, its signed `attitude_timestamp - camera_timestamp`
offset, and whether time alignment was applied.

When the localizer publishes `initial_pose_generation = N` in
`initial_pose_ready`, the controller resets the EKF from the published yaw and
then writes a valid quaternion with `ekf_reset_generation = N`. The localizer
will not use shared attitude before this exact acknowledgement. `N` is unique
to the localizer process so stale acknowledgement data cannot bypass an EKF
reset after the localizer restarts.

## Localizer output

The localizer writes:

- state, pose validity, source, and MyGrid request;
- frame and pose sequences plus capture timestamp;
- drone position in world FLU metres;
- drone-to-world quaternion in `x,y,z,w` order;
- initial yaw in radians;
- selected tile, feature count, reprojection RMS, and processing time.
- conservative HyperGrid acquisition height in metres.
- a yaw-correction candidate containing synchronized `FC EKF yaw - PnP yaw`,
  independent-PnP reprojection RMS, minimum image span, and validity. The sign
  matches Crazyflie firmware's `yawErrorMeasurement_t` convention. It is valid
  only for accepted HyperGrid PnP solutions paired with a synchronized FC
  attitude; the controller applies the final quality and temporal gates.

`shared_memory_pose_technique` in the localizer JSON chooses whether normal
tracking samples contain the accepted `shared_attitude` or `pnp` pose. The
initial pose always comes from PnP because it is the input used to initialize
the controller EKF; only after that reset can the controller provide shared
attitude. The shared-memory ABI is unchanged by this selection.

`pose_source` is `0=none`, `1=MyGrid`, `2=HyperGrid`.
`mygrid_request` is `0=blink`, `1=static`, `2=off`.

## Sequence/checksum protocol

Each writable block and each attitude-ring slot uses the same protocol:

1. Store an odd `sequence_begin` with release ordering.
2. Write the payload.
3. Store the next even value in `sequence_end`.
4. Store the FNV-1a checksum of the bytes between the two sequence fields.
5. Store that even value in `sequence_begin` with release ordering.

A reader accepts a snapshot only when both sequence values are equal and even,
the begin value remained unchanged across the copy, and the checksum matches.
