# Shared-memory ABI

The POSIX shared-memory name defaults to `/fls_localizer_v2`. The ABI is the
1280-byte `flsloc::shared::Layout` declared in
`include/fls_localizer/shared_memory.hpp`.

The first cache line is an immutable header:

- magic: `0x324c5346`
- ABI version: `2`
- layout size: `1280`

The second cache line is controller metadata. The next 16 cache lines are the
controller's attitude history, and the final two belong to the localizer.
Keeping ownership separate avoids both processes writing the same fields.

## Controller input

The controller writes a 16-entry ring of individually committed samples. Each
sample contains:

- host `CLOCK_MONOTONIC` timestamp in camera-compatible seconds, recorded when
  the Crazyflie log callback receives the attitude;
- normalized drone-to-world quaternion in `x,y,z,w` order;
- `attitude_valid`;
- EKF reset generation being acknowledged;
- its monotonically increasing sample sequence.

The controller metadata contains the newest attitude sequence, landing request,
and known landing tile `(i,j)`. At a 10 ms Crazyflie log period the ring covers
approximately 150 ms before the newest sample.

For every camera frame, the localizer reads the stable samples and chooses the
one minimizing `abs(camera_capture_timestamp - attitude_timestamp)`. It does
not interpolate. An exactly equidistant tie selects the later sample. The
existing maximum-attitude-age check is then applied to the selected sample.
Each frame log records the selected sequence, attitude timestamp, and signed
`attitude_timestamp - camera_timestamp` offset for timing diagnostics.

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
