# Shared-memory ABI

The POSIX shared-memory name defaults to `/fls_localizer_v2`. The ABI is the
256-byte `flsloc::shared::Layout` declared in
`include/fls_localizer/shared_memory.hpp`.

The first cache line is an immutable header:

- magic: `0x324c5346`
- ABI version: `1`
- layout size: `256`

The second cache line belongs exclusively to the controller. The third and
fourth belong exclusively to the localizer. Keeping ownership separate avoids
both processes writing the same pose or quaternion fields.

## Controller input

The controller writes:

- timestamp in camera-compatible seconds;
- normalized drone-to-world quaternion in `x,y,z,w` order;
- `attitude_valid`;
- EKF reset generation being acknowledged;
- landing request and known landing tile `(i,j)`.

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

`pose_source` is `0=none`, `1=MyGrid`, `2=HyperGrid`.
`mygrid_request` is `0=blink`, `1=static`, `2=off`.

## Sequence/checksum protocol

Each writable block uses the same protocol:

1. Store an odd `sequence_begin` with release ordering.
2. Write the payload.
3. Store the next even value in `sequence_end`.
4. Store the FNV-1a checksum of the bytes between the two sequence fields.
5. Store that even value in `sequence_begin` with release ordering.

A reader accepts a snapshot only when both sequence values are equal and even,
the begin value remained unchanged across the copy, and the checksum matches.
