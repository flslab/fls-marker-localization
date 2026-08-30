# FLS Pose Scope

A local-first web viewer for `eye` output logs. It loads a `log.json` in the
browser, keeps the file on the device, and synchronizes a 3D scene, image-plane
overlays, telemetry plots, frame tables, and complete JSON inspection.

## Run locally

```sh
npm install
npm run dev
```

Open the printed local URL, then drag in a log or use **Open log**. A synthetic
blob-grid run is shown before a file is selected.

## Supported records

- Current blob-grid camera poses and complete localization diagnostics.
- Current ArUco camera poses and marker world poses.
- Legacy marker-in-camera / camera-in-marker pose pairs.
- Historical `tvec`, `tvec_filtered`, and `yaw_pitch_roll` logs.
- Relative seconds, live epoch seconds, and historical epoch milliseconds.

Unknown and future fields remain available through **Raw JSON**, while all
known run metadata and selected-frame arrays are rendered as tables.

## Checks

```sh
npm test
npm run build
```
