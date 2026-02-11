# Focus Stack .NET (C#) Port

This folder contains a .NET 8 C# port of the `focus-stack` CLI built on top of **OpenCvSharp**.

Native OpenCV runtime packages are selected by OS in the project file: **Windows** uses `OpenCvSharp4.runtime.win` and **Linux** uses `OpenCvSharp4.runtime.ubuntu.22.04-x64`.

## Added feature coverage

Compared to the initial port, the CLI now supports a much broader set of options from the original tool:

- Output flags: `--output`, `--depthmap`, `--3dview`, `--save-steps`, `--jpgquality`, `--nocrop`
- Alignment flags: `--reference`, `--global-align`, `--full-resolution-align`, `--no-whitebalance`, `--no-contrast`, `--no-transform`, `--align-only`, `--align-keep-size`, `--no-align`
- Merge flags: `--consistency`, `--denoise`
- Depth map flags: `--depthmap-threshold`, `--depthmap-smooth-xy`, `--depthmap-smooth-z`, `--remove-bg`, `--halo-radius`, `--3dviewpoint`
- Performance/info flags: `--threads`, `--batchsize`, `--no-opencl`, `--wait-images`, `--verbose`, `--opencv-version`

## Build

```bash
dotnet build dotnet/FocusStack.Cli/FocusStack.Cli.csproj
```

## Run

```bash
dotnet run --project dotnet/FocusStack.Cli/FocusStack.Cli.csproj -- \
  --output=output.jpg \
  --depthmap=depthmap.png \
  --3dview=3dview.png \
  examples/pcb/pcb_001.jpg examples/pcb/pcb_002.jpg examples/pcb/pcb_003.jpg
```
