# Focus Stack .NET (C#) Port

This folder contains a .NET 8 C# port of the `focus-stack` CLI built on top of **OpenCvSharp**.

Native OpenCV runtime packages are included for both **Windows** (`OpenCvSharp4.runtime.win`) and **Ubuntu 22.04 x64** (`OpenCvSharp4.runtime.ubuntu.22.04-x64`).

## What is currently ported

- CLI-oriented batch processing flow (`input -> align -> merge -> optional depthmap`).
- ECC-based image registration (OpenCV `FindTransformECC`) with either neighbour chaining or global-to-reference alignment.
- Focus selection via Laplacian response maps.
- Pixel reassignment into merged output image.
- Optional depth map output derived from selected source index.
- Simple consistency cleanup + bilateral denoise controls.

> Note: The original C++ project contains more algorithms and options (wavelet merge variants, OpenCL paths, richer depth-map postprocessing, etc.).
> This C# port intentionally focuses on the core stacker pipeline first.

## Build

```bash
dotnet build dotnet/FocusStack.Cli/FocusStack.Cli.csproj
```

## Run

```bash
dotnet run --project dotnet/FocusStack.Cli/FocusStack.Cli.csproj -- \
  --output=output.jpg \
  --depthmap=depthmap.png \
  examples/pcb/pcb_001.jpg examples/pcb/pcb_002.jpg examples/pcb/pcb_003.jpg
```

## CLI options

```text
Usage: focus-stack-cs [options] file1.jpg file2.jpg ...

Output options:
  --output=<file>              Set output filename (default output.jpg)
  --depthmap=<file>            Write depth map image
  --jpgquality=<0-100>         JPG quality (default 95)

Alignment options:
  --reference=<index>          Reference frame index (default middle frame)
  --global-align               Align all frames directly to reference frame
  --no-align                   Disable frame alignment
  --align-only                 Save aligned stack and exit

Merge options:
  --consistency=<0-2>          Morphological cleanup level (default 2)
  --denoise=<value>            Denoise level (default 1.0)

Information options:
  --help                       Show this help
  --version                    Show version info
```
