namespace FocusStack.Cli;

public sealed class CliOptions
{
    public static readonly string HelpText = """
Usage: focus-stack-cs [options] file1.jpg file2.jpg ...

Output file options:
  --output=output.jpg           Set output filename
  --depthmap=depthmap.png       Write a depth map image
  --3dview=3dview.png           Write a 3D preview image
  --save-steps                  Save intermediate images from processing steps
  --jpgquality=95               JPG quality (0-100, default 95)
  --nocrop                      Save full image, including extrapolated border data

Image alignment options:
  --reference=0                 Reference frame index (default middle one)
  --global-align                Align directly against reference
  --full-resolution-align       Use full resolution images in alignment (default max 2048 px)
  --no-whitebalance             Don't correct white balance differences
  --no-contrast                 Don't correct contrast/exposure differences
  --no-transform                Don't do geometric alignment transform
  --align-only                  Only align and write aligned stack
  --align-keep-size             Keep full image size by not cropping alignment borders
  --no-align                    Skip alignment completely

Image merge options:
  --consistency=2               Neighbour consistency filter level 0..2
  --denoise=1.0                 Merged image denoise level
  --merge-method=wavelet        Merge method: wavelet or laplacian
  --wavelet-levels=5            Wavelet pyramid levels (default 5)

Depth map options:
  --depthmap-threshold=10       Threshold to accept depth points (0-255)
  --depthmap-smooth-xy=20       Smoothing of depthmap in X/Y directions
  --depthmap-smooth-z=40        Smoothing of depthmap depth transitions
  --remove-bg=0                 Positive removes black background, negative white
  --halo-radius=20              Radius of halo effects to remove from depthmap
  --3dviewpoint=x:y:z:zscale    Viewpoint for 3D preview (default 1:1:1:2)

Performance options:
  --threads=0                   Number of OpenCV threads (0 = auto)
  --batchsize=8                 Merge batch size (reserved, default 8)
  --no-opencl                   Disable OpenCL acceleration if available
  --wait-images=0.0             Wait seconds for input files to appear

Information options:
  --verbose                     Verbose output
  --help                        Show this help
  --version                     Show version info
  --opencv-version              Show OpenCV library version
""";

    public List<string> InputFiles { get; } = [];

    public string OutputPath { get; private set; } = "output.jpg";
    public string? DepthMapPath { get; private set; }
    public string? View3DPath { get; private set; }
    public bool SaveSteps { get; private set; }
    public int JpegQuality { get; private set; } = 95;
    public bool NoCrop { get; private set; }

    public int? ReferenceIndex { get; private set; }
    public bool GlobalAlign { get; private set; }
    public bool FullResolutionAlign { get; private set; }
    public bool NoWhiteBalance { get; private set; }
    public bool NoContrast { get; private set; }
    public bool NoTransform { get; private set; }
    public bool DisableAlignment { get; private set; }
    public bool AlignOnly { get; private set; }
    public bool AlignKeepSize { get; private set; }

    public int ConsistencyLevel { get; private set; } = 2;
    public double DenoiseLevel { get; private set; } = 1.0;
    public MergeMethod MergeMethod { get; private set; } = MergeMethod.Wavelet;
    public int WaveletLevels { get; private set; } = 5;

    public int DepthMapThreshold { get; private set; } = 10;
    public int DepthMapSmoothXy { get; private set; } = 20;
    public int DepthMapSmoothZ { get; private set; } = 40;
    public int RemoveBackground { get; private set; }
    public int HaloRadius { get; private set; } = 20;
    public ViewPoint3D ViewPoint { get; private set; } = new(1, 1, 1, 2);

    public int Threads { get; private set; }
    public int BatchSize { get; private set; } = 8;
    public bool DisableOpenCl { get; private set; }
    public double WaitImagesSeconds { get; private set; }

    public bool Verbose { get; private set; }
    public bool ShowHelp { get; private set; }
    public bool ShowVersion { get; private set; }
    public bool ShowOpenCvVersion { get; private set; }

    public static CliOptions Parse(string[] args)
    {
        var options = new CliOptions();

        foreach (var arg in args)
        {
            if (arg is "--help" or "-h") options.ShowHelp = true;
            else if (arg == "--version") options.ShowVersion = true;
            else if (arg == "--opencv-version") options.ShowOpenCvVersion = true;
            else if (arg == "--verbose") options.Verbose = true;
            else if (arg.StartsWith("--output=")) options.OutputPath = arg[9..];
            else if (arg.StartsWith("--depthmap=")) options.DepthMapPath = arg[11..];
            else if (arg.StartsWith("--3dview=")) options.View3DPath = arg[9..];
            else if (arg.StartsWith("--jpgquality=")) options.JpegQuality = ParseInt(arg[13..], "jpgquality", 0, 100);
            else if (arg == "--save-steps") options.SaveSteps = true;
            else if (arg == "--nocrop") options.NoCrop = true;
            else if (arg.StartsWith("--reference=")) options.ReferenceIndex = ParseInt(arg[12..], "reference", 0, int.MaxValue);
            else if (arg == "--global-align") options.GlobalAlign = true;
            else if (arg == "--full-resolution-align") options.FullResolutionAlign = true;
            else if (arg == "--no-whitebalance") options.NoWhiteBalance = true;
            else if (arg == "--no-contrast") options.NoContrast = true;
            else if (arg == "--no-transform") options.NoTransform = true;
            else if (arg == "--align-only") options.AlignOnly = true;
            else if (arg == "--align-keep-size") options.AlignKeepSize = true;
            else if (arg == "--no-align") options.DisableAlignment = true;
            else if (arg.StartsWith("--consistency=")) options.ConsistencyLevel = ParseInt(arg[14..], "consistency", 0, 2);
            else if (arg.StartsWith("--denoise=")) options.DenoiseLevel = ParseDouble(arg[10..], "denoise", 0.0, 100.0);
            else if (arg.StartsWith("--merge-method=")) options.MergeMethod = ParseMergeMethod(arg[15..]);
            else if (arg.StartsWith("--wavelet-levels=")) options.WaveletLevels = ParseInt(arg[17..], "wavelet-levels", 1, 10);
            else if (arg.StartsWith("--depthmap-threshold=")) options.DepthMapThreshold = ParseInt(arg[21..], "depthmap-threshold", 0, 255);
            else if (arg.StartsWith("--depthmap-smooth-xy=")) options.DepthMapSmoothXy = ParseInt(arg[21..], "depthmap-smooth-xy", 0, 1000);
            else if (arg.StartsWith("--depthmap-smooth-z=")) options.DepthMapSmoothZ = ParseInt(arg[20..], "depthmap-smooth-z", 0, 1000);
            else if (arg.StartsWith("--remove-bg=")) options.RemoveBackground = ParseInt(arg[12..], "remove-bg", -255, 255);
            else if (arg.StartsWith("--halo-radius=")) options.HaloRadius = ParseInt(arg[14..], "halo-radius", 0, 1000);
            else if (arg.StartsWith("--3dviewpoint=")) options.ViewPoint = ParseViewPoint(arg[13..]);
            else if (arg.StartsWith("--threads=")) options.Threads = ParseInt(arg[10..], "threads", 0, 1024);
            else if (arg.StartsWith("--batchsize=")) options.BatchSize = ParseInt(arg[12..], "batchsize", 1, 1024);
            else if (arg == "--no-opencl") options.DisableOpenCl = true;
            else if (arg.StartsWith("--wait-images=")) options.WaitImagesSeconds = ParseDouble(arg[14..], "wait-images", 0.0, 36000.0);
            else if (arg.StartsWith("--")) throw new ArgumentException($"Unknown option: {arg}");
            else options.InputFiles.Add(arg);
        }

        return options;
    }

    private static int ParseInt(string value, string name, int min, int max)
    {
        if (!int.TryParse(value, out var parsed) || parsed < min || parsed > max)
            throw new ArgumentException($"Invalid --{name} value: {value}");
        return parsed;
    }

    private static double ParseDouble(string value, string name, double min, double max)
    {
        if (!double.TryParse(value, out var parsed) || parsed < min || parsed > max)
            throw new ArgumentException($"Invalid --{name} value: {value}");
        return parsed;
    }


    private static MergeMethod ParseMergeMethod(string value)
    {
        return value.ToLowerInvariant() switch
        {
            "wavelet" => MergeMethod.Wavelet,
            "laplacian" => MergeMethod.Laplacian,
            _ => throw new ArgumentException($"Invalid --merge-method value: {value}")
        };
    }

    private static ViewPoint3D ParseViewPoint(string value)
    {
        var parts = value.Split(':');
        if (parts.Length != 4)
            throw new ArgumentException($"Invalid --3dviewpoint value: {value}");

        if (!double.TryParse(parts[0], out var x) ||
            !double.TryParse(parts[1], out var y) ||
            !double.TryParse(parts[2], out var z) ||
            !double.TryParse(parts[3], out var zScale))
        {
            throw new ArgumentException($"Invalid --3dviewpoint value: {value}");
        }

        return new ViewPoint3D(x, y, z, zScale);
    }
}

public readonly record struct ViewPoint3D(double X, double Y, double Z, double ZScale);

public enum MergeMethod
{
    Wavelet,
    Laplacian
}
