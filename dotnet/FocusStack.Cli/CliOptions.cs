namespace FocusStack.Cli;

public sealed class CliOptions
{
    public static readonly string HelpText = """
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
""";

    public List<string> InputFiles { get; } = [];
    public string OutputPath { get; private set; } = "output.jpg";
    public string? DepthMapPath { get; private set; }
    public int JpegQuality { get; private set; } = 95;
    public int? ReferenceIndex { get; private set; }
    public bool GlobalAlign { get; private set; }
    public bool DisableAlignment { get; private set; }
    public bool AlignOnly { get; private set; }
    public int ConsistencyLevel { get; private set; } = 2;
    public double DenoiseLevel { get; private set; } = 1.0;
    public bool ShowHelp { get; private set; }
    public bool ShowVersion { get; private set; }

    public static CliOptions Parse(string[] args)
    {
        var options = new CliOptions();

        foreach (var arg in args)
        {
            if (arg == "--help" || arg == "-h")
            {
                options.ShowHelp = true;
            }
            else if (arg == "--version")
            {
                options.ShowVersion = true;
            }
            else if (arg.StartsWith("--output="))
            {
                options.OutputPath = arg[9..];
            }
            else if (arg.StartsWith("--depthmap="))
            {
                options.DepthMapPath = arg[11..];
            }
            else if (arg.StartsWith("--jpgquality="))
            {
                options.JpegQuality = ParseInt(arg[13..], "jpgquality", 0, 100);
            }
            else if (arg.StartsWith("--reference="))
            {
                options.ReferenceIndex = ParseInt(arg[12..], "reference", 0, int.MaxValue);
            }
            else if (arg == "--global-align")
            {
                options.GlobalAlign = true;
            }
            else if (arg == "--no-align")
            {
                options.DisableAlignment = true;
            }
            else if (arg == "--align-only")
            {
                options.AlignOnly = true;
            }
            else if (arg.StartsWith("--consistency="))
            {
                options.ConsistencyLevel = ParseInt(arg[14..], "consistency", 0, 2);
            }
            else if (arg.StartsWith("--denoise="))
            {
                options.DenoiseLevel = ParseDouble(arg[10..], "denoise", 0.0, 100.0);
            }
            else if (arg.StartsWith("--"))
            {
                throw new ArgumentException($"Unknown option: {arg}");
            }
            else
            {
                options.InputFiles.Add(arg);
            }
        }

        return options;
    }

    private static int ParseInt(string value, string name, int min, int max)
    {
        if (!int.TryParse(value, out var parsed) || parsed < min || parsed > max)
        {
            throw new ArgumentException($"Invalid --{name} value: {value}");
        }

        return parsed;
    }

    private static double ParseDouble(string value, string name, double min, double max)
    {
        if (!double.TryParse(value, out var parsed) || parsed < min || parsed > max)
        {
            throw new ArgumentException($"Invalid --{name} value: {value}");
        }

        return parsed;
    }
}
