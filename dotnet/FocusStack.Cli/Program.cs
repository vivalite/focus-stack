using OpenCvSharp;

namespace FocusStack.Cli;

public static class Program
{
    public static int Main(string[] args)
    {
        CliOptions options;
        try
        {
            options = CliOptions.Parse(args);
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine(ex.Message);
            Console.Error.WriteLine(CliOptions.HelpText);
            return 2;
        }

        if (options.ShowHelp)
        {
            Console.WriteLine(CliOptions.HelpText);
            return 0;
        }

        if (options.ShowVersion)
        {
            Console.WriteLine("focus-stack-cs 0.1.0");
            Console.WriteLine($"OpenCV (OpenCvSharp): {Cv2.GetVersionString()}");
            return 0;
        }

        if (options.InputFiles.Count == 0)
        {
            Console.Error.WriteLine("No input files provided.");
            return 2;
        }

        try
        {
            using var inputStack = ImageStack.Load(options.InputFiles);
            using var alignedStack = options.DisableAlignment
                ? inputStack.Clone()
                : ImageAligner.Align(inputStack, options);

            if (options.AlignOnly)
            {
                alignedStack.SaveAligned(options.OutputPath);
                return 0;
            }

            using var mergeResult = FocusMerger.Merge(alignedStack, options);
            ImageStack.WriteImage(options.OutputPath, mergeResult.MergedImage, options.JpegQuality);

            if (!string.IsNullOrWhiteSpace(options.DepthMapPath))
            {
                ImageStack.WriteImage(options.DepthMapPath, mergeResult.DepthMap8U, options.JpegQuality);
            }

            Console.WriteLine($"Saved merged image to: {options.OutputPath}");
            if (!string.IsNullOrWhiteSpace(options.DepthMapPath))
            {
                Console.WriteLine($"Saved depth map to: {options.DepthMapPath}");
            }

            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Focus stacking failed: {ex.Message}");
            return 1;
        }
    }
}
