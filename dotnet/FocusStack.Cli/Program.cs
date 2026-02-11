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
            Console.WriteLine("focus-stack-cs 0.2.0");
            return 0;
        }

        if (options.ShowOpenCvVersion)
        {
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
            WaitForInputsIfRequested(options);

            if (options.Threads > 0)
            {
                Cv2.SetNumThreads(options.Threads);
            }

            if (options.DisableOpenCl && options.Verbose)
            {
                Console.WriteLine("--no-opencl requested (OpenCV OpenCL toggle is runtime-dependent in OpenCvSharp build).");
            }

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

            if (!string.IsNullOrWhiteSpace(options.View3DPath))
            {
                using var preview = ThreeDPreviewRenderer.Render(mergeResult.MergedImage, mergeResult.DepthMap8U, options.ViewPoint);
                ImageStack.WriteImage(options.View3DPath, preview, options.JpegQuality);
            }

            Console.WriteLine($"Saved merged image to: {options.OutputPath}");
            if (!string.IsNullOrWhiteSpace(options.DepthMapPath))
            {
                Console.WriteLine($"Saved depth map to: {options.DepthMapPath}");
            }

            if (!string.IsNullOrWhiteSpace(options.View3DPath))
            {
                Console.WriteLine($"Saved 3D preview to: {options.View3DPath}");
            }

            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Focus stacking failed: {ex.Message}");
            return 1;
        }
    }

    private static void WaitForInputsIfRequested(CliOptions options)
    {
        if (options.WaitImagesSeconds <= 0)
        {
            return;
        }

        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(options.WaitImagesSeconds);
        foreach (var input in options.InputFiles)
        {
            while (!File.Exists(input) || new FileInfo(input).Length == 0)
            {
                if (DateTime.UtcNow > deadline)
                {
                    throw new InvalidOperationException($"Timed out waiting for input image: {input}");
                }

                System.Threading.Thread.Sleep(200);
            }
        }
    }
}
