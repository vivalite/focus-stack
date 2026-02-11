using OpenCvSharp;

namespace FocusStack.Cli;

public sealed class ImageStack : IDisposable
{
    public IReadOnlyList<Mat> Frames => _frames;

    private readonly List<Mat> _frames;

    internal ImageStack(List<Mat> frames)
    {
        _frames = frames;
    }

    public static ImageStack Load(IReadOnlyList<string> paths)
    {
        var frames = new List<Mat>(paths.Count);
        Size? firstSize = null;

        foreach (var path in paths)
        {
            var frame = Cv2.ImRead(path, ImreadModes.Color);
            if (frame.Empty())
            {
                throw new InvalidOperationException($"Unable to load input image: {path}");
            }

            firstSize ??= frame.Size();
            if (frame.Size() != firstSize.Value)
            {
                throw new InvalidOperationException("All input images must have exactly the same dimensions.");
            }

            frames.Add(frame);
        }

        return new ImageStack(frames);
    }

    public ImageStack Clone()
    {
        return new ImageStack(_frames.Select(m => m.Clone()).ToList());
    }

    public void SaveAligned(string outputPath)
    {
        if (_frames.Count == 0)
        {
            return;
        }

        var outputDir = Path.GetDirectoryName(outputPath);
        var fileName = Path.GetFileNameWithoutExtension(outputPath);
        var ext = Path.GetExtension(outputPath);

        if (string.IsNullOrWhiteSpace(outputDir))
        {
            outputDir = Directory.GetCurrentDirectory();
        }

        Directory.CreateDirectory(outputDir);

        for (var i = 0; i < _frames.Count; i++)
        {
            var path = Path.Combine(outputDir, $"{fileName}_aligned_{i:D3}{ext}");
            WriteImage(path, _frames[i], 95);
        }
    }

    public static void WriteImage(string path, Mat image, int jpgQuality)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
        var ext = Path.GetExtension(path).ToLowerInvariant();
        if (ext is ".jpg" or ".jpeg")
        {
            Cv2.ImWrite(path, image, [new ImageEncodingParam(ImwriteFlags.JpegQuality, jpgQuality)]);
            return;
        }

        Cv2.ImWrite(path, image);
    }

    public void Dispose()
    {
        foreach (var frame in _frames)
        {
            frame.Dispose();
        }
    }
}
