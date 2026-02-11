using OpenCvSharp;

namespace FocusStack.Cli;

public static class ImageAligner
{
    public static ImageStack Align(ImageStack source, CliOptions options)
    {
        var frames = source.Frames;
        var refIndex = options.ReferenceIndex ?? (frames.Count / 2);
        if (refIndex < 0 || refIndex >= frames.Count)
        {
            throw new InvalidOperationException($"Reference frame index out of range: {refIndex}");
        }

        var aligned = frames.Select(f => f.Clone()).ToList();
        using var refGray = ToGray32(frames[refIndex]);

        if (options.GlobalAlign)
        {
            for (var i = 0; i < aligned.Count; i++)
            {
                if (i == refIndex)
                {
                    continue;
                }

                AlignOne(aligned[i], refGray);
            }

            return new ImageStack(aligned);
        }

        for (var i = refIndex - 1; i >= 0; i--)
        {
            using var target = ToGray32(aligned[i + 1]);
            AlignOne(aligned[i], target);
        }

        for (var i = refIndex + 1; i < aligned.Count; i++)
        {
            using var target = ToGray32(aligned[i - 1]);
            AlignOne(aligned[i], target);
        }

        return new ImageStack(aligned);
    }

    private static void AlignOne(Mat movingColor, Mat targetGray32)
    {
        using var movingGray32 = ToGray32(movingColor);
        using var warp = Mat.Eye(2, 3, MatType.CV_32FC1).ToMat();
        var criteria = new TermCriteria(CriteriaTypes.Eps | CriteriaTypes.Count, 100, 1e-6);

        Cv2.FindTransformECC(
            targetGray32,
            movingGray32,
            warp,
            MotionTypes.Euclidean,
            criteria);

        var aligned = new Mat();
        Cv2.WarpAffine(
            movingColor,
            aligned,
            warp,
            movingColor.Size(),
            InterpolationFlags.Linear | InterpolationFlags.WarpInverseMap,
            BorderTypes.Reflect);

        movingColor.SetTo(Scalar.Black);
        aligned.CopyTo(movingColor);
        aligned.Dispose();
    }

    private static Mat ToGray32(Mat color)
    {
        var gray = new Mat();
        Cv2.CvtColor(color, gray, ColorConversionCodes.BGR2GRAY);

        var normalized = new Mat();
        gray.ConvertTo(normalized, MatType.CV_32F, 1.0 / 255.0);
        gray.Dispose();
        return normalized;
    }
}
