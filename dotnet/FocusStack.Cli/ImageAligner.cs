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
        var validMasks = aligned.Select(f => new Mat(f.Size(), MatType.CV_8U, Scalar.All(255))).ToList();

        using var refImage = aligned[refIndex].Clone();
        if (options.GlobalAlign)
        {
            for (var i = 0; i < aligned.Count; i++)
            {
                if (i == refIndex)
                {
                    continue;
                }

                AlignOne(aligned[i], validMasks[i], refImage, options);
            }
        }
        else
        {
            for (var i = refIndex - 1; i >= 0; i--)
            {
                AlignOne(aligned[i], validMasks[i], aligned[i + 1], options);
            }

            for (var i = refIndex + 1; i < aligned.Count; i++)
            {
                AlignOne(aligned[i], validMasks[i], aligned[i - 1], options);
            }
        }

        if (!options.AlignKeepSize && !options.NoCrop)
        {
            using var cropMask = validMasks[0].Clone();
            for (var i = 1; i < validMasks.Count; i++)
            {
                Cv2.BitwiseAnd(cropMask, validMasks[i], cropMask);
            }

            var cropRect = LargestValidRect(cropMask);
            if (cropRect.Width > 0 && cropRect.Height > 0)
            {
                for (var i = 0; i < aligned.Count; i++)
                {
                    var cropped = new Mat(aligned[i], cropRect).Clone();
                    aligned[i].Dispose();
                    aligned[i] = cropped;
                }
            }
        }

        foreach (var mask in validMasks)
        {
            mask.Dispose();
        }

        return new ImageStack(aligned);
    }

    private static void AlignOne(Mat movingColor, Mat movingValidMask, Mat targetColor, CliOptions options)
    {
        if (!options.NoWhiteBalance)
        {
            MatchWhiteBalance(movingColor, targetColor);
        }

        if (!options.NoContrast)
        {
            MatchContrast(movingColor, targetColor);
        }

        if (options.NoTransform)
        {
            return;
        }

        using var movingGray32 = ToGray32(movingColor);
        using var targetGray32 = ToGray32(targetColor);

        var scale = options.FullResolutionAlign ? 1.0 : ComputeAlignmentScale(targetColor.Size(), 2048);
        using var movingForEcc = scale < 0.999 ? ResizeFloat(movingGray32, scale) : movingGray32.Clone();
        using var targetForEcc = scale < 0.999 ? ResizeFloat(targetGray32, scale) : targetGray32.Clone();

        using var warp = Mat.Eye(2, 3, MatType.CV_32FC1).ToMat();
        var criteria = new TermCriteria(CriteriaTypes.Eps | CriteriaTypes.Count, 150, 1e-6);
        Cv2.FindTransformECC(targetForEcc, movingForEcc, warp, MotionTypes.Euclidean, criteria);

        if (scale < 0.999)
        {
            warp.Set(0, 2, warp.At<float>(0, 2) / scale);
            warp.Set(1, 2, warp.At<float>(1, 2) / scale);
        }

        var aligned = new Mat();
        Cv2.WarpAffine(
            movingColor,
            aligned,
            warp,
            movingColor.Size(),
            InterpolationFlags.Linear | InterpolationFlags.WarpInverseMap,
            BorderTypes.Reflect);

        var alignedMask = new Mat();
        Cv2.WarpAffine(
            movingValidMask,
            alignedMask,
            warp,
            movingValidMask.Size(),
            InterpolationFlags.Nearest | InterpolationFlags.WarpInverseMap,
            BorderTypes.Constant,
            Scalar.All(0));

        aligned.CopyTo(movingColor);
        alignedMask.CopyTo(movingValidMask);
        aligned.Dispose();
        alignedMask.Dispose();
    }

    private static void MatchWhiteBalance(Mat movingColor, Mat targetColor)
    {
        var src = Cv2.Mean(movingColor);
        var dst = Cv2.Mean(targetColor);
        var eps = 1e-6;
        var gains = new Scalar(
            dst.Val0 / Math.Max(src.Val0, eps),
            dst.Val1 / Math.Max(src.Val1, eps),
            dst.Val2 / Math.Max(src.Val2, eps));

        using var floatImg = new Mat();
        movingColor.ConvertTo(floatImg, MatType.CV_32FC3);
        Cv2.Multiply(floatImg, gains, floatImg);
        Cv2.Min(floatImg, new Scalar(255, 255, 255), floatImg);
        Cv2.Max(floatImg, Scalar.All(0), floatImg);
        floatImg.ConvertTo(movingColor, MatType.CV_8UC3);
    }

    private static void MatchContrast(Mat movingColor, Mat targetColor)
    {
        using var srcLab = new Mat();
        using var dstLab = new Mat();
        Cv2.CvtColor(movingColor, srcLab, ColorConversionCodes.BGR2Lab);
        Cv2.CvtColor(targetColor, dstLab, ColorConversionCodes.BGR2Lab);

        Cv2.MeanStdDev(srcLab, out var srcMean, out var srcStd);
        Cv2.MeanStdDev(dstLab, out var dstMean, out var dstStd);

        var srcMeanL = srcMean.Val0;
        var srcStdL = Math.Max(srcStd.Val0, 1e-3);
        var dstMeanL = dstMean.Val0;
        var dstStdL = Math.Max(dstStd.Val0, 1e-3);

        var gain = dstStdL / srcStdL;
        var bias = dstMeanL - gain * srcMeanL;

        var channels = srcLab.Split();

        using var l32 = new Mat();
        channels[0].ConvertTo(l32, MatType.CV_32F);
        Cv2.Multiply(l32, gain, l32);
        Cv2.Add(l32, bias, l32);
        Cv2.Min(l32, Scalar.All(255), l32);
        Cv2.Max(l32, Scalar.All(0), l32);
        l32.ConvertTo(channels[0], MatType.CV_8U);

        using var mergedLab = new Mat();
        Cv2.Merge(channels, mergedLab);
        Cv2.CvtColor(mergedLab, movingColor, ColorConversionCodes.Lab2BGR);

        foreach (var ch in channels)
        {
            ch.Dispose();
        }
    }

    private static double ComputeAlignmentScale(Size size, int maxDim)
    {
        var largest = Math.Max(size.Width, size.Height);
        return largest <= maxDim ? 1.0 : (double)maxDim / largest;
    }

    private static Mat ResizeFloat(Mat input, double scale)
    {
        var resized = new Mat();
        Cv2.Resize(input, resized, Size.Zero, scale, scale, InterpolationFlags.Area);
        return resized;
    }

    private static Rect LargestValidRect(Mat mask)
    {
        Cv2.FindContours(mask, out var contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
        Rect best = default;
        var bestArea = 0;
        foreach (var contour in contours)
        {
            var rect = Cv2.BoundingRect(contour);
            var area = rect.Width * rect.Height;
            if (area > bestArea)
            {
                bestArea = area;
                best = rect;
            }
        }

        return best;
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
