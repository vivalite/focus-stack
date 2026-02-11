using OpenCvSharp;

namespace FocusStack.Cli;

public sealed class MergeResult : IDisposable
{
    public Mat MergedImage { get; }
    public Mat DepthMap8U { get; }

    public MergeResult(Mat mergedImage, Mat depthMap8U)
    {
        MergedImage = mergedImage;
        DepthMap8U = depthMap8U;
    }

    public void Dispose()
    {
        MergedImage.Dispose();
        DepthMap8U.Dispose();
    }
}

public static class FocusMerger
{
    public static MergeResult Merge(ImageStack stack, CliOptions options)
    {
        var frames = stack.Frames;
        var rows = frames[0].Rows;
        var cols = frames[0].Cols;

        ComputeFocusLabels(frames, out var labels, out var bestResponse);
        try
        {
            ApplyConsistencyFilter(labels, options.ConsistencyLevel);

            Mat merged;
            if (options.MergeMethod == MergeMethod.Wavelet)
            {
                merged = MergeByWavelet(frames, options.WaveletLevels);
            }
            else
            {
                merged = MergeByLabelSelection(frames, labels, rows, cols);
            }

            if (options.SaveSteps)
            {
                SaveIntermediateSteps(options.OutputPath, labels, bestResponse, merged);
            }

            if (options.DenoiseLevel > 0)
            {
                var sigma = 5.0 + options.DenoiseLevel * 5.0;
                Cv2.BilateralFilter(merged, merged, 0, sigma, sigma);
            }

            var depthMap = DepthMapFromLabels(labels, bestResponse, merged, frames.Count, options);
            return new MergeResult(merged, depthMap);
        }
        finally
        {
            labels.Dispose();
            bestResponse.Dispose();
        }
    }

    private static void ComputeFocusLabels(IReadOnlyList<Mat> frames, out Mat labels, out Mat bestResponse)
    {
        var rows = frames[0].Rows;
        var cols = frames[0].Cols;

        var responseMaps = new List<Mat>(frames.Count);
        try
        {
            foreach (var frame in frames)
            {
                responseMaps.Add(FocusMeasure(frame));
            }

            labels = new Mat(rows, cols, MatType.CV_8U, Scalar.All(0));
            bestResponse = responseMaps[0].Clone();

            for (var i = 1; i < responseMaps.Count; i++)
            {
                using var mask = new Mat();
                Cv2.Compare(responseMaps[i], bestResponse, mask, CmpType.GT);
                responseMaps[i].CopyTo(bestResponse, mask);
                labels.SetTo(new Scalar(i), mask);
            }
        }
        finally
        {
            foreach (var response in responseMaps)
            {
                response.Dispose();
            }
        }
    }

    private static Mat MergeByLabelSelection(IReadOnlyList<Mat> frames, Mat labels, int rows, int cols)
    {
        var merged = new Mat(rows, cols, MatType.CV_8UC3, Scalar.All(0));
        for (var i = 0; i < frames.Count; i++)
        {
            using var mask = new Mat();
            Cv2.Compare(labels, i, mask, CmpType.EQ);
            frames[i].CopyTo(merged, mask);
        }

        return merged;
    }

    private static Mat MergeByWavelet(IReadOnlyList<Mat> frames, int levels)
    {
        var floatFrames = new List<Mat>(frames.Count);
        var pyramids = new List<List<Mat>>(frames.Count);
        try
        {
            foreach (var frame in frames)
            {
                var f32 = new Mat();
                frame.ConvertTo(f32, MatType.CV_32FC3, 1.0 / 255.0);
                floatFrames.Add(f32);
                pyramids.Add(BuildLaplacianPyramid(f32, levels));
            }

            var mergedPyramid = new List<Mat>(levels + 1);
            for (var level = 0; level <= levels; level++)
            {
                mergedPyramid.Add(SelectBestCoefficientsAtLevel(pyramids, level));
            }

            var reconstructed = ReconstructFromLaplacianPyramid(mergedPyramid);
            Cv2.Min(reconstructed, new Scalar(1, 1, 1), reconstructed);
            Cv2.Max(reconstructed, Scalar.All(0), reconstructed);

            var outMat = new Mat();
            reconstructed.ConvertTo(outMat, MatType.CV_8UC3, 255.0);

            reconstructed.Dispose();
            foreach (var m in mergedPyramid)
            {
                m.Dispose();
            }

            return outMat;
        }
        finally
        {
            foreach (var p in pyramids)
            {
                foreach (var m in p)
                {
                    m.Dispose();
                }
            }

            foreach (var m in floatFrames)
            {
                m.Dispose();
            }
        }
    }

    private static List<Mat> BuildLaplacianPyramid(Mat src, int levels)
    {
        var gaussian = new List<Mat>(levels + 1) { src.Clone() };
        for (var i = 0; i < levels; i++)
        {
            var next = new Mat();
            Cv2.PyrDown(gaussian[i], next);
            gaussian.Add(next);
        }

        var laplacian = new List<Mat>(levels + 1);
        for (var i = 0; i < levels; i++)
        {
            using var up = new Mat();
            Cv2.PyrUp(gaussian[i + 1], up, gaussian[i].Size());
            var lap = new Mat();
            Cv2.Subtract(gaussian[i], up, lap);
            laplacian.Add(lap);
        }

        laplacian.Add(gaussian[^1].Clone());

        foreach (var g in gaussian)
        {
            g.Dispose();
        }

        return laplacian;
    }

    private static Mat SelectBestCoefficientsAtLevel(IReadOnlyList<List<Mat>> pyramids, int level)
    {
        var best = pyramids[0][level].Clone();
        using var bestScore = CoefficientScore(best);

        for (var i = 1; i < pyramids.Count; i++)
        {
            var candidate = pyramids[i][level];
            using var candidateScore = CoefficientScore(candidate);
            using var mask = new Mat();
            Cv2.Compare(candidateScore, bestScore, mask, CmpType.GT);
            candidate.CopyTo(best, mask);
            candidateScore.CopyTo(bestScore, mask);
        }

        return best;
    }

    private static Mat CoefficientScore(Mat coeff)
    {
        using var absCoeff = new Mat();
        Cv2.Absdiff(coeff, Scalar.All(0), absCoeff);
        var score = new Mat();
        Cv2.CvtColor(absCoeff, score, ColorConversionCodes.BGR2GRAY);
        return score;
    }

    private static Mat ReconstructFromLaplacianPyramid(IReadOnlyList<Mat> pyramid)
    {
        var current = pyramid[^1].Clone();
        for (var i = pyramid.Count - 2; i >= 0; i--)
        {
            using var up = new Mat();
            Cv2.PyrUp(current, up, pyramid[i].Size());
            var next = new Mat();
            Cv2.Add(up, pyramid[i], next);
            current.Dispose();
            current = next;
        }

        return current;
    }

    private static Mat FocusMeasure(Mat frame)
    {
        using var gray = new Mat();
        Cv2.CvtColor(frame, gray, ColorConversionCodes.BGR2GRAY);

        using var lap = new Mat();
        Cv2.Laplacian(gray, lap, MatType.CV_32F, ksize: 3);

        var absLap = new Mat();
        Cv2.Absdiff(lap, Scalar.All(0), absLap);

        var smooth = new Mat();
        Cv2.GaussianBlur(absLap, smooth, new Size(0, 0), 1.2);
        absLap.Dispose();
        return smooth;
    }

    private static void SaveIntermediateSteps(string outputPath, Mat labels, Mat bestResponse, Mat merged)
    {
        var outputDir = Path.GetDirectoryName(outputPath) ?? ".";
        var name = Path.GetFileNameWithoutExtension(outputPath);

        var labelsVis = new Mat();
        labels.ConvertTo(labelsVis, MatType.CV_8U, 16);
        ImageStack.WriteImage(Path.Combine(outputDir, $"{name}_step_labels.png"), labelsVis, 95);
        labelsVis.Dispose();

        var responseNorm = new Mat();
        Cv2.Normalize(bestResponse, responseNorm, 0, 255, NormTypes.MinMax);
        responseNorm.ConvertTo(responseNorm, MatType.CV_8U);
        ImageStack.WriteImage(Path.Combine(outputDir, $"{name}_step_focus_response.png"), responseNorm, 95);
        responseNorm.Dispose();

        ImageStack.WriteImage(Path.Combine(outputDir, $"{name}_step_merged_raw.png"), merged, 95);
    }

    private static void ApplyConsistencyFilter(Mat labels, int level)
    {
        if (level <= 0)
        {
            return;
        }

        var kernelSize = level == 1 ? 3 : 5;
        using var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(kernelSize, kernelSize));

        using var opened = new Mat();
        Cv2.MorphologyEx(labels, opened, MorphTypes.Open, kernel);
        Cv2.MorphologyEx(opened, labels, MorphTypes.Close, kernel);
    }

    private static Mat DepthMapFromLabels(Mat labels, Mat bestResponse, Mat merged, int frameCount, CliOptions options)
    {
        var depth = new Mat();
        var scale = frameCount <= 1 ? 0 : 255.0 / (frameCount - 1);
        labels.ConvertTo(depth, MatType.CV_8U, scale);

        if (options.DepthMapThreshold > 0)
        {
            using var responseNorm = new Mat();
            Cv2.Normalize(bestResponse, responseNorm, 0, 255, NormTypes.MinMax);
            using var lowConfidenceMask = new Mat();
            Cv2.Threshold(responseNorm, lowConfidenceMask, options.DepthMapThreshold, 255, ThresholdTypes.BinaryInv);
            depth.SetTo(Scalar.All(0), lowConfidenceMask);
        }

        if (options.DepthMapSmoothXy > 0)
        {
            var sigma = Math.Max(0.1, options.DepthMapSmoothXy / 10.0);
            Cv2.GaussianBlur(depth, depth, new Size(0, 0), sigma);
        }

        if (options.DepthMapSmoothZ > 0)
        {
            var sigmaColor = Math.Max(1, options.DepthMapSmoothZ);
            Cv2.BilateralFilter(depth, depth, 0, sigmaColor, 5);
        }

        if (options.HaloRadius > 0)
        {
            var radius = Math.Max(1, options.HaloRadius / 2);
            using var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(radius * 2 + 1, radius * 2 + 1));
            Cv2.MorphologyEx(depth, depth, MorphTypes.Close, kernel);
        }

        if (options.RemoveBackground != 0)
        {
            using var gray = new Mat();
            Cv2.CvtColor(merged, gray, ColorConversionCodes.BGR2GRAY);
            using var bgMask = new Mat();

            if (options.RemoveBackground > 0)
            {
                Cv2.Threshold(gray, bgMask, options.RemoveBackground, 255, ThresholdTypes.BinaryInv);
            }
            else
            {
                Cv2.Threshold(gray, bgMask, 255 + options.RemoveBackground, 255, ThresholdTypes.Binary);
            }

            depth.SetTo(Scalar.All(0), bgMask);
        }

        return depth;
    }
}
