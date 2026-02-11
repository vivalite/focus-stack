using OpenCvSharp;

namespace FocusStack.Cli;

public static class ThreeDPreviewRenderer
{
    public static Mat Render(Mat mergedBgr, Mat depthMap8U, ViewPoint3D view)
    {
        var depth32 = new Mat();
        depthMap8U.ConvertTo(depth32, MatType.CV_32F, 1.0 / 255.0);

        var gradX = new Mat();
        var gradY = new Mat();
        Cv2.Sobel(depth32, gradX, MatType.CV_32F, 1, 0, 3);
        Cv2.Sobel(depth32, gradY, MatType.CV_32F, 0, 1, 3);

        var nx = new Mat();
        var ny = new Mat();
        var nz = new Mat(depth32.Size(), MatType.CV_32F, Scalar.All(1.0 / Math.Max(0.1, view.ZScale)));
        Cv2.Multiply(gradX, -view.X, nx);
        Cv2.Multiply(gradY, -view.Y, ny);

        var normalLen = new Mat();
        Cv2.Multiply(nx, nx, normalLen);
        using (var tmp = new Mat())
        {
            Cv2.Multiply(ny, ny, tmp);
            Cv2.Add(normalLen, tmp, normalLen);
        }

        using (var tmp = new Mat())
        {
            Cv2.Multiply(nz, nz, tmp);
            Cv2.Add(normalLen, tmp, normalLen);
        }

        Cv2.Sqrt(normalLen, normalLen);
        Cv2.Divide(nx, normalLen, nx);
        Cv2.Divide(ny, normalLen, ny);
        Cv2.Divide(nz, normalLen, nz);

        var dot = new Mat();
        Cv2.Multiply(nz, view.Z, dot);
        using (var tmp = new Mat())
        {
            Cv2.Multiply(nx, view.X, tmp);
            Cv2.Add(dot, tmp, dot);
        }

        using (var tmp = new Mat())
        {
            Cv2.Multiply(ny, view.Y, tmp);
            Cv2.Add(dot, tmp, dot);
        }

        Cv2.Normalize(dot, dot, 0.35, 1.0, NormTypes.MinMax);

        var dot3 = new Mat();
        Cv2.Merge([dot, dot, dot], dot3);

        var color32 = new Mat();
        mergedBgr.ConvertTo(color32, MatType.CV_32FC3, 1.0 / 255.0);
        Cv2.Multiply(color32, dot3, color32);

        var outMat = new Mat();
        color32.ConvertTo(outMat, MatType.CV_8UC3, 255.0);

        depth32.Dispose();
        gradX.Dispose();
        gradY.Dispose();
        nx.Dispose();
        ny.Dispose();
        nz.Dispose();
        normalLen.Dispose();
        dot.Dispose();
        dot3.Dispose();
        color32.Dispose();

        return outMat;
    }
}
