#include <iostream>
#include <algorithm>
#include <cmath>
#include <string>
#include "options.hh"
#include "focusstack.hh"
#include "task_wavelet.hh"
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifndef GIT_VERSION
#define GIT_VERSION "unknown"
#endif

using namespace focusstack;

namespace {

// Parse tiles spec like "2x2" or "2:2"
bool parse_tiles(const std::string &spec, int &nx, int &ny)
{
  if (spec.empty()) return false;

  char sep = 0;
  if (spec.find('x') != std::string::npos) sep = 'x';
  else if (spec.find('X') != std::string::npos) sep = 'X';
  else if (spec.find(':') != std::string::npos) sep = ':';
  else return false;

  size_t pos = spec.find(sep);
  if (pos == std::string::npos) return false;

  try
  {
    nx = std::stoi(spec.substr(0, pos));
    ny = std::stoi(spec.substr(pos + 1));
    return nx > 0 && ny > 0;
  }
  catch (...)
  {
    return false;
  }
}

struct TilePlan
{
  int tx = 0;
  int ty = 0;
  int index = 0;

  cv::Rect base_rect;
  cv::Rect expanded_rect;

  int left_pad = 0;
  int right_pad = 0;
  int top_pad = 0;
  int bottom_pad = 0;

  int start_idx = 0;
  int end_idx = 0;
  int peak_idx = 0;
};

cv::Mat ensure_gray_u8(const cv::Mat &img)
{
  cv::Mat gray;
  if (img.channels() == 1)
  {
    gray = img;
  }
  else if (img.channels() == 3)
  {
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  }
  else if (img.channels() == 4)
  {
    cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
  }
  else
  {
    // Fallback: treat as single channel
    gray = img;
  }

  if (gray.depth() == CV_8U) return gray;

  cv::Mat out;
  double minv = 0.0, maxv = 0.0;
  cv::minMaxLoc(gray, &minv, &maxv);
  if (maxv <= minv) maxv = minv + 1.0;
  double scale = 255.0 / (maxv - minv);
  gray.convertTo(out, CV_8U, scale, -minv * scale);
  return out;
}

// Simple focus metric for choosing the best Z range in a tile:
// variance of Laplacian on a downsampled grayscale crop.
double focus_metric_for_roi(const cv::Mat &img_anydepth, const cv::Rect &roi, int max_dim)
{
  cv::Rect clipped = roi & cv::Rect(0, 0, img_anydepth.cols, img_anydepth.rows);
  if (clipped.width <= 0 || clipped.height <= 0) return 0.0;

  cv::Mat crop = img_anydepth(clipped);
  cv::Mat gray8 = ensure_gray_u8(crop);

  int w = gray8.cols;
  int h = gray8.rows;
  int m = std::max(w, h);
  if (max_dim > 0 && m > max_dim)
  {
    double scale = (double)max_dim / (double)m;
    cv::resize(gray8, gray8, cv::Size(), scale, scale, cv::INTER_AREA);
  }

  cv::Mat lap;
  cv::Laplacian(gray8, lap, CV_32F, 3);
  cv::Scalar mean, stddev;
  cv::meanStdDev(lap, mean, stddev);
  return stddev[0] * stddev[0];
}


// Focus metric on a pre-downsampled grayscale image. The ROI is specified in full-resolution coordinates,
// and scaled by `scale` to match the downsampled image.
double focus_metric_for_roi_small_gray8(const cv::Mat &gray8_small, const cv::Rect &roi_full, double scale)
{
  if (!gray8_small.data) return 0.0;
  if (gray8_small.channels() != 1) return 0.0;

  // Scale ROI to match downsampled image space
  int x = (int)std::round(roi_full.x * scale);
  int y = (int)std::round(roi_full.y * scale);
  int w = (int)std::round(roi_full.width * scale);
  int h = (int)std::round(roi_full.height * scale);

  w = std::max(1, w);
  h = std::max(1, h);

  cv::Rect roi_small(x, y, w, h);
  roi_small &= cv::Rect(0, 0, gray8_small.cols, gray8_small.rows);
  if (roi_small.width <= 0 || roi_small.height <= 0) return 0.0;

  cv::Mat crop = gray8_small(roi_small);
  cv::Mat lap;
  cv::Laplacian(crop, lap, CV_32F, 3);
  cv::Scalar mean, stddev;
  cv::meanStdDev(lap, mean, stddev);
  return stddev[0] * stddev[0];
}

// Convert a full-resolution ROI into downsampled coordinates (matching the scaling used for focus_gray_small).
// Returns false if ROI becomes empty after clipping.
bool roi_full_to_small(const cv::Rect &roi_full, double scale, const cv::Size &small_size, cv::Rect &out_small)
{
  int x = (int)std::round(roi_full.x * scale);
  int y = (int)std::round(roi_full.y * scale);
  int w = (int)std::round(roi_full.width * scale);
  int h = (int)std::round(roi_full.height * scale);

  w = std::max(1, w);
  h = std::max(1, h);

  cv::Rect r(x, y, w, h);
  r &= cv::Rect(0, 0, small_size.width, small_size.height);
  if (r.width <= 0 || r.height <= 0) return false;
  out_small = r;
  return true;
}

// Fast variance-of-Laplacian for many ROIs: uses integral images of Laplacian and Laplacian^2.
// `sum` and `sqsum` are integral images with size (h+1, w+1), type CV_64F.
double focus_metric_from_integrals(const cv::Mat &sum, const cv::Mat &sqsum, const cv::Rect &roi)
{
  // integral images use +1 indexing
  int x0 = roi.x;
  int y0 = roi.y;
  int x1 = roi.x + roi.width;
  int y1 = roi.y + roi.height;

  auto at = [](const cv::Mat &m, int y, int x) -> double {
    return m.at<double>(y, x);
  };

  double s = at(sum, y1, x1) - at(sum, y0, x1) - at(sum, y1, x0) + at(sum, y0, x0);
  double ss = at(sqsum, y1, x1) - at(sqsum, y0, x1) - at(sqsum, y1, x0) + at(sqsum, y0, x0);

  double area = (double)roi.width * (double)roi.height;
  if (area <= 0.0) return 0.0;

  double mean = s / area;
  double var = ss / area - mean * mean;
  if (var < 0.0) var = 0.0;
  return var;
}

// Median smoothing for a 1D signal (robust against spikes).
std::vector<double> smooth_median_1d(const std::vector<double> &v, int win)
{
  if ((int)v.size() <= 1) return v;
  if (win <= 1) return v;
  if (win % 2 == 0) win += 1;
  int half = win / 2;

  std::vector<double> out(v.size(), 0.0);
  std::vector<double> tmp;
  tmp.reserve(win);

  for (int i = 0; i < (int)v.size(); i++)
  {
    tmp.clear();
    int a = std::max(0, i - half);
    int b = std::min((int)v.size() - 1, i + half);
    for (int j = a; j <= b; j++) tmp.push_back(v[j]);
    if (tmp.empty())
    {
      out[i] = v[i];
      continue;
    }
    size_t mid = tmp.size() / 2;
    std::nth_element(tmp.begin(), tmp.begin() + mid, tmp.end());
    out[i] = tmp[mid];
  }
  return out;
}

// Select a contiguous slice band around the best-focus index.
// - scores: per-slice focus scores (higher is sharper)
// - rel_threshold: include slices where smoothed_score >= peak * rel_threshold
// - smooth_win: median smoothing window (odd recommended, e.g. 3 or 5)
// - min_slices: ensure at least this many slices are selected
// - max_slices: cap selection width (<=0 means no cap)
// Outputs start/end inclusive, peak index and peak value (on smoothed signal).
void select_focus_band(const std::vector<double> &scores,
                       double rel_threshold,
                       int smooth_win,
                       int min_slices,
                       int max_slices,
                       int &out_start,
                       int &out_end,
                       int &out_peak_idx,
                       double &out_peak_val)
{
  int n = (int)scores.size();
  out_start = 0;
  out_end = std::max(0, n - 1);
  out_peak_idx = 0;
  out_peak_val = 0.0;
  if (n <= 0) return;

  std::vector<double> sm = smooth_median_1d(scores, smooth_win);

  // Find peak on smoothed scores
  int peak = 0;
  double peakv = sm[0];
  for (int i = 1; i < n; i++)
  {
    if (sm[i] > peakv)
    {
      peakv = sm[i];
      peak = i;
    }
  }

  out_peak_idx = peak;
  out_peak_val = peakv;

  // If peak is non-positive, fall back to a small centered band.
  if (!(peakv > 0.0))
  {
    int want = std::max(1, min_slices);
    if (max_slices > 0) want = std::min(want, max_slices);
    int s = std::max(0, peak - want / 2);
    int e = std::min(n - 1, s + want - 1);
    s = std::max(0, e - want + 1);
    out_start = s;
    out_end = e;
    return;
  }

  double thr = peakv * rel_threshold;
  int s = peak;
  int e = peak;

  while (s > 0 && sm[s - 1] >= thr) s--;
  while (e + 1 < n && sm[e + 1] >= thr) e++;

  // Enforce minimum slices by expanding around the peak.
  int cur = e - s + 1;
  if (cur < min_slices)
  {
    int want = min_slices;
    if (max_slices > 0) want = std::min(want, max_slices);
    int hs = want / 2;
    s = std::max(0, peak - hs);
    e = std::min(n - 1, s + want - 1);
    s = std::max(0, e - want + 1);
  }

  // Enforce maximum slices by shrinking around the peak.
  if (max_slices > 0 && (e - s + 1) > max_slices)
  {
    int want = max_slices;
    int hs = want / 2;
    s = std::max(0, peak - hs);
    e = std::min(n - 1, s + want - 1);
    s = std::max(0, e - want + 1);
  }

  out_start = s;
  out_end = e;
}


// Crop the center valid area that Task_LoadImg would mark, matching the padding strategy.
cv::Mat crop_wavelet_padded_center(const cv::Mat &padded, const cv::Size &orig_size)
{
  cv::Size expanded;
  Task_Wavelet::levels_for_size(orig_size, &expanded);

  int expand_x = expanded.width - orig_size.width;
  int expand_y = expanded.height - orig_size.height;

  // If padded image size differs from our computed size for any reason, fall back to center crop.
  int px = std::max(0, std::min(expand_x / 2, padded.cols - orig_size.width));
  int py = std::max(0, std::min(expand_y / 2, padded.rows - orig_size.height));

  cv::Rect r(px, py, orig_size.width, orig_size.height);
  if (r.x < 0 || r.y < 0 || r.x + r.width > padded.cols || r.y + r.height > padded.rows)
  {
    // Clamp again if needed.
    int x = std::max(0, (padded.cols - orig_size.width) / 2);
    int y = std::max(0, (padded.rows - orig_size.height) / 2);
    r = cv::Rect(x, y, std::min(orig_size.width, padded.cols - x), std::min(orig_size.height, padded.rows - y));
  }

  return padded(r).clone();
}

// Create a feather blending weight map for a tile expanded rect.
// Pads describe how much the expanded rect extends beyond the base rect on each side.
cv::Mat make_feather_weight(int width, int height,
                            int left_pad, int right_pad, int top_pad, int bottom_pad,
                            bool has_left, bool has_right, bool has_top, bool has_bottom)
{
  // Build separable ramps for speed: weight(x,y) = wx(x) * wy(y)
  std::vector<float> wx(width, 1.0f);
  std::vector<float> wy(height, 1.0f);

  if (has_left && left_pad > 0)
  {
    int ramp = std::max(1, 2 * left_pad);
    int lim = std::min(width, ramp);
    for (int x = 0; x < lim; x++) wx[x] = (float)x / (float)ramp;
  }
  if (has_right && right_pad > 0)
  {
    int ramp = std::max(1, 2 * right_pad);
    int lim = std::min(width, ramp);
    for (int k = 0; k < lim; k++)
    {
      int x = width - 1 - k;
      float v = (float)k / (float)ramp;
      wx[x] = std::min(wx[x], v);
    }
  }

  if (has_top && top_pad > 0)
  {
    int ramp = std::max(1, 2 * top_pad);
    int lim = std::min(height, ramp);
    for (int y = 0; y < lim; y++) wy[y] = (float)y / (float)ramp;
  }
  if (has_bottom && bottom_pad > 0)
  {
    int ramp = std::max(1, 2 * bottom_pad);
    int lim = std::min(height, ramp);
    for (int k = 0; k < lim; k++)
    {
      int y = height - 1 - k;
      float v = (float)k / (float)ramp;
      wy[y] = std::min(wy[y], v);
    }
  }

  cv::Mat weight(height, width, CV_32F);
  for (int y = 0; y < height; y++)
  {
    float wyv = wy[y];
    float *row = weight.ptr<float>(y);
    for (int x = 0; x < width; x++)
    {
      row[x] = wx[x] * wyv;
    }
  }
  return weight;
}

} // namespace

int main(int argc, const char *argv[])
{
  // OpenCV can be very noisy at INFO level (e.g., optional plugin DLLs not found).
  // Default to WARNING to avoid confusing "... => FAILED" messages that are not fatal.
  // Allow override via: --opencv-log=info|warn|error|silent
  {
    cv::utils::logging::LogLevel lvl = cv::utils::logging::LOG_LEVEL_WARNING;

    auto to_lower = [](std::string s) {
      std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)std::tolower(c); });
      return s;
    };

    for (int i = 1; i < argc; i++)
    {
      std::string a = argv[i] ? argv[i] : "";
      const std::string p1 = "--opencv-log=";
      const std::string p2 = "--opencv-loglevel=";
      if (a.rfind(p1, 0) == 0 || a.rfind(p2, 0) == 0)
      {
        std::string v = a.substr(a.find('=') + 1);
        v = to_lower(v);
        if (v == "info") lvl = cv::utils::logging::LOG_LEVEL_INFO;
        else if (v == "warn" || v == "warning") lvl = cv::utils::logging::LOG_LEVEL_WARNING;
        else if (v == "error") lvl = cv::utils::logging::LOG_LEVEL_ERROR;
        else if (v == "silent" || v == "quiet" || v == "off") lvl = cv::utils::logging::LOG_LEVEL_SILENT;
      }
    }

    cv::utils::logging::setLogLevel(lvl);
  }

  Options options(argc, argv);
  FocusStack stack;

  if (options.has_flag("--version"))
  {
    std::cerr << "focus-stack " GIT_VERSION ", built " __DATE__ " " __TIME__ "\n"
                 "Compiled with OpenCV version " CV_VERSION "\n"
                 "Copyright (c) 2019 Petteri Aimonen\n\n"

"Permission is hereby granted, free of charge, to any person obtaining a copy\n"
"of this software and associated documentation files (the \"Software\"), to\n"
"deal in the Software without restriction, including without limitation the\n"
"rights to use, copy, modify, merge, publish, distribute, sublicense, and/or\n"
"sell copies of the Software, and to permit persons to whom the Software is\n"
"furnished to do so, subject to the following conditions:\n\n"

"The above copyright notice and this permission notice shall be included in all\n"
"copies or substantial portions of the Software.\n\n"

"THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
"IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
"FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
"AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
"LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
"OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
"SOFTWARE."
              << std::endl;
    return 0;
  }

  if (options.has_flag("--opencv-version"))
  {
    std::cerr << cv::getBuildInformation().c_str() << std::endl;
    return 0;
  }

  // Parse positional filenames once.
  std::vector<std::string> input_files = options.get_filenames();

  if (options.has_flag("--help") || input_files.size() < 2)
  {
    std::cerr << "Usage: " << argv[0] << " [options] file1.jpg file2.jpg ...\n";
    std::cerr << "\n";
    std::cerr << "Output file options:\n"
                 "  --output=output.jpg           Set output filename\n"
                 "  --depthmap=depthmap.png       Write a depth map image (default disabled)\n"
                 "  --3dview=3dview.png           Write a 3D preview image (default disabled)\n"
                 "  --save-steps                  Save intermediate images from processing steps\n"
                 "  --jpgquality=95               Quality for saving in JPG format (0-100, default 95)\n"
                 "  --nocrop                      Save full image, including extrapolated border data\n";
    std::cerr << "\n";
    std::cerr << "Image alignment options:\n"
                 "  --reference=0                 Set index of image used as alignment reference (default middle one)\n"
                 "  --global-align                Align directly against reference (default with neighbour image)\n"
                 "  --full-resolution-align       Use full resolution images in alignment (default max 2048 px)\n"
                 "  --no-whitebalance             Don't attempt to correct white balance differences\n"
                 "  --no-contrast                 Don't attempt to correct contrast and exposure differences\n"
                 "  --no-transform                Don't attempt to correct image position alignment\n"
                 "  --align-only                  Only align the input image stack and exit\n"
                 "  --align-keep-size             Keep original image size by not cropping alignment borders\n"
                 "  --no-align                    Skips the alignment completely, overrides all other alignment options\n";
    std::cerr << "\n";
    std::cerr << "Image merge options:\n"
                 "  --consistency=2               Neighbour pixel consistency filter level 0..2 (default 2)\n"
                 "  --denoise=1.0                 Merged image denoise level (default 1.0)\n";
    std::cerr << "\n";
    std::cerr << "Depth map generation options:\n"
                 "  --depthmap-threshold=10       Threshold to accept depth points (0-255, default 10)\n"
                 "  --depthmap-smooth-xy=20       Smoothing of depthmap in X and Y directions (default 20)\n"
                 "  --depthmap-smooth-z=40        Smoothing of depthmap in Z direction (default 40)\n"
                 "  --remove-bg=0                 Positive value removes black background, negative white\n"
                 "  --halo-radius=20              Radius of halo effects to remove from depthmap\n"
                 "  --3dviewpoint=x:y:z:zscale    Viewpoint for 3D view (default 1:1:1:2)\n";
    std::cerr << "\n";
    std::cerr << "Tilt-compensation options:\n"
                 "  --tilt=2x2                    Enable tilt-compensated tiling (e.g. 2x2 or 3x3)\n"
                 "  --tilt-overlap=0.15           Tile overlap fraction (0.0..0.45, default 0.15)\n"
                 "  --tilt-cache=1                Cache decoded slices used by any tile (faster, uses RAM; default 1)\n"
                 "  --tilt-cache-maxmb=1200       Disable cache if estimated RAM exceeds this (default 1200)\n"
                 "  --tilt-autorange=1            Auto-select best focus band per tile (default 1)\n"
                 "  --tilt-tile-rel=0.45          Per-tile relative threshold vs peak (default 0.45)\n"
                 "  --tilt-global-rel=0.30        Global relative threshold vs peak (default 0.30)\n"
                 "  --tilt-smooth=3               Median smoothing window along Z (default 3)\n"
                 "  --tilt-minslices=3            Minimum slices kept per tile (default 3)\n"
                 "  --tilt-maxslices=13           Maximum slices kept per tile (default 13; set -1 for unlimited)\n"
                 "  --tilt-weak=0.25              Weak-tile fallback to global band (default 0.25)\n"
                 "  --tilt-halfwindow=3           Legacy fixed window: (2*halfwindow+1) slices around best focus (default 3; set -1 to use all)\n"
                 "  --tilt-focus-maxdim=512       Max dimension for focus metric downsampling (default 512)\n";
    std::cerr << "\n";
    std::cerr << "Performance options:\n"
                 "  --threads=2                   Select number of threads to use (default number of CPUs + 1)\n"
                 "  --batchsize=8                 Images per merge batch (default 8)\n"
                 "  --no-opencl                   Disable OpenCL GPU acceleration (default enabled)\n"
                 "  --opencv-log=warn             OpenCV log level: info|warn|error|silent (default warn)\n"
                 "  --wait-images=0.0             Wait for image files to appear (allows simultaneous capture and processing)\n";
    std::cerr << "\n";
    std::cerr << "Information options:\n"
                 "  --verbose                     Verbose output from steps\n"
                 "  --version                     Show application version number\n"
                 "  --opencv-version              Show OpenCV library version and build info\n";
    return 1;
  }

  // Common options parsing
  std::string output_file = options.get_arg("--output", "output.jpg");
  std::string depthmap_file = options.get_arg("--depthmap", "");
  std::string view3d_file = options.get_arg("--3dview", "");

  int jpgquality = std::stoi(options.get_arg("--jpgquality", "95"));
  bool save_steps = options.has_flag("--save-steps");
  bool nocrop = options.has_flag("--nocrop");

  // Image alignment flags
  int flags = FocusStack::ALIGN_DEFAULT;
  if (options.has_flag("--global-align"))             flags |= FocusStack::ALIGN_GLOBAL;
  if (options.has_flag("--full-resolution-align"))    flags |= FocusStack::ALIGN_FULL_RESOLUTION;
  if (options.has_flag("--no-whitebalance"))          flags |= FocusStack::ALIGN_NO_WHITEBALANCE;
  if (options.has_flag("--no-contrast"))              flags |= FocusStack::ALIGN_NO_CONTRAST;
  if (options.has_flag("--no-transform"))             flags |= FocusStack::ALIGN_NO_TRANSFORM;
  if (options.has_flag("--align-keep-size"))          flags |= FocusStack::ALIGN_KEEP_SIZE;
  if (options.has_flag("--no-align"))                 flags  = FocusStack::ALIGN_NONE;

  bool align_only = (options.has_flag("--align-only") && !options.has_flag("--no-align"));

  int reference = -1;
  if (options.has_flag("--reference"))
  {
    reference = std::stoi(options.get_arg("--reference"));
  }

  int consistency = std::stoi(options.get_arg("--consistency", "2"));
  float denoise = std::stof(options.get_arg("--denoise", "1.0"));

  // Depth map generation options
  int depthmap_smooth_xy = std::stoi(options.get_arg("--depthmap-smooth-xy", "20"));
  int depthmap_smooth_z  = std::stoi(options.get_arg("--depthmap-smooth-z", "40"));
  int depthmap_threshold = std::stoi(options.get_arg("--depthmap-threshold", "10"));
  int halo_radius        = std::stoi(options.get_arg("--halo-radius", "20"));
  int remove_bg          = std::stoi(options.get_arg("--remove-bg", "0"));
  std::string viewpoint  = options.get_arg("--3dviewpoint", "1:1:1:2");

  int threads = -1;
  if (options.has_flag("--threads")) threads = std::stoi(options.get_arg("--threads"));

  int batchsize = -1;
  if (options.has_flag("--batchsize")) batchsize = std::stoi(options.get_arg("--batchsize"));

  bool disable_opencl = options.has_flag("--no-opencl");
  float wait_images = std::stof(options.get_arg("--wait-images", "0.0"));
  bool verbose = options.has_flag("--verbose");

  // Tilt-compensation options
  bool tilt_mode = options.has_flag("--tilt");
  std::string tilt_spec = options.get_arg("--tilt", "2x2");
  float tilt_overlap = std::stof(options.get_arg("--tilt-overlap", "0.15"));

  // Performance: cache decoded slices used by any tile to avoid re-reading the same files multiple times.
  bool tilt_cache = (std::stoi(options.get_arg("--tilt-cache", "1")) != 0);
  int tilt_cache_maxmb = std::stoi(options.get_arg("--tilt-cache-maxmb", "1200"));

  // Autorange options (effective only when --tilt is enabled)
  bool tilt_autorange = (std::stoi(options.get_arg("--tilt-autorange", "1")) != 0);
  double tilt_tile_rel = std::stod(options.get_arg("--tilt-tile-rel", "0.45"));
  double tilt_global_rel = std::stod(options.get_arg("--tilt-global-rel", "0.30"));
  int tilt_smooth = std::stoi(options.get_arg("--tilt-smooth", "3"));
  int tilt_minslices = std::stoi(options.get_arg("--tilt-minslices", "3"));
  int tilt_maxslices = std::stoi(options.get_arg("--tilt-maxslices", "13"));
  double tilt_weak = std::stod(options.get_arg("--tilt-weak", "0.25"));

  // Legacy fixed-window option (used when autorange is disabled)
  int tilt_halfwindow = std::stoi(options.get_arg("--tilt-halfwindow", "3"));

  int tilt_focus_maxdim = std::stoi(options.get_arg("--tilt-focus-maxdim", "512"));

  // Check for any unhandled options
  std::vector<std::string> unparsed = options.get_unparsed();
  if (unparsed.size())
  {
    std::cerr << "Warning: unknown options: ";
    for (std::string arg: unparsed)
    {
      std::cerr << arg << " ";
    }
    std::cerr << std::endl;
  }

  if (align_only)
  {
    // Align-only mode doesn't make sense with tilt tiling, use normal pipeline.
    tilt_mode = false;
  }

  if (tilt_mode)
  {
    int nx = 2, ny = 2;
    if (!parse_tiles(tilt_spec, nx, ny))
    {
      std::cerr << "Error: invalid --tilt spec '" << tilt_spec << "'. Use e.g. --tilt=2x2\n";
      return 1;
    }

    if (tilt_overlap < 0.0f) tilt_overlap = 0.0f;
    if (tilt_overlap > 0.45f) tilt_overlap = 0.45f;

    // Load first image to determine full frame size
    cv::Mat first = cv::imread(input_files.front(), cv::IMREAD_ANYCOLOR);
    if (!first.data)
    {
      std::cerr << "Error: could not load " << input_files.front() << "\n";
      return 1;
    }
    int full_w = first.cols;
    int full_h = first.rows;

    // Accumulation buffers
    cv::Mat accum(full_h, full_w, CV_32FC3, cv::Scalar(0,0,0));
    cv::Mat wsum(full_h, full_w, CV_32F, cv::Scalar(0));

    // Tile base size
    int base_w = full_w / nx;
    int base_h = full_h / ny;

    // Clamp autorange parameters
    if (tilt_tile_rel < 0.0) tilt_tile_rel = 0.0;
    if (tilt_tile_rel > 1.0) tilt_tile_rel = 1.0;
    if (tilt_global_rel < 0.0) tilt_global_rel = 0.0;
    if (tilt_global_rel > 1.0) tilt_global_rel = 1.0;
    if (tilt_weak < 0.0) tilt_weak = 0.0;
    if (tilt_weak > 1.0) tilt_weak = 1.0;
    if (tilt_smooth < 1) tilt_smooth = 1;
    if (tilt_minslices < 1) tilt_minslices = 1;

    int nslices = (int)input_files.size();
    int max_slices_cap = 0;
    if (tilt_maxslices < 0)
    {
      max_slices_cap = 0; // unlimited
    }
    else
    {
      max_slices_cap = std::max(1, tilt_maxslices);
      max_slices_cap = std::min(max_slices_cap, nslices);
    }

    // Precompute tile base rectangles (non-overlapped), used for focus scoring.
    std::vector<cv::Rect> tile_base_rects;
    tile_base_rects.reserve(nx * ny);
    for (int ty = 0; ty < ny; ty++)
    {
      for (int tx = 0; tx < nx; tx++)
      {
        int x0 = tx * base_w;
        int y0 = ty * base_h;
        int x1 = (tx == nx - 1) ? full_w : (tx + 1) * base_w;
        int y1 = (ty == ny - 1) ? full_h : (ty + 1) * base_h;
        tile_base_rects.emplace_back(x0, y0, x1 - x0, y1 - y0);
      }
    }

    // Build a grayscale, downsampled cache of each slice for robust focus scoring.
    std::vector<cv::Mat> focus_gray_small(nslices);
    std::vector<double> focus_scale(nslices, 1.0);

    for (int i = 0; i < nslices; i++)
    {
      cv::Mat img;
      if (i == 0)
      {
        img = first;
      }
      else
      {
        img = cv::imread(input_files[i], cv::IMREAD_ANYCOLOR);
      }

      if (!img.data)
      {
        // Keep empty; score will be zero for this slice.
        focus_gray_small[i] = cv::Mat();
        focus_scale[i] = 1.0;
        continue;
      }

      cv::Mat g = ensure_gray_u8(img);

      double sc = 1.0;
      int m = std::max(g.cols, g.rows);
      if (tilt_focus_maxdim > 0 && m > tilt_focus_maxdim)
      {
        sc = (double)tilt_focus_maxdim / (double)m;
        cv::resize(g, g, cv::Size(), sc, sc, cv::INTER_AREA);
      }

      focus_gray_small[i] = g;
      focus_scale[i] = sc;
    }

    // Release the full-resolution first frame now that we have the size and downsampled cache.
    first.release();

    // Compute per-tile and global (across tiles) focus scores for each slice.
    // Speed optimization: for each slice, compute Laplacian once on the downsampled grayscale,
    // then use integral images to evaluate variance-of-Laplacian for many tile ROIs.
    std::vector<std::vector<double>> tile_scores(tile_base_rects.size(), std::vector<double>(nslices, 0.0));
    std::vector<double> global_scores(nslices, 0.0);

    for (int i = 0; i < nslices; i++)
    {
      if (!focus_gray_small[i].data)
      {
        global_scores[i] = 0.0;
        continue;
      }

      cv::Mat lap;
      cv::Laplacian(focus_gray_small[i], lap, CV_32F, 3);

      cv::Mat sum, sqsum;
      cv::integral(lap, sum, sqsum, CV_64F, CV_64F);

      double acc = 0.0;
      int cnt = 0;
      for (int t = 0; t < (int)tile_base_rects.size(); t++)
      {
        cv::Rect roi_small;
        if (!roi_full_to_small(tile_base_rects[t], focus_scale[i], focus_gray_small[i].size(), roi_small))
        {
          tile_scores[t][i] = 0.0;
          acc += 0.0;
          cnt++;
          continue;
        }

        // Integral images are (h+1, w+1), so ROI can be used directly.
        double s = focus_metric_from_integrals(sum, sqsum, roi_small);
        tile_scores[t][i] = s;
        acc += s;
        cnt++;
      }

      global_scores[i] = (cnt > 0) ? (acc / (double)cnt) : 0.0;
    }

    // Global band: prevents tiles from selecting globally useless slices.
    int global_start = 0;
    int global_end = std::max(0, nslices - 1);
    int global_peak = 0;
    double global_peakv = 0.0;

    if (tilt_autorange)
    {
      select_focus_band(global_scores, tilt_global_rel, tilt_smooth, tilt_minslices, 0,
                        global_start, global_end, global_peak, global_peakv);

      if (verbose)
      {
        std::cerr << "Tilt autorange global band: peak=" << global_peak
                  << " range=" << global_start << ".." << global_end
                  << " (" << (global_end - global_start + 1) << " slices)" << std::endl;
      }
    }


    // Build per-tile plans first (slice bands + expanded rects). This lets us cache decoded images
    // only for the union of slices actually used by any tile, which is typically much smaller than
    // the raw input count.
    std::vector<TilePlan> plans;
    plans.reserve(nx * ny);

    for (int ty = 0; ty < ny; ty++)
    {
      for (int tx = 0; tx < nx; tx++)
      {
        TilePlan p;
        p.tx = tx;
        p.ty = ty;
        p.index = ty * nx + tx;
        p.base_rect = tile_base_rects.at(p.index);

        int ox = (int)std::round(p.base_rect.width * tilt_overlap);
        int oy = (int)std::round(p.base_rect.height * tilt_overlap);
        ox = std::min(ox, p.base_rect.width / 2);
        oy = std::min(oy, p.base_rect.height / 2);

        p.expanded_rect = cv::Rect(p.base_rect.x - ox, p.base_rect.y - oy,
                                   p.base_rect.width + 2 * ox, p.base_rect.height + 2 * oy);
        p.expanded_rect &= cv::Rect(0, 0, full_w, full_h);

        p.left_pad = p.base_rect.x - p.expanded_rect.x;
        p.top_pad = p.base_rect.y - p.expanded_rect.y;
        p.right_pad = (p.expanded_rect.x + p.expanded_rect.width) - (p.base_rect.x + p.base_rect.width);
        p.bottom_pad = (p.expanded_rect.y + p.expanded_rect.height) - (p.base_rect.y + p.base_rect.height);

        // Choose slice range for this tile
        int start_idx = 0;
        int end_idx = (int)input_files.size() - 1;
        int best_idx = 0;

        if (tilt_autorange)
        {
          int t_start = 0, t_end = (int)input_files.size() - 1, t_peak = 0;
          double t_peakv = 0.0;

          select_focus_band(tile_scores[p.index], tilt_tile_rel, tilt_smooth, tilt_minslices,
                            max_slices_cap, t_start, t_end, t_peak, t_peakv);

          best_idx = t_peak;
          start_idx = t_start;
          end_idx = t_end;

          // Weak-signal fallback: if a tile has too little detail, use the global band instead.
          if (global_peakv > 0.0 && tilt_weak > 0.0 && (t_peakv < global_peakv * tilt_weak))
          {
            best_idx = global_peak;
            start_idx = global_start;
            end_idx = global_end;
          }

          // Clamp to the global band to drop globally useless slices (e.g. totally blurred tail slices).
          start_idx = std::max(start_idx, global_start);
          end_idx = std::min(end_idx, global_end);
          if (end_idx < start_idx)
          {
            start_idx = global_start;
            end_idx = global_end;
          }

          // Enforce min/max after clamping
          int want_min = std::max(1, tilt_minslices);
          if (end_idx - start_idx + 1 < want_min)
          {
            int hs = want_min / 2;
            start_idx = std::max(global_start, best_idx - hs);
            end_idx = std::min(global_end, start_idx + want_min - 1);
            start_idx = std::max(global_start, end_idx - want_min + 1);
          }

          if (max_slices_cap > 0 && (end_idx - start_idx + 1) > max_slices_cap)
          {
            int want = max_slices_cap;
            int hs = want / 2;
            start_idx = std::max(global_start, best_idx - hs);
            end_idx = std::min(global_end, start_idx + want - 1);
            start_idx = std::max(global_start, end_idx - want + 1);
          }
        }
        else
        {
          // Legacy fixed-window selection around the best-focus index (using cached focus scores)
          int best_i = 0;
          double best_s = -1.0;
          for (int i = 0; i < (int)input_files.size(); i++)
          {
            double s = tile_scores[p.index][i];
            if (s > best_s)
            {
              best_s = s;
              best_i = i;
            }
          }
          best_idx = best_i;

          if (tilt_halfwindow >= 0)
          {
            start_idx = std::max(0, best_idx - tilt_halfwindow);
            end_idx = std::min((int)input_files.size() - 1, best_idx + tilt_halfwindow);
            if (end_idx - start_idx + 1 < 2)
            {
              // Ensure at least 2 images
              if (start_idx > 0) start_idx--;
              if (end_idx < (int)input_files.size() - 1) end_idx++;
            }
          }
        }

        p.start_idx = start_idx;
        p.end_idx = end_idx;
        p.peak_idx = best_idx;

        if (verbose)
        {
          std::cerr << "Tile (" << tx << "," << ty << ") "
                    << (tilt_autorange ? "autorange" : "legacy")
                    << " peak=" << p.peak_idx
                    << " range=" << p.start_idx << ".." << p.end_idx
                    << " (" << (p.end_idx - p.start_idx + 1) << " slices)\n";
        }

        plans.push_back(p);
      }
    }

    // Free focus-cache memory early to reduce peak RAM when slice caching is enabled.
    focus_gray_small.clear();
    focus_scale.clear();

    // Decide which slices are needed by any tile.
    std::vector<unsigned char> slice_needed(nslices, 0);
    for (const auto &p : plans)
    {
      for (int i = p.start_idx; i <= p.end_idx; i++) slice_needed[i] = 1;
    }

    int needed_count = 0;
    for (int i = 0; i < nslices; i++) if (slice_needed[i]) needed_count++;

    // Cache decoded full-resolution slices for speed (avoid repeated disk I/O and PNG decode).
    // Auto-disable if estimated RAM exceeds tilt_cache_maxmb.
    if (tilt_cache_maxmb <= 0) tilt_cache = false;
    if (tilt_cache)
    {
      // Estimate memory: assume 3 bytes per pixel (RGB/BGR 8-bit). This is conservative enough
      // for most microscope PNGs. Users can clamp with --tilt-cache-maxmb.
      double bytes_per_slice = (double)full_w * (double)full_h * 3.0;
      double est_mb = (bytes_per_slice * (double)needed_count) / (1024.0 * 1024.0);
      if (est_mb > (double)tilt_cache_maxmb)
      {
        tilt_cache = false;
        if (verbose)
        {
          std::cerr << "Tilt cache disabled (estimated " << (int)std::round(est_mb)
                    << " MB > limit " << tilt_cache_maxmb << " MB).\n";
        }
      }
      else if (verbose)
      {
        std::cerr << "Tilt cache enabled for " << needed_count << " slices (estimated "
                  << (int)std::round(est_mb) << " MB).\n";
      }
    }

    std::vector<cv::Mat> slice_cache(nslices);
    if (tilt_cache)
    {
      for (int i = 0; i < nslices; i++)
      {
        if (!slice_needed[i]) continue;
        slice_cache[i] = cv::imread(input_files[i], cv::IMREAD_ANYCOLOR);
        if (!slice_cache[i].data)
        {
          std::cerr << "Error: could not load " << input_files[i] << "\n";
          return 1;
        }
      }
    }

    // Process tiles in row-major order
    for (const auto &p : plans)
    {
      // Run focus stacking on this tile
      FocusStack tile_stack;
      tile_stack.set_output(""); // no file output
      tile_stack.set_depthmap(""); // disable depthmap generation by default in tile mode
      tile_stack.set_3dview("");
      tile_stack.set_jpgquality(jpgquality);
      tile_stack.set_save_steps(false); // avoid producing per-tile intermediate files
      tile_stack.set_nocrop(nocrop);
      tile_stack.set_align_only(false);
      tile_stack.set_consistency(consistency);
      tile_stack.set_denoise(denoise);
      tile_stack.set_disable_opencl(disable_opencl);
      tile_stack.set_wait_images(wait_images);
      tile_stack.set_verbose(verbose);

      // Force keep-size in tile mode so each tile output matches its input crop size.
      int tile_flags = flags | FocusStack::ALIGN_KEEP_SIZE;
      tile_stack.set_align_flags(tile_flags);

      if (reference >= 0) tile_stack.set_reference(reference);
      if (threads > 0) tile_stack.set_threads(threads);
      if (batchsize > 0) tile_stack.set_batchsize(batchsize);

      // Add selected slice crops (no clone): pass ROI views to avoid expensive copies.
      std::vector<cv::Mat> loaded_full; // keeps full images alive when caching is disabled
      if (!tilt_cache)
      {
        loaded_full.reserve((size_t)std::max(0, p.end_idx - p.start_idx + 1));
      }

      for (int i = p.start_idx; i <= p.end_idx; i++)
      {
        const cv::Mat *src = nullptr;
        if (tilt_cache)
        {
          src = &slice_cache[i];
        }
        else
        {
          loaded_full.push_back(cv::imread(input_files[i], cv::IMREAD_ANYCOLOR));
          if (!loaded_full.back().data)
          {
            std::cerr << "Error: could not load " << input_files[i] << "\n";
            return 1;
          }
          src = &loaded_full.back();
        }

        cv::Rect r = p.expanded_rect & cv::Rect(0, 0, src->cols, src->rows);
        tile_stack.add_image((*src)(r));
      }

      tile_stack.start();
      tile_stack.do_final_merge();

      bool ok = false;
      std::string errmsg;
      tile_stack.wait_done(ok, errmsg, -1);
      if (!ok)
      {
        std::cerr << "Error in tile (" << p.tx << "," << p.ty << "): " << errmsg << "\n";
        return 1;
      }

      cv::Mat tile_out = tile_stack.get_result_image();
      cv::Mat tile_cropped = crop_wavelet_padded_center(tile_out, p.expanded_rect.size());

      if (tile_cropped.channels() == 1)
      {
        cv::cvtColor(tile_cropped, tile_cropped, cv::COLOR_GRAY2BGR);
      }
      else if (tile_cropped.channels() == 4)
      {
        cv::cvtColor(tile_cropped, tile_cropped, cv::COLOR_BGRA2BGR);
      }

      // Feather blend this tile into the full frame
      cv::Mat weight = make_feather_weight(p.expanded_rect.width, p.expanded_rect.height,
                                           p.left_pad, p.right_pad, p.top_pad, p.bottom_pad,
                                           p.tx > 0, p.tx < nx - 1, p.ty > 0, p.ty < ny - 1);

      cv::Mat tile_f;
      tile_cropped.convertTo(tile_f, CV_32FC3);

      cv::Mat w3;
      {
        cv::Mat ch[] = {weight, weight, weight};
        cv::merge(ch, 3, w3);
      }

      cv::Mat tile_weighted;
      cv::multiply(tile_f, w3, tile_weighted);

      cv::Mat acc_roi = accum(p.expanded_rect);
      cv::add(acc_roi, tile_weighted, acc_roi);

      cv::Mat wsum_roi = wsum(p.expanded_rect);
      cv::add(wsum_roi, weight, wsum_roi);
    }

    // Finalize: divide by weights and save
    cv::Mat wsum_safe;
    cv::max(wsum, cv::Scalar(1e-6), wsum_safe);

    cv::Mat w3;
    {
      cv::Mat ch[] = {wsum_safe, wsum_safe, wsum_safe};
      cv::merge(ch, 3, w3);
    }

    cv::Mat result_f;
    cv::divide(accum, w3, result_f);

    cv::Mat result_u8;
    result_f.convertTo(result_u8, CV_8UC3);

    // Save output
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(jpgquality);

    if (!cv::imwrite(output_file, result_u8, compression_params))
    {
      std::cerr << "Error: failed to save " << output_file << "\n";
      return 1;
    }

    std::printf("\rSaved to %-40s\n", output_file.c_str());
    if (depthmap_file != "" || view3d_file != "")
    {
      std::printf("Note: --depthmap and --3dview are currently ignored in --tilt mode.\n");
    }
    return 0;
  }

  // Normal pipeline (unchanged behavior)
  // Output file options
  stack.set_inputs(input_files);
  stack.set_output(output_file);
  stack.set_depthmap(depthmap_file);
  stack.set_3dview(view3d_file);
  stack.set_jpgquality(jpgquality);
  stack.set_save_steps(save_steps);
  stack.set_nocrop(nocrop);

  // Image alignment options
  stack.set_align_flags(flags);

  if (reference >= 0)
  {
    stack.set_reference(reference);
  }

  if (align_only && !options.has_flag("--no-align"))
  {
    stack.set_align_only(true);
    stack.set_output(options.get_arg("--output", "aligned_"));
  }

  // Image merge options
  stack.set_consistency(consistency);
  stack.set_denoise(denoise);

  // Depth map generation options
  stack.set_depthmap_smooth_xy(depthmap_smooth_xy);
  stack.set_depthmap_smooth_z(depthmap_smooth_z);
  stack.set_depthmap_threshold(depthmap_threshold);
  stack.set_halo_radius(halo_radius);
  stack.set_remove_bg(remove_bg);
  stack.set_3dviewpoint(viewpoint);

  // Performance options
  if (threads > 0)
  {
    stack.set_threads(threads);
  }

  if (batchsize > 0)
  {
    stack.set_batchsize(batchsize);
  }

  stack.set_disable_opencl(disable_opencl);
  stack.set_wait_images(wait_images);

  // Information options (some are handled at beginning of this function)
  stack.set_verbose(verbose);

  if (!stack.run())
  {
    std::printf("\nError exit due to failed steps\n");
    return 1;
  }

  std::printf("\rSaved to %-40s\n", stack.get_output().c_str());

  if (stack.get_depthmap() != "")
  {
    std::printf("\rSaved depthmap to %s\n", stack.get_depthmap().c_str());
  }

  if (stack.get_3dview() != "")
  {
    std::printf("\rSaved 3D preview to %s\n", stack.get_3dview().c_str());
  }

  return 0;
}
