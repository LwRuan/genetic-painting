#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <ctime>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

using namespace cv;

RNG rng((unsigned)time(NULL));
// std::default_random_engine re((unsigned)time(NULL));

class Stroke {
 public:
  Scalar color_;
  Point pos_;
  Size size_;
  float rotation_;
  float alpha_;
  int type_;
  float loss_;

  Stroke() {}
  Stroke(const Scalar& color, const Point& pos, const Size& size,
         const float rot, const float alpha)
      : color_(color), pos_(pos), size_(size), rotation_(rot), alpha_(alpha) {}
};

class Painter {
 public:
  Mat original_img_;
  Mat img_gray_;
  Mat img_grad_mag_;
  Mat img_grad_angle_;
  Mat sample_mask_;
  Mat color_mask_;
  bool enable_mask_;
  Mat canvas_;
  int height_;
  int width_;

  int pop_size_;
  Size min_stroke_size_;
  Size max_stroke_size_;

  std::array<std::vector<Stroke>, 2> population_;  // swap buffer
  int cur_;
  std::vector<float> pp_;
  std::vector<Point> indices_;
  std::vector<double> mask_pp_;

  Painter(const std::string& img_path, int pop_size, const Size& mins,
          const Size& maxs)
      : cur_(0),
        pop_size_(pop_size),
        min_stroke_size_(mins),
        max_stroke_size_(maxs),
        enable_mask_(false) {
    original_img_ = imread(img_path);
    original_img_.copyTo(canvas_);
    canvas_.setTo(mean(original_img_));
    height_ = original_img_.size().height;
    width_ = original_img_.size().width;
    population_[0].resize(pop_size_);
    population_[1].resize(pop_size_);
    InitGradientMap();
    img_gray_.copyTo(sample_mask_);
    sample_mask_.setTo(0.0f);
    original_img_.copyTo(color_mask_);
    indices_.resize(width_ * height_);
    for (int i = 0; i < height_; ++i) {
      for (int j = 0; j < width_; ++j) {
        indices_[i * width_ + j] = Point(j, i);
      }
    }
  }

  void Paint(int stages, int generations, float xi, float decay, float prob_crs,
             float prob_mut, bool verbose);
  void InitGradientMap();
  void InitPopulation();
  void ComputeCrossProb(float xi, float loss_worst);
  void EvaluatePopulation(int& idx_best, int& idx_worst);
  void CreateSamplingMask(int s, int stages);
  void CreateColorMask(int s, int stages);
  float Evaluate(const Mat& src);
  void DrawStroke(const Stroke& stroke, const Mat& src, Mat& dst);
  Stroke Mutate(const Stroke& src);
  Point MutatePos(const Point& pos);
  Size MutateSize(const Size& size);
  float MutateRot(const float& rot);
  Stroke CrossOver(const Stroke& p1, const Stroke& p2);
};

void Painter::Paint(int stages, int generations, float xi, float decay,
                    float prob_crs, float prob_mut, bool verbose) {
  Size c_min_st = min_stroke_size_;
  Size c_max_st = max_stroke_size_;
  for (int s = 0; s < stages; ++s) {
    int start_stage = int(stages * 0.1);
    int end_stage = int(stages * 0.6);
    if (s >= start_stage && s <= end_stage) {
      float t = (1.0 - float(s - start_stage) /
                           std::max(end_stage - start_stage, 1)) *
                    0.9 +
                0.1;
      min_stroke_size_ =
          Size(int(c_min_st.width * t), int(c_min_st.height * t));
      max_stroke_size_ =
          Size(int(c_max_st.width * t), int(c_max_st.height * t));
    } else if (s > end_stage) {
      min_stroke_size_ =
          Size(int(c_min_st.width * 0.1), int(c_min_st.height * 0.1));
      max_stroke_size_ =
          Size(int(c_max_st.width * 0.1), int(c_max_st.height * 0.1));
    }
    CreateSamplingMask(s, stages);
    CreateColorMask(s, stages);
    if (verbose) {
      Mat mask_img;
      sample_mask_.convertTo(mask_img, CV_8U, 255);
      imwrite("output/sampling_mask" + std::to_string(s) + ".png", mask_img);
      imwrite("output/color_mask" + std::to_string(s) + ".png", color_mask_);
    }
    InitPopulation();
    // Evaluate fitness
    int p_best, p_worst;
    EvaluatePopulation(p_best, p_worst);
    float txi = xi;
    for (int g = 0; g < generations; ++g) {
      // Init sampling probability
      ComputeCrossProb(txi, population_[cur_][p_worst].loss_);
      txi /= decay;
      // cross over
      for (int i = 0; i < pop_size_; ++i) {
        int p1 = std::upper_bound(pp_.data(), pp_.data() + pop_size_,
                                  rng.uniform(0.0, 1.0)) -
                 pp_.data();
        int p2 = std::upper_bound(pp_.data(), pp_.data() + pop_size_,
                                  rng.uniform(0.0, 1.0)) -
                 pp_.data();
        population_[1 - cur_][i] = population_[cur_][p1];
        if (rng.uniform(0.0, 1.0) < prob_crs) {
          population_[1 - cur_][i] =
              CrossOver(population_[cur_][p1], population_[cur_][p2]);
        }
      }
      // mutate
      for (int i = 0; i < pop_size_; ++i) {
        if (rng.uniform(0.0, 1.0) < prob_mut) {
          Stroke new_stroke = Mutate(population_[1 - cur_][i]);
          population_[1 - cur_][i] = new_stroke;
        }
      }
      int c_best, c_worst;
      cur_ = 1 - cur_;
      EvaluatePopulation(c_best, c_worst);
      if (population_[1 - cur_][p_best].loss_ <
          population_[cur_][c_best].loss_) {
        population_[cur_][c_worst] = population_[1 - cur_][p_best];
        p_best = c_worst;
      } else
        p_best = c_best;
      float loss_worst = 0;
      for (int i = 0; i < pop_size_; ++i) {
        if (population_[cur_][i].loss_ > loss_worst) {
          loss_worst = population_[cur_][i].loss_;
          p_worst = i;
        }
      }
    }
    Mat tmp;
    DrawStroke(population_[cur_][p_best], canvas_, tmp);
    tmp.copyTo(canvas_);
    // imshow("test", canvas_);
    imwrite("output/stage" + std::to_string(s) + ".png", canvas_);
    std::cout << "stage: " << s << " loss: " << population_[cur_][p_best].loss_
              << std::endl;
  }
}

void Painter::ComputeCrossProb(float xi, float loss_worst) {
  pp_.resize(pop_size_);
  float sum = 0;
  for (int i = 0; i < pop_size_; ++i) {
    pp_[i] = loss_worst - population_[cur_][i].loss_ + xi;
    sum += pp_[i];
  }
  pp_[0] /= sum;
  for (int i = 1; i < pop_size_; ++i) {
    pp_[i] = pp_[i - 1] + pp_[i] / sum;
  }
}

void Painter::InitGradientMap() {
  cvtColor(original_img_, img_gray_, COLOR_BGR2GRAY);
  Mat tmp, gx, gy;
  normalize(img_gray_, tmp, 0.0, 1.0, NORM_MINMAX, CV_32F);
  Sobel(tmp, gx, CV_32F, 1, 0, 1);
  Sobel(tmp, gy, CV_32F, 0, 1, 1);
  cartToPolar(gx, gy, img_grad_mag_, img_grad_angle_, true);
  pow(img_grad_mag_, 0.3, img_grad_mag_);
}

void Painter::InitPopulation() {
  // population_[cur_].clear();
  Point pos;
  if (enable_mask_) {
    int idx =
        std::upper_bound(mask_pp_.begin(), mask_pp_.end(),
                         rng.uniform(0.0, mask_pp_[width_ * height_ - 1])) -
        mask_pp_.begin();
    pos = indices_[idx];
  } else
    pos = Point(rng.uniform(0, width_), rng.uniform(0, height_));
  Scalar color = color_mask_.at<Vec3b>(pos.y, pos.x);
  float local_mag = img_grad_mag_.at<float>(pos.y, pos.x);
  float local_angle = img_grad_angle_.at<float>(pos.y, pos.x) + 90.0;
  float rotation = rng.uniform(-180, 180) * (1.0 - local_mag) + local_angle;
  population_[cur_][0] = Stroke(
      color, pos,
      Size(rng.uniform(min_stroke_size_.width, max_stroke_size_.width),
           rng.uniform(min_stroke_size_.height, max_stroke_size_.height)),
      rotation, 0.7);
  for (int i = 1; i < pop_size_; ++i) {
    // Point pos = indices_[distr(re)];
    Point tpos = MutatePos(pos);
    float t_mag = img_grad_mag_.at<float>(tpos.y, tpos.x);
    float t_angle = img_grad_angle_.at<float>(tpos.y, tpos.x) + 90.0;
    float t_rot = rng.uniform(-180, 180) * (1.0 - t_mag) + t_angle;

    population_[cur_][i] = Stroke(
        color, tpos,
        Size(rng.uniform(min_stroke_size_.width, max_stroke_size_.width),
             rng.uniform(min_stroke_size_.height, max_stroke_size_.height)),
        t_rot, 0.7);
  }
}

void Painter::EvaluatePopulation(int& idx_best, int& idx_worst) {
  Mat tmp;
  float loss_best = 1e12;
  float loss_worst = 0;
  for (int i = 0; i < pop_size_; ++i) {
    Stroke& stroke = population_[cur_][i];
    DrawStroke(stroke, canvas_, tmp);
    stroke.loss_ = Evaluate(tmp);
    if (stroke.loss_ < loss_best) {
      loss_best = stroke.loss_;
      idx_best = i;
    }
    if (stroke.loss_ > loss_worst) {
      loss_worst = stroke.loss_;
      idx_worst = i;
    }
  }
}

void Painter::CreateSamplingMask(int s, int stages) {
  enable_mask_ = false;
  int start_stage = int(stages * 0.2);
  if (s >= start_stage) {
    enable_mask_ = true;
    float t =
        1.0 - float(s - start_stage) / std::max(stages - start_stage - 1, 1);
    GaussianBlur(img_grad_mag_, sample_mask_, Size(0, 0),
                 std::max(width_ * (t * 0.25f + 0.005f), 1.0f));
    Mat diff1, diff2, total_diff, diff_gray;
    subtract(canvas_, original_img_, diff1);
    subtract(original_img_, canvas_, diff2);
    add(diff1, diff2, total_diff);
    cvtColor(total_diff, diff_gray, COLOR_BGR2GRAY);
    normalize(diff_gray, diff_gray, 0, 1, NORM_MINMAX, CV_32F);
    normalize(sample_mask_, sample_mask_, 0, 1, NORM_MINMAX, CV_32F);
    addWeighted(diff_gray, 1-t, sample_mask_, t, 0, sample_mask_);
    mask_pp_.resize(width_ * height_);
    mask_pp_.assign((float*)sample_mask_.data,
                    (float*)sample_mask_.data + width_ * height_);
    for (int i = 1; i < height_ * width_; ++i) {
      mask_pp_[i] += mask_pp_[i - 1];
    }
  }
}

void Painter::CreateColorMask(int s, int stages) {
  int end_stage = int(stages * 0.7);
  original_img_.copyTo(color_mask_);
  if (s < end_stage) {
    float t = (1.0 - float(s) / std::max(end_stage, 1)) * 0.01;
    GaussianBlur(original_img_, color_mask_, Size(0, 0),
                 std::max(width_ * t, 1.0f));
  }
}

float Painter::Evaluate(const Mat& src) {
  Mat diff1, diff2, total_diff;
  subtract(src, original_img_, diff1);
  subtract(original_img_, src, diff2);
  add(diff1, diff2, total_diff);
  Scalar tmp = sum(total_diff);
  return (tmp[0] + tmp[1] + tmp[2]) / (height_ * width_ * 3.0);
}

void Painter::DrawStroke(const Stroke& stroke, const Mat& src, Mat& dst) {
  src.copyTo(dst);
  ellipse(dst, stroke.pos_, stroke.size_, stroke.rotation_, 0, 360,
          stroke.color_, -1, LINE_AA);
  addWeighted(dst, stroke.alpha_, src, 1 - stroke.alpha_, 0, dst);
}

Stroke Painter::Mutate(const Stroke& src) {
  Stroke tmp = src;
  std::vector<double> mut_pp = {0.2, 0.6, 1.0};
  int idx =
      std::upper_bound(mut_pp.begin(), mut_pp.end(), rng.uniform(0.0, 1.0)) -
      mut_pp.begin();
  switch (idx) {
    case 0:  // pos
      tmp.pos_ = MutatePos(src.pos_);
      break;
    case 1:  // size
      tmp.size_ = MutateSize(src.size_);
      break;
    case 2:  // rot
      tmp.rotation_ = MutateRot(src.rotation_);
      break;
    default:
      break;
  }
  return tmp;
}

Point Painter::MutatePos(const Point& pos) {
  int p_mut = std::min(width_ / 8, height_ / 8);
  int x = pos.x + rng.uniform(-p_mut, p_mut);
  int y = pos.y + rng.uniform(-p_mut, p_mut);
  return Point(std::clamp(x, 0, width_ - 1), std::clamp(y, 0, height_ - 1));
}

Size Painter::MutateSize(const Size& size) {
  Size tmp[4] = {Size(int(size.width * 1.5), size.height),
                 Size(int(size.width * 0.67), size.height),
                 Size(size.width, int(size.height * 1.5)),
                 Size(size.width, int(size.height * 0.67))};
  Size& ret = tmp[rng.uniform(0, 5)];
  return Size(
      std::clamp(ret.width, min_stroke_size_.width, max_stroke_size_.width),
      std::clamp(ret.height, min_stroke_size_.height, max_stroke_size_.height));
}

float Painter::MutateRot(const float& rot) {
  float rot_mut = 90;
  float tmp = rot + rng.uniform(-rot_mut, rot_mut);
  return float(int(tmp) % 360);
}

Stroke Painter::CrossOver(const Stroke& p1, const Stroke& p2) {
  Stroke c = p1;
  // if (rng.uniform(0.0, 1.0) < 0.5) {
  //   c.color_ = p2.color_;
  // }
  if (rng.uniform(0.0, 1.0) < 0.5) {
    c.pos_ = p2.pos_;
    // c.color_ = p2.color_;
  }
  if (rng.uniform(0.0, 1.0) < 0.5) {
    c.size_ = p2.size_;
  }
  if (rng.uniform(0.0, 1.0) < 0.5) {
    c.rotation_ = p2.rotation_;
  }
  return c;
}

int main() {
  Painter painter("../assets/seagull.jpg", 20, Size(20, 20), Size(70, 100));
  painter.Paint(1000, 40, 20, 0.9, 0.8, 0.2, true);
  waitKey(0);
}