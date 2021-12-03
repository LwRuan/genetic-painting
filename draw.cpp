#include <algorithm>
#include <array>
#include <cmath>
#include <ctime>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;

RNG rng((unsigned)time(NULL));

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
  Mat mask_;
  Mat canvas_;
  int height_;
  int width_;

  int pop_size_;
  Size min_stroke_size_;
  Size max_stroke_size_;

  std::array<std::vector<Stroke>, 2> population_;  // swap buffer
  int cur_;
  std::vector<float> pp_;

  Painter(const std::string& img_path, int pop_size, const Size& mins,
          const Size& maxs)
      : cur_(0),
        pop_size_(pop_size),
        min_stroke_size_(mins),
        max_stroke_size_(maxs) {
    original_img_ = imread(img_path);
    original_img_.copyTo(canvas_);
    canvas_.setTo(mean(original_img_));
    height_ = original_img_.size().height;
    width_ = original_img_.size().width;
    population_[0].resize(pop_size_);
    population_[1].resize(pop_size_);
    InitGradientMap();
  }

  void Paint(int stages, int generations, float xi, float decay, float prob_crs,
             float prob_mut, bool show_progress);
  void InitGradientMap();
  void InitPopulation();
  void ComputeCrossProb(float xi, float loss_worst);
  void EvaluatePopulation(int& idx_best, int& idx_worst);
  void CreateSamplingMask(int s, int stages);
  float Evaluate(const Mat& src);
  void DrawStroke(const Stroke& stroke, const Mat& src, Mat& dst);
  Stroke Mutate(const Stroke& src);
  Stroke CrossOver(const Stroke& p1, const Stroke& p2);
};

void Painter::Paint(int stages, int generations, float xi, float decay,
                    float prob_crs, float prob_mut, bool show_progress) {
  for (int s = 0; s < stages; ++s) {
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
  for (int i = 0; i < pop_size_; ++i) {
    Point pos(rng.uniform(0, width_), rng.uniform(0, height_));
    float local_mag = img_grad_mag_.at<float>(pos.x, pos.y);
    float local_angle = img_grad_angle_.at<float>(pos.x, pos.y) + 90.0;
    float rotation = rng.uniform(-180, 180) * (1.0 - local_mag) + local_angle;
    population_[cur_][i] = Stroke(
        // Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0,
        // 256)),
        original_img_.at<Vec3b>(rng.uniform(0, width_),
                                rng.uniform(0, height_)),
        pos,
        Size(rng.uniform(min_stroke_size_.width, max_stroke_size_.width),
             rng.uniform(min_stroke_size_.height, max_stroke_size_.height)),
        rotation, 0.7);
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
  int start_stage = int(stages * 0.2);
  if (s >= start_stage) {
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
  int color_mut = 128;
  int px_mut = width_ / 8;
  int py_mut = height_ / 8;
  int sx_mut = (max_stroke_size_.width - min_stroke_size_.width) / 8;
  int sy_mut = (max_stroke_size_.height - min_stroke_size_.height) / 8;
  float rot_mut = 180;
  float alpha_mut = 0.2;
  int idx = rng.uniform(0, 4);  // no alpha mutation
  switch (idx) {
    case 0:  // color
      tmp.color_ = src.color_ + Scalar(rng.uniform(-color_mut, color_mut),
                                       rng.uniform(-color_mut, color_mut),
                                       rng.uniform(-color_mut, color_mut));
      tmp.color_ = Scalar(std::clamp(int(tmp.color_[0]), 0, 256),
                          std::clamp(int(tmp.color_[1]), 0, 256),
                          std::clamp(int(tmp.color_[2]), 0, 256));
      break;
    case 1:  // pos
      tmp.pos_ = src.pos_ + Point(rng.uniform(-px_mut, px_mut),
                                  rng.uniform(-py_mut, py_mut));
      tmp.pos_ = Point(std::clamp(int(tmp.pos_.x), 0, width_),
                       std::clamp(int(tmp.pos_.y), 0, height_));
      break;
    case 2:  // size
      tmp.size_ = src.size_ + Size(rng.uniform(-sx_mut, sx_mut),
                                   rng.uniform(-sy_mut, sy_mut));
      tmp.size_ = Size(std::clamp(tmp.size_.width, min_stroke_size_.width,
                                  max_stroke_size_.width),
                       std::clamp(tmp.size_.height, min_stroke_size_.height,
                                  max_stroke_size_.height));
      break;
    case 3:  // rotation
      tmp.rotation_ = src.rotation_ + rng.uniform(-rot_mut, rot_mut);
      tmp.rotation_ = float(int(tmp.rotation_) % 360);
      break;
    case 4:  // alpha
      tmp.alpha_ = src.alpha_ + rng.uniform(-alpha_mut, alpha_mut);
      tmp.alpha_ = std::clamp(tmp.alpha_, 0.5f, 1.0f);
      break;
    default:
      break;
  }
  return tmp;
}

Stroke Painter::CrossOver(const Stroke& p1, const Stroke& p2) {
  Stroke c = p1;
  if (rng.uniform(0.0, 1.0) < 0.5) {
    c.color_ = p2.color_;
  }
  if (rng.uniform(0.0, 1.0) < 0.5) {
    c.pos_ = p2.size_;
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
  Painter painter("../assets/monalisa-322px.jpg", 10, Size(10, 10),
                  Size(100, 100));
  painter.InitPopulation();
  painter.Paint(1000, 50, 1e3, 0.6, 0.8, 0.1, false);
  waitKey(0);
}