#include <algorithm>
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
  Mat canvas_;
  int height_;
  int width_;

  int pop_size_;
  Size min_stroke_size_;
  Size max_stroke_size_;

  std::vector<Stroke> population_;

  Painter(const std::string& img_path, int pop_size, const Size& mins,
          const Size& maxs)
      : pop_size_(pop_size), min_stroke_size_(mins), max_stroke_size_(maxs) {
    original_img_ = imread(img_path);
    original_img_.copyTo(canvas_);
    canvas_.setTo(mean(original_img_));
    height_ = original_img_.size().height;
    width_ = original_img_.size().width;
    InitGradientMap();
  }

  void Paint(int stages, int generations, float prob_crs, float prob_mut,
             bool show_progress);
  void InitGradientMap();
  void InitPopulation();
  void CreateSamplingMask(int s, int stages);
  float Evaluate(const Mat& src);
  void DrawStroke(const Stroke& stroke, const Mat& src, Mat& dst);
  Stroke Mutate(const Stroke& src);
};

void Painter::Paint(int stages, int generations, float prob_crs, float prob_mut,
                    bool show_progress) {
  for (int s = 0; s < stages; ++s) {
    InitPopulation();
    float loss_best = Evaluate(canvas_);
    Mat img_best, tmp1, tmp2;
    canvas_.copyTo(img_best);
    for (int g = 0; g < generations; ++g) {
      // cross over
      // mutate
      for (int i = 0; i < pop_size_; ++i) {
        if (rng.uniform(0.0, 1.0) < prob_mut) {
          Stroke new_stroke = Mutate(population_[i]);
          DrawStroke(population_[i], canvas_, tmp1);
          DrawStroke(new_stroke, canvas_, tmp2);
          float loss1 = Evaluate(tmp1);
          float loss2 = Evaluate(tmp2);
          if(loss2 < loss1) { // mutation succeed
            population_[i] = new_stroke;
            if(loss2 < loss_best) {
              loss_best = loss2;
              tmp2.copyTo(img_best);
            }
          }
          else { // mutation fail
            if(loss1 < loss_best) {
              loss_best = loss1;
              tmp1.copyTo(img_best);
            }
          }
        }
      }
    }
    img_best.copyTo(canvas_);
    // imshow("test", canvas_);
    imwrite("output/stage"+std::to_string(s)+".png", canvas_);
    std::cout << "stage: " << s << " loss: " << loss_best << std::endl;
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
  population_.clear();
  for (int i = 0; i < pop_size_; ++i) {
    Point pos(rng.uniform(0, width_), rng.uniform(0, height_));
    float local_mag = img_grad_mag_.at<float>(pos.x, pos.y);
    float local_angle = img_grad_angle_.at<float>(pos.x, pos.y) + 90.0;
    float rotation = rng.uniform(-180, 180) * (1.0 - local_mag) + local_angle;
    population_.push_back(Stroke(
        //Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)),
        original_img_.at<Vec3b>(rng.uniform(0, width_), rng.uniform(0, height_)),
        pos,
        Size(rng.uniform(min_stroke_size_.width, max_stroke_size_.width),
             rng.uniform(min_stroke_size_.height, max_stroke_size_.height)),
        rotation, rng.uniform(0.5, 1.0)));
  }
}

void Painter::CreateSamplingMask(int s, int stages) {

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
  int idx = rng.uniform(0, 4); // no alpha mutation
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

int main() {
  Painter painter("../assets/owl.png", 4, Size(50, 50),
                  Size(200, 200));
  painter.InitPopulation();
  painter.Paint(100, 100, 0.5, 0.8, false);
  waitKey(0);
}