import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
Stroke: float array
 0 1 2     | 3 4    | 5 6     | 7        | 8    | 9 10 11     |
 color:bgr | pos:yx | size:yx | rotation | type | reserved    |
'''


class Painter:
    def __init__(self, img_path, stroke_cnt=10, min_stroke_size=np.array([1e-2, 1e-2]), max_stroke_size=np.array([1, 1])):
        self.original_img = cv2.imread(img_path)
        self.img_gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        self.img_grads = self._imgGradient(self.img_gray)
        self.img_shape = self.img_gray.shape
        self.min_stroke_size = min_stroke_size * max(self.img_shape)
        self.max_stroke_size = max_stroke_size * max(self.img_shape)
        self.stroke_cnt = 10

        self.img_buffer = [np.zeros(
            (self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8)]

    def initPopulation(self, pop_size=100):
        self.population = np.zeros((pop_size, 12), dtype=float)
        for p in range(pop_size):
            color = np.random.rand(3) * 256.0
            

    def _imgGradient(self, img):
        # convert to 0 to 1 float representation
        img = np.float32(img) / 255.0
        # Calculate gradient
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        # Python Calculate gradient magnitude and direction ( in degrees )
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        # normalize magnitudes
        mag /= np.max(mag)
        # lower contrast
        mag = np.power(mag, 0.3)
        return mag, angle


def main():
    np.random.seed(0)
    painter = Painter('assets/monalisa-515px.jpg')

    plt.axis('off')
    # plt.imshow(painter.original_img[:,:,::-1]) # bgr to rgb
    # plt.imshow(painter.img_gray, cmap="gray")
    plt.imshow(painter.img_grads[0], cmap="gray")
    plt.show()


if __name__ == '__main__':
    main()
