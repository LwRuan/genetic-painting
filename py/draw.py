import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

'''
Stroke: float array
| 0 1 2     | 3 4    | 5 6     | 7        | 8     | 9    | 10 11    |
| color:bgr | pos:xy | size:xy | rotation | alpha | type | reserved |
'''


class Painter:
    def __init__(self, img_path, stroke_cnt=4, pop_size=10, min_stroke_size=np.array([1e-2, 1e-2]), max_stroke_size=np.array([0.5, 0.5])):
        self.original_img = cv2.imread(img_path)
        self.img_gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        self.img_grads = self._imgGradient(self.img_gray)
        self.img_shape = np.array(self.img_gray.shape)
        self.stroke_cnt = stroke_cnt
        self.pop_size = pop_size
        self.min_stroke_size = (
            min_stroke_size * max(self.img_shape)).astype(int)
        self.max_stroke_size = (
            max_stroke_size * max(self.img_shape)).astype(int)
        self.img_buffer = [np.zeros(
            (self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8)]

    def initPopulation(self):
        self.population = []
        for p in range(self.pop_size):
            dna = []
            for s in range(self.stroke_cnt):
                color = np.random.randint(low=0, high=256, size=3, dtype=int)
                posx, posy = np.random.randint(
                    low=[0, 0], high=self.img_shape, size=2)  # no mask
                size = np.random.randint(
                    low=self.min_stroke_size, high=self.max_stroke_size, size=2)
                local_mag = self.img_grads[0][posx][posy]
                local_angle = self.img_grads[1][posx][posy] + 90
                rotation = int((np.random.rand() * 360 - 180) *
                               (1 - local_mag) + local_angle)
                alpha = np.random.rand()
                dna.append(
                    [int(color[0]), int(color[1]), int(color[2]), posx, posy, size[0], size[1], rotation, alpha, 0, 0, 0])
            self.population.append(dna)

    def paint(self, stages=20, generations=100, show_progress=True):
        canvas = np.copy(self.img_buffer[-1])
        for s in range(stages):
            self.initPopulation()
            # canvas = np.copy(self.img_buffer[-1])
            error_cache = self._evaluate(canvas)
            img_cache = canvas
            for g in range(generations):
                # cross over
                # mutation
                for i in range(len(self.population)):
                    new_dna, img = self._mutateDNA(self.population[i], canvas, 0.8)
                    self.population[i] = new_dna
                    error = self._evaluate(img)
                    if error < error_cache:
                        error_cache = error
                        img_cache = img
            # self.img_buffer.append(img_cache)      
            canvas = img_cache      
            if show_progress:
                plt.imshow(img_cache[:, :, ::-1])
                plt.show()
            print(f"stage: {s}")
            plt.imsave(f"output/stage{s}.png", img_cache[:, :, ::-1])

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

    def _mutateSeg(self, seg, prob):
        if np.random.rand() > prob:
            return seg
        color_mutate_range = 128
        pos_mutate_range = (self.img_shape/16).astype(int)
        size_mutate_range = ((self.max_stroke_size - self.min_stroke_size) / 4).astype(int)
        rotation_mutate_range = 180
        alpha_mutate_range = 0.5

        seg_copy = copy.deepcopy(seg)
        index_options = [0, 1, 2, 3, 4]
        change_indices = []
        change_cnt = np.random.randint(low=1, high=len(index_options)+1)
        for i in range(change_cnt):
            idx = np.random.randint(low=0, high=len(index_options))
            change_indices.append(index_options.pop(idx))
        np.sort(change_indices)
        for idx in change_indices:
            if idx == 0:  # color
                color = np.array(seg[0:3], dtype=int) + \
                    np.random.randint(low=-color_mutate_range, high=color_mutate_range, size=3, dtype=int)
                seg_copy[0:3] = np.clip(color, 0, 256).astype(int)
            elif idx == 1:  # pos
                pos = np.array(seg[3:5], dtype=int) + \
                    np.random.randint(low=-pos_mutate_range, high=pos_mutate_range, size=2, dtype=int)
                seg_copy[3:5] = np.clip(pos, 0, self.img_shape)
            elif idx == 2:  # size
                size = np.array(seg[5:7], dtype=int) + \
                    np.random.randint(low=-size_mutate_range, high=size_mutate_range, size=2, dtype=int)
                seg_copy[5:7] = np.clip(size, self.min_stroke_size, self.max_stroke_size)
            elif idx == 3: # rotation
                rotation = seg[7] + \
                    random.randint(-rotation_mutate_range, rotation_mutate_range)
                seg_copy[7] = rotation % 360
            elif idx == 4: # alpha
                alpha = seg[8] + np.random.rand() * 2 * alpha_mutate_range - alpha_mutate_range
                seg_copy[8] = np.clip(alpha, 0.0, 1.0)
        return seg_copy

    def _mutateDNA(self, dna, img, prob):
        img_cache = self._drawDNA(img, dna)
        error_cache = self._evaluate(img_cache)
        dna_copy = copy.deepcopy(dna)
        for i in range(len(dna)):
            new_seg = self._mutateSeg(dna[i], prob)
            dna_copy[i] = new_seg
            new_img = self._drawDNA(img, dna_copy)
            new_error = self._evaluate(new_img)
            if new_error < error_cache: # mutation success
                error_cache = new_error
                img_cache = new_img
            else: # mutation failed
                dna_copy[i] = dna[i]
        return (dna_copy, img_cache)
    
    def _drawStroke(self, img, stroke):
        img_copy = np.copy(img)
        img_copy = cv2.ellipse(img=img_copy, center=stroke[4:2:-1], axes=stroke[5:7],
                               angle=stroke[7], startAngle=0, endAngle=360, color=[int(stroke[0]), int(stroke[1]), int(stroke[2])], thickness=-1)
        return cv2.addWeighted(img_copy, stroke[8], img, 1-stroke[8], 0)

    def _drawDNA(self, img, dna):
        img_copy = np.copy(img)
        for stroke in dna:
            img_copy = self._drawStroke(img_copy, stroke)
        return img_copy

    def _evaluate(self, img):
        diff1 = cv2.subtract(img, self.original_img)
        diff2 = cv2.subtract(self.original_img, img)
        total_diff = cv2.add(diff1, diff2)
        total_diff = np.sum(total_diff)
        return total_diff

def main():
    np.random.seed(0)
    plt.axis('off')

    painter = Painter(img_path='assets/monalisa-515px.jpg', pop_size=1)
    painter.paint(stages=100, show_progress=False)    
    # plt.imshow(painter.original_img[:,:,::-1]) # bgr to rgb
    # plt.imshow(painter.img_gray, cmap="gray")
    # plt.imshow(painter.img_grads[0], cmap="gray")


if __name__ == '__main__':
    main()
