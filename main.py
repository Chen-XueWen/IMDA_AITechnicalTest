import os
import numpy as np
from collections import defaultdict

class Captcha(object):
    def __init__(self, input_folder='sampleCaptchas/input', output_folder='sampleCaptchas/output'):
        """
        Preprocess training data: load mean character templates.
        """
        self.mean_cells = self._build_mean_cells(input_folder, output_folder)

    def __call__(self, im_path, save_path):
        """
        Inference on a single .txt test image and save the result.
        Args:
            im_path: .txt image path
            save_path: file path to save the one-line output
        """
        result = self._recognize_characters(im_path)
        with open(save_path, 'w') as f:
            f.write(result)

    def _load_image_from_txt(self, path):
        with open(path, 'r') as file:
            lines = file.readlines()
        height, width = map(int, lines[0].strip().split())
        pixels = []
        for line in lines[1:]:
            for pixel_str in line.strip().split():
                r, g, b = map(int, pixel_str.split(','))
                pixels.append([r, g, b])
        return np.array(pixels, dtype=np.uint8).reshape((height, width, 3))

    def _build_mean_cells(self, input_folder, output_folder):
        char_to_cells = defaultdict(list)

        for filename in sorted(os.listdir(input_folder)):
            if not filename.endswith('.txt'):
                continue

            image_path = os.path.join(input_folder, filename)
            output_index = filename.replace('input', '').replace('.txt', '')
            label_path = os.path.join(output_folder, f'output{output_index}.txt')
            if not os.path.exists(label_path):
                continue

            image = self._load_image_from_txt(image_path)
            with open(label_path, 'r') as f:
                label = f.read().strip()

            cropped_image = image[11:21, 5:49, :]
            cells = [(0, 8), (9, 17), (18, 26), (27, 35), (36, 44)]

            for i, (start, end) in enumerate(cells):
                if i >= len(label):
                    continue
                char = label[i]
                cell_img = cropped_image[:, start:end, :]
                char_to_cells[char].append(cell_img)

        mean_cells = {}
        for char, cell_list in char_to_cells.items():
            stacked = np.stack(cell_list, axis=0)
            mean_rgb = np.mean(stacked, axis=0)
            gray = 0.2989 * mean_rgb[:, :, 0] + 0.5870 * mean_rgb[:, :, 1] + 0.1140 * mean_rgb[:, :, 2]
            thresholded = np.where(gray > 70, 255, 0).astype(np.uint8)
            mean_cells[char] = thresholded
        return mean_cells

    def _recognize_characters(self, test_image_path):
        image = self._load_image_from_txt(test_image_path)
        cropped_image = image[11:21, 5:49, :]
        cells = [(0, 8), (9, 17), (18, 26), (27, 35), (36, 44)]

        recognized = ''
        for start, end in cells:
            cell_img = cropped_image[:, start:end, :]
            gray = 0.2989 * cell_img[:, :, 0] + 0.5870 * cell_img[:, :, 1] + 0.1140 * cell_img[:, :, 2]
            gray = np.where(gray > 70, 255, 0).astype(np.uint8)

            best_char = None
            best_score = float('inf')
            for char, template in self.mean_cells.items():
                if gray.shape != template.shape:
                    continue
                score = np.sum(np.abs(gray.astype(int) - template.astype(int)))
                if score < best_score:
                    best_score = score
                    best_char = char
            recognized += best_char if best_char else '?'

        return recognized

if __name__ == "__main__":
    captcha = Captcha()
    captcha('sampleCaptchas/input/input01.txt', 'output.txt')
