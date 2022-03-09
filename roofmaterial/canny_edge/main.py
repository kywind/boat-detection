import cv2
import numpy as np
import os
from tqdm import tqdm


def gaussian_filter(shape, var):
    z = np.array([(shape[0]-1)/2, (shape[1]-1)/2])
    res = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            x = np.array([i,j])
            res[i,j] = np.exp((-(np.linalg.norm(x-z)**2))/(2*var**2))
    res /= res.sum()
    return res


def normalize(img, xmin=0.0, xmax=255.0):
    img = np.float32(img)
    img = (xmax - xmin) * (img - np.min(img)) / (np.max(img) - np.min(img)) + xmin
    return img


def conv2d(img, kernel, conv_type='same'):

    def get_padding(k, mode):
        if mode == 'full':
            padding = (k-1, k-1)
        elif mode == 'same':
            half = (k-1) // 2
            padding = (half, half) if (k-1) % 2 == 0 else (half, half+1)
        elif mode == 'valid':
            padding = (0, 0)
        return padding

    kernel = kernel[::-1, ::-1]
    h_pad = get_padding(kernel.shape[0], mode=conv_type)
    w_pad = get_padding(kernel.shape[1], mode=conv_type)

    X = np.pad(img, (h_pad, w_pad), 'reflect')
    result = np.zeros((X.shape[0]-kernel.shape[0]+1, X.shape[1]-kernel.shape[1]+1))
    for r in range(result.shape[0]):
        for c in range(result.shape[1]):
            result[r,c] = (kernel * X[r:r+kernel.shape[0], c:c+kernel.shape[1]]).sum()
    return result


def nms(grad_x, grad_y):

    def get_interp_grad(x1, y1, x2, y2, g_y, g_x):
        if (y1 == y2 and x2 == 0) or (x1 == x2 and y2 == 0):
            x1, x2, y1, y2 = x2, x1, y2, y1
        gx1 = grad_x[y1, x1]
        gy1 = grad_y[y1, x1]
        gx2 = grad_x[y2, x2]
        gy2 = grad_y[y2, x2]
        k = abs(g_y) / (abs(g_x) + 1e-8)
        k = k if k <= 1 else 1/k
        gx = k * gx2 + (1-k) * gx1
        gy = k * gy2 + (1-k) * gy1
        return np.sqrt(gx ** 2 + gy ** 2)

    grad_x = np.pad(grad_x, ((1,1), (1,1)), 'constant')
    grad_y = np.pad(grad_y, ((1,1), (1,1)), 'constant')
    nl = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1), (1,0)]
    norm = np.sqrt(grad_x ** 2 + grad_y ** 2)
    index = 4 * (grad_y<0) + 2 * (grad_y*grad_x<0) + (np.abs(grad_y)<np.abs(grad_x)) * (grad_y*grad_x<0)
    for r in range(1, grad_x.shape[0]-1):
        for c in range(1,grad_x.shape[1]-1):
            i = index[r,c]
            g1 = get_interp_grad(c+nl[i][0], r-nl[i][1], c+nl[i+1][0], r-nl[i+1][1], grad_y[r,c], grad_x[r,c])
            g2 = get_interp_grad(c-nl[i][0], r+nl[i][1], c-nl[i+1][0], r+nl[i+1][1], grad_y[r,c], grad_x[r,c])
            norm[r,c] = norm[r,c] * (norm[r,c] >= max(g1,g2))
    norm = 255. * (norm - np.min(norm)) / (np.max(norm) - np.min(norm)) 
    return norm


def hysteresis(img, lo, hi):
    img = np.pad(img, ((1,1),(1,1)), 'constant')
    res = 255. * (img >= hi)
    q = list(np.argwhere(res))
    point = 0
    while len(q) > point:
        r, c = q[point][0], q[point][1]
        point += 1
        for rr in range(r-1, r+2):
            for cc in range(c-1, c+2):
                if img[rr, cc] >= lo and res[rr, cc] == 0:
                    q.append([rr, cc])
                    res[rr, cc] = 255.
    return res[1:-1,1:-1]


def do_canny_edge_detection(img):
    """
    Implement canny edge detection for 2D image.
    :param img: float/int array, given image, shape: (height, width)
    :return detection results, a 2D image numpy array.
    """
    sobel_x = np.array([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]])
    sobel_y = np.array([[1., 2., 1.],
                        [0., 0., 0.],
                        [-1., -2., -1.]]) 
    img = img.astype('float32')
    img = conv2d(img, gaussian_filter((5,5), 0.001))
    grad_x = conv2d(img, sobel_x)
    grad_y = conv2d(img, sobel_y)
    res = normalize(np.sqrt(grad_x**2 + grad_y**2))
    # res = nms(grad_x, grad_y)
    # res = hysteresis(res, 10, 30)
    return res


if __name__ == '__main__':
    img_path='../result/2018_pred_single_png/'
    # img_path = 'grad_vis_kmeans/'
    out_path = 'grad_vis_new_png_png/'
    os.makedirs(out_path, exist_ok=True)
    indices = range(3601)
    for i in tqdm(indices):
        f = img_path + '{}.png'.format(i)
        img = cv2.imread(f)

        width_range = list(range(img.shape[1]))  # horizontal
        height_range = list(range(img.shape[0]))  # vertical
        for j in range(img.shape[0]):
            if img[j].mean() <= 20:
                height_range.remove(j)
        for j in range(img.shape[1]):
            if img[:, j].mean() <= 20:
                width_range.remove(j)
        img = img[height_range]
        img = img[:, width_range]

        scale = 4
        width  = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        
        # img = cv2.GaussianBlur(img, (5,5), 1)
        # grad = do_canny_edge_detection(img)
        # grad = cv2.Canny(img, 10, 100)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1)
        
        (ret, gray) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        edge = cv2.Canny(gray, 100, 200)
        kernel = np.ones((7, 7), np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=2)
        edge = cv2.erode( edge, kernel, iterations=2)
        # edge = cv2.dilate(edge, kernel, iterations=2)
        # edge = cv2.erode( edge, kernel, iterations=2)

        # (cnts, _) = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # for cnt in cnts[:3]:
        #     epsilon = 0.03 * cv2.arcLength(cnt, True)
        #     approx = cv2.approxPolyDP(cnt, epsilon, False)
        #     cv2.drawContours(img, [approx], -1, (0,255,0), 1)

        # cnts.sort(key=cv2.contourArea)
        # for cnt in cnts[::-1]:
        #     area = cv2.contourArea(cnt)
        #     if area <= 10 or area >= 0.5 * img.shape[0] * img.shape[1]:
        #         continue
        #     cv2.drawContours(img, [cnt], -1, (0,255,0), 1)
        #     break

        cv2.imwrite(out_path + '{}.png'.format(i), img)
        cv2.imwrite(out_path + '{}_grad.png'.format(i), edge)
            

    