import os
import cv2
import numpy as np
from pathlib import Path #to create system-idependent paths
from random import randint, choice, seed
from scipy.ndimage import rotate
from scipy import ndimage
from skimage.draw import ellipse

#seed(42) #number for random generator to start with for generation. Default: sys-time

output_folder_name = ("dataset_outlined_rotated1")

# Configuration
output_dir = Path(output_folder_name)  #pathlib.Path instantiates either PosixPath or WindowsPath
img_size = 64
num_images_per_class = 500
polygon_number_of_vertices = 5
shape_classes = ["circle", "ellipse", "triangle", "rectangle", "square"]#, "polygon"] #"kite"]

thickness = 1
#thickness = -1
# color = (0, 1, 0) #BGR
# color = (randint(0, 255), randint(0, 255), randint(0, 255))
#color = "random"
color = "black"

def color_function (color):
    if color is None or color == "black" or color == 0 or color == (0, 0, 0):
        return (0, 1, 0)
    else:
        return (randint(0, 255), randint(0, 255), randint(0, 255))

def create_blank_image():
    #create 3D-array filled with ones of datatype between 0-255, set all values to 255 (=white)
    return 255 * np.ones((img_size, img_size, 3), dtype=np.uint8) #unsigned integer 8bit

def draw_circle(img, thickness, color):
    color = color_function(color)
    radius = randint(2, int(img_size//2-1)) #returns number between a and b
    center = (randint(radius, img_size - radius), randint(radius, img_size - radius))
    cv2.circle(img, center, radius, color, thickness) #b+w, filled - thickness=-1 fills the circle
    #cv2.circle(img, center, radius, color, thickness) #b+w, outlined
    #cv2.circle(img, center, radius, color, thickness) #colored, filled

def draw_ellipse(img, thickness, color):
    color = color_function(color)
    axes_width = randint(4, int(img_size//2-1)) #returns number between a and b
    center = (randint(axes_width, img_size - axes_width), randint(axes_width, img_size - axes_width))
    angle = randint(0, 180)
    axes_height = randint(2, axes_width-1)
    cv2.ellipse(img, center, [axes_width, axes_height], angle, 0, 360, color, thickness) #b+w, filled
    #cv2.ellipse(img, center, [axes_width, axes_height], angle, 0, 360, color, 1) #b+w, outlined
    #cv2.ellipse(img, center, [axes_width, axes_height], angle, 0, 360, color, -1) #colored, filled

def draw_rectangle(img, thickness, color):
    color = color_function(color)
    rectangle_height = randint(2, int(img_size//(2**.5)-1))
    rectangle_width = randint(2, int(img_size//(2**.5)-1))
    rotation_angle = randint(0, 360)
    image_mask = np.zeros_like(img)
    x1, y1 = randint(0, img_size-rectangle_width), randint(0, img_size-rectangle_height)
    pts = np.array([
        [x1, y1],
        [x1, y1 + rectangle_height],
        [x1 + rectangle_width, y1 + rectangle_height],
        [x1 + rectangle_width, y1],
    ], dtype=np.int32)

    mask, rotated = draw_polygon(image_mask, pts, color, thickness, rotation_angle)
    #x2, y2 = x1 + rectangle_width, y1 + rectangle_height

    #img_with_rectangle = cv2.rectangle(image_mask, (img_size//2-rectangle_width//2, img_size//2-rectangle_height//2),(img_size//2+rectangle_width//2, img_size//2+rectangle_height//2), color, thickness) #filled, colored
    #img_with_rectangle = cv2.rectangle(image_mask, (x1, y1), (x2, y2), color, thickness) # b+w, outline
    #img_with_rectangle = cv2.rectangle(image_mask, (img_size//2-rectangle_width//2, img_size//2-rectangle_height//2), (img_size//2+rectangle_width//2, img_size//2+rectangle_height//2), color, thickness) #filled, colored
    #rotated = ndimage.rotate(img_with_rectangle, rotation_angle, reshape=False, mode='constant', cval=0)
    #rotated = np.clip(rotated, 0, 255).astype(np.uint8)
    #cv2.addWeighted(img, 1, rotated, 1, 0)
    #mask = np.any(rotated > 0, axis=-1)
    img[mask > 0] = rotated[mask > 0]

def draw_square(img, thickness, color):
    color = color_function(color)
    size = randint(2, int(img_size//(2**.5)-1))
    rotation_angle = randint(0, 360)
    image_mask = np.zeros_like(img)
    x1, y1 = randint(0, img_size - size), randint(0, img_size - size)
    pts = np.array([
        [x1, y1],
        [x1, y1 + size],
        [x1 + size, y1 + size],
        [x1 + size, y1]
    ], dtype=np.int32)

    mask, rotated = draw_polygon(image_mask, pts, color, thickness, rotation_angle)

    #x1, y1 = randint(0, img_size-size), randint(0, img_size-size)
    #img_with_square = cv2.rectangle(image_mask, (img_size//2-size//2, img_size//2-size//2), (img_size//2 + size//2, img_size//2 + size//2), color, thickness) #filled, black
    #img_with_square = cv2.rectangle(image_mask, (x1, y1), (x1 + size, y1 + size), color, thickness) # b+w, outline
    #(h, w) = img_with_square.shape[:2]
    #center = (w / 2, h / 2)
    #M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    #rotated = cv2.warpAffine(img_with_square, M, (w, h))
    # filled, colored
    #img_with_square = cv2.rectangle(image_mask, (img_size//2-size//2, img_size//2-size//2), (img_size//2 + size//2, img_size//2 + size//2), color, thickness)
    #rotated = ndimage.rotate(img_with_square, rotation_angle, reshape=False, mode='constant', cval=0)
    #rotated = np.clip(rotated, 0, 255).astype(np.uint8)
    #cv2.addWeighted(img, 0.1, rotated, 1, 0)
    #mask = np.any(rotated > 0, axis=-1)
    img[mask >0] = rotated[mask > 0]

def draw_triangle(img, thickness, color):
    color = color_function(color)
    pts = []
    for i in range(3):
        point = [randint(0, img_size), randint(0, img_size)]
        pts.append(point)
    pts = np.array(pts)#, dtype=np.int32)
    draw_polygon(img, pts, color, thickness)
    #cv2.fillPoly(img, [pts], color) #filled, black
    #cv2.polylines(img, [pts], True, color, 1) #outline
    #cv2.fillPoly(img, [pts], color)

def draw_polygon (img, pts, color, thickness, angle = 0):
    if thickness == -1:
        cv2.fillPoly(img, [pts], color)
    else:
        cv2.polylines(img, [pts], True, color, thickness)

    if pts.size < 3:
        return

    #(img_height, img_width) = img.shape[:2]
    x, y = np.mean(pts, axis=0).astype(int)
    center = (int(x), int(y))

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img_size, img_size), flags=cv2.INTER_NEAREST)

    # Mask and overlay (keep all non-zero pixels, even if black)
    mask = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY) > 0
    rotated_mask = cv2.warpAffine(mask.astype(np.uint8), M, (img_size, img_size), flags=cv2.INTER_NEAREST)
    return mask, rotated

'''
def draw_triangle(img, color = (0, 1, 0), thickness):
    pts = np.array([
        [randint(10, 54), randint(10, 54)],
        [randint(10, 54), randint(10, 54)],
        [randint(10, 54), randint(10, 54)]
    ])
    cv2.drawContours(img, [pts], 0, color, thickness)
    # As opencv does not have a funtion triangle ontours can be explained simply as a curve joining all the
    # continuous points (along the boundary), having same color or intensity.
    # cv.drawContours function is used to draw any shape if you have its boundary points. Its first
    # argument is source image, second argument is the contours which should be passed as a Python
    # list, third argument is index of contours (useful when drawing individual contour. To draw
    # all contours, pass -1) and remaining arguments are color, thickness etc.
    '''

'''
def draw_kite(img, color = (0, 1, 0), thickness):
    cx, cy = randint(20, 44), randint(20, 44)
    dx, dy = randint(10, 15), randint(10, 15)
    pts = np.array([
        [cx, cy - dy],
        [cx - dx, cy],
        [cx, cy + dy],
        [cx + dx, cy]
    ])
    cv2.drawContours(img, [pts], 0, color, thickness)
'''

# Shape drawing map/dictionary: key = name of shape, value = shape-creating function
draw_funcs = {
    "circle": draw_circle,
    "ellipse": draw_ellipse,
    "triangle": draw_triangle,
    "rectangle": draw_rectangle,
    "square": draw_square
    #"polygon": draw_polygon
    #"kite": draw_kite
}

# Dataset generation
for shape in shape_classes:
    shape_dir = output_dir / shape
    all_shapes_dir = output_dir / "all_shapes_dir"
    shape_dir.mkdir(parents=True, exist_ok=True)
    all_shapes_dir.mkdir(parents=True, exist_ok=True)
    draw_fn = draw_funcs[shape]

    for i in range(num_images_per_class):
        img = create_blank_image()
        draw_fn(img, thickness, color)
        filepath1 = shape_dir / f"{shape}_{i:03d}.png"
        filepath2 = all_shapes_dir / f"{shape}_{i:03d}.png"
        cv2.imwrite(str(filepath1), img)
        cv2.imwrite(str(filepath2), img)

print("✅ Dataset created in '", output_folder_name, "/' folder.")