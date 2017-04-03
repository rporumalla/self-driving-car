#**Finding Lane Lines on the Road** 

In this project, I used Python3 and OpenCV to find lane lines in the road images. The following techniques were used:

* Color Selection
* Gaussian Smoothing
* Canny Edge Detection
* Region of Interest Selection
* Hough Transform Line Detection

## Color Selection
The images from RGB to HLS color space as yellow and white colors were clearly recognizable.
* Use cv2.inRange to filter the white color and the yellow color seperately.
* Use cv2.bitwise_or to combine these two binary masks.
* Use cv2.bitwise_and to apply the combined mask onto the original RGB image

```def select_rgb_white_yellow(image): 
    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    white_mask = cv2.inRange(hls_image, np.uint8([20,200,0]), np.uint8([255,255,255])) 
    yellow_mask = cv2.inRange(hls_image, np.uint8([10,50,100]), np.uint8([100,255,255]))

    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked
```
     
## Gaussian Smoothing
cv2.GaussianBlur is used to smooth out rough edges. The GaussianBlur takes a kernel_size parameter which needs to be selected appropriately. 

```def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```

## Canny Edge Detection
cv2.Canny takes two threshold values that are defined by trial and error. Canny recommended a upper:lower ratio between 2:1 and 3:1.

```def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)
```    

## Region of Interest Selection
The image mask is applied to only the region defined by the polygon formed from the vertices. The rest of the image is set to black.

```def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```

## Hough Transform Line Detection
cv2.HoughLinesP is used to detect lines in the edge images.

The following parameters need to be adjusted accordingly:

rho: distance resolution in pixels of the Hough grid
theta: angular resolution in radians of the Hough grid
threshold: minimum number of votes (intersections in Hough grid cell)
min_line_len: minimum number of pixels making up a line
max_line_gap: maximum gap in pixels between connectable line segments

<img src="test_images/solidWhiteCurve_Output1.jpg" width="480" alt="Hough Lines Image1" />
<img src="test_images/solidWhiteCurve_Output2.jpg" width="480" alt="Hough Lines Image1" />
