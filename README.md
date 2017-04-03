#**Finding Lane Lines on the Road** 

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

In this project, I used Python3 and OpenCV to find lane lines in the road images. The following techniques were used:

* Color Selection
* Gaussian Smoothing
* Canny Edge detection
* Region of Interest Selection
* Hough Transform Line Detection

## Color Selection
The images from RGB to HLS color space as yellow and white colors were clearly recognizable. The following command is used:
cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
* Use cv2.inRange to filter the white color and the yellow color seperately.
* Use cv2.bitwise_or to combine these two binary masks.
* Use cv2.bitwise_and to apply the combined mask onto the original RGB image

This is described in select_rgb_white_yellow() below:

```def select_rgb_white_yellow(image): 
    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    white_mask = cv2.inRange(hls_image, np.uint8([20,200,0]), np.uint8([255,255,255])) 
    yellow_mask = cv2.inRange(hls_image, np.uint8([10,50,100]), np.uint8([100,255,255]))

    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked
```
     
