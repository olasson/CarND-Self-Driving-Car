# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

The pipeline consists of the following steps:

- **Convert to grayscale**
- **Apply Gaussian Blurr**
- **Apply Canny Edge detection**
- **Apply Region of Interest mask**
- **Apply Hough Transform**
- **Combine Line image and Original image**

Details on the pipeline can be found in the notebook. 

The **draw_lines** function was improved/modified by first implementing a **draw_line** function that uses least squares to draw a single solid line between two points. Next, the **draw_lines** function then simply calculates the slope of the lines found by the hough TF, sorts them into "left" and "right" lines, and passes them on to **draw_line**. Details can be found in the notebook. 
 




### 2. Identify potential shortcomings with your current pipeline


-**Shaky Lines**: As can be seen from the output videos in the notebook, the lines tend to shake quite a bit. 

-**Lines extends too far**: The lines intersect at the top. Not sure if this is a major issue, but in the example videos there is clearly a gap between the lines at the top.

-**Hilarious, abysmal failure when applied to the challenge video**: The video output pretty much speaks for itself. I suspect this has to do with curved lane lines, since the pipline preforms decently when applied to both yellow and white straight lane lines (I don't think it is a color related issue). 

### 3. Suggest possible improvements to your pipeline

-**Shaky Lines**: Honestly a little clueless on what to do with this one, perhaps better averaging? 

-**Lines extends too far**: Better tuning of the Hough TF parameters could possibly remedy this. 

-**Hilarious, abysmal failure when applied to the challenge video**: The draw lines function could be modified to couple line "segments" instead of drawing a single solid line. Other modifications might be necessary aswell, but i completely ran out of time on this one. 
