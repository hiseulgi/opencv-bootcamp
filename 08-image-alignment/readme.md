1. Image Alignment is used to make images from different sources line up with each other, for various applications like building panoramic images or comparing medical scans.

2. Transformations in image alignment:
    - Translation: Shift the pixel coordinates in the original image to the translated image.
    - Euclidean Transformations: Include rotation while preserving size and shape.
    - Affine Transformations: Include rotation, shear, and scale changes, preserving parallel lines.
    - Homography: The most general transformation for 2D images, allows warping an image to change its perspective.

3. Document Alignment Example: Aligning a filled-out form image to its original template using homography, making tasks like optical character recognition easier.

4. Key Points and Descriptors: Extracting meaningful information from images using feature extraction algorithms like ORB (Oriented FAST and Rotated BRIEF).

5. Matching Key Points: Identifying matching key points between two images to compute the homography.

6. Homography Computation: Computing the homography using RANSAC (RANdom SAmple Consensus), a robust algorithm for filtering out outliers.

7. Warp Perspective: Applying the homography transformation to the image to align it with the template, effectively changing its perspective.

8. Applications: Image alignment has various applications in computer vision, such as panoramic image stitching, object tracking, and medical image analysis.

9. Conclusion: Image alignment is a powerful technique that can be easily implemented using libraries like OpenCV and offers many possibilities for experimentation with your own images.