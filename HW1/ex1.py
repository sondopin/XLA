import cv2 
import matplotlib.pyplot as plt
import numpy as np

# Load an image from file as function
def load_image(image_path):
    """
    Load an image from file, using OpenCV
    """
    return  cv2.imread(image_path)

# Display an image as function
def display_image(image, title):
    """
    Display an image using matplotlib. Rembember to use plt.show() to display the image
    """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
# grayscale an image as function
def grayscale_image(image):
    """
    Convert an image to grayscale. Convert the original image to a grayscale image. In a grayscale image, the pixel value of the
    3 channels will be the same for a particular X, Y coordinate. The equation for the pixel value
    [1] is given by:
        p = 0.299R + 0.587G + 0.114B
    Where the R, G, B are the values for each of the corresponding channels. We will do this by
    creating an array called img_gray with the same shape as img
    """
    img_gray = np.zeros(image.shape[:-1], dtype=np.uint8)
    weights = np.array([0.299, 0.587, 0.114])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img_gray[i, j] = np.dot(image[i, j], weights)
    return img_gray


# Save an image as function
def save_image(image, output_path):
    """
    Save an image to file using OpenCV
    """
    cv2.imwrite(output_path,image)


# flip an image as function 
def flip_image(image):
    """
    Flip an image horizontally using OpenCV
    """
    return cv2.flip(image, 1)


# rotate an image as function
def rotate_image(image, angle):
    """
    Rotate an image using OpenCV. The angle is in degrees
    """
    num_rows, num_cols = image.shape[:2]    
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))


if __name__ == "__main__":
    # Load an image from file
    img = load_image('hw1/images/uet.png')

    # Display the image
    display_image(img, "Original Image")

    # Convert the image to grayscale
    img_gray = grayscale_image(img)

    # Display the grayscale image
    display_image(img_gray, "Grayscale Image")

    # Save the grayscale image
    save_image(img_gray, "hw1/images/lena_gray.jpg")

    # Flip the grayscale image
    img_gray_flipped = flip_image(img_gray)
    save_image(img_gray_flipped, "hw1/images/flipped.jpg")
    # Display the flipped grayscale image
    display_image(img_gray_flipped, "Flipped Grayscale Image")

    # Rotate the grayscale image
    img_gray_rotated = rotate_image(img_gray, 45)

    # Display the rotated grayscale image
    display_image(img_gray_rotated, "Rotated Grayscale Image")

    # Save the rotated grayscale image
    save_image(img_gray_rotated, "hw1/images/lena_gray_rotated.jpg")

    # Show the images
    plt.show() 
