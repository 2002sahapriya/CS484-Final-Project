import os
from PIL import Image
import exifread
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import least_squares
import random
import argparse
import sys

# Step 1: Data Preparation
def load_bracketed_images(directory):
    """
    Load all images in the given directory, assuming they are part of a bracketed set.
    """
    images = []
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg', '.HEIC'))]
    for file in sorted(image_files):
        # Read the image
        img = cv2.imread(os.path.join(directory, file))
        # Convert image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images

def extract_exposure_times(directory):
    """
    Extract exposure times from the images' EXIF data, handling both fractional and whole number formats.
    """
    exposure_times = []
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for file in sorted(image_files):
        with open(os.path.join(directory, file), 'rb') as f:
            tags = exifread.process_file(f)
            exposure_time_tag = tags.get('EXIF ExposureTime')
            if exposure_time_tag:
                exposure_time_str = str(exposure_time_tag)
                if '/' in exposure_time_str:
                    # Handle fractional exposure time
                    numerator, denominator = map(int, exposure_time_str.split('/'))
                    exposure_time_sec = numerator / denominator
                else:
                    # Handle whole number exposure time
                    exposure_time_sec = float(exposure_time_str)
                exposure_times.append(exposure_time_sec)
            else:
                print(f"Exposure time not found in {file}.")
    return exposure_times

def extract_channels(images):
    """
    Extracts the blue, green, and red channels from a list of RGB images.
    
    Parameters:
    - images: list of numpy.ndarray, where each ndarray is an RGB image.
    
    Returns:
    - Tuple of three lists: (blue_channels, green_channels, red_channels),
      where each list contains the respective channel images of all input images.
    """
    blue_channels = []
    green_channels = []
    red_channels = []
    
    for img in images:
        # Split the image into its three channels
        blue, green, red = cv2.split(img)
        
        # Append each channel to its respective list
        blue_channels.append(blue)
        green_channels.append(green)
        red_channels.append(red)
    
    return blue_channels, green_channels, red_channels

# Function to display an image
def display_images(images, save_name ='input_images', titles=None, cols=3, cmap=None):
    """
    Display a list of images in a grid.

    Args:
    - images (list of ndarray): List of images to display.
    - titles (list of str): Optional list of titles for each image.
    - cols (int): Number of columns in the display grid.
    - cmap (str): Colormap used for displaying grayscale images.
    """
    n_images = len(images)
    rows = n_images // cols + int(n_images % cols != 0)
    fig = plt.figure(figsize=(15, rows * 5))

    for i in range(1, n_images + 1):
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(cv2.cvtColor(images[i-1], cv2.COLOR_BGR2RGB), cmap=cmap)
        ax.axis('off')
        if titles and i <= len(titles):
            ax.set_title(f'Exposure Time: {titles[i-1]} secs')
    plt.tight_layout()
    plt.savefig(f'{save_name}_input_imgs.png')
    plt.show()

## Step 2: Estimating response curves
# Weighting function
def weights(z):
    """
    Define the weighting function for each pixel.
    """
    z_min, z_max = 0., 255.
    if z <= (z_min + z_max) / 2:
        return z - z_min
    return z_max - z

# Helper: Function to select pixels for sampling
def select_random_pixels(num_pixels, num_samples):
    # Randomly select 'num_samples' unique pixel indices from 'num_pixels' total pixels
    return random.sample(range(num_pixels), num_samples)

# Helper: Plot the pixel samples
def plot_pixel_samples(image, pixel_indices, save_name):
    # Plot an image with red dots at the locations of 'pixel_indices'
    # Convert list to numpy array for element-wise operations
    pixel_indices = np.array(pixel_indices)
    
    x = pixel_indices % image.shape[1] # Calculate the x-coordinates
    y =  pixel_indices // image.shape[1] # Calculate the y-coordinates
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.scatter(x, y, c='r')
    plt.title('Samples')
    plt.axis('off')
    plt.savefig(f'{save_name}_samples.png')
    plt.show()
    
# Function to get the pixel samples
def get_pixel_samples(image, save_name='results/samples', num_samples=100, save_samples=True):
    # Get 'num_samples' random pixel indices from 'image' and optionally plot the sample locations
    pixel_indices = select_random_pixels(image.size, num_samples)
    if save_samples:
        plot_pixel_samples(image, pixel_indices, save_name)
    return pixel_indices

def get_pixel_values_from_indices(images, indices):
    # Retrieve pixel values from 'images' at the locations specified by 'indices'
    # The list comprehension traverses each image and collects values from the flattened image array
    return [[img.ravel()[index] for index in indices] for img in images]

    """
    Construct the system of equations derived from the CRF model.
    """
    num_samples = pixel_samples.shape[0]
    num_images = pixel_samples.shape[1]
    num_pixels = num_samples * num_images
    num_equations = num_pixels + 255  # 256 intensities - 1
    A = np.zeros((num_equations, num_pixels + 256))  # 256 for the g function values
    b = np.zeros(num_equations)
    
    # Add data-fitting equations
    k = 0
    for i in range(num_samples):
        for j in range(num_images):
            wij = weighting_function(pixel_samples[i, j])
            z = int(pixel_samples[i, j])  # Ensure this is an integer for indexing
            A[k, i * num_images + j] = wij
            A[k, num_pixels + z] = -wij  # Ensure z is used as an index
            b[k] = wij * log_exposure_times[j]
            k += 1


    # Add smoothing equations
    for z in range(254):  # Exclude the first and last intensity value
        A[k, num_pixels + z] = smoothing_lambda * weighting_function(z + 1)
        A[k, num_pixels + z + 1] = -2 * smoothing_lambda * weighting_function(z + 1)
        A[k, num_pixels + z + 2] = smoothing_lambda * weighting_function(z + 1)
        k += 1

    # Fix the curve by setting its middle value to 0
    A[k, num_pixels + 128] = 1

    return A, b
# Function to calculate the camera response function/ curve
def compute_response_curve(pixel_values, log_exposures, lambda_smoothing=100):
    """
    Computes the camera response curve using a system of linear equations that incorporate data fitting and smoothness.
    
    Parameters:
    - pixel_values (numpy.ndarray): 2D array of pixel intensities across different exposures.
    - log_exposures (numpy.array): Array of logarithmic exposure times.
    - weights (numpy.array): Weight function values for each pixel intensity.
    - lambda_smoothing (float): Regularization parameter for smoothness.

    Returns:
    - numpy.array: The estimated camera response curve.
    """
    pixel_values = np.array(pixel_values)  # Convert list to numpy array if not already
    log_exposures = np.array(log_exposures)  # Ensure log_exposures is also an array
    
    max_pixel_value = 256 
    num_images = pixel_values.shape[0] # number of images
    num_samples = pixel_values.shape[1] # number of samples
    
    # Initialize the system matrix A and vector b
    A = np.zeros((num_samples * num_images + max_pixel_value + 1, max_pixel_value + num_samples), dtype=np.float32)
    b = np.zeros((A.shape[0], 1), dtype=np.float32)
    
    # Build the data-fitting term
    k = 0
    for sample in range(num_samples):
        for image in range(num_images):
            pixel = pixel_values[image][sample]
            weight = weights(pixel)
            A[k][pixel] = weight
            A[k][max_pixel_value + sample] = -weight
            b[k] = weight * log_exposures[image]
            k += 1
    
    # Center the curve - fix the curve by setting its middle value to 0 (middle = 256/2)
    A[k][128] = 1
    k += 1
    
    # Adding smoothness constraints
    for i in range(max_pixel_value - 1):
        A[k][i] = lambda_smoothing * weights(i + 1)
        A[k][i + 1] = -2 * lambda_smoothing * weights(i + 1)
        A[k][i + 2] = lambda_smoothing * weights(i + 1)
        k += 1
    
    # Solve the system to find the CRF
    solution = np.linalg.lstsq(A, b, rcond=None)[0]
    response_curve = solution[:max_pixel_value]
    lE = solution[max_pixel_value:np.size(solution, 0)]
    
    return response_curve, lE

# Plot the response curves
def plot_response_curves(blue_curve, green_curve, red_curve, save_name='unknown'):
    """
    Plots the camera response curves for the blue, green, and red channels individually and together in a grid layout.
    
    Parameters:
    - blue_curve (numpy.array): Camera response curve for the blue channel.
    - green_curve (numpy.array): Camera response curve for the green channel.
    - red_curve (numpy.array): Camera response curve for the red channel.
    """
    # Create a grid layout
    gs = gridspec.GridSpec(3, 2, width_ratios=[1,2], height_ratios=[3, 3, 3])
    
    plt.figure(figsize=(12, 8))
    
    ax0 = plt.subplot(gs[0, 0])
    ax0.plot(blue_curve, np.arange(256), 'b')
    ax0.set_xlabel('Log Exposure X')
    ax0.set_ylabel('Pixel Value Z')
    ax0.set_title('Blue Channel')
    ax0.grid(True)

    ax1 = plt.subplot(gs[1, 0])
    ax1.set_xlabel('Log Exposure X')
    ax1.set_ylabel('Pixel Value Z')
    ax1.plot(green_curve, np.arange(256), 'g')
    ax1.set_title('Green Channel')
    ax1.grid(True)

    ax2 = plt.subplot(gs[2, 0])
    ax2.plot(red_curve, np.arange(256), 'r')
    ax2.set_xlabel('Log Exposure X')
    ax2.set_ylabel('Pixel Value Z')
    ax2.set_title('Red Channel')
    ax2.grid(True)

    # Larger subplot for combined curves
    ax3 = plt.subplot(gs[0:3, 1])
    ax3.plot(blue_curve, np.arange(256), 'b', label='Blue')
    ax3.plot(green_curve,np.arange(256),  'g', label='Green')
    ax3.plot(red_curve, np.arange(256), 'r', label='Red')
    ax3.set_xlabel('Log Exposure X')
    ax3.set_ylabel('Pixel Value Z')
    ax3.set_title('Combined Response Curves')
    ax3.legend()
    ax3.grid(True)

    # Adjust layout for spacing
    plt.tight_layout()
    plt.savefig(f'{save_name}_response_curves.png')
    plt.show()

# Step 3: Build Radiance Maps
# Function to construct radiance map
def create_radiance_map(img_list, response_curve, log_exposures):
    # Get the dimensions of the images.
    img_height, img_width = img_list[0].shape
    
    # Create an empty array for the radiance map.
    radiance_map = np.zeros((img_height, img_width), dtype=np.float64)
    
    # Number of images in the list.
    num_images = len(img_list)
    
    # Process each pixel in the image.
    for i in range(img_height):
        for j in range(img_width):
            weighted_sum = 0
            total_weight = 0
            
            # Accumulate the weighted sum of the radiance for each image at this pixel.
            for img_index in range(num_images):
                pixel_value = img_list[img_index][i, j]
                g_value = response_curve[pixel_value][0]  # First element from the response curve.
                weight = weights(pixel_value)
                
                weighted_sum += weight * (g_value - log_exposures[img_index])
                total_weight += weight
            
            # Avoid division by zero; if no valid weights, use a fallback value.
            if total_weight > 0:
                radiance_map[i, j] = weighted_sum / total_weight
            else:
                middle_index = num_images // 2
                radiance_map[i, j] = response_curve[img_list[middle_index][i, j]][0] - log_exposures[middle_index]
                
    return radiance_map

def plot_radiance_histograms(radiance_maps, save_name):
    """
    Plots histograms for each color channel in a radiance map given as a list of 2D arrays.

    Args:
    radiance_maps (list of numpy.ndarray): List of 2D arrays where each array represents the radiance map of a color channel.
    """
    # Define channel colors assuming RGB order.
    colors = ['Red', 'Green', 'Blue']
    channel_colors = ['red', 'green', 'blue']

    # Create a figure and a set of subplots.
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # Plot histogram for each channel.
    for i, channel_data in enumerate(radiance_maps):
        ax = axes[i]
        # Flatten the 2D array to 1D for histogram plotting.
        data_flat = channel_data.flatten()
        # Plot the histogram with 256 bins.
        ax.hist(data_flat, bins=256, color=channel_colors[i], alpha=0.75)
        ax.set_title(f'{colors[i]} Channel Histogram')
        ax.set_xlabel('Radiance Value')
        ax.set_ylabel('Frequency')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_name}_histogram.png')
    plt.show()

# Display radiance maps for each channel 
def display_radiance_map(radiance_red, radiance_green, radiance_blue, save_name):
    """
    Displays a radiance map in pseudocolor.

    Args:
    radiance_red (numpy.ndarray): 2D array of radiance values for the red channel.
    radiance_green (numpy.ndarray): 2D array of radiance values for the green channel.
    radiance_blue (numpy.ndarray): 2D array of radiance values for the blue channel.
    """
    # Combine the radiance channels into a single image
    radiance_rgb = np.stack((radiance_red, radiance_green, radiance_blue), axis=-1)
    
    # Convert the radiance RGB image to grayscale to represent intensity
    radiance_gray = np.mean(radiance_rgb, axis=2)
    
    # Apply a colormap to the grayscale radiance map
    plt.imshow(radiance_gray, cmap='jet')
    plt.colorbar()  # Display a colorbar showing the scale
    plt.axis('off')  # Turn off axis labels and ticks
    plt.title('Radiance Map Pseudocolor')
    plt.legend()
    plt.savefig(f'{save_name}_psuedocolor.png')
    plt.show()

# Step 4: Construct HDR images + Tone Mapping
# Function to construct HDR image
def construct_hdr_image(E_b, E_g, E_r, width, height, channel):
    """
    Constructs an HDR image from logarithmic radiance estimates for each color channel.

    Args:
    E_b, E_g, E_r (numpy.ndarray): Logarithmic radiance estimates for the blue, green, and red channels.
    width (int): Width of the HDR image.
    height (int): Height of the HDR image.

    Returns:
    numpy.ndarray: Constructed HDR image in linear radiance scale.
    """
    hdr = np.zeros((width, height, channel), 'float32')
    hdr[..., 0] = np.reshape(np.exp(E_b), (width, height))
    hdr[..., 1] = np.reshape(np.exp(E_g), (width, height))
    hdr[..., 2] = np.reshape(np.exp(E_r), (width, height))
    return hdr

# Function to apply tone mapping
def tone_map_and_normalize(hdr_image):
    """
    Applies tone mapping using logarithmic scaling and normalization.

    Args:
    hdr_image (numpy.ndarray): HDR image in linear radiance scale.

    Returns:
    numpy.ndarray: Tone-mapped and normalized image.
    """
    # Apply logarithmic transformation and normalize
    # log transform
    logarithmic_image = np.log(hdr_image)
    # normalization
    normalize = lambda zi: (zi - zi.min()) / (zi.max() - zi.min())
    normalized_image = normalize(logarithmic_image)
    tonned_image = normalized_image / normalized_image.max()
    return tonned_image

# Function to plot and save HDR image
def plot_and_save_hdr_image(image, filename, lambda_smooth):
    """
    Displays and saves the image.

    Args:
    image (numpy.ndarray): The image to be displayed and saved.
    filename (str): The filename for saving the image.
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'HDR Tone-Mapped Image @ smoothing_lambda = {lambda_smooth}')
    plt.savefig(f'{filename}_hdr_{lambda_smooth}.png')
    plt.show()

# Step 6: Estimate Relative Exposures
def estimate_relative_exposure(images, ref_index=0):
    """
    Estimates relative exposure times based on image intensities.

    Args:
    images (list of ndarray): List of images (all images must be aligned and of the same size).
    ref_index (int): Index of the reference image in the list.

    Returns:
    list: Relative exposure times for each image with respect to the reference image.
    """
    ref_image = images[ref_index]
    num_images = len(images)
    height, width, _ = ref_image.shape

    # Initialize exposure ratios
    exposure_ratios = np.ones(num_images)

    # For each image, compute the ratio of its median brightness to that of the reference image
    for i in range(num_images):
        if i != ref_index:
            # Convert both images to grayscale
            gray_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
            gray_target = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

            # Avoid zero division and extremely dark areas
            median_ref = np.median(gray_ref[gray_ref > 10])
            median_target = np.median(gray_target[gray_target > 10])

            # Compute the relative exposure time (ratio)
            exposure_ratios[i] = median_target / median_ref

    return exposure_ratios.tolist()

# Step 5: Putting it all together
def HDR_imaging(images_directory, results_directory, smooth_lambda, sample_image_index, tone_map_function, num_samples, exposure = True):
    # Step 1: Load images and exposure from directory 
    # Load images from dataset
    bracketed_images = load_bracketed_images(images_directory)
    exposure_times = None
    if exposure:
        # Get exposures from file name
        exposure_times = extract_exposure_times(images_directory)
    else:
        exposure_times = estimate_relative_exposure(bracketed_images)

    print('Exposure_times:', exposure_times)
    
    # Extract BGR channels for the images
    blue_channels, green_channels, red_channels = extract_channels(bracketed_images)
    # Display and save the images
    display_images(bracketed_images, f'{results_directory}_{smooth_lambda}', exposure_times)
    display_images(red_channels, f'{results_directory}_{smooth_lambda}_red', exposure_times)
    display_images(blue_channels, f'{results_directory}_{smooth_lambda}_blue', exposure_times)
    display_images(green_channels, f'{results_directory}_{smooth_lambda}_green', exposure_times)

    # Step 2: Estimate response curves
    log_exposure_times = np.log(exposure_times)
    # Get sample image index
    image_index = 0
    if sample_image_index == 0:
        image_index = 0 # first
    elif sample_image_index == 1:
        image_index = len(bracketed_images) // 2 # middle
    elif sample_image_index == 2:
        image_index = len(bracketed_images) - 1 # Last
    else:
        image_index = len(bracketed_images) // 2 # middle
    
    # Sample image
    sample_image = red_channels[image_index]
    # Generate random samples with num_samples = 100
    samples = get_pixel_samples(sample_image, save_name=f'{results_directory}_{smooth_lambda}', num_samples=num_samples)
    # Channel sample pixel values with dimenstions (num_samples, num_images)
    pixels_blue = get_pixel_values_from_indices(blue_channels, samples) # for blue
    pixels_green = get_pixel_values_from_indices(green_channels, samples) # for green
    pixels_red = get_pixel_values_from_indices(red_channels, samples) # for red

    # Construct Response Curves
    blue_response_curve, _ = compute_response_curve(pixels_blue, log_exposure_times, smooth_lambda) # for blue
    green_response_curve, _ = compute_response_curve(pixels_green, log_exposure_times, smooth_lambda) # for green
    red_response_curve, _ = compute_response_curve(pixels_red, log_exposure_times, smooth_lambda) # for red
    
    # Plot the response curve
    plot_response_curves(blue_response_curve, green_response_curve, red_response_curve, f'{results_directory}_{smooth_lambda}')

    # Step 3: Create the radiance maps
    radiance_blue = create_radiance_map(blue_channels, blue_response_curve, log_exposure_times)
    radiance_green = create_radiance_map(green_channels, green_response_curve, log_exposure_times)
    radiance_red = create_radiance_map(red_channels, red_response_curve, log_exposure_times)

    # Plot radiance histogram
    radiance_map_combined = [radiance_red, radiance_green, radiance_blue]
    plot_radiance_histograms(radiance_map_combined, results_directory)

    # Display radiance map
    display_radiance_map(radiance_red, radiance_green, radiance_blue, f'{results_directory}_{smooth_lambda}')

    # Step 4: Construct HDR images + apply tone mapping
    width, height, channel = bracketed_images[0].shape
    hdr_image = construct_hdr_image(radiance_blue, radiance_green, radiance_red, width, height, channel)
    tone_mapped_image = tone_map_function(hdr_image)
    plot_and_save_hdr_image(tone_mapped_image, results_directory, smooth_lambda)

    print('HDR image successfully generated.')


def HDR_estimate_exposure(images_directory, results_directory, smooth_lambda, sample_image_index, tone_map_function, num_samples, align = True, exposure = False):
    
    # Step 1: Load images and exposure from directory 
    # Load images from dataset and align them 
    images = load_bracketed_images(images_directory)
    
    # Align images if misaligned here:
    # images = None
    # if align:
    #     images = align_images(bracketed_images)
    # else:
    #     images = bracketed_images
    
    # Get exposures from file name
    estimated_exposure_times = estimate_relative_exposure(images, ref_index=(len(images) // 2))
    print('Estimated Exposure Times:', estimated_exposure_times)
    
    # Extract BGR channels for the images
    blue_channels, green_channels, red_channels = extract_channels(images)
    # Display and save the images
    display_images(images, f'{results_directory}_{smooth_lambda}', estimated_exposure_times)
    display_images(red_channels, f'{results_directory}_{smooth_lambda}_red', estimated_exposure_times)
    display_images(blue_channels, f'{results_directory}_{smooth_lambda}_blue', estimated_exposure_times)
    display_images(green_channels, f'{results_directory}_{smooth_lambda}_green', estimated_exposure_times)

    # Step 2: Estimate response curves
    log_exposure_times = np.log(estimated_exposure_times)
    # Get sample image index
    image_index = 0
    if sample_image_index == 0:
        image_index = 0 # first
    elif sample_image_index == 1:
        image_index = len(images) // 2 # middle
    elif sample_image_index == 2:
        image_index = len(images) - 1 # Last
    else:
        image_index = len(images) // 2 # middle
    
    # Sample image
    sample_image = red_channels[image_index]
    # Generate random samples with num_samples = 100
    samples = get_pixel_samples(sample_image, save_name=f'{results_directory}_{smooth_lambda}', num_samples=num_samples)
    # Channel sample pixel values with dimenstions (num_samples, num_images)
    pixels_blue = get_pixel_values_from_indices(blue_channels, samples) # for blue
    pixels_green = get_pixel_values_from_indices(green_channels, samples) # for green
    pixels_red = get_pixel_values_from_indices(red_channels, samples) # for red

    # Construct Response Curves
    blue_response_curve, _ = compute_response_curve(pixels_blue, log_exposure_times, smooth_lambda) # for blue
    green_response_curve, _ = compute_response_curve(pixels_green, log_exposure_times, smooth_lambda) # for green
    red_response_curve, _ = compute_response_curve(pixels_red, log_exposure_times, smooth_lambda) # for red
    
    # Plot the response curve
    plot_response_curves(blue_response_curve, green_response_curve, red_response_curve, f'{results_directory}_{smooth_lambda}')

    # Step 3: Create the radiance maps
    radiance_blue = create_radiance_map(blue_channels, blue_response_curve, log_exposure_times)
    radiance_green = create_radiance_map(green_channels, green_response_curve, log_exposure_times)
    radiance_red = create_radiance_map(red_channels, red_response_curve, log_exposure_times)

    # Plot radiance histogram
    radiance_map_combined = [radiance_red, radiance_green, radiance_blue]
    plot_radiance_histograms(radiance_map_combined, results_directory)

    # Display radiance map
    display_radiance_map(radiance_red, radiance_green, radiance_blue, f'{results_directory}_{smooth_lambda}')

    # Step 4: Construct HDR images + apply tone mapping
    width, height, channel = images[0].shape
    hdr_image = construct_hdr_image(radiance_blue, radiance_green, radiance_red, width, height, channel)
    tone_mapped_image = tone_map_function(hdr_image)
    plot_and_save_hdr_image(tone_mapped_image, results_directory, smooth_lambda)

    print('Estimate Exposure HDR image successfully generated.')

# Step 6: Collect arguments
def main():
    parser = argparse.ArgumentParser(description = 'Process images for HDR and display the radiance map.')

    parser.add_argument('images_directory', type=str, help='Directory where input images are stored:')
    parser.add_argument('result_directory', type=str, help='Directory where results will be saved:')
    parser.add_argument('smooth_lambda', type=int, help='Smoothing lambda for HDR processing:')
    parser.add_argument('sample_image_index', type=int, help='Index of the sample image for HDR processing:') # 0 = first, 1 = middle, 2 = last
    parser.add_argument('file_name', type=str, help='Filename for saving the result:')
    parser.add_argument('tone_map_function', type=str, help='Tone Map function:') 
    parser.add_argument('num_samples', type=int, help='Number of samples:') 

    args = parser.parse_args()

    # Ensure the image directory exists
    if not os.path.isdir(args.images_directory):
        print(f"Error: The directory {args.images_directory} does not exist.")
        sys.exit(1)

    # Ensure the result directory exists or create it
    os.makedirs(args.result_directory, exist_ok=True)

    print(args)

    # Define tone map functions:
    # Assume that 'tone_map_linear' and 'tone_map_gamma' are defined tone mapping functions
    tone_mapping_functions = {
        'tone_map_normalize': tone_map_and_normalize
    }

     # Map the tone_map_function argument to an actual function object
    if args.tone_map_function in tone_mapping_functions:
        tone_map_func = tone_mapping_functions[args.tone_map_function]
    else:
        print(f"Error: '{args.tone_map_function}' is not a recognized tone mapping function.")
        sys.exit(1)    

    # construct hdr image
    HDR_imaging(images_directory=args.images_directory, results_directory=f'{args.result_directory}/{args.file_name}',
                smooth_lambda=args.smooth_lambda, sample_image_index=args.sample_image_index,
                tone_map_function=tone_map_func,
                num_samples=args.num_samples, exposure=True)

    print(f"Result saved successfully in {os.path.join(args.result_directory, args.file_name)}")


if __name__ == "__main__":
    main()
## Way to run it: python hdr_imaging.py /path/to/images /path/to/results 20 1 file_name tone_map num_samples
## python hdr_imaging.py datasets/images1 results/madison_capitol1_est 100 1 mc1 tone_map_normalize 100
## python hdr_imaging.py datasets/images2 results/madison_capitol2 100 1 mc2 tone_map_normalize 100

## python hdr_imaging.py datasets/images3 results/normalize/madison_capitol3 1 1 mc3 tone_map_normalize 100
## python hdr_imaging.py datasets/images3 results/normalize/madison_capitol3 10 1 mc3 tone_map_normalize 100
## python hdr_imaging.py datasets/images3 results/normalize/madison_capitol3 50 1 mc3 tone_map_normalize 100
## python hdr_imaging.py datasets/images3 results/normalize/madison_capitol3 100 1 mc3 tone_map_normalize 100
## python hdr_imaging.py datasets/images3 results/normalize/madison_capitol3 1000 1 mc3 tone_map_normalize 100

## python hdr_imaging.py datasets/images4 results/normalize/madison_capitol4 100 1 mc4 tone_map_normalize 100