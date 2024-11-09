import cv2 
import numpy as np 
from typing import Tuple, List


def preprocess_the_image(input_image: np.array | Tuple[np.array]) -> np.array | Tuple[np.array]: 
    '''
    Preprocessing image of image batch for contour detection

    Args:
        image (np.array | Tuple[np.array]): A single image in RGB format or a batch of images

    Returns:
        np.array | Tuple[np.array]:  A single image in RGB format or a batch of preprocessed images
    '''
    def _preproc_single_image(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 5
        )
        thresh = cv2.dilate(thresh.copy(), None, iterations=4) / 255
        return cv2.convertScaleAbs(thresh)

    if not isinstance(input_image, tuple | list):
        return _preproc_single_image(input_image)
    
    return tuple([_preproc_single_image(image) for image in input_image])

def find_image_contours(
    input_image: np.array | Tuple[np.array], 
    epsilon_factor: float = 0.02, 
    min_area: int = 30000
    ) -> Tuple[np.array | Tuple[np.array], Tuple[List[np.array]]]:
    '''
    Find big contours on the image. They are highly likely objects of interest

    Args:
        input_image (np.array | Tuple[np.array]): A single preprocessed image or a batch of images
        epsilon_factor (float, optional): Epsilon factor to approximates a contour shape. Defaults to 0.04.
        min_area (int, optional): Min detected area. Defaults to 5000.

    Returns:
        Tuple[np.array | Tuple[np.array], Tuple[List[np.array]]]: a tuple where: 
            - The first component is a single image with contours drawn or batch of images with contours drawn
            - The second component is a Tuple of contours detected for each images
    '''

    def _find_contours_single_image(image_processed, original_image, epsilon_factor, min_area):
        contours, _ = cv2.findContours(image_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangular_contours = []

        for contour in contours:
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) != 4:
                continue 
            
            if cv2.contourArea(contour) >=  min_area:
                rectangular_contours.append(contour)

        contour_image = original_image.copy()
        cv2.drawContours(contour_image, rectangular_contours, -1, (0, 255, 255), 5)  # draw in green color
        
        return contour_image, rectangular_contours

    if not isinstance(input_image, tuple | list):
        processed_image = preprocess_the_image(input_image.copy())
        return _find_contours_single_image(processed_image, input_image, epsilon_factor, min_area)
    
    contour_image_list, contour_coordinates_list = [], []

    for image in input_image:
        processed_image = preprocess_the_image(image)
        contour_image, contours_detected = _find_contours_single_image(processed_image, image, epsilon_factor, min_area)
        contour_image_list.append(contour_image)
        contour_coordinates_list.append(contours_detected)

    return tuple(contour_image_list), contour_coordinates_list

def warp_contour(image: np.array, contours: List[np.array]) -> List[np.array]:
    '''
    Warp contours detected from the image

    Args:
        image (np.array): A single image in RGB format
        contours (List[np.array]): A list of contours detected

    Returns:
        List[np.array]: A List of warped images
    '''
    warped_images = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y + h, x:x + w]
        mask = np.ones_like(image, dtype=np.uint8) * 255
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        mask_roi = mask[y:y + h, x:x + w]
        mask_roi = cv2.cvtColor(mask_roi, cv2.COLOR_RGB2GRAY)
        warped = cv2.bitwise_and(roi, roi, mask=mask_roi)
        warped_images.append(warped)
    
    return warped_images

def pipeline(input_image: np.array | Tuple[np.array]) -> Tuple[np.array | Tuple[np.array], List[np.array]]:
    '''
    Pipeline for background detection

    Args:
        input_image (np.array | Tuple[np.array]): A single image or a Tuple of images

    Returns:
        Tuple[np.array | Tuple[np.array], List[np.array]]: 
            if single image: 
                return an image with contours and all warped images as a list
            if Tuple of images:
                return a tuple of images with contours and all warped images as a list
    '''
    if not isinstance(input_image, tuple | list):
        print('Processing a single image')
        
        contour_image, rectangular_contours = find_image_contours(input_image)
        warped = warp_contour(input_image, rectangular_contours)

        print(f'{len(warped)} contours was found sucessfully!')
        print('Processed successfully!')
        return contour_image, warped
    
    print(f'Processing a batch of images. Total {len(input_image)} images')

    warped = []
    contour_images_list, rectangular_contours_list = find_image_contours(input_image)

    for i, (image, rectangular_contours) in enumerate(zip(input_image, rectangular_contours_list)):
        
        warped_ = warp_contour(image, rectangular_contours)
        warped += warped_
        ordinal = lambda n: '%d%s' % (n, 'tsnrhtdd'[(n//10 % 10 != 1)*(n % 10 < 4)*n % 10::4]) # omfg

        print(f'{len(warped_)} contours was found sucessfully in {ordinal(i + 1)} image!')
    
    print('Processed successfully!')
    return contour_images_list, warped