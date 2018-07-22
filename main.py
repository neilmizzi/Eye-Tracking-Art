"""
main.py

This script contains the main system.

Steps:
    -   Get obtaining the corner points from the .csv file
    -   Plot the points to get the resultant figure
    -   Obtain and fine-tune saccadic path & world view figures, i.e:
        -   Crop off any whitespace in saccadic path figure
        -   Resize world view to same dimensions of saccadic path
    -   Filter the world view to get solely the line sculpture.
        - Threshold applied is a Bilateral Filter, or sharpening filter, depending on params
    -   Apply the Harris Corner Detection algorithm to obtain the corner points of both the world view & saccadic path
    -   Compare both sets of corner points and remove any points which exceed Euclidean distance threshold
        -   Points are sorted depending on which algorithm we decide to choose
    -   Represent the results achieved if asked for & store result
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import functions as fn
import copy


def main_system(folder, show_corners, show_images, use_sim_anneal,
                value_of_d, value_of_md, value_of_max_eucl,
                print_stuff, result_index):

    """         STEP 0: PARAMETERS & SETUP      """

    # Bilateral filter parameters
    # source: https://goo.gl/rfhWS2
    #
    # d          – Diameter of each pixel neighborhood used during filtering.
    #              If it is non-positive, it is computed from sigmaSpace.
    #
    # sigmaColor – Filter sigma in the color space.
    #              A larger value of the parameter means that farther colors within the pixel neighborhood
    #              (see sigmaSpace) will be mixed together, resulting in larger areas of semi-equal color.
    #
    # sigmaSpace – Filter sigma in the coordinate space.
    #              A larger value of the parameter means that farther pixels will influence each other as long
    #              as their colors are close enough (see sigmaColor). When d>0,
    #              it specifies the neighborhood size regardless of sigmaSpace.
    #              Otherwise, d is proportional to sigmaSpace.

    d = value_of_d      # diameter of pixel neighbourhood to consider
    sigma_colour = 75   # standard deviation of colour space
    sigma_space = 75    # standard deviation of sigma space

    # Harris Corner detection algorithms
    # Source of descriptions: https://goo.gl/CBPq4g
    max_cp = 500      # Maximum number of corners to return
    ql = 0.001        # The minimal accepted quality level of corners
    md = value_of_md  # Minimum possible Euclidean distance between returned corners
    k = 0.01          # Free Parameter of Harris Detector

    # max_eucl: The maximum distance for a pt in the sacc_corners list not to be considered as an outlier,
    #           when compared to the world_corners list of points
    max_eucl = value_of_max_eucl

    """     STEP 1: GETTING THE SACCADIC PATH       """

    sacc_path, world_view = fn.get_images(folder)

    """     STEP 2: FILTERING WORLD VIEW            """

    # filtering world view
    before = copy.copy(world_view)
    if d > 0:
        # The Bilateral filter will
        world_view = cv2.bilateralFilter(world_view, d, sigma_colour, sigma_space)
    elif d < 0:
        # utilising a generally known kernel to sharpen image and extract more feature points of interest
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        world_view = cv2.filter2D(world_view, -1, kernel)

    if show_images:
        bleh = np.concatenate((before, world_view), 1)
        cv2.imshow('test', bleh)

    """     STEP 3: CORNER DETECTION                """

    # Obtaining the coordinates of the corner points in both images
    sacc_corners = fn.format_list(np.squeeze(cv2.goodFeaturesToTrack(sacc_path, maxCorners=max_cp, qualityLevel=ql,
                                  minDistance=md, useHarrisDetector=True, k=k)))

    world_corners = fn.format_list(np.squeeze(cv2.goodFeaturesToTrack(world_view, maxCorners=max_cp, qualityLevel=ql,
                                   minDistance=md, useHarrisDetector=True, k=k)))

    if show_images:
        sacc_x = cv2.cvtColor(sacc_path, cv2.COLOR_GRAY2BGR)
        for i in sacc_corners:
            x, y = i
            cv2.circle(sacc_x, (x, y), 2, (0, 0, 255), -1)

        world_x = cv2.cvtColor(world_view, cv2.COLOR_GRAY2BGR)
        for i in world_corners:
            x, y = i
            cv2.circle(world_x, (x, y), 2, (0, 0, 255), -1)

        cnct = np.concatenate((sacc_x, world_x), 1)
        cv2.imshow('corners', cnct)

    """  STEP 4: OUTLIER REMOVAL & POINT SORTING    """

    # remove_outliers
    new_path = fn.remove_outliers(sacc_corners, world_corners, max_eucl)

    if use_sim_anneal:
        # only use Simulated Annealing if path is longer than just 2 points
        if len(new_path) > 1:
            new_path = fn.sim_anneal_sort(new_path)
        else:
            new_path = fn.eucl_sort(new_path)
    else:
        new_path = fn.eucl_sort(new_path)

    """     STEP 5: RESULTS                         """

    # some numbers on how many corners were detected, retained & removed
    if print_stuff:
        print()
        print("No. of corners in original Saccadic Path: "+str(len(sacc_corners)))
        print("No. of corners in World View:             "+str(len(world_corners)))
        print("No. of corners retained in the new path:  "+str(len(new_path)))

    # plotting our new and improved figure
    _, plot = plt.subplots()
    if new_path:
        x, y = zip(*new_path)
    else:
        x = []
        y = []

    y = [-i for i in y]
    plot.plot(x, y, '-o', ms=0, lw=1, alpha=1, color='black')
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(folder+'results/'+str(result_index)+'.png')

    # plotting the figures, depending on what we want to show

    if show_images:
        new_shape = fn.crop_img(cv2.imread(folder+'results/'+str(result_index)+'.png'))  # new Path
        if show_corners:
            concat_1 = np.concatenate((sacc_path, before), 1)
            concat_1 = np.concatenate((concat_1, world_view), 1)

            for i in sacc_corners:
                x, y = i
                cv2.circle(sacc_path, (x, y), 2, (0, 0, 0), -1)

            for i in world_corners:
                x, y = i
                cv2.circle(world_view, (x, y), 2, (0, 0, 0), -1)

            concat_2 = np.concatenate((sacc_path, world_view), 1)
            concat_2 = np.concatenate((concat_2, new_shape), 1)

            concat_1 = np.concatenate((concat_1, concat_2), 1)

        else:
            concat_1 = np.concatenate((sacc_path, new_shape), 1)
            concat_1 = np.concatenate((world_view, concat_1), 1)

        cv2.imshow('Results', concat_1)

        # Exit sequence after displaying images
        c = cv2.waitKey(0)
        quit_list = ['q', ' ', 'e']
        if chr(c & 255) in quit_list:
            exit()

    # close any remaining instances of matplotlib
    plt.close('all')
