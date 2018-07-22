import main
import os

# parameters to tune (implement as cross-parameter testing to auto-generate results)
show_corners = False               # whether or not to show images with Harris Features
show_images = True                # whether or not to show images (TURN OFF WHEN EVALUATING WHOLE DATASET)
print_stuff = True                # whether or not to display results in console
# use_sim_anneal                     False to use Euclidean sort, True to try to use Simulated Annealing
# value_of_d                         Diameter of pixels to consider in Bilateral Filter
# value_of_md                        Minimum Distance between Harris points detected
# value_of_max_eucl                  Maximum distance for a saccadic path pt. not to be considered as an outlier

# all combinations of these parameters will be tested out. Each combination no. is stored in an Excel File
value_of_d = [-2, 0, 2, 5]
value_of_md = [2, 5]
value_of_max_eucl = [5, 7, 8, 10, 13, 15, 20]
use_sim_anneal = [True, False]

# Directory of data record
folder = 'dataset/'


folders = [os.path.join(folder, f) for f in os.listdir(folder)]

folders = ['dataset/6']
value_of_d = [-2]
value_of_md = [5]
value_of_max_eucl = [5]
use_sim_anneal = [True]

# testing out all combinations for all folders. Results stored in dataset folder
for f in range(0, len(folders)):
    if folders[f] != 'dataset/.DS_Store':
        folders[f] = folders[f]+'/'
        i = 1
        print(folders[f])
        for use_sort in use_sim_anneal:
            for max_eucl in value_of_max_eucl:
                for md in value_of_md:
                    for d in value_of_d:
                        main.main_system(folders[f], show_corners, show_images, use_sort, d,  md,
                                         max_eucl, print_stuff, i)
                        i += 1
