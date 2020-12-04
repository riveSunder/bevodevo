import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Plotting instructions") 

    parser.add_argument("-f", "--filepath", type=str, \
           help="filepath to experiment folder", \
           default="./results/test_exp/")
    parser.add_argument("-x", "--independent_variable", type=str,\
           default="wall_time", \
           help="x variable options: wall_time, total_env_interacts, generation")
    parser.add_argument("-s", "--save_fig", type=bool, default=False)

    args = parser.parse_args()

    my_dir = os.listdir(args.filepath)

    for filename in my_dir:
        if "progress" in filename:
            my_data = np.load(args.filepath + filename, allow_pickle=True)

            my_data = my_data[np.newaxis][0]
            print("exp hyperparameters: \n", my_data["args"])

            x = my_data[args.independent_variable]
            y = np.array(my_data["mean_fitness"])

            max_y = np.array(my_data["max_fitness"])
            min_y = np.array(my_data["min_fitness"])
            std_dev_y = np.array(my_data["std_dev_fitness"])

            plt.figure(figsize=(8,6))

            plt.plot(x, y, 'k', label="Mean fitness", lw=3)
            plt.plot(x, max_y, '--r', label="Max fitness", lw=3)
            plt.fill_between(x, y-std_dev_y, y+std_dev_y, alpha=0.5, label="+/- standard deviation")
            plt.xlabel(args.independent_variable, fontsize=14)
            plt.ylabel("fitness", fontsize=14)
            plt.title("{}".format(filename[:-4]),\
                    fontsize=16)
            plt.legend()


            if args.save_fig:
                plt.savefig(args.filepath + filename[:-4] + ".png")

    plt.show()

