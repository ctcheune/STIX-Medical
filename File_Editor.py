import os
import pandas as pd
import numpy as np
import csv
import PIL
from PIL import Image
from PIL import ImageFile

# Function to rename multiple files

def main():

    image_prep("Validation")
    #rename_and_csv("Validation")


def image_prep(folder):
    pics = os.listdir(folder)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # used rotation in 90,180,and 270, image flip left/right and top/bot, colour conversion to Gray/MGY
    # also used transposed
    for count, picname in enumerate(pics):

        image = Image.open(folder+'/' + picname)
        image = image.resize((200, 200))
        image.save(folder + '/' + picname)



def rename_and_csv(folder):

    # values_list = pd.read_csv("Healthy_sorted/trainLabels.csv", encoding='utf-8')
    #name_list = pd.DataFrame(values_list[:, 1:2].values)
    #values_list = pd.DataFrame(values_list[:, 1:2].values)
    pics = os.listdir(folder)
    index = 0
    with open('Validation_values.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Value"])
        for count, picname in enumerate(pics):

            if picname[0] == 'G':
                value = 1
            else:
                value = 0
            # no_extension = picname[0:len(picname) - 5]
            # intermediate_row = values_list.loc[values_list['image'] == no_extension]
            #if intermediate_row.iloc[0][1] == 0:

            dst = "valid_" + str(index) + ".jpeg"
            src = folder + "/" + picname
            dst = folder + "/" + dst
            print("renamed " + picname + " to " + str(index))

            # rename() function will
            # rename all the files
            writer.writerow(["valid_" + str(index), value])
            index += 1
            os.rename(src, dst)




def rename(folder, name):
    pics = os.listdir(folder)

    for count, picname in enumerate(pics):
        dst = "Valid_" + str(count) + ".jpeg"
        src = folder + "/" + picname
        dst = folder + "/" + dst
        print("renamed " + picname + " to " + str(count))

        # rename() function will
        # rename all the files
        os.rename(src, dst)


if __name__ == '__main__':
    # Calling main() function
    main()