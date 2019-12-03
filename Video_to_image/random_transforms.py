import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import argparse
import os
import random
import progressbar

def random_transforms(directory,output):

    # rotate = iaa.Affine(rotate=(0,360))
    # shear = iaa.Affine(shear = (-35,35))
    # fliplr = iaa.Fliplr(0.5)
    # flipup = iaa.Flipud(0.5)
    if !os.path.exists(output):
        os.mkdir(output)


    transforms = iaa.Sequential(
        [iaa.Affine(rotate=(-25,25)),
        ##iaa.Affine(shear = (-19,19)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Grayscale(alpha=(0,0.5)),
       ## iaa.AdditiveGaussianNoise(scale=0.05*255, per_channel=0.6),
        ]
        )
    file_list = os.listdir(directory)
    bar = progressbar.ProgressBar(maxval=file_list.__len__(),
                                  widgets=[progressbar.Bar("=",'[',']'," "),progressbar.Percentage()])
    bar.start()
    current_status = 0;

    for file in file_list:
        current_status+=1
        bar.update(current_status)
        rand_num = random.randint(1,101)
        image = imageio.imread(os.path.join(directory,file))

        if image is not None and rand_num < 75:
            new_image = transforms.augment_image(image)
    ##        if not os.path.exists('output'):
    ##            os.mkdir('output')
            imageio.imwrite(output+file, new_image)
        else:
            imageio.imwrite(
                output + file,
                image)
    bar.finish()

if __name__=="__main__":
    parsArgs = argparse.ArgumentParser()
    parsArgs.add_argument('-i', '--input', type=str, default="", help="specify directory of images to transform")
    parsArgs.add_argument('-o','--output' type=str, defaults="output/", help="specify out put directory")
    args = parsArgs.parse_args()
    random_transforms(args.input, args.output)
