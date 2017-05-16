from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from skimage.measure import compare_psnr as psnr
import matplotlib.pyplot as plt
import cv2




def compare_images_silent(imageA, imageB):
    imageA = cv2.cvtColor(imageA,cv2.COLOR_RGB2GRAY)
    imageB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    p = psnr(imageA, imageB)

    print("PSNR: %.2f MSE: %.2f SSIM: %.2f" % (p, m, s))
    return m



def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    p = psnr(imageA, imageB)

    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("PSNR: %.2f MSE: %.2f SSIM: %.2f" %(p,m,s))
    print("PSNR: %.2f MSE: %.2f SSIM: %.2f" %(p,m,s))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    # show the images
    plt.show()


    # load the images -- the original, the original + contrast,
    # and the original + photoshop

def test():
    original = cv2.imread("D:\original.jpg")
    contrast = cv2.imread("D:\other.jpg")
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    images = ("Original", original), ("Contrast", contrast)
    compare_images(original, contrast, "Comparison")


