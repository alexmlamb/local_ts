"""
Tools for plotting / visualization
"""

import matplotlib
matplotlib.use('Agg')  # no displayed figures -- need to call before loading pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import warnings

def is_square(shp, n_colors=1):
    """
    Test whether entries in shp are square numbers, or are square numbers after divigind out the
    number of color channels.
    """
    is_sqr = (shp == np.round(np.sqrt(shp))**2)
    is_sqr_colors = (shp == n_colors*np.round(np.sqrt(np.array(shp)/float(n_colors)))**2)
    return is_sqr | is_sqr_colors

def show_receptive_fields(theta, P=None, n_colors=None, max_display=100, grid_wa=None,title=""):
    """
    Display receptive fields in a grid. Tries to intelligently guess whether to treat the rows,
    the columns, or the last two axes together as containing the receptive fields. It does this
    by checking which axes are square numbers -- so you can get some unexpected plots if the wrong
    axis is a square number, or if multiple axes are. It also tries to handle the last axis
    containing color channels correctly.
    """

    shp = np.array(theta.shape)
    if n_colors is None:
        n_colors = 1
        if shp[-1] == 3:
            n_colors = 3
    # multiply colors in as appropriate
    if shp[-1] == n_colors:
        shp[-2] *= n_colors
        theta = theta.reshape(shp[:-1])
        shp = np.array(theta.shape)
    if len(shp) > 2:
        # merge last two axes
        shp[-2] *= shp[-1]
        theta = theta.reshape(shp[:-1])
        shp = np.array(theta.shape)
    if len(shp) > 2:
        # merge leading axes
        theta = theta.reshape((-1,shp[-1]))
        shp = np.array(theta.shape)
    if len(shp) == 1:
        theta = theta.reshape((-1,1))
        shp = np.array(theta.shape)

    # figure out the right orientation, by looking for the axis with a square
    # number of entries, up to number of colors. transpose if required
    is_sqr = is_square(shp, n_colors=n_colors)
    if is_sqr[0] and is_sqr[1]:
        warnings.warn("Unsure of correct matrix orientation. "
            "Assuming receptive fields along first dimension.")
    elif is_sqr[1]:
        theta = theta.T
    elif not is_sqr[0] and not is_sqr[1]:
        # neither direction corresponds well to an image
        # NOTE if you delete this next line, the code will work. The rfs just won't look very
        # image like
        return False

    theta = theta[:,:max_display].copy()

    if P is None:
        img_w = int(np.ceil(np.sqrt(theta.shape[0]/float(n_colors))))
    else:
        img_w = int(np.ceil(np.sqrt(P.shape[0]/float(n_colors))))
    nf = theta.shape[1]
    if grid_wa is None:
        grid_wa = int(np.ceil(np.sqrt(float(nf))))
    grid_wb = int(np.ceil(nf / float(grid_wa)))

    if P is not None:
        theta = np.dot(P, theta)

    vmin = np.min(theta)
    vmax = np.max(theta)


    for jj in range(nf):
        plt.subplot(grid_wa, grid_wb, jj+1)
        
        if jj == int(np.sqrt(nf)/2) - 1:
            plt.title(title)

        ptch = np.zeros((n_colors*img_w**2,))
        ptch[:theta.shape[0]] = theta[:,jj]
        if n_colors==3:
            ptch = ptch.reshape((n_colors, img_w, img_w))
            ptch = ptch.transpose((1,2,0)) # move color channels to end
        else:
            ptch = ptch.reshape((img_w, img_w))
        #ptch -= vmin
        #ptch /= vmax-vmin

        plt.imshow(ptch, interpolation='nearest', cmap=cm.Greys_r, vmin = 0.0, vmax = 1.0)

        plt.axis('off')

    

    return True


def plot_images(X, fname, title=""):
    """
    Plot images in a grid.
    X is expected to be a 4d tensor of dimensions [# images]x[# colors]x[height]x[width]
    """
    
    X = X.clip(0.0,1.0)
    
    if X.shape == (64,784):
        X = X.reshape((64,1,28,28))
    elif X.shape == (64,96*96*3):
        X = X.reshape((64,3,96,96))
        X = X[:25]
    elif X.shape == (64,32*32*3):
        X = X.reshape((64,3,32,32))
    elif X.shape == (64,64*64*3):
        X = X.reshape((64,3,64,64))
    else:
        raise Exception("INVALID SHAPE OPTION")

    ## plot
    # move color to end
    Xcol = X.reshape((X.shape[0],-1,)).T
    plt.figure(figsize=[8,8])
    if show_receptive_fields(Xcol, n_colors=X.shape[1], title=title):
        #plt.savefig(fname + '.pdf')
        plt.savefig(fname + ".png")
    else:
        warnings.warn('Images unexpected shape.')
    
    plt.close()

    ## save as a .npz file
    #np.savez(fname + '.npz', X=X)

if __name__ == "__main__":

    import numpy.random as rng

    x = rng.normal(size = (64,1,28,28))

    plot_images(x, fname = "derp.png", title = "DERP DERP DERP")

