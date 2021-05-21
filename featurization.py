import numpy as np

def make_cosine_bases_kernels(y, y_resolution=500, y_range=None, n_kernels=6, warping_curve=None, plot_pref=True):
    if y_range is None:
        y_range = np.array([np.min(y) , np.max(y)])

    if warping_curve is None:
        warping_curve = np.arange(1000000)

    y_resolution_highRes = y_resolution * 10
    bases_highRes = np.zeros((y_resolution_highRes, n_kernels+1))

    cos_width_highRes = int((bases_highRes.shape[0] / (n_kernels+1))*2)
    cos_kernel_highRes = (np.cos(np.linspace(-np.pi, np.pi, cos_width_highRes)) + 1)/2

    for ii in range(n_kernels):
        bases_highRes[int(cos_width_highRes*(ii/2)) : int(cos_width_highRes*((ii/2)+1)) , ii] = cos_kernel_highRes

    bases_highRes_cropped = bases_highRes[int(cos_width_highRes/2):-int(cos_width_highRes/2)]

    WC_norm = warping_curve - np.min(warping_curve)
    WC_norm = (WC_norm/np.max(WC_norm)) * (bases_highRes_cropped.shape[0]-1)

    f_interp = scipy.interpolate.interp1d(np.arange(bases_highRes_cropped.shape[0]),
                                          bases_highRes_cropped, axis=0)

    bases_interp = f_interp(WC_norm[np.uint64(np.round(np.linspace(0, len(WC_norm)-1, y_resolution)))])

    if plot_pref:
        fig, axs = plt.subplots(6)
        axs[0].plot(bases_highRes)
        axs[1].plot(bases_highRes_cropped)
        axs[2].plot(warping_curve)
        axs[3].plot(WC_norm)
        axs[4].plot(bases_interp)
        axs[5].plot(np.sum(bases_interp, axis=1))
        axs[5].set_ylim([0,1.1])
    
    xAxis_of_curves = np.linspace(y_range[0] , y_range[1], y_resolution)
    
    return bases_interp , xAxis_of_curves