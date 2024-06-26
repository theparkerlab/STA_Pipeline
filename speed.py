def speedPlots(dlc_df, phy_df):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal
    from scipy import stats
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib

    fps = 59.99
    model_dt = 1/59.99  # Frame duration in seconds
    dlc_df['time'] = np.arange(len(dlc_df))/fps
    model_t = np.arange(0, dlc_df['time'].iloc[-1], model_dt)

    def calcVelocity(keys):
        threshold = 0.95
        for key in keys:
            x_vals = dlc_df[key + ' x'].copy()
            y_vals = dlc_df[key + ' y'].copy()
            x_vals[dlc_df[key + ' likelihood']<threshold] = np.nan
            y_vals[dlc_df[key + ' likelihood']<threshold] = np.nan
            x_vals = interpolateMovement(x_vals)
            y_vals = interpolateMovement(y_vals)

        threshold = 0.9
        top_right_corner_x_vals = dlc_df['top_right_corner x'].copy()
        top_right_corner_x_vals[dlc_df['top_right_corner likelihood']<threshold] = np.nan
        top_right_x = np.nanmean(top_right_corner_x_vals)
        top_left_corner_x_vals = dlc_df['top_left_corner x'].copy()
        top_left_corner_x_vals[dlc_df['top_left_corner likelihood']<threshold] = np.nan
        top_left_x = np.nanmean(top_left_corner_x_vals)
        pixels_per_cm = (top_right_x - top_left_x) / 60
        
        velocityX = np.diff(x_vals/pixels_per_cm) * fps #cm per sec
        velocityY = np.diff(y_vals/pixels_per_cm) * fps #cm per sec
        totalVelocity = np.sqrt((velocityX ** 2) + (velocityY ** 2)) #cm per sec
        totalVelocity[0] = totalVelocity[2]
        totalVelocity[1] = totalVelocity[2]
        totalVelocity[-1] = totalVelocity[-3]
        totalVelocity[-2] = totalVelocity[-3]

        for i in range(0, len(totalVelocity)):
            if totalVelocity[i] > 45:
                totalVelocity[i] = np.nan

        return totalVelocity

    def interpolateMovement(arr):
        arr = pd.Series(arr).interpolate().to_numpy()
        arr = pd.Series(arr).fillna(method='bfill').to_numpy()
        arr = signal.medfilt(arr,kernel_size = 5)
        box_size = 5
        box = np.ones(box_size) / box_size
        arr = np.convolve(arr, box, mode='same')
        return arr
    
    #Calculate Binned Firing Rates
    def calcFiringRate(n, timeDiff, time_array):
        data = phy_df['spikeT'][n]
        
        j = data[0]
        while (j < 0):
            data = np.delete(data,0)
            j = data[0]
        j = data[-1]
        while (j > time_array[-1] + timeDiff):
            data = np.delete(data,-1)
            j=data[-1]
    
        bins = time_array
        digitized = np.digitize(data,bins)

        bin_counts= [0]*len(bins)
        for i in digitized:
            bin_counts[i-1]+=1

        return bin_counts
    
    def calcAverageFiringRates(nfr, bI, aV):
        start = 0
        average_firing_rates = []
        standard_deviations = []
        standard_errors = []
        aV = np.array(aV)

        while start < max(aV):
            # Find indices of x-values within the specified range
            indices = np.where((aV >= start) & (aV < start + bI))

            # Extract corresponding y-values
            y_values = np.array(nfr)[indices]

            # Calculate the average of y-values
            average_y = np.mean(y_values)
            std_y = np.std(y_values)

            average_firing_rates.append(average_y)
            standard_deviations.append(std_y)
            standard_errors.append(std_y / np.sqrt(len(y_values)))

            average_firing_rates = pd.Series(average_firing_rates).interpolate().to_list() #interpolate from NaN

            start += bI

        return average_firing_rates, standard_deviations, standard_errors
    
    velocity = calcVelocity(['right_haunch'])
    adjustedVelocity = []
    binTime = 0
    i = 0

    while i < len(velocity):
        j = i
        binTime+=0.5
        while j < len(model_t) and model_t[j] < binTime:
            binEnd = j
            j+=1
        adjustedVelocity.append(np.nanmean(velocity[i: binEnd]))
        i=j

    timeInSeconds = np.arange(0,model_t[-1],0.5)

    fig_list = []

    for i in range(0, len(phy_df.index)):
        firing_rate = calcFiringRate(i, 0.5, timeInSeconds)
        final_average_firing_rates, std_list, std_errors = calcAverageFiringRates(firing_rate, 5, adjustedVelocity)


        
        fig, axs = plt.subplots(1,3,figsize=(9,3))
        axs = axs.ravel()
        ax = axs[0]
        ax.plot(adjustedVelocity, firing_rate, '.', markersize = 2, color='black')
        ax.set_xlabel('Velocity')
        ax.set_ylabel('Firing rate')


        ax = axs[1]

        #The smaller graph appears in front for visibility
        if np.nanmean(firing_rate) > np.nanmean(velocity):
            plot_firing_rate, = ax.plot(timeInSeconds, firing_rate,'-', label = "Firing Rate", color='#0072BD')
            plot_velocity, = ax.plot(timeInSeconds, adjustedVelocity, label = "Velocity", color="#FFAC1C")
        else:
            plot_velocity, = ax.plot(timeInSeconds, adjustedVelocity, label = "Velocity", color="#FFAC1C") 
            plot_firing_rate, = ax.plot(timeInSeconds, firing_rate,'-', label = "Firing Rate", color='#0072BD')

        ax.set_xlabel("Time (s)")
        ax.legend(handles=[plot_firing_rate, plot_velocity])

        ax=axs[2]
        ax.plot(np.arange(0, max(adjustedVelocity), 5), final_average_firing_rates,'k')
        lower_graphed_firing_rates = []
        upper_graphed_firing_rates = []
        for i in range (0, len(std_errors)):
            lower_graphed_firing_rates.append(final_average_firing_rates[i]-std_errors[i])
            upper_graphed_firing_rates.append(final_average_firing_rates[i]+std_errors[i])
        ax.fill_between(np.arange(0, max(adjustedVelocity), 5), lower_graphed_firing_rates, upper_graphed_firing_rates,color='#ADD8E6')
        ax.set_xlabel("Velocity")
        ax.set_ylabel("Firing Rate")
        ax.set_ylim(bottom=0)


        neuron_number_official = phy_df.index[i]

        velocityCleaned = np.array(adjustedVelocity)[~np.isnan(adjustedVelocity)]
        firing_rate_cleaned = np.array(firing_rate)[~np.isnan(adjustedVelocity)]

        rcoeff, pval = stats.pearsonr(velocityCleaned, firing_rate_cleaned)

        fig.suptitle('Neuron %s (r=%0.3f, p=%s)' % (neuron_number_official, rcoeff, str(pval)))
        fig.tight_layout()
        fig_list.append(fig)
        print("Created Figure")

    return fig_list