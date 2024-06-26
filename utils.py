def set_to_nan_based_on_likelihood(df, threshold):
    for col in df.columns:
        if 'likelihood' in col:
            prefix = col.replace(' likelihood', '')
            df.loc[df[col] < threshold, [f'{prefix} x', f'{prefix} y']] = np.nan
    return df

def plot_polar_plot(bin_counts,bin_edges,i):

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Create a polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Convert bin centers from degrees to radians
    bin_centers_rad = np.deg2rad(bin_centers)

    # Create the bar plot
    bars = ax.bar(bin_centers_rad, bin_counts, width=np.deg2rad(bin_width), edgecolor='k')

    # Add labels and title
    ax.set_title('Cell Number: '+str(i))
    ax.set_theta_zero_location('N')  # Set 0 degrees to the top
    ax.set_theta_direction(-1)  # Clockwise

    plt.show()