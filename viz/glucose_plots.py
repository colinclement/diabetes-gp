import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import itertools
import datetime
import matplotlib.dates as pltdate

def blood_glucose_plot(times, bgs, mode='full', ls=':'):
    """Plot blood glucoes on pretty plot.
    
    Parameters
    ----------
    times : list of datetime objects of observations
    bgs   : list of blood glucose records, in mg/dL
    mode  : the mode of the plot
            choose from 'full', 'model_day', and 'model_week'
    """

    fig = plt.figure(figsize=(12, 6), )
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_frame_on(False)
    
    # Format the time axis.
    ax.get_xaxis().tick_bottom()
    timemin, timemax = min(times), max(times)
    if mode=='full':
        ax.xaxis.set_major_formatter(pltdate.DateFormatter('%b %d'))
        xmin, xmax = timemin, timemax
    elif mode=='model_day':
        ax.xaxis.set_major_formatter(pltdate.DateFormatter('%H:%M'))
        # Collapse all data onto single day
        times, xmin, xmax = maptoday(times)
    elif mode=='model_week':
        ax.xaxis.set_major_formatter(pltdate.DateFormatter('%A'))
        times, xmin, xmax = maptoweek(times)

    timeinds = np.argsort(times)
    times = times[timeinds]
    bgs = bgs[timeinds]


    ax.set_xlim((xmin, xmax))
    ax.xaxis_date()
    
    # Format the y axis, show typical range.
    ax.set_ylim(bottom=20, top=max(bgs))
    ax.axes.get_yaxis().set_visible(False)
    ax.axhspan(70, 150, alpha=0.1, color = '#FF4775')
    ax.text(-0.01, 70, '70', va='center', ha='right', transform=ax.get_yaxis_transform())
    ax.text(-0.01, 150, '150', va='center', ha='right', transform=ax.get_yaxis_transform())

    ax.set_title('Blood Glucose Records (mg/dL) for {0} to {1}'.format(
                    timemin.strftime('%b %d'), timemax.strftime('%b %d')),
                    loc='left')
    
    # Plot the data.
    ax.plot(times, bgs, linewidth=1.0, color='k', linestyle=ls, marker='o', markersize=3.)
    
    # Label statistics.
    labeled_boxplot(ax, bgs, 1., transform=ax.get_yaxis_transform()) 


    #ymin, ymax = ax.get_ylim()
    #xmin, xmax = ax.get_xlim()
    #axis_top = ymin+(ymax-ymin)/100.
    #spread = [x for x in timerange(min(datax), max(datax), datetime.timedelta(hours=1))]
    #ax.fill_between(spread, ymin*np.ones(len(spread)), axis_top*np.ones(len(spread)), 
    #    where=[dataxi.hour>=23 or dataxi.hour<=6 for dataxi in spread] , color='#BF00FF', alpha=0.3)


    #x_coverage(ax, datax, datetime.timedelta(hours=1.5), 70, 150, 0.1)
    #x_coverage(ax, datax, datetime.timedelta(hours=1.5), 0.01, 0.02, 0.1, transform = ax.get_xaxis_transform())
    
    return plt.gca()

def x_coverage(ax, data, spread, bottom = 0, top = 1, alpha=0.1, transform=None):
    """Add vertical bars around the points of a scatterplot to easily identify regions that are well covered.
    
    Parameters
    ----------
        handle - the handle of the graph to be amended
        data - the abscissa points
        spread - either a singleton, for equal spread to the left and right, or else a tuple (left, right)
        top - the ordinate for the top of the bars
        bottom - the ordinate for the bottom of the bars
        alphs - the opacity of the bars
        transform - 
    """
    # Get the appropriate transform for coordinates
    if transform == None:
        transform = ax.transData
    # set the left and right spread for the coverage bars
    if np.size(spread)==2:
        leftpread = spread[0]
        rightspread = spread[1]
    else:
        leftspread = rightspread = spread
    
    for x in data:
        # put the bars between bottom and top
        ax.fill_between([x-leftspread, x+rightspread], [top, top], [bottom, bottom], 
                            alpha=alpha, color = '#FF4775', transform=transform)


def labeled_boxplot(ax, data, xloc, fontsize=12, transform=None): 
    if transform == None:
        transform = ax.transData
        
    # collect the quartile information
    y100 = stats.scoreatpercentile(data, 100)    
    y75 = stats.scoreatpercentile(data, 75)
    y50 = np.mean(data)
    y25 = stats.scoreatpercentile(data, 25)
    y0 = stats.scoreatpercentile(data, 0)
    
    # convert the xloc and location of the mean to the figure scale coordinates
    # so that the gap around the mean and labels (if label==True) are drawn in the
    # correct location independent of data scale
    pt_scale = transform.transform( [[xloc, y50], 
                                     [xloc, y50]] )
    pt_scale[:,1] += [2, -2]                         # make a 4 pixel gap
    transform = ax.get_yaxis_transform()             # reset the transform
    pt_ax = transform.inverted().transform(pt_scale) # retransform
    xloc = pt_ax[0,0]
    mean_gap_up = pt_ax[0,1]
    mean_gap_down = pt_ax[1,1]
    
    # draw the boxplot lines
    ax.vlines(xloc, mean_gap_up, y75, lw=2., transform=transform)
    ax.vlines(xloc, y25, mean_gap_down, lw=2., transform=transform)
    ax.vlines(xloc, y75, y100, lw=1., transform=transform)
    ax.vlines(xloc, y0, y25, lw=1., transform=transform)
    
    # label the data points
    ax.text(xloc, y50, ('%1.f' % y50), va='center', fontsize=fontsize, transform=transform)
    ax.text(xloc, y75, ('%1.f' % y75), va='center', fontsize=0.75*fontsize, transform=transform)
    ax.text(xloc, y25, ('%1.f' % y25), va='center', fontsize=0.75*fontsize, transform=transform)
    ax.text(xloc, y100, ('%1.f' % y100), va='center', fontsize=0.83*fontsize, transform=transform)
    ax.text(xloc, y0, ('%1.f' % y0), va='center', fontsize=0.83*fontsize, transform=transform)

def timerange(start, end, step):
    date = start
    while date<=end:
        yield date
        date += step

def maptoweek(times): 
    times = np.array([datetime.datetime(1999, 11, 8, x.hour, x.minute, x.second) +\
                datetime.timedelta(days=x.weekday()) for x in times])
    xmin = datetime.datetime(1999, 11, 8, 0, 0, 0)
    xmax = datetime.datetime(1999, 11, 14, 23, 59, 59)
    return times, xmin, xmax

def maptoday(times):
    times = np.array([datetime.datetime(year=1999, month=12, day=1, 
                                hour=x.hour, minute=x.minute) 
                                for x in times])
    xmin = datetime.datetime(year=1999, month=12, day=1,
                                hour=0, minute=0)
    xmax = datetime.datetime(year=1999, month=12, day=1,
                                hour=23, minute=59, second=59)
    return times, xmin, xmax





