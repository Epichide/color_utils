
import matplotlib.pyplot as plt
import random
from scipy.stats import norm
import numpy as np

# plot standard normal distribution cdf curve
def plot_standard_normal_cdf():
    mu = 0
    sigma = 1
    # ==> standard normal cdf
    xs= np.linspace(-5, 5, 1000)
    cdf = norm.cdf(xs, mu, sigma) 
    
    # find x(1JOD) where  0.75 CDF,  ==> JOD cdf
    cdf_y=0.75
    x_75 = norm.ppf(cdf_y, mu, sigma)
    
    # scale x axis so that CDF(1JOD)=0.75
    xscale=1/x_75
    scaled_xs = xs * xscale
    
    # Thurstone scaling factor , eliminate the tail effect (==>approximately JOD CDF)
    thurstone_xs=(12/np.pi * np.arcsin( np.sqrt(cdf) ) - 3)
    thurstone_x_75=(12/np.pi * np.arcsin( np.sqrt(cdf_y) ) - 3)
    
    
    # plot all three cdf curves
    plt.figure(figsize=(10, 6))
    plt.plot(scaled_xs, cdf, label='Scaled JOD CDF', color='blue')
    plt.plot(xs, cdf, label='Standard Normal CDF', color='orange', linestyle=':')
    plt.plot(thurstone_xs, cdf, label='Thurstone approximate CDF', color='cyan', linestyle='-')
    plt.axhline(y=cdf_y, color='k', linestyle='--', label=f'CDF = {cdf_y}')
    plt.axvline(x=x_75, color='orange', linestyle='--', label=f'x = {x_75:.5f}')
    plt.axvline(x=1, color='blue', linestyle='--', label='x = 1 (JOD)')
    plt.axvline(x=thurstone_x_75, color='cyan', linestyle='--', label=f'x = {thurstone_x_75:.5f} (Thurstone)')
    
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.title('Standard & Scaled Normal  Distribution CDF')
    plt.grid()
    plt.xlim([-5.3, 5.3])
    # set x label showing ticks at every 1 unit
    plt.xticks(np.arange(-5, 6, 1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    
    # plot all three distribution pdf curves
    plt.figure(figsize=(10, 6))
    pdf = norm.pdf(xs, mu, sigma)
    JOD_pdf = norm.pdf(scaled_xs, mu, sigma* xscale) 
    plt.plot(scaled_xs, JOD_pdf, label='Scaled JOD PDF', color='blue')
    plt.plot(xs, pdf, label='Standard Normal PDF', color='orange', linestyle=':')
    thurstone_pdf= 2*np.sin(np.pi/12 * (thurstone_xs +3)) * np.cos(np.pi/12 * (thurstone_xs +3)) * (np.pi/12)
    thurstone_pdf=np.pi/12 * np.sin(np.pi/6 * (thurstone_xs +3))
    plt.plot(thurstone_xs, thurstone_pdf, label='Thurstone approximate PDF', color='cyan', linestyle='-')
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.title('Standard & Scaled Normal Distribution PDF')
    plt.grid()
    plt.xlim([-5.3, 5.3])
    # set x label showing ticks at every 1 unit
    plt.xticks(np.arange(-5, 6, 1))
    plt.legend()
    
    # test some values
    cdf_y2=np.array([0.1,0.9,0.23333,0.76666])
    x2 = norm.ppf(cdf_y2, mu, sigma)
    x3=np.array([1.0792,-1.0792,2.7190,-2.7190,2]) /xscale
    cdf_y3=norm.cdf(x3, mu, sigma)
    print(xscale)
    print(f"Probability x for CDF={cdf_y2} is {x2}","scaled to",x2*xscale)
    print(f"CDF for x={x3} is {cdf_y3}")
    
    
    plt.show()
    
if __name__ == "__main__":
    plot_standard_normal_cdf()
    # sum( -(12/pi * asin( sqrt(M) ) - 3)/2, 1 )
    # ys=np.linspace(0, 1, 100)
    # xs=(12/np.pi * np.arcsin( np.sqrt(ys) ) - 3)
    # # plt.figure(figsize=(10, 6))
    # plt.plot(xs, ys)
    # plt.ylabel('M')
    # plt.xlabel('JOD Distance')
    # plt.title('M to JOD Distance Conversion')
    # plt.show()