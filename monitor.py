import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import my_logger, logging
from matplotlib.mlab import specgram
from scipy.fftpack import fft
from collections import deque
import threading 
import time
import sys

logger = logging.getLogger("logger.plot")
logger.setLevel(logging.WARN)

class SerialPlot:

    def __init__(self, str_port, baud_rate, max_len, num_of_sensors, fft_len):
        self.__max_len = max_len
        self.__num_of_sensors = num_of_sensors
        try:
            self.__ser = serial.Serial(str_port, baud_rate)
            self.__ser.flushInput()
        except:
            print("{} not found".format(str_port))
        self.__data = [deque([0.0]*max_len) for num in range(num_of_sensors)]
        self.__lock = True
        self.cnt = 0
        self.ti = time.time()         
        self.time = 0
        self.ann = None
        self.fft_len = fft_len
        self.__sampling_freq = 25

    def __add_to_buf(self, buf, val):
        if len(buf) < self.__max_len:
            buf.append(val)
        else:
            buf.pop()
            buf.appendleft(val)
    
    def __add(self, data):
        assert(len(data) == self.__num_of_sensors)
        for (rcv_data, deque) in zip(data, self.__data):
            self.__add_to_buf(deque, rcv_data)
    
    def __get_serial_data(self, p):
        try:
            data = []
            while  self.__ser.in_waiting>0:
                line = self.__ser.readline()
                if p == True:
                    print(line.strip().split(b','))
                data = [float(val) for val in line.strip().split(b',')]
                self.__add(data)
            return data
        except KeyboardInterrupt:
            print("exitiing")

    def update_raw_data(self,  frame, sub_plot):
            self.__get_serial_data(False)
            min_average = np.inf
            min_plot = None
            for plot, ax, d in zip(sub_plot[0], sub_plot[1], self.__data):
                plot.set_data(range(self.__max_len), d)
                plot.set_color('b')
                ave = np.average(d)
                ax.set_title("ave={:.2f}".format(ave))
                if(ave < min_average):
                    min_average = ave
                    min_plot = plot
            min_plot.set_color('r')
            self.__lock = False

    def update_sum_plot(self, plot, ax):
        sum = self.get_sum_buffered_data()
        plot.set_data(range(len(sum)),sum)
        ax.set_title("average = {:.2f}".format(np.average(sum))) 
        # ax.grid(color='k', linestyle='--', linewidth=0.5)
        return sum

    def update_fft_plot(self, plot, ax, data, cnt, t):
        fft_data = self.get_fft(data, length=self.fft_len)
        dc = fft_data[0]
        #get maximum from first
        first_zero = int(self.fft_len/self.__max_len)
        sub_data =list(fft_data[first_zero:int(self.fft_len/2)])
        max_harmonic = max(sub_data)
        ratio = max_harmonic
        max_harmonic_number = first_zero+sub_data.index(max_harmonic)
        max_harmonic_freq = self.__sampling_freq*max_harmonic_number/self.fft_len
        plot.set_data(range(len(fft_data)), fft_data)
        ax.set_title("rate={:.2f} f={:.2f} cnt={} t={:.2f}".format(
                      ratio, max_harmonic_freq, int(cnt), t))
        if self.ann != None:
            self.ann.remove()
        self.ann = ax.annotate('local max',
                               xy=(max_harmonic_number, max_harmonic),
                               xytext=(max_harmonic_number, max_harmonic+10),
                               arrowprops=dict(facecolor='red', shrink=0.01),
                               )
        return ratio, max_harmonic_freq

    def update_analyse(self, frame, sub_plot):
        if not self.__lock:
            tim = time.time() - self.ti
            self.ti = time.time()
            sum_array = self.update_sum_plot(sub_plot[0][1], sub_plot[1][1])
            threshold, max_harmonic_freq = self.update_fft_plot(sub_plot[0][0],
                                                                sub_plot[1][0],
                                                                sum_array,
                                                                self.cnt, 
                                                                self.time)
            if threshold>80:
                 c = tim*max_harmonic_freq
                 self.cnt += c
                 self.time = self.time + tim
                 logger.debug(c)
            else:
                 cnt = 0
            self.__lock = True

    def get_sum_buffered_data(self):
        sum = []
        for i in range(self.__max_len):
            s = 0
            for d in range(self.__num_of_sensors):
                s = s + self.__data[d][i]
            sum.append(s)
        return sum

    def get_average(self):
        return np.average(self.get_sum_buffered_data())

    def get_spectrogram(self, data):
        Pxx, freqs, bins  = specgram(data, NFFT=64, Fs=25, noverlap=5)
        return Pxx, freqs, bins
    
    def get_fft(self, data, length):
        data = data - np.average(data)
        fft_data = 20*np.log(np.abs(fft(data, n=length)))
        return fft_data

    def add_ma_filter(self, data, ma_len):
        return np.convolve(data, np.ones(ma_len))

    def get_row_data(self):
        return self.__data

class Draw:
    
    def __init__(self, serial_port, baud, num_of_sensors, fft_length):
        fft_len = fft_length
        sp = SerialPlot("/dev/"+str(serial_port), baud, 25, num_of_sensors,
                        fft_length)
        #define fig
        self.rawd_fig, self.rawd_ax = plt.subplots(2,2)
        self.fft_fig, self.fft_ax = plt.subplots(2,1)
        self.rawd_plot = []
        self.fft_plot = []
        rawd_color = ['r', 'g', 'b', 'c']
        for c,a in zip(rawd_color,self.rawd_ax.reshape(4,)):
                a.set_xlim([0,25])
                a.set_ylim([0,50])
                b, = a.plot([], [], color = c)
                self.rawd_plot.append(b)
        fft_label = ['fft' , 'sum']
        fft_color = ['g' , 'r']
        for l,c,ax in zip(fft_label,fft_color,self.fft_ax):
            ax.set_xlim([0,25])
            ax.set_ylim([0,200])
            b, = ax.plot([], [], label=l, color=c)
            self.fft_plot.append(b)    
    
        ax.grid(color='k', linestyle='--', linewidth=0.5)
        self.fft_ax[0].set_xlim([0, fft_len])
        self.fft_ax[0].set_ylim([0, 200])
        b, = self.fft_ax[0].plot([], [], label='fft', color='g')
        self.fft_fig.canvas.set_window_title(
                'FFT len={} & Average'.format(fft_len))
        self.rawd_fig.canvas.set_window_title('Raw Data')
        anim = animation.FuncAnimation(self.rawd_fig, sp.update_raw_data, 
                                       fargs=((self.rawd_plot, 
                                               self.rawd_ax.reshape(4,)),),
                                               interval=1)
        anim1 = animation.FuncAnimation(self.fft_fig, sp.update_analyse,
                                        fargs=((self.fft_plot, self.fft_ax),),
                                        interval=1)
        plt.show()

def main():
    try:
        d = Draw(serial_port=sys.argv[1], baud=int(sys.argv[2]), 
                 fft_length=int(sys.argv[3]), num_of_sensors=4)
    except KeyboardInterrupt:
        exit()
        
if __name__ == '__main__':
    main()


