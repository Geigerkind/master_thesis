import matplotlib as mpl

from sources.config import BIN_FOLDER_PATH

mpl.use('Agg')
import matplotlib.pyplot as plt

distinct_number_of_locations = [9, 16, 17, 25, 32, 48, 51, 102]
dt_acc_wo_fb = [0.965, 0.9429, 0.9649, 0.952, 0.9248, 0.8905, 0.8927, 0.8611]
knn_acc_wo_fb = [0.9416, 0.8892, 0.9346, 0.9244, 0.8254, 0.8153, 0.8516, 0.7325]
dt_acc_w_fb = [0.9376, 0.8352, 0.9441, 0.9242, 0.8556, 0.8604, 0.8497, 0.8437]
knn_acc_w_fb = [0.8977, 0.6394, 0.8722, 0.7905, 0.5944, 0.4222, 0.6898, 0.4504]

mian_distinct_loc = [6, 9, 14]
mian_wffnn = [0.9913, 0.9441, 0.9451]
mian_wfbnn = [0, 0, 0.9326]
mian_fbnn = [0.5662, 0.8556, 0.3357]


fig = plt.figure(figsize=(30 / 2.54, 15 / 2.54))
plt.plot(distinct_number_of_locations, dt_acc_wo_fb, "o-", color="green")
plt.plot(distinct_number_of_locations, dt_acc_w_fb, "*-", color="lime")
plt.plot(distinct_number_of_locations, knn_acc_wo_fb, "o-", color="blue")
plt.plot(distinct_number_of_locations, knn_acc_w_fb, "*-", color="darkblue")
plt.plot(mian_distinct_loc, mian_wffnn, "o-", color="orange")
plt.plot(mian_distinct_loc, mian_wfbnn, "*-", color="darkgoldenrod")
plt.plot(mian_distinct_loc, mian_fbnn, "X-", color="saddlebrown")
plt.xlabel("Anzahl Standorte (Diskret)", fontsize=16)
plt.ylabel("Klassifizierungsgenauigkeit", fontsize=16)
plt.ylim([0, 1])
"""
fig.legend(['Entscheidungswald (ohne Rückwärtskante) P(A)',
            'Entscheidungswald (mit Rückwärtskante) P(A) (cont)',
            'FFNN (ohne Rückwärtskante) P(A)',
            'FFNN (mit Rückwärtskante) P(A) (cont)',
            'Mian WFFNN',
            'Mian WFBNN',
            'Mian FBNN'],
           loc=[0.475, 0.11], fontsize=12)
"""
plt.savefig("{0}/best_dt_vs_knn_fb_vs_no_fb.png".format(BIN_FOLDER_PATH))
plt.clf()
plt.close(fig)