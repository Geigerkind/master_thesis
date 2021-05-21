import matplotlib as mpl

from sources.config import BIN_FOLDER_PATH

mpl.use('Agg')
import matplotlib.pyplot as plt

distinct_number_of_locations = [9, 16, 17, 25, 32, 48, 51, 102]
dt_pb_5 = [0.9862, 0.9596, 0.9871, 0.9787, 0.9395, 0.9042, 0.9208, 0.8735]
knn_pb_5 = [0.9776, 0.9208, 0.9724, 0.9605, 0.8451, 0.8318, 0.8876, 0.7472]
dt_pb_5_cont = [0.9824, 0.8949, 0.9864, 0.9639, 0.8834, 0.8932, 0.896, 0.8777]
knn_pb_5_cont = [0.9617, 0.7177, 0.9333, 0.8411, 0.6524, 0.4508, 0.7539, 0.4882]

fig = plt.figure(figsize=(30 / 2.54, 15 / 2.54))
plt.plot(distinct_number_of_locations, dt_pb_5, "o-", color="green")
plt.plot(distinct_number_of_locations, dt_pb_5_cont, "*-", color="lime")
plt.plot(distinct_number_of_locations, knn_pb_5, "o-", color="blue")
plt.plot(distinct_number_of_locations, knn_pb_5_cont, "*-", color="darkblue")
plt.xlabel("Anzahl Standorte (Diskret)", fontsize=16)
plt.ylabel("Klassifizierungsgenauigkeit", fontsize=16)
plt.ylim([0, 1])
fig.legend(['Entscheidungswald (ohne Rückwärtskante) P(B≤5)',
            'Entscheidungswald (mit Rückwärtskante) P(B≤5) (cont)',
            'FFNN (ohne Rückwärtskante) P(B≤5)',
            'FFNN (mit Rückwärtskante) P(B≤5) (cont)'],
           loc=[0.13, 0.12], fontsize=12)
plt.savefig("{0}/best_dt_vs_knn_pb_5_vs_pb_5_cont.png".format(BIN_FOLDER_PATH))
plt.clf()
plt.close(fig)
