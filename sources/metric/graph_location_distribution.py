import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class GraphLocationDistribution:
    def __init__(self, path, labels):
        self.labels = labels
        self.path = path

        self.__generate_graph()

    def __calculate_location_distribution(self):
        result = dict()
        for label in self.labels:
            if label in result:
                result[label] = result[label] + 1
            else:
                result[label] = 1

        log_file = open(self.path + "log_location_distribution.csv", "w")
        log_file.write("location,amount\n")
        if 0 in result:
            log_file.write("{0},{1}\n".format(0, result[0]))
        total = sum([result[i] for i in range(1, len(result.values()))])
        keys = range(1, len(result.values()))
        values = []
        for i in range(1, len(result.values())):
            values.append(result[i] / total)
            log_file.write("{0},{1}\n".format(i, result[i]))
        log_file.close()
        return keys, values

    def __generate_graph(self):
        fig = plt.figure(figsize=(30 / 2.54, 15 / 2.54))
        x, y = self.__calculate_location_distribution()
        plt.bar(x, y)
        plt.xlabel("Standort (Diskret)")
        plt.ylabel("Anteil")
        plt.savefig("{0}location_distribution.png".format(self.path))
        plt.clf()
        plt.close(fig)
