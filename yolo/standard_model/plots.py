import matplotlib.pyplot as plt
mean_loss = [0.5, 0.4, 0.3, 0.2, 0.1]  # Example data
plt.plot(mean_loss, label="Mean Loss")
plt.xlabel("Batch")
plt.ylabel("Mean Loss")
plt.title("Mean Loss per Batch")
plt.legend()
# To show the plot in the terminal
# To save the plot to a file (e.g., "plot.png")
# Uncomment the following line if you want to save the plot instead of showing it
plt.savefig("plot.png")
# /cs/student/kennethtan/cs190i/yolo/

plt.show()

