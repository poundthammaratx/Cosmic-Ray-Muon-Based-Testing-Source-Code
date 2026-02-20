# eventHist.py

class eventHist:
    def __init__(self, xmin, xmax, nbins, xlabel="", ylabel="", zlabel=""):
        """
        Initializes a histogram object.

        Args:
            xmin (float): The minimum value of the histogram range.
            xmax (float): The maximum value of the histogram range.
            nbins (int): The number of bins in the histogram.
            xlabel (str, optional): Label for the x-axis. Defaults to "".
            ylabel (str, optional): Label for the y-axis. Defaults to "".
            zlabel (str, optional): Label for the z-axis (for 3D histograms, not used here). Defaults to "".
        """
        if nbins <= 0:
            raise ValueError("Number of bins must be positive.")
        if xmax <= xmin:
            raise ValueError("xmax must be greater than xmin.")

        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.nbins = int(nbins)
        self.dx = (self.xmax - self.xmin) / self.nbins
        self.counts = [0] * self.nbins
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel

    def increment(self, value):
        """
        Increments the count of the bin corresponding to the given value.
        Values outside the xmin-xmax range are ignored.

        Args:
            value (float): The value to be added to the histogram.
        """
        if self.xmin <= value < self.xmax:
            bin_index = int((value - self.xmin) / self.dx)
            # Ensure bin_index is within valid range, especially for values very close to xmax
            if bin_index == self.nbins:
                bin_index -= 1
            self.counts[bin_index] += 1

    def getCounts(self, bin_index):
        """
        Returns the count of a specific bin.

        Args:
            bin_index (int): The index of the bin.

        Returns:
            int: The count of the specified bin, or 0 if the index is out of range.
        """
        if 0 <= bin_index < self.nbins:
            return self.counts[bin_index]
        return 0

    def get_bin_centers(self):
        """
        Returns a list of the center values for each bin.

        Returns:
            list: A list of bin center values.
        """
        return [self.xmin + (i + 0.5) * self.dx for i in range(self.nbins)]

    def get_edges(self):
        """
        Returns a list of the bin edges.

        Returns:
            list: A list of bin edge values.
        """
        return [self.xmin + i * self.dx for i in range(self.nbins + 1)]

    def get_counts_array(self):
        """
        Returns the raw counts array.

        Returns:
            list: A list containing the counts for each bin.
        """
        return self.counts