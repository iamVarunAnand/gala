# import the necessary packages
import h5py
import os


class HDF5DatasetWriter:
    def __init__(self, dims, output_path, data_key = "images", buf_size = 1024):
        # raise an exception if the output path exists
        if os.path.exists(output_path):
            raise ValueError(
                "The supplied ‘outputPath‘ already exists and cannot be"
                "overwritten.Manually delete the file before continuing.",
                output_path)

        # open the database for writing and create the necessary datasets
        self.db = h5py.File(output_path, "w")
        self.data = self.db.create_dataset(data_key, dims, dtype = "float")
        self.labels = self.db.create_dataset(
            "labels", (dims[0],), dtype = "int")

        # initialize the instance variables
        self.buf_size = buf_size
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        # add the rows and the labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) > self.buf_size:
            self.flush()

    def flush(self):
        # write the buffer to disk
        end_idx = self.idx + len(self.buffer["data"])
        self.data[self.idx: end_idx] = self.buffer["data"]
        self.labels[self.idx: end_idx] = self.buffer["labels"]

        # reset the buffer
        self.idx = end_idx
        self.buffer = {"data": [], "labels": []}

    def store_class_labels(self, class_labels):
        # create a dataset to store the class labels
        dt = h5py.special_dtype(vlen = str)
        label_set = self.db.create_dataset(
            "label_names", (len(class_labels), ), dtype = dt)
        label_set[:] = class_labels

    def close(self):
        # check if anything needs to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # close the connection
        self.db.close()
