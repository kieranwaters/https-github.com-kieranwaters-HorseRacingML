using Tensorflow;
using Tensorflow.NumPy;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using HorseRacingML.Models;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Builds and trains a simple TensorFlow model using supplied hyperparameters.
    /// GPU is used when available.
    /// </summary>
    public (double TrainAccuracy, double TrainLoss, double ValidationAccuracy, double ValidationLoss) Train(MLParameter param)
    {
        var gpus = tf.config.list_physical_devices("GPU"); if (gpus.Length > 0) { try { tf.config.experimental.set_memory_growth(gpus[0], true); } catch { } } // enable GPU if present
        var x = np.random.uniform(0f, 1f, new Shape(1000, 10)); var y = np.random.randint(0, 2, new Shape(1000, 1)).astype(np.float32); // dummy data
        var xTrain = x[new Slice(0, 800)]; var yTrain = y[new Slice(0, 800)]; // split train
        var model = keras.Sequential(); // build model
        model.add(keras.layers.Dense(units: param.Units, input_shape: new Shape(10))); model.add(keras.layers.ReLU()); if (param.Dropout > 0) model.add(keras.layers.Dropout((float)param.Dropout)); // input block (no activation arg to avoid overload issues)
        for (int i = 1; i < param.Layers; i++) { model.add(keras.layers.Dense(units: param.Units)); model.add(keras.layers.ReLU()); if (param.Dropout > 0) model.add(keras.layers.Dropout((float)param.Dropout)); } // hidden blocks
        model.add(keras.layers.Dense(units: 1)); // output logits (no sigmoid)
        var optimizer = keras.optimizers.Adam((float)param.LearningRate); // optimizer
        model.compile(optimizer: optimizer, loss: keras.losses.BinaryCrossentropy(from_logits: true), metrics: new Tensorflow.Keras.Metrics.IMetricFunc[] { keras.metrics.BinaryAccuracy() }); // compile with loss object + metric object
        var history = model.fit(x: xTrain, y: yTrain, batch_size: param.BatchSize, epochs: param.Epochs, validation_split: 0.2f, verbose: 0); // train (use validation_split to avoid absent validation_data overload)
        var hist = history.history; // grab metrics
        double trainLoss = ((NDArray)hist["loss"])[-1].AsScalar<double>(); // last epoch train loss
        string accKey = hist.ContainsKey("binary_accuracy") ? "binary_accuracy" : "accuracy"; double trainAcc = ((NDArray)hist[accKey])[-1].AsScalar<double>(); // last epoch train acc
        double valLoss = ((NDArray)hist["val_loss"])[-1].AsScalar<double>(); // last epoch val loss
        string valAccKey = hist.ContainsKey("val_binary_accuracy") ? "val_binary_accuracy" : "val_accuracy"; double valAcc = ((NDArray)hist[valAccKey])[-1].AsScalar<double>(); // last epoch val acc
        return (trainAcc, trainLoss, valAcc, valLoss); // results
    }

}

