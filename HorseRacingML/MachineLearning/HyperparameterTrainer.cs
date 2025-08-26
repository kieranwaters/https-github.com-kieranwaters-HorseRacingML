using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using HorseRacingML.Models;

namespace HorseRacingML.ML
{
    /// <summary>
    /// Builds and trains a simple TensorFlow model using supplied hyperparameters.
    /// GPU is used when available.
    /// </summary>
    public class HyperparameterTrainer
    {
        public (double TrainAccuracy, double TrainLoss, double ValidationAccuracy, double ValidationLoss) Train(MLParameter param)
        {
            // Enable GPU if available
            var gpus = tf.config.list_physical_devices("GPU");
            if (gpus.Length > 0)
            {
                try { tf.config.experimental.set_memory_growth(gpus[0], true); } catch { }
            }

            // Generate dummy data for demonstration purposes
            var x = np.random.uniform(0f, 1f, new Shape(1000, 10));
            var y = np.random.randint(0, 2, new Shape(1000, 1)).astype(np.float32);
            var xTrain = x[new Slice(0, 800)];
            var yTrain = y[new Slice(0, 800)];

            // Build a simple sequential model
            var model = keras.Sequential();
            model.add(keras.layers.Dense(units: param.Units, activation: keras.activations.Relu, input_shape: new Shape(10)));
            if (param.Dropout > 0)
                model.add(keras.layers.Dropout((float)param.Dropout));

            for (int i = 1; i < param.Layers; i++)
            {
                model.add(keras.layers.Dense(units: param.Units, activation: keras.activations.Relu));
                if (param.Dropout > 0)
                    model.add(keras.layers.Dropout((float)param.Dropout));
            }

            // Output layer (logits)
            model.add(keras.layers.Dense(units: 1));

            var optimizer = keras.optimizers.Adam((float)param.LearningRate);
            var loss = keras.losses.BinaryCrossentropy(from_logits: true);
            var metric = keras.metrics.BinaryAccuracy();

            model.compile(optimizer: optimizer, loss: loss, metrics: new Tensorflow.Keras.Metrics.IMetricFunc[] { metric });

            // Train the model (use validation_split to get validation metrics)
            var history = model.fit(x: xTrain, y: yTrain,
                                   batch_size: param.BatchSize,
                                   epochs: param.Epochs,
                                   validation_split: 0.2f,
                                   verbose: 0);

            var hist = history.history;
            double trainLoss = ((NDArray)hist["loss"])[-1].AsScalar<double>();
            string accKey = hist.ContainsKey("binary_accuracy") ? "binary_accuracy" : "accuracy";
            double trainAcc = ((NDArray)hist[accKey])[-1].AsScalar<double>();
            double valLoss = ((NDArray)hist["val_loss"])[-1].AsScalar<double>();
            string valAccKey = hist.ContainsKey("val_binary_accuracy") ? "val_binary_accuracy" : "val_accuracy";
            double valAcc = ((NDArray)hist[valAccKey])[-1].AsScalar<double>();

            return (trainAcc, trainLoss, valAcc, valLoss);
        }
    }
}