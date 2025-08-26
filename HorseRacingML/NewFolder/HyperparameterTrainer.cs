using Tensorflow;
using Tensorflow.NumPy;
using HorseRacingML.Models;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using OpenQA.Selenium;

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
            // Enable GPU if one is present
            var gpus = tf.config.list_physical_devices("GPU");
            if (gpus.Length > 0)
            {
                try { tf.config.experimental.set_memory_growth(gpus[0], true); } catch { }
            }

            // Example dataset - random data for demonstration purposes
            var x = np.random.rand(1000, 10).astype(np.float32);
            var y = np.random.randint(0, 2, (1000, 1)).astype(np.float32);

            var xTrain = x[new Slice(0, 800)];
            var yTrain = y[new Slice(0, 800)];
            var xVal = x[new Slice(800, 1000)];
            var yVal = y[new Slice(800, 1000)];

            var layers = new Tensorflow.Keras.LayersApi();
            var model = keras.Sequential();
            model.add(layers.Dense(param.Units, activation: keras.activations.Relu, input_shape: new Shape(10)));
            if (param.Dropout > 0)
                model.add(layers.Dropout(param.Dropout));

            for (int i = 1; i < param.Layers; i++)
            {
                model.add(layers.Dense(param.Units, activation: keras.activations.Relu));
                if (param.Dropout > 0)
                    model.add(layers.Dropout(param.Dropout));
            }

            model.add(layers.Dense(1, activation: keras.activations.Sigmoid));

            var optimizer = keras.optimizers.Adam(param.LearningRate);
            model.compile(optimizer, keras.losses.BinaryCrossentropy(), new[] { "accuracy" });

            model.fit(xTrain, yTrain, batch_size: param.BatchSize, epochs: param.Epochs,
                      validation_data: (xVal, yVal), verbose: 0);

            var trainEval = model.evaluate(xTrain, yTrain, verbose: 0);
            var valEval = model.evaluate(xVal, yVal, verbose: 0);

            double trainLoss = trainEval[0];
            double trainAcc = trainEval[1];
            double valLoss = valEval[0];
            double valAcc = valEval[1];

            return (trainAcc, trainLoss, valAcc, valLoss);
        }
    }
}
