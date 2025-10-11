using System;
using System.Linq;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace HorseRacingML.ML
{
    internal static class GpuMath
    {
        private static readonly Lazy<bool> _isGpuAvailable = new Lazy<bool>(InitializeGpu, true);
        private static string? _initializationError;

        public static bool IsGpuAvailable => _isGpuAvailable.Value;

        public static string? InitializationError => _initializationError;

        private static bool InitializeGpu()
        {
            try
            {
                if (!tf.executing_eagerly())
                {
                    tf.enable_eager_execution();
                }
            }
            catch (Exception ex)
            {
                _initializationError = ex.Message;
                return false;
            }

            try
            {
                var devices = tf.config.list_physical_devices("GPU");
                return devices != null && devices.Count() > 0;
            }
            catch (Exception ex)
            {
                _initializationError = ex.Message;
                return false;
            }
        }

        public static bool TryMatMul(double[] inputs, double[][] weights, double[] bias, out double[] result)
        {
            result = Array.Empty<double>();
            if (!IsGpuAvailable)
            {
                return false;
            }

            if (weights.Length != inputs.Length)
            {
                return false;
            }

            var outputCount = weights.Length == 0 ? 0 : (weights[0]?.Length ?? 0);
            if (outputCount == 0)
            {
                return true;
            }

            try
            {
                var weightMatrix = ToTransposedMatrix(weights, outputCount);
                var matrixTensor = tf.constant(ToFloatMatrix(weightMatrix), TF_DataType.TF_FLOAT);
                var inputTensor = tf.constant(ToFloatArray(inputs), TF_DataType.TF_FLOAT);
                var inputColumn = tf.reshape(inputTensor, new long[] { inputs.Length, 1 });
                var product = tf.matmul(matrixTensor, inputColumn);
                var flattened = tf.reshape(product, new long[] { outputCount });
                var biasVector = tf.constant(ToFloatArray(PadBias(bias, outputCount)), TF_DataType.TF_FLOAT);
                var summed = tf.add(flattened, biasVector);
                NDArray output = summed.numpy();
                var values = output.ToArray<float>();
                result = values.Select(v => (double)v).ToArray();
                return true;
            }
            catch
            {
                return false;
            }
        }

        public static bool TryRelu(double[] values, out double[] result)
        {
            result = Array.Empty<double>();
            if (!IsGpuAvailable)
            {
                return false;
            }

            try
            {
                var tensor = tf.constant(ToFloatArray(values), TF_DataType.TF_FLOAT);
                var relu = tf.nn.relu(tensor);
                NDArray output = relu.numpy();
                var reluValues = output.ToArray<float>();
                result = reluValues.Select(v => (double)v).ToArray();
                return true;
            }
            catch
            {
                return false;
            }
        }

        private static double[,] ToTransposedMatrix(double[][] weights, int outputCount)
        {
            int inputCount = weights.Length;
            var matrix = new double[outputCount, inputCount];
            for (int i = 0; i < inputCount; i++)
            {
                var row = weights[i] ?? Array.Empty<double>();
                for (int j = 0; j < Math.Min(outputCount, row.Length); j++)
                {
                    matrix[j, i] = row[j];
                }
            }

            return matrix;
        }

        private static double[] PadBias(double[] bias, int length)
        {
            if (bias.Length >= length)
            {
                return bias.Take(length).ToArray();
            }

            var result = new double[length];
            Array.Copy(bias, result, bias.Length);
            return result;
        }

        private static float[,] ToFloatMatrix(double[,] source)
        {
            int dim0 = source.GetLength(0);
            int dim1 = source.GetLength(1);
            var result = new float[dim0, dim1];
            for (int i = 0; i < dim0; i++)
            {
                for (int j = 0; j < dim1; j++)
                {
                    result[i, j] = (float)source[i, j];
                }
            }

            return result;
        }

        private static float[] ToFloatArray(double[] source)
        {
            return source.Select(v => (float)v).ToArray();
        }
    }
}