using UnityEngine;
using System.IO;
using System;

namespace MachineLearningMath
{
    /// <summary>
    /// Custom Matrix Class developped for working on the GPU and with DNANeuralNetworks
    /// </summary>
    [Serializable]
    public class Matrix
    {
        // 0--------> Width
        // |
        // |
        // |
        // Height

        public struct GPUMatrix
        {
            public double[] Values;
            public int Height;
            public int Width;
        }

        public struct GPUMatrixDimensions
        {
            public uint Height;
            public uint Width;
        }

        /// <summary>
        /// Shader Script that runs Matrix Multiplication on the GPU
        /// </summary>
        public static ComputeShader matrixMultScript;

        /// <summary>
        /// Shader Script that runs Matrix Multiplication on the GPU using Floats
        /// </summary>
        public static ComputeShader matrixMultFloatScript;

        /// <summary>
        /// Shader Script that runs Matrix Addition on the GPU
        /// </summary>
        public static ComputeShader matrixAdditionScript;

        /// <summary>
        /// Shader Script that runs Matrix Substraction on the GPU
        /// </summary>
        public static ComputeShader matrixSubstractionScript;

        /// <summary>
        /// 
        /// </summary>
        public static ComputeShader transposeScript;


        /// <summary>
        /// Load the Shader Scripts associated for speeding up the mathematics
        /// </summary>
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterAssembliesLoaded)]
        public static void loadAssets()
        {
            matrixMultScript = Resources.Load<ComputeShader>("MatrixMultiplicationGPU");
            matrixMultFloatScript = Resources.Load<ComputeShader>("MatrixMultiplicationGPUFloat");
            matrixAdditionScript = Resources.Load<ComputeShader>("MatrixAdditionGPU");
            matrixSubstractionScript = Resources.Load<ComputeShader>("MatrixSubstractionGPU");
            transposeScript = Resources.Load<ComputeShader>("TransposeGPU");
        }

        /// <summary>
        /// Describes the number of rows the matrix has
        /// </summary>
        [SerializeField]
        private int _height;

        /// <summary>
        /// Describes the number of rows the matrix has
        /// </summary>
        public int Height { get => _height; set => _height = value; }


        /// <summary>
        /// Describes the number of columns the matrix has
        /// </summary>
        [SerializeField]
        private int _width;

        /// <summary>
        /// Describes the number of columns the matrix has
        /// </summary>
        public int Width { get => _width; set => _width = value; }

        /// <summary>
        /// Gets the Dimensions of the Matrix in String Form (HeightxWidth)
        /// </summary>
        public string DebugDimension { get => GetDebugDimension(); }

        /// <summary>
        /// Gets the Dimensions of the Matrix in Array format
        /// </summary>
        public int[] Dimensions { get => new int[] { Height, Width }; }

        /// <summary>
        /// Describes the number of values in the Matrix
        /// </summary>
        public int Length { get => Width * Height; }

        /// <summary>
        /// A list of all values contained in the matrix
        /// </summary>
        [SerializeField]
        private double[] _values;

        /// <summary>
        /// A list of all values contained in the matrix
        /// </summary>
        public double[] Values
        {
            get
            {
                return _values;
            }
            set
            {
                _values = value;
            }
        }

        /// <summary>
        /// Constructor function initializing the Matrix
        /// </summary>
        /// <param name="height"></param>
        /// <param name="width"></param>
        public Matrix(int height, int width)
        {
            this.Width = width;
            this.Height = height;

            _values = new double[width * height];
        }

        /// <summary>
        /// Constructor function initializing the Matrix
        /// </summary>
        /// <param name="matrix"></param>
        public Matrix(MatrixFloat matrix)
        {
            this.Width = matrix.Width;
            this.Height = matrix.Height;

            _values = new double[Width * Height];

            for (int i = 0; i < matrix.Length; i++)
            {
                _values[i] = (double)matrix[i];
            }
        }

        /// <summary>
        /// Indexer allowing us to get access and sey  to a value using array notation
        /// </summary>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <returns></returns>
        public double this[int height, int width]
        {
            get
            {
                return _values[GetFlatIndex(height, width)];
            }
            set
            {
                _values[GetFlatIndex(height, width)] = value;
            }
        }

        /// <summary>
        /// Indexer allowing us to access and set the value in array notation
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public double this[int index]
        {
            get
            {
                return _values[index];
            }
            set
            {
                _values[index] = value;
            }
        }

        /// <summary>
        /// Indexer allowing us to get a vector from the matrix
        /// </summary>
        /// <param name="index"></param>
        /// <param name="idk"></param>
        /// <returns></returns>
        public double[] this[int index, bool row]
        {
            get
            {
                double[] vector;
                if (row)
                {
                    //Get a row
                    vector = new double[Width];

                    for (int i = 0; i < Width; i++)
                    {
                        vector[i] = this[index, i];
                    }

                }
                else
                {
                    //Get a column
                    vector = new double[Height];

                    for (int i = 0; i < Height; i++)
                    {
                        vector[i] = this[i, index];
                    }
                }

                return vector;
            }
        }


        /// <summary>
        /// Initializes a matrix that counts from 1 - the number of values based on the dimensions
        /// </summary>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <returns></returns>
        public static Matrix Increment(int height, int width)
        {
            Matrix matrix = new Matrix(height, width);

            for (int i = 0; i < width * height; i++)
            {
                matrix[i] = i;
            }

            return matrix;
        }

        /// <summary>
        /// Sets the value at the given height and width index
        /// </summary>
        /// <param name="heightIndex"></param>
        /// <param name="widthIndex"></param>
        /// <param name="val"></param>
        public void SetValue(int heightIndex, int widthIndex, double val)
        {
            this._values[GetFlatIndex(heightIndex, widthIndex)] = val;
        }

        /// <summary>
        /// Add to the values at the given height and width index
        /// </summary>
        /// <param name="heightIndex"></param>
        /// <param name="widthIndex"></param>
        /// <param name="val"></param>
        public void AddValue(int heightIndex, int widthIndex, double val)
        {
            this._values[GetFlatIndex(heightIndex, widthIndex)] += val;
        }

        /// <summary>
        /// Gets the value at the given height and width index
        /// </summary>
        /// <param name="heightIndex"></param>
        /// <param name="widthIndex"></param>
        /// <returns></returns>
        public double GetValue(int heightIndex, int widthIndex)
        {
            return _values[GetFlatIndex(heightIndex, widthIndex)];
        }

        /// <summary>
        /// Returns the flat index of a value 
        /// </summary>
        /// <param name="heightIndex"></param>
        /// <param name="widthIndex"></param>
        /// <returns></returns>
        public int GetFlatIndex(int heightIndex, int widthIndex)
        {
            return heightIndex * Width + widthIndex;
        }

        /// <summary>
        /// Returns calculated indeces for the matrix based on a length index
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public (int height, int width) GetIndex(int index)
        {
            int height = index / Width;
            int width = index % Width;

            return (height, width);
        }

        /// <summary>
        /// Returns calculated indeces for the matrix based on a length index for static functions
        /// </summary>
        /// <param name="index"></param>
        /// <param name="mat"></param>
        /// <returns></returns>
        public static (int height, int width) GetIndex(int index, Matrix mat)
        {
            int height = index / mat.Width;
            int width = index % mat.Width;

            return (height, width);
        }

        /// <summary>
        /// Determines if the Tensors have the Same Dimension
        /// </summary>
        /// <param name="dim1"></param>
        /// <param name="dim2"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        private static bool IsSameDimension(Matrix matrix1, Matrix matrix2)
        {
            int[] dim1 = matrix1.Dimensions;
            int[] dim2 = matrix2.Dimensions;

            if (dim1.Length == dim2.Length)
            {
                for (int i = 0; i < dim1.Length; i++)
                {
                    if (dim1[i] != dim2[i])
                        return false;
                }
            }
            else
                throw new InvalidOperationException($"Matrix dimensions do not match. ({dim1.Length}, {dim2.Length})");

            return true;
        }

        /// <summary>
        /// Returns the Dot Product
        /// </summary>
        /// <param name="vector1"></param>
        /// <param name="vector2"></param>
        /// <returns></returns>
        double DotProduct(double[] vector1, double[] vector2)
        {
            double value = 0;

            if (vector1.Length == vector2.Length)
            {
                for (int i = 0; i < vector1.Length; i++)
                {
                    value += vector1[i] * vector2[i];
                }
            }

            return value;
        }

        /// <summary>
        /// Returns the Dot Product for Static Functions
        /// </summary>
        /// <param name="vector1"></param>
        /// <param name="vector2"></param>
        /// <returns></returns>
        static double DotProductStatic(double[] vector1, double[] vector2)
        {
            double value = 0;

            if (vector1.Length == vector2.Length)
            {
                for (int i = 0; i < vector1.Length; i++)
                {
                    value += vector1[i] * vector2[i];
                }
            }

            return value;
        }

        /// <summary>
        /// Returns the Transpose of the matrix
        /// </summary>
        public Matrix Transpose()
        {
            Matrix transpose = new Matrix(this.Width, this.Height);

            if (transposeScript != null)
                transpose = TransposeGPU(this);
            else
            {
                for (int width = 0; width < this.Width; width++)
                {
                    for (int height = 0; height < this.Height; height++)
                    {
                        transpose[width, height] = this[height, width];
                    }
                }
            }

            return transpose;
        }

        /// <summary>
        /// Stacks 2 Matrices on top of each other Creating a Tensor
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public static Tensor operator ^(Matrix matrixA, Matrix matrixB)
        {
            if (IsSameDimension(matrixA, matrixB))
            {
                Tensor outputTensor = new Tensor(new int[] { 2, matrixA.Height, matrixB.Width });

                outputTensor.MatrixProperties[0] = matrixA;
                outputTensor.MatrixProperties[1] = matrixB;

                return outputTensor;
            }
            else
                throw new InvalidOperationException("Matrix Dimensions do not match.");
        }

        /// <summary>
        /// Operator for adding two matrices together
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        public static Matrix operator +(Matrix matrixA, Matrix matrixB)
        {
            Matrix newMat = new Matrix(0, 0);

            // if (matrixAdditionScript != null)
            //    newMat = matrixAdditionGPU(matrixA, matrixB);
            //else
            // {
            if (matrixA.Height == matrixB.Height && matrixA.Width == matrixB.Width)
            {
                newMat = new Matrix(matrixA.Height, matrixA.Width);

                for (int i = 0; i < matrixA.Values.Length; i++)
                {
                    newMat[i] = matrixA[i] + matrixB[i];
                }
            }
            else
                Debug.Log("Error, Dimensions don't match");

            return newMat;
        }

        /// <summary>
        /// Operator for Adding a Scalar to the Matrix
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        public static Matrix operator +(Matrix matrixA, double scalar)
        {
            Matrix newMat = new Matrix(matrixA.Height, matrixA.Width);

            for (int i = 0; i < matrixA.Values.Length; i++)
                newMat[i] = matrixA[i] + scalar;

            return newMat;
        }

        /// <summary>
        /// Operator handling subtractions between 2 matrices
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        public static Matrix operator -(Matrix matrixA, Matrix matrixB)
        {
            Matrix newMat = new Matrix(0, 0);

            if (matrixA.Height == matrixB.Height && matrixA.Width == matrixB.Width)
            {
                newMat = new Matrix(matrixA.Height, matrixA.Width);

                for (int i = 0; i < matrixA.Values.Length; i++)
                {
                    newMat[i] = matrixA[i] - matrixB[i];
                }
            }
            else
            {
                Debug.Log("Error, Dimensions don't match");
            }

            return newMat;
        }

        /// <summary>
        /// Operator for Subtracting a Scalar from the Matrix
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        public static Matrix operator -(Matrix matrixA, double scalar)
        {
            Matrix newMat = new Matrix(matrixA.Height, matrixA.Width);

            for (int i = 0; i < matrixA.Values.Length; i++)
                newMat[i] = matrixA[i] - scalar;

            return newMat;
        }

        /// <summary>
        /// Multiplication operation, multiplying matrices together
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        public static Matrix operator *(Matrix matrixA, Matrix matrixB)
        {
            Matrix newMat = new Matrix(0, 0);

            if (matrixMultScript != null && SystemInfo.deviceType == DeviceType.Desktop)
                newMat = multMatrixGPU(matrixA, matrixB);
            else if (matrixMultFloatScript != null && SystemInfo.deviceType == DeviceType.Handheld)
                newMat = multMatrixGPUFloat(matrixA, matrixB);
            else
            {
                //Check if matrixA Width is equal to matrixB Height
                if (matrixA.Width == matrixB.Height)
                {
                    newMat = new Matrix(matrixA.Height, matrixB.Width);

                    System.Threading.Tasks.Parallel.For(0, newMat.Values.Length, (index) =>
                    {
                        (int height, int width) = GetIndex(index, newMat);

                        newMat[index] = DotProductStatic(matrixA[height, true], matrixB[width, false]);

                    });

                }
                else
                    Debug.Log("Error, Dimensions don't match");
            }

            return newMat;
        }

        /// <summary>
        /// Multiplication operation with a factor
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="factor"></param>
        /// <returns></returns>
        public static Matrix operator *(Matrix matrixA, double scalar)
        {
            Matrix newMat = new Matrix(0, 0);

            newMat = new Matrix(matrixA.Height, matrixA.Width);

            for (int i = 0; i < matrixA.Values.Length; i++)
            {
                newMat[i] = matrixA[i] * scalar;
            }

            return newMat;
        }

        /// <summary>
        /// Division operation with a Scalar
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="factor"></param>
        /// <returns></returns>
        public static Matrix operator /(Matrix matrixA, double scalar)
        {
            Matrix newMat = new Matrix(0, 0);

            newMat = new Matrix(matrixA.Height, matrixA.Width);

            for (int i = 0; i < matrixA.Values.Length; i++)
            {
                newMat[i] = matrixA[i] / scalar;
            }

            return newMat;
        }

        /// <summary>
        /// Handles a Matrix Multiplication by handing it off to the GPU, this makes it crazy fast
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        public static Matrix multMatrixGPU(Matrix matrixA, Matrix matrixB)
        {
            Matrix newMat = new Matrix(0, 0);
            if (matrixA.Width == matrixB.Height)
            {
                newMat = new Matrix(matrixA.Height, matrixB.Width);

                ComputeShader computeShader = matrixMultScript;

                // Create compute buffers
                ComputeBuffer matrixAVals = new ComputeBuffer(matrixA.Length, sizeof(double));
                ComputeBuffer matrixBVals = new ComputeBuffer(matrixB.Length, sizeof(double));
                ComputeBuffer newMatrixVals = new ComputeBuffer(newMat.Length, sizeof(double));

                ComputeBuffer newMatrixDim = new ComputeBuffer(1, sizeof(uint) * 2);
                ComputeBuffer matrixADim = new ComputeBuffer(1, sizeof(uint) * 2);
                ComputeBuffer matrixBDim = new ComputeBuffer(1, sizeof(uint) * 2);


                matrixAVals.SetData(matrixA.Values);
                matrixBVals.SetData(matrixB.Values);

                matrixADim.SetData(new uint[] { (uint)matrixA.Width, (uint)matrixA.Height });
                matrixBDim.SetData(new uint[] { (uint)matrixB.Width, (uint)matrixB.Height });
                newMatrixDim.SetData(new uint[] { (uint)newMat.Width, (uint)newMat.Height });


                computeShader.SetBuffer(0, "matrixA", matrixAVals);
                computeShader.SetBuffer(0, "matrixB", matrixBVals);
                computeShader.SetBuffer(0, "newMatrix", newMatrixVals);

                computeShader.SetBuffer(0, "matrixDimsA", matrixADim);
                computeShader.SetBuffer(0, "matrixDimsB", matrixBDim);
                computeShader.SetBuffer(0, "newMatrixDims", newMatrixDim);

                //Calculate
                computeShader.Dispatch(0, newMat.Width, newMat.Height, 1);

                //Receaive Result
                newMatrixVals.GetData(newMat.Values);

                //Get rid of memory
                matrixAVals.Dispose();
                matrixBVals.Dispose();
                newMatrixVals.Dispose();

                matrixADim.Dispose();
                matrixBDim.Dispose();
                newMatrixDim.Dispose();
            }
            else
                Debug.Log("Error, Dimensions don't match");

            return newMat;
        }

        /// <summary>
        /// Handles a Matrix Multiplication by handing it off to the GPU, this makes it crazy fast
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        public static Matrix multMatrixGPUFloat(Matrix matrixA, Matrix matrixB)
        {
            Matrix newMat = new Matrix(0, 0);
            if (matrixA.Width == matrixB.Height)
            {
                newMat = new Matrix(matrixA.Height, matrixB.Width);

                ComputeShader computeShader = matrixMultScript;

                // Create compute buffers
                ComputeBuffer matrixAVals = new ComputeBuffer(matrixA.Length, sizeof(float));
                ComputeBuffer matrixBVals = new ComputeBuffer(matrixB.Length, sizeof(float));
                ComputeBuffer newMatrixVals = new ComputeBuffer(newMat.Length, sizeof(float));

                ComputeBuffer newMatrixDim = new ComputeBuffer(1, sizeof(uint) * 2);
                ComputeBuffer matrixADim = new ComputeBuffer(1, sizeof(uint) * 2);
                ComputeBuffer matrixBDim = new ComputeBuffer(1, sizeof(uint) * 2);

                matrixAVals.SetData(new MatrixFloat(matrixA).Values);
                matrixBVals.SetData(new MatrixFloat(matrixB).Values);

                matrixADim.SetData(new uint[] { (uint)matrixA.Width, (uint)matrixA.Height });
                matrixBDim.SetData(new uint[] { (uint)matrixB.Width, (uint)matrixB.Height });
                newMatrixDim.SetData(new uint[] { (uint)newMat.Width, (uint)newMat.Height });


                computeShader.SetBuffer(0, "matrixA", matrixAVals);
                computeShader.SetBuffer(0, "matrixB", matrixBVals);
                computeShader.SetBuffer(0, "newMatrix", newMatrixVals);

                computeShader.SetBuffer(0, "matrixDimsA", matrixADim);
                computeShader.SetBuffer(0, "matrixDimsB", matrixBDim);
                computeShader.SetBuffer(0, "newMatrixDims", newMatrixDim);

                //Calculate
                computeShader.Dispatch(0, newMat.Width, newMat.Height, 1);

                MatrixFloat floatMatrix = new MatrixFloat(newMat);

                //Receaive Result
                newMatrixVals.GetData(newMat.Values);

                newMat = new Matrix(floatMatrix);

                //Get rid of memory
                matrixAVals.Release();
                matrixBVals.Release();
                newMatrixVals.Release();

                matrixADim.Release();
                matrixBDim.Release();
                newMatrixDim.Release();
            }
            else
                Debug.Log("Error, Dimensions don't match");

            return newMat;
        }

        public static Matrix TransposeGPU(Matrix matrix)
        {
            Matrix transposedMatrix = new Matrix(matrix.Width, matrix.Height);

            //Dispatch to GPU
            ComputeShader computeShader = transposeScript;

            ComputeBuffer matrixVals = new ComputeBuffer(matrix.Length, sizeof(double));
            ComputeBuffer matrixDim = new ComputeBuffer(1, sizeof(uint) * 2);

            ComputeBuffer transposeVals = new ComputeBuffer(matrix.Length, sizeof(double));

            matrixVals.SetData(matrix.Values);
            matrixDim.SetData(new uint[] { (uint)matrix.Width, (uint)matrix.Height });

            computeShader.SetBuffer(0, "matrixVals", matrixVals);
            computeShader.SetBuffer(0, "matrixDim", matrixDim);
            computeShader.SetBuffer(0, "transposedMatrix", transposeVals);

            computeShader.Dispatch(0, matrix.Width, matrix.Height, 1);

            transposeVals.GetData(transposedMatrix.Values);

            matrixDim.Release();
            matrixVals.Release();
            transposeVals.Release();

            return transposedMatrix;
        }

        /// <summary>
        /// Handles Matrix Additions through the GPU
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        public static Matrix matrixAdditionGPU(Matrix matrixA, Matrix matrixB)
        {
            Matrix newMat = new Matrix(0, 0);

            if (SameDimension(matrixA, matrixB))
            {
                ComputeShader computeShader = matrixAdditionScript;
                newMat = new Matrix(matrixA.Height, matrixB.Width);

                /*
                int sizeOfDimensions = sizeof(int) * 2;
                int sizeOfDouble = sizeof(double);

                int sizeOfMatrixAValues = matrixA.Length * sizeOfDouble;
                int sizeOfMatrixBValues = matrixB.Length * sizeOfDouble;
                int sizeOfMatrixOutputValues = newMat.Length * sizeOfDouble;

                int totalSizeA = sizeOfDimensions + sizeOfMatrixAValues;
                int totalSizeB = sizeOfDimensions + sizeOfMatrixBValues;
                int totalSizeOutput = sizeOfDimensions + sizeOfMatrixOutputValues;

                GPUMatrix matrixAGPU = new GPUMatrix { Values = matrixA.Values, Height = matrixA.Height, Width = matrixA.Width};
                GPUMatrix matrixBGPU = new GPUMatrix { Values = matrixB.Values, Height = matrixB.Height, Width = matrixB.Width};
                GPUMatrix matrixOutputGPU = new GPUMatrix { Values = newMat.Values, Height = newMat.Height, Width = newMat.Width };
                */



                ComputeBuffer matrixAValues = new ComputeBuffer(matrixA.Values.Length, sizeof(double));
                matrixAValues.SetData(matrixA.Values);

                ComputeBuffer matrixBValues = new ComputeBuffer(matrixB.Values.Length, sizeof(double));
                matrixBValues.SetData(matrixB.Values);

                ComputeBuffer newMatrixValues = new ComputeBuffer(newMat.Values.Length, sizeof(double));

                computeShader.SetBuffer(0, "matrixAValues", matrixAValues);
                computeShader.SetBuffer(0, "matrixBValues", matrixBValues);
                computeShader.SetBuffer(0, "newMatrixValues", newMatrixValues);

                //Run Calculations
                computeShader.Dispatch(0, (newMat.Values.Length / 1024) + 1, 1, 1);

                //Receive Data
                newMatrixValues.GetData(newMat.Values);

                //Get rid of memory
                matrixAValues.Dispose();
                matrixBValues.Dispose();
                newMatrixValues.Dispose();

            }
            else
                Debug.Log("Error, Dimensions don't match");

            return newMat;
        }

        public static Matrix matrixSubstractionGPU(Matrix matrixA, Matrix matrixB)
        {
            Matrix newMat = new Matrix(0, 0);

            if (SameDimension(matrixA, matrixB))
            {
                ComputeShader computeShader = matrixSubstractionScript;
                newMat = new Matrix(matrixA.Height, matrixB.Width);

                ComputeBuffer matrixAValues = new ComputeBuffer(matrixA.Values.Length, sizeof(double));
                matrixAValues.SetData(matrixA.Values);

                ComputeBuffer matrixBValues = new ComputeBuffer(matrixB.Values.Length, sizeof(double));
                matrixBValues.SetData(matrixB.Values);

                ComputeBuffer newMatrixValues = new ComputeBuffer(newMat.Values.Length, sizeof(double));
                newMatrixValues.SetData(newMat.Values);

                computeShader.SetBuffer(0, "matrixAValues", matrixAValues);
                computeShader.SetBuffer(0, "matrixBValues", matrixBValues);
                computeShader.SetBuffer(0, "newMatrixValues", newMatrixValues);

                int groupCount = 0;
                if ((newMat.Values.Length / 1024) >= 1)
                    groupCount = (newMat.Values.Length / 1024);
                else
                    groupCount = 1;

                //Run Calculations
                computeShader.Dispatch(0, groupCount, 1, 1);

                //Receive Data
                newMatrixValues.GetData(newMat.Values);

                //Get rid of memory
                matrixAValues.Dispose();
                matrixBValues.Dispose();
                newMatrixValues.Dispose();

            }
            else
                Debug.Log("Error, Dimensions don't match");

            return newMat;
        }

        /// <summary>
        /// Checks if the Matrices are the same Dimension
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        private static bool SameDimension(Matrix matrixA, Matrix matrixB)
        {
            if (matrixA.Height == matrixB.Height && matrixA.Width == matrixB.Width)
                return true;
            else
                return false;
        }

        /// <summary>
        // /// Returns the Max Value in the Matrix
        /// </summary>
        /// <returns></returns>
        public double GetMaxValue()
        {
            double max = Values[0];

            foreach (double val in Values)
            {
                if (val >= max)
                    max = val;
            }

            return max;
        }

        /// <summary>
        /// Displays the Matrix in the Debug Log, for debugging purposes
        /// </summary>
        public void DisplayMat()
        {
            //Display the matrix
            string line = "\n";
            for (int height = 0; height < this.Height; height++)
            {
                for (int width = 0; width < this.Width; width++)
                    line += $"{this[height, width]}   ";

                line += "\n";
            }

            Debug.Log(line);
        }

        /// <summary>
        /// Returns the Matrix in a string format to display
        /// </summary>
        /// <returns></returns>
        public string GetDisplayMat()
        {
            //Display the matrix
            string line = "\n";

            for (int height = 0; height < this.Height; height++)
            {
                for (int width = 0; width < this.Width; width++)
                    line += $"{this[height, width]} ";

                line += "\n";
            }

            return line;
        }

        /// <summary>
        /// Returns the Output Matrix Dimension of a Matrix Multiplication
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        public static string GetMultOutputDimensions(Matrix matrixA, Matrix matrixB)
        {
            return $"({matrixA.Height} x {matrixB.Width})";
        }

        /// <summary>
        /// Returns the Dimensions of the Matrix in a string format 
        /// </summary>
        /// <returns></returns>
        public string GetDebugDimension()
        {
            return $"({Height} x {Width})";
        }

        /// <summary>
        /// Saves the Difference Matrix to the device for debugging purposes
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <param name="name"></param>
        public static void SaveDifference(Matrix matrixA, Matrix matrixB, string name)
        {
            var dir = "Assets/Resources/matrix" + "/" + $"{name}" + ".json";

            Matrix difference = matrixA - matrixB;

            string jsonData = JsonUtility.ToJson(difference, true);
            jsonData += JsonUtility.ToJson(matrixA, true);
            jsonData += JsonUtility.ToJson(matrixB, true);

            File.WriteAllText(dir, jsonData);
        }
    }
}

