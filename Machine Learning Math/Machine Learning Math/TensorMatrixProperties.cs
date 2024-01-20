using System;

namespace MachineLearningMath
{
    public class TensorMatrixProperties
    {
        /// <summary>
        /// Stored Reference to the Tensor
        /// </summary>
        private Tensor Tensor { get; set; }

        /// <summary>
        /// Returns the Tensors Stored Matrix Width
        /// </summary>
        public int Width
        {
            get => Tensor.Dimensions[Tensor.Dimensions.Length - 1];
        }

        /// <summary>
        /// Returns the Tensors Stored Matrix Height
        /// </summary>
        public int Height
        {
            get => Tensor.Dimensions[Tensor.Dimensions.Length - 2];
        }

        /// <summary>
        /// Returns the Tensors Stored Matrix Dimension
        /// </summary>
        public int[] MatrixDimension
        {
            get => new int[] { Height, Width };
        }

        /// <summary>
        /// Getter for the Debug Dimension
        /// </summary>
        public string DebugDimension
        {
            get => GetDebugDimension();
        }

        /// <summary>
        /// Returns the Number of Matrices in the Tensor
        /// </summary>
        public int NumberOfMatrices
        {
            get => Tensor.Length / MatrixLength;
        }

        /// <summary>
        /// Returns the Length of the Tensors Matrix
        /// </summary>
        public int MatrixLength
        {
            get => Width * Height;
        }

        /// <summary>
        /// Indexer for Getting and Setting the Matrix at the Specified Matrix Index
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public Matrix this[int index]
        {
            get => GetMatrix(index);
            set => SetMatrix(index, value);
        }

        /// <summary>
        /// Indexer for Getting and Setting the Matrix at the Specified Indexes
        /// </summary>
        /// <param name="indexes"></param>
        /// <returns></returns>
        public Matrix this[int[] indexes]
        {
            get => GetMatrix(GetMatrixIndex(indexes));
            set => SetMatrix(indexes, value);
        }

        /// <summary>
        /// Initializes the Tensors Matrix Properties
        /// </summary>
        /// <param name="tensor"></param>
        public TensorMatrixProperties(Tensor tensor)
        {
            Tensor = tensor;
        }

        /// <summary>
        /// Getter for a List of all Matrices in the Tensor
        /// </summary>
        public Matrix[] Matrices
        {
            get
            {
                Matrix[] matrices = new Matrix[NumberOfMatrices];

                for (int i = 0; i < NumberOfMatrices; i++)
                    matrices[i] = GetMatrix(i);

                return matrices;
            }
        }

        /// <summary>
        /// Returns the Indexes in each Dimension given the Matrix Index
        /// </summary>
        /// <param name="matrixIndex"></param>
        /// <returns></returns>
        /// <exception cref="IndexOutOfRangeException"></exception>
        public int[] GetIndexes(int matrixIndex)
        {
            if (matrixIndex < NumberOfMatrices)
                return Tensor.GetIndex(matrixIndex * MatrixLength);
            else
                throw new IndexOutOfRangeException("The Index Specified is out of the Tensors Range");
        }

        /// <summary>
        /// Returns the Matrix Index that the Specified Indexes is within
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public int GetMatrixIndex(int[] indexes)
        {
            return GetMatrixIndex(Tensor.GetFlatIndex(indexes));
        }

        /// <summary>
        /// Returns the Matrix Index that the Specified Index is within
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public int GetMatrixIndex(int index)
        {
            if (index < Tensor.Length)
                return index / MatrixLength;
            else
                throw new IndexOutOfRangeException("The Index Specified is out of the Tensors Range.");
        }

        /// <summary>
        /// Returns the Matrix Stored in the Tensor at the Specified Matrix Index
        /// </summary>
        /// <param name="matrixIndex"></param>
        /// <returns></returns>
        /// <exception cref="IndexOutOfRangeException"></exception>
        public Matrix GetMatrix(int matrixIndex)
        {
            if (matrixIndex < NumberOfMatrices)
                return CopyMatrix(matrixIndex * MatrixLength);
            else
                throw new IndexOutOfRangeException("The Index Specified is out of the Tensors Range.");
        }

        /// <summary>
        /// Copies the Values from the Tensor of the Length of a Matrix Starting at the Specified Indexes
        /// </summary>
        /// <param name="indexes"></param>
        /// <returns></returns>
        /// <exception cref="IndexOutOfRangeException"></exception>
        private Matrix CopyMatrix(int[] indexes)
        {
            if (Tensor.IsWithinTensor(indexes))
                return CopyMatrix(Tensor.GetFlatIndex(indexes));
            else
                throw new IndexOutOfRangeException("The Index Specified is out of the Tensors Range.");
        }

        /// <summary>
        /// Copies the Values from the Tensor of the Length of a Matrix Starting at the Specified Index
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        /// <exception cref="IndexOutOfRangeException"></exception>
        private Matrix CopyMatrix(int index)
        {
            bool SAFE_LENGTH = index < Tensor.Length && (index + MatrixLength) <= Tensor.Length;

            if (SAFE_LENGTH)
            {
                Matrix matrix = new Matrix(Height, Width);

                Array.Copy(Tensor.Values, index, matrix.Values, 0, matrix.Values.Length);

                return matrix;
            }
            else
                throw new IndexOutOfRangeException($"The Index Specified is out of the Tensors Range. ({index}, {Tensor.Length})");
        }

        /// <summary>
        /// Sets the Matrixs Values in the Tensor at the Specified Indexes
        /// </summary>
        /// <param name="indexes"></param>
        /// <param name="matrix"></param>
        /// <exception cref="IndexOutOfRangeException"></exception>
        private void SetMatrix(int[] indexes, Matrix matrix)
        {
            int[] fullIndex = GetFullIndex(indexes);

            if (Tensor.IsWithinTensor(fullIndex))
                SetMatrix(GetMatrixIndex(fullIndex), matrix);
            else
                throw new IndexOutOfRangeException("The Indexes Specified are out of the Tensors Range.");
        }

        /// <summary>
        /// Sets the Matrixs Values in the Tensor at the Specified Matrix Index
        /// </summary>
        /// <param name="index"></param>
        /// <param name="matrix"></param>
        /// <exception cref="IndexOutOfRangeException"></exception>
        /// <exception cref="InvalidOperationException"></exception>
        private void SetMatrix(int matrixIndex, Matrix matrix)
        {
            bool MATCHING_MATRIX_DIMENSIONS = Tensor.IsSameDimension(Tensor, matrix);

            if (MATCHING_MATRIX_DIMENSIONS)
                PasteMatrix(matrixIndex * MatrixLength, matrix);
            else
                throw new InvalidOperationException($"The Tensors and the Matrix Dimensions don't Match. ({DebugDimension}, {matrix.DebugDimension})");
        }

        /// <summary>
        /// Returns the Full index of the Tensor
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        private int[] GetFullIndex(int[] index)
        {
            int[] fullIndex = new int[Tensor.Dimensions.Length];

            for (int i = 0; i < index.Length; i++)
                fullIndex[i] = index[i];

            return fullIndex;
        }

        /// <summary>
        /// Pastes the Values of the Matrix into the Tensor starting at the Specified Indexes
        /// </summary>
        /// <param name="index"></param>
        /// <param name="matrix"></param>
        private void PasteMatrix(int[] indexes, Matrix matrix)
        {
            PasteMatrix(Tensor.GetFlatIndex(indexes), matrix);
        }

        /// <summary>
        /// Pastes the Values of the Matrix into the Tensor starting at the Specified Index
        /// </summary>
        /// <param name="index"></param>
        /// <param name="matrix"></param>
        private void PasteMatrix(int index, Matrix matrix)
        {
            bool SAFE_LENGTH = index < Tensor.Length && (index + matrix.Length) <= Tensor.Length;

            if (SAFE_LENGTH)
                Array.Copy(matrix.Values, 0, Tensor.Values, index, matrix.Length);
            else
                throw new IndexOutOfRangeException("The Indexes Specified are out of the Tensors Range.");
        }

        /// <summary>
        /// Returns the Dimensions of the Matrix in a string format 
        /// </summary>
        /// <returns></returns>
        public string GetDebugDimension()
        {
            return $"({Height} x {Width})";
        }
    }
}