using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using Unity.VisualScripting;
using UnityEditor.PackageManager;
using UnityEngine;
using UnityEngine.UIElements.Experimental;

namespace MachineLearningMath
{
    /// <summary>
    /// Custom Tensor Class developped for working on the GPU and with DNANeuralNetworks
    /// </summary>
    public class Tensor
    {
        // [depth , height, width]
        // 0--------> Width
        // |
        // |
        // |
        // Height

        /// <summary>
        /// Array Describing the Dimension of the Tensor
        /// </summary>
        private int[] _dimensions;

        /// <summary>
        /// Getter and Setter for the Dimensions of the Tensor
        /// </summary>
        public int[] Dimensions
        {
            get => _dimensions;
            private set => _dimensions = value;
        }

        /// <summary>
        /// Getter for the Matrix Properties of the Tensor
        /// </summary>
        public TensorMatrixProperties MatrixProperties { get; private set;}

        /// <summary>
        /// Returns the Tensors Dimension in String Format
        /// </summary>
        public string DisplayDimensions
        {
            get => GetDisplayIndex(Dimensions);
        }

        /// <summary>
        /// Getter for the Length of the Tensor
        /// </summary>
        public int Length { get => GetLength(); }

        /// <summary>
        /// Array of all the Values in the Tensor
        /// </summary>
        private double[] _values;

        /// <summary>
        /// Getter and Setter of the Values in the Tensor
        /// </summary>
        public double[] Values
        {
            get => _values;
            set => _values = value;
        }

        /// <summary>
        /// Indexer for Getting and Setting the Values of the Tensor at a certain index
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public double this[int index]
        {
            get => _values[index];
            set => _values[index] = value;
        }

        /// <summary>
        /// Indexer for Getting and Setting the Values of the Tensor at the provided Dimensions
        /// </summary>
        /// <param name="indexes"></param>
        /// <returns></returns>
        public double this[int[] indexes]
        {
            get => _values[GetFlatIndex(indexes)];
            set => _values[GetFlatIndex(indexes)] = value;
        }

        /// <summary>
        /// Initializes a new Tensor Based off the Suggested Dimensions
        /// </summary>
        /// <param name="dimensions"></param>
        public Tensor(int[] dimensions)
        {
            _dimensions = dimensions;
            _values = new double[GetLength()];
            MatrixProperties = new TensorMatrixProperties(this);
        }

        /// <summary>
        /// Initializes a new Tensor Based off a List of Matrices that Populates it
        /// </summary>
        /// <param name="matrices"></param>
        public Tensor(Matrix[] matrices)
        {
            _dimensions = new int[] { matrices.Length, matrices[0].Height, matrices[0].Width };
            _values = new double[GetLength()];
            MatrixProperties = new TensorMatrixProperties(this);

            int count = 0;
            foreach (Matrix matrix in matrices)
            {
                Array.Copy(matrix.Values, 0, _values, count, matrix.Length);
                count += matrix.Length;
            }
        }

        /// <summary>
        /// Returns an Tensor of the Specified Dimensions with Values that Increment
        /// </summary>
        /// <param name="dimensions"></param>
        /// <returns></returns>
        public static Tensor Increment(int[] dimensions)
        {
            Tensor tensor = new Tensor(dimensions);

            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = i;

            return tensor;
        }

        /// <summary>
        /// Gets the Indexes in each Dimensions given a Flat index
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public int[] GetIndex(int index)
        {
            if (index < Length)
            {
                int[] indexes = new int[_dimensions.Length];
                int leftover = index;

                for (int i = 0; i < _dimensions.Length; i++)
                {
                    int mult = 1;

                    for (int j = i + 1; j < _dimensions.Length; j++)
                        mult *= _dimensions[j];

                    indexes[i] = leftover / mult;
                    leftover = leftover % mult;
                }

                return indexes;
            }
            else
                return new int[_dimensions.Length];
        }

        /// <summary>
        /// Gets the Indexes in each Dimensions given a Flat index and a Tensor
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public static int[] GetIndex(int index, Tensor tensor)
        {
            int[] indexes = new int[tensor.Dimensions.Length];
            int leftover = index;

            for (int i = 0; i < tensor.Dimensions.Length; i++)
            {
                int mult = 1;

                for (int j = i + 1; j < tensor.Dimensions.Length; j++)
                    mult *= tensor.Dimensions[j];

                indexes[i] = leftover / mult;
                leftover = leftover % mult;
            }

            return indexes;
        }

        /// <summary>
        /// Checks to see if the Specified Indexes are within the Tensors Dimension
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public bool IsWithinTensor(int[] index)
        {
            if (index.Length == Dimensions.Length)
            {
                for (int i = 0; i < Dimensions.Length; i++)
                {
                    if (index[i] >= Dimensions[i])
                        return false;
                }
            }
            else
                throw new InvalidOperationException("Dimensions don't Match");

            return true;
        }

        /// <summary>
        /// Determines if the Tensors have the Same Dimension
        /// </summary>
        /// <param name="dim1"></param>
        /// <param name="dim2"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public static bool IsSameDimension(Tensor tensor1, Tensor tensor2)
        {
            int[] dim1 = tensor1.Dimensions;
            int[] dim2 = tensor2.Dimensions;

            if (dim1.Length == dim2.Length)
            {
                for (int i = 0; i < dim1.Length; i++)
                {
                    if (dim1[i] != dim2[i])
                        return false;
                }
            }
            else
                throw new InvalidOperationException($"Tensor dimensions do not match. ({dim1.Length}, {dim2.Length})");

            return true;
        }

        /// <summary>
        /// Determines if the Tensors have the Same Dimension
        /// </summary>
        /// <param name="dim1"></param>
        /// <param name="dim2"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public static bool IsSameDimension(Tensor tensor, Matrix matrix)
        {
            int[] dim1 = tensor.MatrixProperties.MatrixDimension;
            int[] dim2 = matrix.Dimensions;

            if (dim1.Length == dim2.Length)
            {
                for (int i = 0; i < dim1.Length; i++)
                {
                    if (dim1[i] != dim2[i])
                        return false;
                }
            }
            else
                throw new InvalidOperationException($"The Tensors dimensions do not match with the Matrix. ({dim1.Length}, {dim2.Length})");

            return true;
        }

        /// <summary>
        /// Returns the total Length of the Values in the Tensor
        /// </summary>
        /// <returns></returns>
        private int GetLength()
        {
            int length = 1;
            foreach (int dimension in _dimensions)
                length *= dimension;
            return length;
        }

        /// <summary>
        /// Returns the flat index of a value 
        /// </summary>
        /// <param name="heightIndex"></param>
        /// <param name="widthIndex"></param>
        /// <returns></returns>
        public int GetFlatIndex(int[] indexes)
        {
            if (IsWithinTensor(indexes))
            {
                int index = 0;

                for (int i = 0; i < _dimensions.Length; i++)
                {
                    int mult = 1;
                    for (int j = i + 1; j < _dimensions.Length; j++)
                        mult *= _dimensions[j];

                    index += indexes[i] * mult;
                }

                if (index < Length)
                    return index;
            }
            throw new IndexOutOfRangeException("The Indexes Specified are out of the Tensors Range.");
        }

        /// <summary>
        /// Pastes the Tensor Values and Pastes it in the Current Tensor
        /// </summary>
        /// <param name="tensor"></param>
        /// <exception cref="IndexOutOfRangeException"></exception>
        public void PasteTensor(Tensor tensor)
        {
            if (tensor.Length <= Length)
                Array.Copy(tensor.Values, 0, _values, 0, tensor.Length);
            else
                throw new IndexOutOfRangeException("The Copied Tensor is Longer than the one it is being pasted in");
        }

        /// <summary>
        /// Operator Handling the Addition between 2 Tensors
        /// </summary>
        /// <param name="tensorA"></param>
        /// <param name="tensorB"></param>
        /// <returns></returns>
        public static Tensor operator +(Tensor tensorA, Tensor tensorB)
        {
            if (Tensor.IsSameDimension(tensorA, tensorB))
            {
                Tensor outputTensor = new Tensor(tensorA.Dimensions);

                for (int i = 0; i < tensorA.Length; i++)
                    outputTensor[i] = tensorA[i] + tensorB[i];

                return outputTensor;
            }
            else
                throw new InvalidOperationException($"Tensor dimensions do not match. ({tensorA.DisplayDimensions}, {tensorB.DisplayDimensions})");
        }

        /// <summary>
        /// Operator Handling the Substraction between 2 Tensors
        /// </summary>
        /// <param name="tensorA"></param>
        /// <param name="tensorB"></param>
        /// <returns></returns>
        public static Tensor operator -(Tensor tensorA, Tensor tensorB)
        {
            if (Tensor.IsSameDimension(tensorA, tensorB))
            {
                Tensor outputTensor = new Tensor(tensorA.Dimensions);

                for (int i = 0; i < tensorA.Length; i++)
                    outputTensor[i] = tensorA[i] - tensorB[i];

                return outputTensor;
            }
            else
                throw new InvalidOperationException("Tensor dimensions do not match.");
        }

        /// <summary>
        /// Operator Handling the Scalar Addition of a Tensor
        /// </summary>
        /// <param name="tensorA"></param>
        /// <param name="tensorB"></param>
        /// <returns></returns>
        public static Tensor operator +(Tensor tensor, double additive)
        {
            Tensor outputTensor = new Tensor(tensor.Dimensions);

            for (int i = 0; i < tensor.Length; i++)
                outputTensor[i] = tensor[i] + additive;

            return outputTensor;
        }

        /// <summary>
        /// Operator Handling the Scalar Addition of a Tensor
        /// </summary>
        /// <param name="tensorA"></param>
        /// <param name="tensorB"></param>
        /// <returns></returns>
        public static Tensor operator -(Tensor tensor, double substrator)
        {
            Tensor outputTensor = new Tensor(tensor.Dimensions);

            for (int i = 0; i < tensor.Length; i++)
                outputTensor[i] = tensor[i] - substrator;

            return outputTensor;
        }

        /// <summary>
        /// Operator Handling the Scalar Multiplication of a Tensor
        /// </summary>
        /// <param name="tensorA"></param>
        /// <param name="tensorB"></param>
        /// <returns></returns>
        public static Tensor operator *(Tensor tensor, double multiplier)
        {
            Tensor outputTensor = new Tensor(tensor.Dimensions);

            for (int i = 0; i < tensor.Length; i++)
                outputTensor[i] = tensor[i] * multiplier;

            return outputTensor;
        }

        /// <summary>
        /// Operator Handling the Scalar Division of a Tensor
        /// </summary>
        /// <param name="tensorA"></param>
        /// <param name="tensorB"></param>
        /// <returns></returns>
        public static Tensor operator /(Tensor tensor, double diviser)
        {
            Tensor outputTensor = new Tensor(tensor.Dimensions);

            for (int i = 0; i < tensor.Length; i++)
                outputTensor[i] = tensor[i] / diviser;

            return outputTensor;
        }

        /// <summary>
        /// Adds a Matrix to the Stack of Tensors
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="matrix"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        public static Tensor operator ^(Tensor tensor, Matrix matrix)
        {
            bool SAME_DIMENSIONS = IsSameDimension(tensor, matrix);
            bool TENSOR3D = tensor.Dimensions.Length == 3;

            if (TENSOR3D)
            {
                if (SAME_DIMENSIONS)
                {
                    int[] dims = (int[])(tensor.Dimensions.Clone());
                    dims[0] += 1;

                    Tensor outputTensor = new Tensor(dims);

                    outputTensor.PasteTensor(tensor);
                    outputTensor.MatrixProperties[outputTensor.MatrixProperties.NumberOfMatrices - 1] = matrix;

                    return outputTensor;
                }
                else
                    throw new InvalidOperationException("Matrix Dimensions do not match with the Tensors.");
            }
            else
                throw new InvalidOperationException("The Tensor is not 3D");
        }

        /// <summary>
        /// Returns the Index in String Format
        /// </summary>
        /// <param name="indexes"></param>
        /// <returns></returns>
        public string GetDisplayIndex(int[] indexes)
        {
            string dimension = "[";

            foreach (int index in indexes)
                dimension += $" {index},";

            dimension.Remove(dimension.Length - 1);
            dimension += "]";

            return dimension;
        }

        /// <summary>
        /// Returns the Display Header
        /// </summary>
        /// <returns></returns>
        private string GetTensorHeading()
        {
            string header = "";

            header += $"Dimension: {DisplayDimensions}\n";
            header += $"Length: {Length}\n";
            header += $"Matrices: {MatrixProperties.NumberOfMatrices}\n";
            header += "\n";

            return header;
        }

        /// <summary>
        /// Returns the Formatted Tensor for Displaying in a String
        /// </summary>
        /// <returns></returns>
        public string GetDisplayTensor()
        {
            string tensor = GetTensorHeading();

            for (int i = 0; i < MatrixProperties.NumberOfMatrices; i++)
            {
                tensor += $"Matrix : {i}   Index:{GetDisplayIndex(MatrixProperties.GetIndexes(i))}";
                tensor += "\n";
                tensor += MatrixProperties[i].GetDisplayMat();
                tensor += "\n";
            }

            return tensor;
        }

        /// <summary>
        /// Displays the Tensor in the Debug Log
        /// </summary>
        public void DisplayTensor()
        {
            Debug.Log(GetDisplayTensor());
        }
    }
}