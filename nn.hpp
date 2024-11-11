#pragma once

#include <vector>
#include <iostream>
#include <eigen3/Eigen/Eigen>

typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

class NeuralNetwork
{
public:
  NeuralNetwork(std::vector<uint> topology, float learningRate = 0.005);
  void forward(RowVector &input);
  void backward(RowVector &output);
  void calcErrors(RowVector &output);
  void updateWeights();
  void train(std::vector<RowVector *> data);

  std::vector<RowVector *> neuronLayers; // stores the different layers of out network
  std::vector<RowVector *> cacheLayers;  // stores the unactivated (activation fn not yet applied) values of layers
  std::vector<RowVector *> deltas;       // stores the error contribution of each neurons
  std::vector<Matrix *> weights;         // the connection weights itself
  float learningRate;
};