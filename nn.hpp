#ifndef NN_HPP
#define NN_HPP

#include <eigen3/Eigen/Eigen>
#include <vector>
#include <iostream>
#include <cmath>

typedef Eigen::RowVectorXf RowVector;
typedef Eigen::MatrixXf Matrix;
typedef unsigned int uint;

class NeuralNetwork
{
public:
  NeuralNetwork(const std::vector<uint> &topology, float learningRate = 0.1);
  ~NeuralNetwork(); // Added destructor

  void forward(RowVector &input);
  void backward(RowVector &output);
  void calcErrors(RowVector &output);
  void updateWeights();
  void train(std::vector<RowVector *> input_data, std::vector<RowVector *> output_data);

private:
  std::vector<uint> topology;
  std::vector<RowVector *> neuronLayers;
  std::vector<RowVector *> cacheLayers;
  std::vector<RowVector *> deltas;
  std::vector<Matrix *> weights;
  float learningRate;
};

#endif