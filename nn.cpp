#include "nn.hpp"

// neural network class constructor
NeuralNetwork::NeuralNetwork(std::vector<uint> topology, float learningRate = 0.005)
{
  this->topology = topology;
};