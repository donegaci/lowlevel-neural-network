// nn.cpp
#include "nn.hpp"
#include <functional>

float activationFunction(float x)
{
  return tanhf(x);
}

float activationFunctionDerivative(float x)
{
  return 1 - (tanhf(x) * tanhf(x));
}

NeuralNetwork::NeuralNetwork(const std::vector<uint> &topology, float learningRate)
{
  this->topology = topology;
  this->learningRate = learningRate;

  for (uint i = 0; i < topology.size() - 1; i++)
  {
    // initialize neuron layers
    if (i == topology.size() - 1)
      neuronLayers.push_back(new RowVector(topology[i]));
    else
      neuronLayers.push_back(new RowVector(topology[i] + 1));

    // initialize cache and delta vectors
    cacheLayers.push_back(new RowVector(neuronLayers.size()));
    deltas.push_back(new RowVector(neuronLayers.size()));

    if (i != topology.size() - 1)
    {
      neuronLayers.back()->coeffRef(topology[i]) = 1.0;
      cacheLayers.back()->coeffRef(topology[i]) = 1.0;
    }

    // initialize weights matrix
    if (i > 0)
    {
      if (i != topology.size() - 1)
      {
        weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
        weights.back()->setRandom();
        weights.back()->col(topology[i]).setZero();
        weights.back()->coeffRef(topology[i - 1], topology[i]) = 1.0;
      }
      else
      {
        weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
        weights.back()->setRandom();
      }
    }
  }
}

void NeuralNetwork::forward(RowVector &input)
{
  // assign the input to first layer
  neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;

  // propagate the data forward and apply activation function
  for (uint i = 1; i < topology.size(); i++)
  {
    (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
    // Using a lambda instead of ptr_fun
    neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr([](float x)
                                                           { return activationFunction(x); });
  }
}

void NeuralNetwork::calcErrors(RowVector &output)
{
  // calculate errors for output layer
  *deltas.back() = output - *neuronLayers.back();

  // calculate errors for hidden layers
  for (uint i = topology.size() - 2; i > 0; i--)
  {
    *deltas[i] = *deltas[i + 1] * weights[i]->transpose();
  }
}

void NeuralNetwork::updateWeights()
{
  for (uint i = 0; i < topology.size() - 1; i++)
  {
    uint colLimit = (i != topology.size() - 2) ? weights[i]->cols() - 1 : weights[i]->cols();

    for (uint c = 0; c < colLimit; c++)
    {
      for (uint r = 0; r < weights[i]->rows(); r++)
      {
        float delta = deltas[i + 1]->coeffRef(c);
        float activationDeriv = activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c));
        float neuronOutput = neuronLayers[i]->coeffRef(r);

        weights[i]->coeffRef(r, c) += learningRate * delta * activationDeriv * neuronOutput;
      }
    }
  }
}

void NeuralNetwork::backward(RowVector &output)
{
  calcErrors(output);
  updateWeights();
}

void NeuralNetwork::train(std::vector<RowVector *> input_data, std::vector<RowVector *> output_data)
{
  for (uint i = 0; i < input_data.size(); i++)
  {
    std::cout << "Input to neural network is : " << *input_data[i] << std::endl;
    forward(*input_data[i]);
    std::cout << "Expected output is : " << *output_data[i] << std::endl;
    std::cout << "Output produced is : " << *neuronLayers.back() << std::endl;
    backward(*output_data[i]);
    std::cout << "MSE : " << std::sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back()->size()) << std::endl;
  }
}

// Add destructor to clean up memory
NeuralNetwork::~NeuralNetwork()
{
  for (auto *layer : neuronLayers)
    delete layer;
  for (auto *layer : cacheLayers)
    delete layer;
  for (auto *delta : deltas)
    delete delta;
  for (auto *weight : weights)
    delete weight;
}