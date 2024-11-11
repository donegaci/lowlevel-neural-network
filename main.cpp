#include "nn.hpp"

#include <fstream>

typedef std::vector<RowVector *> data;

void ReadCSV(std::string filename, std::vector<RowVector *> &data)
{
  data.clear();
  std::ifstream file(filename);
  std::string line, word;
  // determine number of columns in file
  std::getline(file, line, '\n');
  std::stringstream ss(line);
  std::vector<float> parsed_vec;
  while (std::getline(ss, word, ','))
  {
    parsed_vec.push_back(float(std::stof(&word[0])));
  }
  uint cols = parsed_vec.size();
  data.push_back(new RowVector(cols));
  for (uint i = 0; i < cols; i++)
  {
    data.back()->coeffRef(1, i) = parsed_vec[i];
  }

  // read the file
  if (file.is_open())
  {
    while (std::getline(file, line, '\n'))
    {
      std::stringstream ss(line);
      data.push_back(new RowVector(1, cols));
      uint i = 0;
      while (std::getline(ss, word, ','))
      {
        data.back()->coeffRef(i) = float(std::stof(&word[0]));
        i++;
      }
    }
  }
}

void genData(std::string filename)
{
  std::ofstream file1(filename + "-in");
  std::ofstream file2(filename + "-out");
  for (uint r = 0; r < 1000; r++)
  {
    float x = rand() / float(RAND_MAX);
    float y = rand() / float(RAND_MAX);
    file1 << x << ", " << y << std::endl;
    file2 << 2 * x + 10 + y << std::endl;
  }
  file1.close();
  file2.close();
}

int main()
{

  // Create neural network with topology {2, 3, 1}
  std::vector<uint> topology = {2, 3, 1};
  NeuralNetwork n(topology);
  data in_dat, out_dat;
  genData("test");
  ReadCSV("test-in", in_dat);
  ReadCSV("test-out", out_dat);
  n.train(in_dat, out_dat);
  return 0;
}
