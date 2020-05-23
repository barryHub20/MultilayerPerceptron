#pragma once
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
using namespace std;
#define TOTAL_EPOCH 50
#define TOTAL_ITERATIONS 60000
#define MOMENTUM 0.9
const double learningRate = 0.001;

/************************************************************************************************************
Helper functions
*************************************************************************************************************/
// Logistic functions
double sigmoldFunction(double x);
double sigmoldDerivative(double sigmoldVal);
double ReLU_Function(double x);
double ReLU_Derivative(double reluVal);

/************************************************************************************************************
Neuron, 0.0 >= value <= 1.0
*************************************************************************************************************/
class Neuron
{
public:

	// indexing
	int layer;
	int index;

	// prev. layer
	vector<double> weights;	// from prev. neuron
	vector<double> weightsGradients;
	vector<double> prevGradients;
	double bias; //	from previous weights
	double biasGradient;

	// value
	double z;
	double a;
	double localGradient;

	// functions
	Neuron();
	~Neuron();

	// init
	void initRandomize(int layer, int index, int totalWeights);
	void initAsPixel(int layer, int index, double pixel);
	void initFromFile(int layer, int index, const vector<double>& weights, double bias);
	void reset();

	// calculate activations
	void calculateActivation(vector<Neuron>& prevLayer);

	// gradient descent L
	void applyDerivativesLast(vector<Neuron>& prevLayer, double Yj);

	// gradient descent L - 1
	void applyDerivatives(vector<Neuron>& nextLayer, vector<Neuron>& prevLayer);

	void apply(int iteration);

	// helper functions
	double logistics(double x);
	double logisticsDerivative(double logisticVal);

	// misc funtions
	void print();
	void writeToFile(ofstream& myFile);
};