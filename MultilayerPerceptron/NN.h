#pragma once
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
using namespace std;
#define TOTAL_SERIES 30
#define TOTAL_ITERATIONS 60000
#define MOMENTUM 0.9
const double learningRate = 0.03;

/************************************************************************************************************
Helper functions
*************************************************************************************************************/
double sigmoldFunction(double x);
double sigmoldDerivative(double rawValue);

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

	// calculate activations
	void calculateActivation(Neuron* prevLayer);

	// gradient descent L
	void applyDerivativesLast(Neuron* prevLayer, double Yj);

	// gradient descent L - 1
	void applyDerivatives(Neuron* nextLayer, Neuron* prevLayer, int nextLayerSize);

	void apply(int iteration);

	// misc funtions
	void print();
	void writeToFile(ofstream& myFile);
};