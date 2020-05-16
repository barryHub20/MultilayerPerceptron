#include "NN.h"

Neuron::Neuron()
{
	// indexing
	layer = index = 0;
	localGradient = bias = biasGradient = z = a = 0.0;
}
Neuron::~Neuron() {}

// called when beginning training
void Neuron::initRandomize(int layer, int index, int totalWeights)
{
	this->layer = layer;
	this->index = index;

	//bias
	bias = (double)(rand() % 100 + 0) / 100.0;

	// weights
	weights.clear();
	weightsGradients.clear();
	prevGradients.clear();
	weights.resize(totalWeights);
	weightsGradients.resize(totalWeights);
	prevGradients.resize(totalWeights);
	double divider = 0.0;
	for (int i = 0; i < totalWeights; ++i)
	{
		if (layer == 1)
		{
			divider = 10000000.0;
		}
		else if (layer == 2)
		{
			divider = 1000000.0;
		}
		else
		{
			divider = 1000.0;
		}

		weights[i] = (double)(rand() % 100 + 0) / divider;

		if (rand() % 2 == 0)
		{
			weights[i] = -weights[i];
		}

		prevGradients[i] = 0.0;
	}
}

void Neuron::initAsPixel(int layer, int index, double pixel)
{
	this->layer = layer;
	this->index = index;
	this->a = pixel;
}

void Neuron::initFromFile(int layer, int index, const vector<double>& _weights, double bias)
{
	this->layer = layer;
	this->index = index;

	weights.clear();
	weights.resize(_weights.size());
	for (int i = 0; i < _weights.size(); ++i)
	{
		weights[i] = _weights[i];
	}

	this->bias = bias;
}

void Neuron::calculateActivation(vector<Neuron>& prevLayer)
{
	// calculate Z
	z = 0.0;
	for (int i = 0; i < weights.size(); ++i)
	{
		z += prevLayer[i].a * weights[i];
	}
	z += bias;

	// calculate a
	a = sigmoldFunction(z);
}

void Neuron::applyDerivativesLast(vector<Neuron>& prevLayer, double Yj)
{
	localGradient = 2.0 * (a - Yj) * sigmoldDerivative(a);

	// for each weight
	for (int i = 0; i < weights.size(); ++i)
	{
		weightsGradients[i] = localGradient * prevLayer[i].a;
	}

	// bias
	biasGradient = localGradient;
}

void Neuron::applyDerivatives(vector<Neuron>& nextLayer, vector<Neuron>& prevLayer)
{
	// get local gradient
	localGradient = 0.0;
	for (int i = 0; i < nextLayer.size(); ++i)
	{
		localGradient += nextLayer[i].localGradient * nextLayer[i].weights[this->index];
	}
	localGradient *= sigmoldDerivative(a);

	// for each weight
	for (int i = 0; i < weights.size(); ++i)
	{
		weightsGradients[i] = localGradient * prevLayer[i].a;
	}

	// bias
	biasGradient = localGradient;
}

void Neuron::print()
{
	cout << layer << ' ' << index << " ======================================================" << endl;
	/*cout << "Weights: ";
	for (int i = 0; i < weights.size(); ++i)
	{
		cout << weights[i] << ", ";
	}*/
	cout << "weightsGradients: ";
	for (int i = 0; i < weightsGradients.size(); ++i)
	{
		cout << weightsGradients[i] << ", ";
	}
	cout << endl;
	cout << "Bias: " << bias << endl;
	cout << "Z: " << z << endl;
	cout << "A: " << a << endl;
	cout << endl;
}

void Neuron::apply(int iteration)
{
	for (int i = 0; i < weights.size(); ++i)
	{
		// normal
		weights[i] = weights[i] - weightsGradients[i] * learningRate;
		prevGradients[i] = weightsGradients[i] * learningRate;
	}

	//if (iteration == 0)
	//{
	//	for (int i = 0; i < weights.size(); ++i)
	//	{
	//		// normal
	//		weights[i] = weights[i] - weightsGradients[i] * learningRate;
	//		prevGradients[i] = weightsGradients[i] * learningRate;
	//	}
	//}
	//else 
	//{
	//	for (int i = 0; i < weights.size(); ++i)
	//	{
	//		// momentum
	//		double v = MOMENTUM * prevGradients[i] + weightsGradients[i] * learningRate;
	//		weights[i] = weights[i] - v;

	//		// Nesterov Accelerated Gradient
	//		/*double v = MOMENTUM * prevGradients[i] - weightsGradients[i] * learningRate;
	//		weights[i] = weights[i] - MOMENTUM * prevGradients[i] + (1 + MOMENTUM) * v;*/

	//		prevGradients[i] = v;
	//	}
	//}

	bias -= biasGradient * learningRate;
}

void Neuron::writeToFile(ofstream& myFile)
{
	// format: w1,........,wn,b
	for (int i = 0; i < weights.size(); ++i)
	{
		myFile << std::to_string(weights[i]) << ",";
	}
	myFile << bias << ",";
}

double sigmoldFunction(double x)
{
	// what is e: https://en.wikipedia.org/wiki/E_(mathematical_constant)
	return 1.0 / (1.0 + exp(-x));
}

double sigmoldDerivative(double sigmoldVal)
{
	return sigmoldVal * (1.0 - sigmoldVal);
}