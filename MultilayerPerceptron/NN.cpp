#include "NN.h"

Neuron::Neuron()
{
	reset();
}
Neuron::~Neuron() {}

// called to reset all values
void Neuron::reset()
{
	layer = index = 0;
	localGradient = bias = biasGradient = z = a = 0.0;
}

// called when beginning training
void Neuron::initRandomize(int layer, int index, int totalWeights)
{
	reset();
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
	double divider = 100000000.0 / layer;

	for (int i = 0; i < totalWeights; ++i)
	{
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
	reset();
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
	a = logistics(z);
}

void Neuron::applyDerivativesLast(vector<Neuron>& prevLayer, double Yj)
{
	localGradient = 2.0 * (a - Yj) * logisticsDerivative(a);

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
	localGradient *= logisticsDerivative(a);

	// for each weight
	for (int i = 0; i < weights.size(); ++i)
	{
		weightsGradients[i] = localGradient * prevLayer[i].a;
	}

	// bias
	biasGradient = localGradient;
}

double Neuron::logistics(double x)
{
	// return sigmoldFunction(x);
	return ReLU_Function(x);
}

double Neuron::logisticsDerivative(double logisticVal)
{
	// return sigmoldDerivative(logisticVal);
	return ReLU_Derivative(logisticVal);
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
	myFile << bias << "~";
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

double ReLU_Function(double x)
{
	return max(x, 0.0);
}

double ReLU_Derivative(double reluVal)
{
	if (reluVal > 0.0)
	{
		return 1.0;
	}
	else
	{
		return 0.0;
	}
}