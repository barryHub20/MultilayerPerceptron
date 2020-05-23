#include <vector>
#include <iterator>
#include <bitset>
#include "NN.h"
using namespace std;
#define TOTAL_TEST_COUNT 1066
#define PIXELS_COUNT 784

void readMnistFile(string fileName, vector<char>& contents)
{
	ifstream in;
	in.open(fileName, ios::in | ios::binary);

	if (in.is_open())
	{
		// get the starting position
		streampos start = in.tellg();

		// go to the end
		in.seekg(0, ios::end);

		// get the ending position
		streampos end = in.tellg();

		// go back to the start
		in.seekg(0, ios::beg);

		// create a vector to hold the data that
		// is resized to the total size of the file
		contents.clear();
		contents.resize(static_cast<size_t>(end - start));

		// read it in
		in.read(&contents[0], contents.size());
		in.close();
	}
}

void printNumber(int imageIdx, vector<char>& contents, vector<char>& labels)
{
	// print out handwritten digit
	int offset = 16;
	for (int z = imageIdx; z < imageIdx + 1; ++z) {
		for (int i = 0; i < 28; ++i) {
			for (int x = 0; x < 28; ++x) {
				int idx = (z * 28 * 28) + (i * 28) + x + offset;
				// convert to signed int
				std::bitset< 8 > value = (std::bitset< 8 >)contents[idx];
				int value2 = static_cast<int>(value.to_ulong());	// the unsigned base 10 value (255 = max, 0 = min)

				if (value2 == 0) {
					cout << '*' << ' ';
				}
				else {
					cout << value2 << ' ';
				}
			}
			cout << endl;
		}
	}

	// print out corr. label
	int labelOffset = 8;

	// convert to signed int
	std::bitset< 8 > value = (std::bitset< 8 >)labels[labelOffset + imageIdx];
	int labelVal = static_cast<int>(value.to_ulong());	// the unsigned base 10 value (255 = max, 0 = min)
	cout << "Label: " << labelVal << endl;
}

void Testing()
{
	// Observations: cost will go lower with every iteration

	// test labels
	/*double pixels[TOTAL_TEST_COUNT] = { 0.91, 0.53, 0.58, 0.26, 0.12, 0.67, 0.4, 0.56 };
	double labels[TOTAL_TEST_COUNT] = { 0, 1, 1, 0, 0, 0, 1, 1 };*/
	double pixels[TOTAL_TEST_COUNT];
	double labels[TOTAL_TEST_COUNT];
	for (int i = 0; i < TOTAL_TEST_COUNT; ++i) {
		pixels[i] = (double)(rand() % 100 + 0) / 100.0;
		labels[i] = (double)(rand() % 2);
		// cout << pixels[i] << " " << labels[i] << endl;
	}

	// test network
	double neuron_0_1 = pixels[0];
	double activation_0_1 = 0.0;

	double weight_0_2 = 0.5;	// random value
	double bias_0_2 = 2.0;	// random value
	double neuron_0_2 = 0.0;

	double weight_0_3 = 0.5;	// random value
	double bias_0_3 = 2.0;	// random value
	double neuron_0_3 = 0.0;

	double y = labels[0];

	for (int i = 0; i < TOTAL_TEST_COUNT; ++i)
	{
		// new training data
		neuron_0_1 = pixels[i];
		y = labels[i];

		cout << "Training Data " << i << " ===================================================/" << endl << endl;

		// layer 2
		double Z_0_2 = (neuron_0_1 * weight_0_2) + bias_0_2;	// multiple Neuron and weights
		double a_0_2 = sigmoldFunction(Z_0_2);

		/*cout << "Layer 1 -> 2" << endl;
		cout << "Weight: " << weight_0_2 << endl;
		cout << "Bias: " << bias_0_2 << endl;
		cout << "Z: " << Z_0_2 << endl;
		cout << "A: " << a_0_2 << endl << endl;*/

		// layer 3
		double Z_0_3 = (neuron_0_2 * weight_0_3) + bias_0_3;	// multiple Neuron and weights
		double a_0_3 = sigmoldFunction(Z_0_3);

		/*cout << "Layer 2 -> 3" << endl;
		cout << "Weight: " << weight_0_3 << endl;
		cout << "Bias: " << bias_0_3 << endl;
		cout << "Z: " << Z_0_3 << endl;
		cout << "A: " << a_0_3 << endl << endl;*/

		// cost
		double cost = (a_0_3 - y) * (a_0_3 - y);	// adds up over all layer L Neuron

		cout << "a_0_3: " << a_0_3 << " Y: " << y << " Cost: " << cost << endl << endl;

		// derivatives layer 3
		double wd_0_3 = a_0_2 * sigmoldDerivative(a_0_3) * (2 * (a_0_3 - y));
		double bd_0_3 = sigmoldDerivative(a_0_3) * (2 * (a_0_3 - y));
		double ad_0_3 = weight_0_3 * sigmoldDerivative(a_0_3) * (2 * (a_0_3 - y));	// sum over layer L
		//double wd_0_3 = cost / weight_0_3;
		//double bd_0_3 = cost / bias_0_3;
		//double ad_0_3 = cost / a_0_2;	// sum over layer L

		/*cout << "wd_0_3: " << wd_0_3 << endl;
		cout << "bd_0_3: " << bd_0_3 << endl;
		cout << "ad_0_3: " << ad_0_3 << endl << endl;*/

		// backpropagation towards layer 2
		weight_0_3 += -wd_0_3;	// average of sum of all gradient(Wk0)
		bias_0_3 += -bd_0_3;	// no average
		a_0_2 += -ad_0_3;	// used to calculate derivatives of next layer

		cout << "Layer 2 <- 3" << endl;
		cout << "Weight: " << weight_0_3 << endl;
		cout << "Bias: " << bias_0_3 << endl;
		cout << "A: " << a_0_3 << endl << endl;

		// derivatives layer 2
		double wd_0_2 = neuron_0_1 * sigmoldDerivative(a_0_2) * (2 * (a_0_2 - y));
		double bd_0_2 = sigmoldDerivative(a_0_2) * (2 * (a_0_2 - y));
		/*double wd_0_2 = cost / weight_0_2;
		double bd_0_2 = cost / bias_0_2;*/

		// backpropagation towards layer 1
		weight_0_2 += -wd_0_2;	// average of sum of all gradient(Wk0)
		bias_0_2 += -bd_0_2;	// no average

		cout << "Layer 1 <- 2" << endl;
		cout << "Weight: " << weight_0_2 << endl;
		cout << "Bias: " << bias_0_2 << endl << endl;
	}
}

/************************************************************************************************************
MLP: helper functions
*************************************************************************************************************/
// tested
void getImage1D(const vector<char>& contents, int imageIdx, vector<Neuron>& layer_0)
{
	int start = 16 + imageIdx * 784;
	int counter = 0;
	for (int i = start; i < start + 784; ++i)
	{
		// convert to signed int
		std::bitset< 8 > value = (std::bitset< 8 >)contents[i];
		layer_0[counter].initAsPixel(0, counter, (double)(static_cast<int>(value.to_ulong())) / 255.0);	// the unsigned base 10 value (255 = max, 0 = min)
		counter++;
	}
}

// tested
void printImage(vector<Neuron>& layer0)
{
	// testing
	for (int i = 0; i < 28; ++i)
	{
		for (int j = 0; j < 28; ++j)
		{
			if (layer0[i * 28 + j].a == 0.0)
				cout << '*';
			else
				cout << '0';
		}
		cout << endl;
	}
}

// tested
int getImageLabel(const vector<char>& labels, int imageIdx)
{
	std::bitset< 8 > value = (std::bitset< 8 >)labels[8 + imageIdx];
	// cout << "Label: " << static_cast<int>(value.to_ulong()) << endl;
	return static_cast<int>(value.to_ulong());	// the unsigned base 10 value (255 = max, 0 = min)
}

// tested
void initNeuron(vector< vector<Neuron> >& layerList)
{
	// each neuron of layer N will have a sum of weights equal to the sum of neurons in Layer N - 1
	for (int i = 1; i < layerList.size(); ++i)	// start from layer 1
	{
		for (int j = 0; j < layerList[i].size(); ++j)
		{
			layerList[i][j].initRandomize(i, j, layerList[i - 1].size());
		}
	}
}

// tested
// read function MUST follow the same format as used here
void saveToTextFile(vector< vector<Neuron> >& layerList)
{
	ofstream myfile("data.txt");
	if (myfile.is_open())
	{
		// layer by index ascending order
		for (int i = 1; i < layerList.size(); ++i)
		{
			for (int j = 0; j < layerList[i].size(); ++j)
			{
				layerList[i][j].writeToFile(myfile);
			}
		}

		myfile.close();
		cout << "Data saved to text file" << endl;
	}
	else
	{
		cout << "Unable to open file" << endl;
	}
}

// tested
void readFromTextFile(vector< vector<Neuron> >& layerList)
{
	string line;
	string line2;
	ifstream myfile("data.txt");

	if (myfile.is_open())
	{
		for (int i = 1; i < layerList.size(); ++i)
		{
			int totalWeights = layerList[i - 1].size();

			// init storage data
			vector<double> weights;
			weights.resize(totalWeights, 0.0);
			double bias = 0.0;

			for (int j = 0; j < layerList[i].size(); ++j)
			{
				getline(myfile, line, '~');	// line is all the data for this neuron

				stringstream neuronStream;	// parse data into stream object
				neuronStream << line;

				// each individual weight/bias
				for (int k = 0; k < totalWeights; ++k)
				{
					getline(neuronStream, line2, ',');
					weights[k] = stod(line2);
				}
				getline(neuronStream, line2, ',');
				bias = stod(line2);

				// init neuron
				layerList[i][j].initFromFile(i, j, weights, bias);
			}
		}
		myfile.close();
	}
	else
	{
		cout << "Unable to open file" << endl;
	}
}

/************************************************************************************************************
MLP: derivative functions
*************************************************************************************************************/
// tested
double getTotalCost(vector<Neuron>& lastLayer, vector<double>& yRow)
{
	double cost = 0.0;
	for (int i = 0; i < lastLayer.size(); ++i)
	{
		cost += (lastLayer[i].a - yRow[i]) * (lastLayer[i].a - yRow[i]);
	}
	return cost;
}

// tested
void getImageAndDetails(const vector<char>& contents, vector<double>& yRow, const vector<char>& labels, vector<Neuron>& layer_0, int index)
{
	// init
	getImage1D(contents, index, layer_0);
	yRow.clear();
	yRow.resize(10, 0.0);	// all to 0
	yRow[getImageLabel(labels, index)] = 1.0;
}

// tested
void applyDerivativesLast(vector<Neuron>& layer_J, vector<Neuron>& layer_K, const vector<double>& Yj)
{
	for (int i = 0; i < layer_J.size(); ++i)
	{
		layer_J[i].applyDerivativesLast(layer_K, Yj[i]);
	}
}

// tested
void applyDerivatives(vector<Neuron>& layer_I, vector<Neuron>& layer_J, vector<Neuron>& layer_K)
{
	for (int i = 0; i < layer_J.size(); ++i)
	{
		layer_J[i].applyDerivatives(layer_I, layer_K);
	}
}

// tested
void backPropagation(vector< vector<Neuron> >& layerList, const vector<double>& final_Yj)
{
	// apply derivatives for last layer
	applyDerivativesLast(layerList.end()[-1], layerList.end()[-2], final_Yj);

	//derivatives for precursor layers
	for (int i = layerList.size() - 2; i > 0; --i)	// from 2nd last layer to 2nd layer
	{
		applyDerivatives(layerList[i + 1], layerList[i], layerList[i - 1]);
	}
}

// tested
void printInfo(vector< vector<Neuron> >& layerList)
{
	for (int i = 1; i < layerList.size(); ++i)
	{
		cout << "Layer " << i << " ========================================================================================================================================" << endl;
		for (int k = 0; k < layerList[i].size(); ++k)
		{
			layerList[i][k].print();
		}
	}
}

/************************************************************************************************************
MLP: main function
*************************************************************************************************************/
void MLP_train(vector< vector<Neuron> >& layerList)
{
	// read pixels and labels
	vector<char> contents;
	vector<double> yRow;
	vector<char> labels;
	readMnistFile("train-images.idx3-ubyte", contents);
	readMnistFile("train-labels.idx1-ubyte", labels);

	// init
	initNeuron(layerList);

	// train for 60k times
	for (int x = 0; x < TOTAL_EPOCH; ++x)
	{
		for (int z = 0; z < TOTAL_ITERATIONS; ++z)
		{
			// init
			getImageAndDetails(contents, yRow, labels, layerList[0], z);

			// calculate activations
			for (int i = 1; i < layerList.size(); ++i)
			{
				for (int j = 0; j < layerList[i].size(); ++j)
				{
					layerList[i][j].calculateActivation(layerList[i - 1]);
				}
			}

			//get total cost
			double cost = getTotalCost(layerList.back(), yRow);
			if (z % 500 == 0)
			{
				bool belowThreshold = cost < 0.01;
				cout << "Epoch: " << x << " Below threshold: " << belowThreshold << "  Training image: " << z << "  Total cost: " << cost << endl;
			}

			// do backpropagation
			backPropagation(layerList, yRow);

			// apply gradient
			int iteration = z + (x * TOTAL_ITERATIONS);
			for (int i = 1; i < layerList.size(); ++i)
			{
				for (int j = 0; j < layerList[i].size(); ++j)
				{
					layerList[i][j].apply(iteration);
				}
			}

			// print
			// printInfo(layerList);
		}
	}

	// save weights and biases to txt
	saveToTextFile(layerList);
	cout << "Final weights and biases saved to text file!" << endl;
}

/************************************************************************************************************
MLP: test function
*************************************************************************************************************/
void MLP_test(vector< vector<Neuron> >& layerList)
{
	// read pixels and labels
	vector<char> contents;
	vector<char> labels;
	readMnistFile("t10k-images.idx3-ubyte", contents);
	readMnistFile("t10k-labels.idx1-ubyte", labels);

	// read off text file
	readFromTextFile(layerList);

	// test
	// printInfo(layerList);

	// init
	int correctCount = 0;
	int testTotal = 10000;
	int startIndex = 0;
	for (int i = startIndex; i < testTotal + startIndex; ++i)
	{
		getImage1D(contents, i, layerList[0]);
		int label = getImageLabel(labels, i);

		// calculate activations
		for (int i = 1; i < layerList.size(); ++i)
		{
			for (int j = 0; j < layerList[i].size(); ++j)
			{
				layerList[i][j].calculateActivation(layerList[i - 1]);
			}
		}

		// the predicted digit corr. to the 'brightest' neuron of last layer
		int brightestNeuron = 0;
		double brightestNeuronVal = 0.0;
		for (int i = 0; i < layerList.back().size(); ++i)
		{
			if (layerList.back()[i].a > brightestNeuronVal)
			{
				brightestNeuron = i;
				brightestNeuronVal = layerList.back()[i].a;
			}
		}

		// results!
		if (brightestNeuron == label)
		{
			// cout << "Correct guess at " << i << ", " << label << " and " << brightestNeuron << endl;
			// printImage(layerList[0]);
			correctCount++;
		}
		else {
			// cout << "Wrong guess at " << i << ", correct label is " << label << ", not " << brightestNeuron << endl;
			// printImage(layerList[0]);
		}

		if (i % 500 == 0)
		{
			cout << "Tested " << i << " of " << testTotal << " images" << endl;
		}
	}
	cout << "Total test images: " << testTotal << "  Correct Predictions: " << correctCount << endl;
}

/************************************************************************************************************
Main
*************************************************************************************************************/
int main()
{
	srand(time(NULL));
	/*
	vector<char> contents;
	vector<char> labels;
	readMnistFile("train-images.idx3-ubyte", contents);
	readMnistFile("train-labels.idx1-ubyte", labels);
	printNumber(156, contents, labels);*/

	//Testing();

	// dynamic
	vector< vector<Neuron> > layerList;

	// how many layers
	layerList.resize(4);

	// each layer
	layerList[0].resize(784);	// image (16 x 16)
	layerList[1].resize(24);
	layerList[2].resize(24);
	layerList[3].resize(10);

	MLP_train(layerList);
	cin.get();
	MLP_test(layerList);
}