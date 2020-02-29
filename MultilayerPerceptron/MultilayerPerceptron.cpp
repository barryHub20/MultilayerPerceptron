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

		cout << "a_0_3: " << a_0_3 <<  " Y: " << y << " Cost: " << cost << endl << endl;

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
void getImage1D(const vector<char>& contents, int imageIdx, Neuron layer_0[784])
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
void printImage(Neuron layer_0[784])
{
	// testing
	for (int i = 0; i < 28; ++i)
	{
		for (int j = 0; j < 28; ++j)
		{
			if (layer_0[i * 28 + j].a == 0.0)
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
void initNeuron(Neuron layer_1[16], Neuron layer_2[16], Neuron layer_3[10])
{
	// layer 0 is the pixels
	// layer 1 (prev. 784 Neuron) and layer 2 (prev. 16 Neuron)
	for (int i = 0; i < 16; ++i)
	{
		layer_1[i].initRandomize(1, i, 784);
		layer_2[i].initRandomize(2, i, 16);
	}

	// layer 3 (prev. 16 Neuron)
	for (int i = 0; i < 10; ++i)
	{
		layer_3[i].initRandomize(3, i, 16);
	}
}

// tested
void printNeurons(Neuron layer_1[16], Neuron layer_2[16], Neuron layer_3[10])
{
	for (int i = 0; i < 16; ++i)
	{
		layer_1[i].print();
	}
	for (int i = 0; i < 16; ++i)
	{
		layer_2[i].print();
	}
	for (int i = 0; i < 10; ++i)
	{
		layer_3[i].print();
	}
}

// tested
// read function MUST follow the same format as used here
void saveToTextFile(Neuron layer_1[16], Neuron layer_2[16], Neuron layer_3[10])
{
	ofstream myfile("data.txt");
	if (myfile.is_open())
	{
		// layer 1
		for (int i = 0; i < 16; ++i)
		{
			layer_1[i].writeToFile(myfile);
		}

		// layer 2
		for (int i = 0; i < 16; ++i)
		{
			layer_2[i].writeToFile(myfile);
		}

		// layer 3
		for (int i = 0; i < 10; ++i)
		{
			layer_3[i].writeToFile(myfile);
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
void readFromTextFile(Neuron layer_1[16], Neuron layer_2[16], Neuron layer_3[10])
{
	string line;
	ifstream myfile("data.txt");
	
	if (myfile.is_open())
	{
		int count = 0;
		int count2 = 0;
		int layer1_total = 784 * 16 + 16;	// weights and biases
		int layer2_total = layer1_total + 16 * 16 + 16;	// weights and biases
		int currentLayer = 1;
		int totalWeights = 784;
		int neuronIdx = 0;

		vector<double> weights;
		weights.resize(784, 0.0);
		double bias = 0.0;

		// loop through all data
		while (getline(myfile, line, ','))
		{
			// layer check
			if (count == layer1_total) { currentLayer = 2; totalWeights = 16; weights.resize(16, 0.0); neuronIdx = 0; }
			if (count == layer2_total) { currentLayer = 3; totalWeights = 16; weights.resize(16, 0.0); neuronIdx = 0; }

			// add weight and bias
			if (count2 < totalWeights)
			{
				weights[count2] = stod(line);
			}
			else
			{
				bias = stod(line);

				// add to neuron
				if (currentLayer == 1)
				{
					layer_1[neuronIdx].initFromFile(currentLayer, neuronIdx, weights, bias);
				}
				else if (currentLayer == 2)
				{
					layer_2[neuronIdx].initFromFile(currentLayer, neuronIdx, weights, bias);
				}
				else
				{
					layer_3[neuronIdx].initFromFile(currentLayer, neuronIdx, weights, bias);
				}

				neuronIdx++;
				count2 = -1;
			}

			// cont
			count++;
			count2++;
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
double getTotalCost(Neuron layer_3[10], vector<double>& yRow)
{
	double cost = 0.0;
	for (int i = 0; i < 10; ++i)
	{
		cost += (layer_3[i].a - yRow[i]) * (layer_3[i].a - yRow[i]);
	}
	return cost;
}

// test
void getImageAndDetails(const vector<char>& contents, vector<double>& yRow, const vector<char>& labels, Neuron* layer_0, int index)
{
	// init
	getImage1D(contents, index, layer_0);
	yRow.clear();
	yRow.resize(10, 0.0);	// all to 0
	yRow[getImageLabel(labels, index)] = 1.0;
}

// tested
void applyDerivativesLast(Neuron* layer_J, Neuron* layer_K, int layer_J_count, const vector<double>& Yj)
{
	for (int i = 0; i < layer_J_count; ++i)
	{
		layer_J[i].applyDerivativesLast(layer_K, Yj[i]);
	}
}

// tested
void applyDerivatives(Neuron* layer_I, Neuron* layer_J, Neuron* layer_K, int layer_J_count, int layer_I_count)
{
	for (int i = 0; i < layer_J_count; ++i)
	{
		layer_J[i].applyDerivatives(layer_I, layer_K, layer_I_count);
	}
}

// tested
void backPropagation(Neuron layer_0[784], Neuron layer_1[16], Neuron layer_2[16], Neuron layer_3[10], const vector<double>& final_Yj)
{
	// calculate derivatives for layer 3
	applyDerivativesLast(layer_3, layer_2, 10, final_Yj);

	// calculate derivatives for layer 2
	applyDerivatives(layer_3, layer_2, layer_1, 16, 10);

	// calculate derivatives for layer 1
	applyDerivatives(layer_2, layer_1, layer_0, 16, 16);
}

// tested
void printInfo(Neuron layer_1[16], Neuron layer_2[16], Neuron layer_3[10])
{
	// layer 1
	cout << "Layer 1 ========================================================================================================================================" << endl;
	for (int i = 0; i < 16; ++i)
	{
		layer_1[i].print();
	}

	// layer 2
	cout << "Layer 2 ========================================================================================================================================" << endl;
	for (int i = 0; i < 16; ++i)
	{
		layer_2[i].print();
	}

	// layer 3
	cout << "Layer 3 ========================================================================================================================================" << endl;
	for (int i = 0; i < 10; ++i)
	{
		layer_3[i].print();
	}
}

/************************************************************************************************************
MLP: main function
*************************************************************************************************************/
void MLP_train()
{
	// read pixels and labels
	vector<char> contents;
	vector<double> yRow;
	vector<char> labels;
	readMnistFile("train-images.idx3-ubyte", contents);
	readMnistFile("train-labels.idx1-ubyte", labels);

	// predefined
	Neuron layer_0[784];	// pixels
	Neuron layer_1[16];	// hidden layer
	Neuron layer_2[16];	// hidden layer
	Neuron layer_3[10];	// activation layer (each neuron = a digit)

	// init
	initNeuron(layer_1, layer_2, layer_3);

	// train for 60k times
	for (int x = 0; x < TOTAL_SERIES; ++x)
	{
		for (int i = 0; i < TOTAL_ITERATIONS; ++i)
		{
			// init
			getImageAndDetails(contents, yRow, labels, layer_0, i);

			// calculate activations
			for (int i = 0; i < 16; ++i)
			{
				layer_1[i].calculateActivation(layer_0);
			}
			for (int i = 0; i < 16; ++i)
			{
				layer_2[i].calculateActivation(layer_1);
			}
			for (int i = 0; i < 10; ++i)
			{
				layer_3[i].calculateActivation(layer_2);
			}

			//get total cost
			double cost = getTotalCost(layer_3, yRow);
			if (i % 500 == 0)
			{
				bool belowThreshold = cost < 0.01;
				cout << "Series: " << x << "  Training image: " << i << "  Total cost: " << cost << " Below threshold: " << belowThreshold << endl;
			}

			// do backpropagation
			backPropagation(layer_0, layer_1, layer_2, layer_3, yRow);

			// apply gradient
			for (int i = 0; i < 16; ++i)
			{
				layer_1[i].apply();
			}
			for (int i = 0; i < 16; ++i)
			{
				layer_2[i].apply();
			}
			for (int i = 0; i < 10; ++i)
			{
				layer_3[i].apply();
			}

			// print
			// printInfo(layer_1, layer_2, layer_3);
		}
	}

	// see results again
	//getImageAndDetails(contents, yRow, labels, layer_0, 56);

	//// calculate activations
	//for (int i = 0; i < 16; ++i)
	//{
	//	layer_1[i].calculateActivation(layer_0);
	//}
	//for (int i = 0; i < 16; ++i)
	//{
	//	layer_2[i].calculateActivation(layer_1);
	//}
	//for (int i = 0; i < 10; ++i)
	//{
	//	layer_3[i].calculateActivation(layer_2);
	//}

	////get total cost
	//double cost = getTotalCost(layer_3, yRow);
	//cout << "Series: 1  Training image: 56  Total cost: " << cost << endl;

	 // save weights and biases to txt
	 saveToTextFile(layer_1, layer_2, layer_3);
	 cout << "Final weights and biases saved to text file!" << endl;
}

/************************************************************************************************************
MLP: test function
*************************************************************************************************************/
void MLP_test()
{
	// read pixels and labels
	vector<char> contents;
	vector<char> labels;
	readMnistFile("t10k-images.idx3-ubyte", contents);
	readMnistFile("t10k-labels.idx1-ubyte", labels);

	// predefined
	Neuron layer_0[784];	// pixels
	Neuron layer_1[16];	// hidden layer
	Neuron layer_2[16];	// hidden layer
	Neuron layer_3[10];	// activation layer (each neuron = a digit)

	// read off text file
	readFromTextFile(layer_1, layer_2, layer_3);

	// test
	// printInfo(layer_1, layer_2, layer_3);

	// init
	int correctCount = 0;
	int testTotal = 500;
	int startIndex = 8000;
	for (int i = startIndex; i < testTotal + startIndex; ++i)
	{
		getImage1D(contents, i, layer_0);
		int label = getImageLabel(labels, i);

		// calculate activations
		for (int i = 0; i < 16; ++i)
		{
			layer_1[i].calculateActivation(layer_0);
		}
		for (int i = 0; i < 16; ++i)
		{
			layer_2[i].calculateActivation(layer_1);
		}
		for (int i = 0; i < 10; ++i)
		{
			layer_3[i].calculateActivation(layer_2);
		}

		// the predicted digit corr. to the 'brightest' neuron of last layer
		int brightestNeuron = 0;
		double brightestNeuronVal = 0.0;
		for (int i = 0; i < 10; ++i)
		{
			if (layer_3[i].a > brightestNeuronVal)
			{
				brightestNeuron = i;
				brightestNeuronVal = layer_3[i].a;
			}
		}

		// results!
		if (brightestNeuron == label)
		{
			cout << "Correct guess at " << i << ", " << label << " and " << brightestNeuron << endl;
			printImage(layer_0);
			correctCount++;
		}
		else {
			cout << "Wrong guess at " << i << ", correct label is " << label << ", not " << brightestNeuron << endl;
			printImage(layer_0);
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

	// MLP_train();
	// cin.get();
	MLP_test();
}