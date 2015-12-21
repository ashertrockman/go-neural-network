package main

import (
	"fmt"
	"github.com/petar/GoMNIST"
	"image"
	"image/color"
	"image/jpeg"
	"math"
	rand "math/rand"
	"os"
	"time"
)

type Neuron struct {
	weights []float64
	output  float64
	inputs  []float64
}

type Layer struct {
	neurons []Neuron
}

type Network struct {
	layers []Layer
}

func MakeNetwork(numInputs int, hiddenLayerSize int, numHiddenLayers int, numOutputs int) Network {

	var layers []Layer

	layers = append(layers, MakeLayer(hiddenLayerSize, numInputs))

	for i := 0; i < numHiddenLayers-1; i++ {
		layers = append(layers, MakeLayer(hiddenLayerSize, hiddenLayerSize))
	}

	layers = append(layers, MakeLayer(numOutputs, hiddenLayerSize))

	return Network{layers}
}

func MakeLayer(numNeurons int, numWeights int) Layer {

	var neurons []Neuron

	for i := 0; i < numNeurons; i++ {
		neurons = append(neurons, Neuron{rands(numNeurons, numWeights+1, numWeights+1), 0, []float64{}})
	}

	return Layer{neurons}
}

func (nn *Network) Train(trainingInputs []float64, trainingOutputs []float64) {
	nn.Outputs(trainingInputs)

	L := len(nn.layers)
	deltas := make([][]float64, L)

	layers := nn.layers

	deltas[L-1] = make([]float64, len(layers[L-1].neurons))

	for i, n := range layers[L-1].neurons {
		deltas[L-1][i] = n.output * (1 - n.output) * -(trainingOutputs[i] - n.output)
	}

	// Deltas for the output layer will be
	// delta_o = n.output * (1 - n.output) * (trainingOutput - n.output)
	// for the hidden layer will be
	// delta_h_1  = output.weight1 * delta_o
	// delta_h_2 = output.weight2 * delta_o
	// delta_h_3

	for i := L - 2; i >= 0; i-- {
		deltas[i] = make([]float64, len(layers[i].neurons)) // Deltas for layer i, last hidden layer
		for k, n := range layers[i].neurons {               // Go through each neuron in last hidden layer

			var deltah float64 // Sum

			for j, _ := range layers[i+1].neurons { // Each of the neurons in the output layer
				deltah += deltas[i+1][j] /* Output layer deltas */ * layers[i+1].neurons[j].weights[k]
			}

			deltas[i][k] = deltah * n.output * (1 - n.output)
		}
	}

	for l := range layers {
		for j := range layers[l].neurons {
			for k := range layers[l].neurons[j].weights {
				delta := deltas[l][j]
				layer := layers[l]
				neuron := layer.neurons[j]
				input := neuron.inputs[k]
				// Learning rate: 0.5
				layers[l].neurons[j].weights[k] += -0.5 * delta * input
			}
		}
	}

}

func (nn Network) Visualize(filename string) {
	inputSize := int(math.Sqrt(float64(len(nn.layers[0].neurons)))) * 28
	img := image.NewRGBA(image.Rect(0, 0, inputSize, inputSize))

	x, y, x2, y2 := 0, 0, 0, 0
	for _, n := range nn.layers[0].neurons {
		for _, w := range n.weights {
			w = sigmoid(w)
			img.Set(x+x2, y+y2, color.RGBA{uint8(w * 255), uint8(w * 255), uint8(w * 255), 255})
			x++

			if x == 28 {
				x = 0
				y++
			}
		}
		x2 += 28
		if x2 == inputSize {
			x2 = 0
			y2 += 28
		}
		x, y = 0, 0
	}
	file, err := os.Create(filename)
	if err != nil {
		fmt.Println("Something happened")
	}
	defer file.Close()

	jpeg.Encode(file, img, &jpeg.Options{80})
}

func (nn Network) Outputs(inputs []float64) []float64 {

	var layerOutputs []float64

	layerOutputs = inputs
	for i := 0; i < len(nn.layers); i++ {

		layerOutputs = nn.layers[i].Outputs(append(layerOutputs, 1))
	}

	return layerOutputs
}

func (n *Neuron) Output(inputs []float64) float64 {
	var output float64

	n.inputs = inputs

	for i := range inputs {
		output += n.weights[i] * inputs[i]
	}

	output = sigmoid(output)
	n.output = output

	return output
}

func (l Layer) Outputs(inputs []float64) []float64 {
	var outputs []float64

	for n := range l.neurons {
		outputs = append(outputs, l.neurons[n].Output(inputs))
	}

	return outputs
}

func sigmoid(input float64) float64 {
	return float64(1 / (1 + math.Exp(float64(-input))))
}

func rands(lout, lin, len int) []float64 {
	var outputs []float64
	epsilon := math.Sqrt(float64(6)) / math.Sqrt(float64(lin+lout))

	for i := 0; i < len; i++ {
		outputs = append(outputs, (rand.Float64()*2*epsilon - epsilon))
	}

	return outputs
}

func label2out(label int) []float64 {
	var outputs []float64

	for i := 0; i < 10; i++ {
		if i == label {
			outputs = append(outputs, 1)
		} else {
			outputs = append(outputs, 0)
		}
	}

	return outputs
}

func main() {

	rand.Seed(time.Now().UTC().UnixNano())
	network := MakeNetwork(784, 300, 2, 10)
	train, test, err := GoMNIST.Load("GoMNIST/data")

	if err != nil {
		fmt.Println("Could not load MNIST data.")
		os.Exit(1)
	}

	network.Visualize("visualize.jpg")

	i := 0
	// Iterate through training set, updating weights for each set of features
	sweeper := train.Sweep()
	for {
		image2, label, present := sweeper.Next()
		if !present {
			break
		}

		floats := make([]float64, len(image2))
		for j := 0; j < 784; j++ {
			floats[j] = float64(image2[j]) / 255
		}

		network.Train(floats, label2out(int(label)))
		i++
		if i%10000 == 0 {
			fmt.Println(fmt.Sprintf("%d / 60000", i))
			network.Visualize(fmt.Sprintf("visualize%d.jpg", i))
		}
	}

	correct, total := 0, 0
	// Iterate through test set
	sweeper = test.Sweep()
	for {
		image, label, present := sweeper.Next()
		if !present {
			break
		}

		floats := make([]float64, len(image))
		for i := 0; i < len(image); i++ {
			floats[i] = float64(image[i]) / 255
		}
		outputs := network.Outputs(floats)
		if outputs[int(label)] > 0.5 {
			correct++
		}
		total++

		// Current % correct
		fmt.Println(float32(correct) / float32(total))
	}
}
