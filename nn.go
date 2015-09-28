package main

import "fmt"
import "math"
import rand "math/rand"
import "time"

type Neuron struct {
	weights []float32
	output  float32
	inputs  []float32
}

type Layer struct {
	neurons []Neuron
}

type Network struct {
	layers []Layer
}

func MakeNetwork(numInputs int, hiddenLayerSize int, numHiddenLayers int, numOutputs int) Network {

	var layers []Layer

	for i := 0; i < numHiddenLayers; i++ {
		layers = append(layers, MakeLayer(hiddenLayerSize, numInputs))
	}

	layers = append(layers, MakeLayer(numOutputs, hiddenLayerSize))

	return Network{layers}
}

func MakeLayer(numNeurons int, numWeights int) Layer {

	var neurons []Neuron

	for i := 0; i < numNeurons; i++ {
		neurons = append(neurons, Neuron{rands(numWeights), 0, []float32{}})
	}

	return Layer{neurons}
}

func (nn *Network) Train(trainingInputs []float32, trainingOutput float32) {
	nn.Outputs(trainingInputs)

	L := len(nn.layers)
	deltas := make([][]float32, L)

	layers := nn.layers

	deltas[L-1] = make([]float32, len(layers[L-1].neurons))

	for i, n := range layers[L-1].neurons {
		deltas[L-1][i] = n.output * (1 - n.output) * -(trainingOutput - n.output)
	}

	// Deltas for the output layer will be
	// delta_o = n.output * (1 - n.output) * (trainingOutput - n.output)
	// for the hidden layer will be
	// delta_h_1  = output.weight1 * delta_o
	// delta_h_2 = output.weight2 * delta_o
	// delta_h_3

	deltas[L-2] = make([]float32, len(layers[L-2].neurons))
	for i, n := range layers[L-2].neurons {

		var deltah float32

		for j, _ := range layers[L-1].neurons {
			deltah += deltas[L-1][j] * layers[L-1].neurons[j].weights[i]
		}

		deltas[L-2][i] = deltah * n.output * (1 - n.output)
	}

	for l := range layers {
		for j := range layers[l].neurons {
			for k := range layers[l].neurons[j].weights {
				delta := deltas[l][j]
				input := layers[l].neurons[j].inputs[k]
				layers[l].neurons[j].weights[k] += -0.5 * delta * input
			}
		}
	}

}

func (nn Network) Inspect() {
	for i := range nn.layers {
		layer := nn.layers[i]
		fmt.Println("===========")
		fmt.Println(len(layer.neurons))

		for j := range layer.neurons {
			neuron := layer.neurons[j]
			fmt.Println(neuron.weights)
		}

		fmt.Println("===========")
	}
}

func (nn Network) Outputs(inputs []float32) []float32 {

	var layerOutputs []float32

	layerOutputs = inputs
	for i := 0; i < len(nn.layers); i++ {
		layerOutputs = nn.layers[i].Outputs(layerOutputs)
	}

	return layerOutputs
}

func (n *Neuron) Output(inputs []float32) float32 {
	var output float32

	n.inputs = inputs

	for i := range inputs {
		output += n.weights[i] * inputs[i]
	}

	output = sigmoid(output + 1)

	n.output = output

	return output
}

func (l Layer) Outputs(inputs []float32) []float32 {
	var outputs []float32

	for n := range l.neurons {
		outputs = append(outputs, l.neurons[n].Output(inputs))
	}

	return outputs
}

func sigmoid(input float32) float32 {
	return float32(1 / (1 + math.Exp(-float64(input))))
}

func rands(len int) []float32 {
	var outputs []float32

	for i := 0; i < len; i++ {
		outputs = append(outputs, rand.Float32())
	}

	return outputs
}

func main() {

	rand.Seed(time.Now().UTC().UnixNano())
	network := MakeNetwork(2, 3, 1, 1)

	for i := 0; i < 10000; i++ {
		network.Train([]float32{0, 1}, 1)
		network.Train([]float32{1, 0}, 1)
		network.Train([]float32{1, 1}, 0)
		network.Train([]float32{0, 0}, 0)
	}

	fmt.Println("HERE WE GO: ")
	fmt.Println(network.Outputs([]float32{1, 0}))
}
