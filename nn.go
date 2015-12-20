package main

import "fmt"
import "math"
import rand "math/rand"
import "time"
import "github.com/petar/GoMNIST"

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
		neurons = append(neurons, Neuron{rands(numNeurons, numWeights, numWeights), 0, []float64{}})
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
				layers[l].neurons[j].weights[k] += -0.5 * delta * input
			}
		}
	}

}

func (nn Network) Inspect() {
	layer := nn.layers[2]
	fmt.Println("===========")
	fmt.Println(len(layer.neurons))

	for j := range layer.neurons {
		neuron := layer.neurons[j]
		fmt.Println(neuron.weights)
	}

	fmt.Println("===========")
}

func (nn Network) Outputs(inputs []float64) []float64 {

	var layerOutputs []float64

	layerOutputs = inputs
	for i := 0; i < len(nn.layers); i++ {
		layerOutputs = nn.layers[i].Outputs(layerOutputs)
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
	return float64(1 / (1 + math.Exp(-float64(input))))
}

func rands(lout, lin, len int) []float64 {
	var outputs []float64
	epsilon := math.Sqrt(float64(6)) / math.Sqrt(float64(lin+lout))

	for i := 0; i < len; i++ {
		outputs = append(outputs, rand.Float64()*2*epsilon-epsilon)
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

	}

	i := 0
	sweeper := train.Sweep()
	for p := 0; p < 1; p++ {
		for {
			image, label, present := sweeper.Next()
			if !present {
				break
			}

			floats := make([]float64, len(image))
			for i := 0; i < len(image); i++ {
				floats[i] = float64(image[i]) / 255
			}
			network.Train(floats, label2out(int(label)))
			i++

			if i%10000 == 0 {
				fmt.Println(i, label, present)
			}

		}
	}
	sweeper = train.Sweep()

	correct, total := 0, 0
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
		fmt.Println(outputs)
		if outputs[int(label)] > 0.5 {
			correct++
		}
		total++

		fmt.Println(float32(correct) / float32(total))
	}

	fmt.Println(len(network.layers))
}
