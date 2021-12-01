# make a VQC as a TensorFlow layer by PennyLane

import numpy
import tensorflow as tf
import pennylane as qml
pi = numpy.pi

def makeVQCLayer(nQubit, depth, outputQubitsId = [0]):
    
    dev = qml.device("default.qubit", wires=nQubit)
    
    def makeCircuit(inputs, weights):
        # inputs must by in [-1,1]
        qml.AngleEmbedding(inputs * 0.5 * pi, wires = range(nQubit), rotation='Y')
        qml.BasicEntanglerLayers(weights, wires=range(nQubit), rotation=qml.RY)
        return [qml.expval(qml.PauliZ(wires=i)) for i in outputQubitsId]
    
    return qml.qnn.KerasLayer(qml.QNode(makeCircuit, dev),
                               {"weights": (depth, nQubit)},
                               output_dim=len(outputQubitsId))
    