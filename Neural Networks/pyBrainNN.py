from pybrain.structure import RecurrentNetwork, LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

n           = RecurrentNetwork()
inLayer     = LinearLayer(2, name="Input")
hiddenLayer = SigmoidLayer(3, name="Hidden")
outLayer    = LinearLayer(1, name="Output")

n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)

n.addConnection(FullConnection(inLayer, hiddenLayer, name="C_IH"))
n.addConnection(FullConnection(hiddenLayer, outLayer, name="C_HO"))
n.addRecurrentConnection(FullConnection(n['Hidden'], n['Hidden'], name='C_HH'))

n.sortModules()
n.reset()
print n.activate((2,2))

