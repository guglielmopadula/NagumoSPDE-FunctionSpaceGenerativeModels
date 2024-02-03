# Function Space Generative Models
An attempt to learn a [Nagumo Stochastic Partial Differential Equation](https://github.com/guglielmopadula/NagumoSPDE) by using  generative models on function space.
The GANO code is a variation of the [original repository](https://github.com/neuraloperator/GANO).

|Model  |Mean RE|COV RE|ACF RE|
|-------|-------|------|------|
|GANO   |1.817  |2.15  |0.75  |
|NFPOD  |1.012  |1.12  |0.86  |
|SBMPOD |1.03   |2.33  |1.03  |
|DeepGPR|0.09   |0.09  |0.03  |
