# Report 
**Course**: Practical Machine Learning and Deep Learning

**Reporter**: Ivan Lyagaev

**Github Repository**: [link](https://github.com/FireFoxIL/simulated-annealing)

**Assignment**: 1

## How to Run
### Task 1
Dependencies:
- Python - `3.6.8`
- Keras - `2.3.1`
- numpy - `1.18.1`
- scikit-learn - `0.20.0`

```bash
cd src && python task1.py
```
### Task 2
Dependencies:
- Python - `3.6.8`
- numpy - `1.18.1`
- pandas - `0.23.4`
- matplotlib - `3.0.1`
```bash
cd src && python task2.py
```

## Results
### Task 1

#### Global Settings:
- test_size: `0.5`

#### Annealing Settings:
- init_temp: `30.0`,
- low_temp: `0.0000001`,
- decay_factor: `0.1`,
- number_of_iterations: `10000`,

#### Annealing Results:
- Loss on test data: `0.3429157045483589`
- Accuracy on test data: `0.9200000166893005`

#### NN Settings:
- epochs: `100`
- batch_size: `32`

#### NN Architecture:
- Dense - `8`
- Dense - `8`
- Dense - `3`

#### NN Results:
- Loss on test data: `0.32666664163271586`
- Accuracy on test data: `0.8799999952316284`

#### Overall Results:
Simulated annealing actually works for neural network weights optimization. 
For some settings, it can outperform classic back-propogation algorithm. 

Overall difference in method's speed can not be compared due to different runtimes. 
Python is really slow

### Task 2

#### Experiments:
| Start Temperature | End Temperature | Decay Factor | Iterations in Epoch |   Total Distance   |
|:-----------------:|:---------------:|:------------:|:-------------------:|:------------------:|
|       25000       |        10       |      0.6     |         200         | 19761.874850558820 |
|       25000       |        10       |      0.1     |         200         | 21906.056463476652 |
|       25000       |        10       |      0.6     |         100         | 25884.497999865434 |
|       15000       |        10       |      0.6     |         200         | 24912.903085002800 |

#### Overall Results:
Seems that initial temperature should be set closer to mean achievable total distance.
Also the convergence should be neither too slow nor too fast.
