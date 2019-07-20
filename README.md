This project uses only python3 and numpy to build a simple neural network. This helps to give a better grasp at what is going on behind the scenes of frameworks and libraries like Keras and TensorFlow.

The problem that we are trying to learn to solve is a simple binary pattern result. We have sequences of three items in an array. Each item is either red or blue. The answer is whatever color shows up more in the array thats the correct color.

Examples:
| Inputs                | Answers        
| --------------------- |:---------:|
| red blue red          | red       |
| blue blue red         | blue      |
| red blue blue stripes | blue      |

Can we teach the computer to solve problems like these? Yes we can!

in order to run this program
```
git clone <project_repo>
cd <project_repo>
python main.py
```
You should see a result close to 0 similar to [.2435...]
The model isn't perfect but decimals close to 0 means that the answer is red, while answers closer to 1 means it is blue. In this code the new pattern we try to predict says the answer is blue which is correct.

That's all. feel free to play around with this and make changes.
