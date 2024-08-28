# Practical Tidbits: To Pickle or Not to Pickle

For those of who aren't aware, pickle is a serialization protocol that can be used to serialize (make into a string) python objects. This is often used to write objects to disk and load them again for later use.
For example after training a model, you might want to store it to use later (who knows).


While there are other critiques of Pickle for storing objects, such as [speed and safety](https://www.benfrederickson.com/dont-pickle-your-data/), one that I haven't seen discussed is maintainability. 
Specifically with regards to storing data long term. Not necessarily just data in the sense of "here's a CSV file", but more as in "weights of a neural network", "contents of a tree" where the class holding the data is what gives it "use".
Ironically, in that content pickle (files) don't have a very good shelf-life.

## Important Limitations of Pickle

Bluntly I don't understand the inner workings of Pickle, but we don't have to to discuss some sources of potential issues.

Pickle is sensitive to changes in the functions used. Specifically it expects function locations and interfaces to stay the same. Let's say I wrote a neural network that uses numpy under the hood. If I want to save the model
as a pickle, this is pretty straight forward:

The neural network file:
```
# ./my_lib/ann.py
import numpy as np

class NeuralNetwork:
    def __init__(self, weights):
        self.weights = weights

    def predict(self, X):
        return np.dot(X, self.weights)



```

The file to save it:
```
# ./save_ann.py
import pickle

import numpy as np

from my_lib.ann import NeuralNetwork

# Initialize the network
network = NeuralNetwork(weights=np.random.rand(3, 3))

# Save the network to a pickle file
with open('network.pkl', 'wb') as f:
    pickle.dump(network, f)


```

If I load the pickle file all I need to do is reload it:
```
# ./load_ann.py
import pickle

from my_lib.new_module.ann import NeuralNetwork

with open('network.pkl', 'rb') as f:
    loaded_network = pickle.load(f)
```

However, if in the original code my neural network class has moved pickle will no longer function throwing an error about not being able to find "ann"

```
# ./load_ann.py
import pickle

# I've since refactored my library because everything in one module was hard to navigate
from my_lib.new_module.ann import NeuralNetwork

with open('network.pkl', 'rb') as f:
    loaded_network = pickle.load(f)
	# Pickle will say it can't find ann and crash
	
```

If I changed the interface of the "predict" function, a similar issue would occur if loading an old object. Obviously I can check for versions, but 
if you are trying to quickly save an object you're probably not doing that.

Moreover it might not be worth putting that effort since the issue extends to underlying dependencies. Let's say I send my neural network to a colleague. 

If I'm using an older version of numpy where `__randomstate_ctor` takes a different number of variables, when my colleague tries to load the pickle file they will get an error. This is an issue I've had before when transfering
an SkLearn model between machines.

## A Rotten Pickle (Lessons Learned)

You might say "version incompatibility is minor, you just need to make sure your environment is the same duh.."

You're not wrong, but what happens if a pickle has been sitting for 2-3 years and the original environment used to create it can't be recreated it anymore (broken wheels, ancient versions, etc..)

I could "repickle" the object with more recent libraries, but the code used to create has long since been broken. Plus will I get the same results, who knows??

Guess what, in that case, the data is essentially locked away AFAIK. You could take the time to recreate the environment, but at that point it might be easier to just 
restart the project and only keep the part you need. IMO you are just kicking the can down the road. You'll still need to update the code and verify 
the older and newer versions give the same result. Especially if you want to deploy, use, and maintain that code.

If you plan on using pickle to store data long term, please don't. Even if you think there is a chance this becomes a long term tool, decouple your data and object. Use a standard data format like JSON and be done with it.
You might want to add versioning on to your data too while you are at it. The next engineer will appreciate it.

## When I Pickle

That's not to say Pickle is the devil incarnate. I like pickle.

Pickle is built-in. I don't need an additional dependency. It's a great fit for saving data quickly.

Here is my checklist for an ideal project to use pickle on:
1. Disentangling data and the object would be involved. Trees are a good example of this (not impossible, but you need to think about it)
2. Simple projects that will only be transferred to identical or similar environments
3. Short term projects that won't be effected by changes in the code base or dependencies
4. No or few dependencies
5. Doesn't have "unpicklable" objects. This would require including non std libs 
6. It probably won't be shared or if it does it will be limited

This isn't by any means definitive, but should give you an idea of the ideal project IMO.


## Conclusion

Pickle isn't necessarily bad. It's a built-in tool that makes saving python objects straightforward. For long term use there are serious short comings. Changes in the interface of dependencies
will break code. If these dependencies have aged to the point where they are depreciated this will cause an issue, especially if it is hard to find that version of the library. If you expect to reuse code, seriously consider taking the time to disentangle data from the object.