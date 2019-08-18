# STrees

A prototype implementation of the C4.5 algorithm for secure decision tree training in MP-SPDZ.

## Setting up, testing, running

Requires Docker to run.

Requires MP-SPDZ repo (tested with latest commit `bd60197`) to be in the parent directory. So if you've cloned this project into `foo/bar/`, MP-SPDZ needs to be under `foo/bar` (or just adjust the shell scripts accordingly). Note that MP-SPDZ only needs to be there, doesn't need to be compiled or even compilable in your system as everything will run in a Docker container.

To set up, run (make sure Docker is running!):

```bash
bash build.sh
```

This might take a while. Once it's done you should get a bash terminal inside a Docker container.

The folder that contains `strees` is mounted into the Docker container. This means that you can modify the source files *outside* the Docker container and the changes will show up inside the container. 

From the Docker terminal prompt `cd ../strees/`.

Run `bash test.sh`. This should eventually print `All tests OK.`

To run all tests and examples with the output shown, run `bast run.sh`. 

The main code is under `c45/c45.py`. A good entry point is the `c45` method, and `main`. 

Note that once you exit the terminal, the Docker container will shut down. You don't need to rebuild it. Just run to restart it and get another bash session:

```bash
docker start mpspdz && docker exec -ti mpspdz /bin/bash
```

## Limitations

This is very much work in progress. Some limitations include:

* A lot of the building blocks are not optimized. The sort I'm using for instance is a naive bubble sort.
* There are probably a lot of MP-SPDZ primitives that we should use instead of the building blocks I hacked together.
* Discrete attributes are not yet supported. Only continuous attributes for now.
* There is only a fully oblivious version of the algorithm implemented, I don't take advantage of revealing some of the tree structure etc. yet.
* The algorithm runs exactly for the specified number of iterations.
* The continuous attribute values are currently just integer values.
* Currently the number of samples must be a power of 2, as required by the underlying oblivious shuffle.
* Not well tested, things will probably break when you run the main algorithm on more complicated inputs.
* Only supports binary discrete attributes, and binary classes.

## Misc

*Representing classes.* There are different ways in which we can represent the class of a sample. The current implementation only deals with binary classes so a single binary column is sufficient. If we have more than two classes, one obvious representation is to assign an integer to each class and have a non-binary class column. However, this forces us to run additional equality checks. We can avoid these by using a separate binary indicator column for each class instead.
 
