import java.util.Random;

/**
 * 
 * @author JonathanLee
 * @created 3 Feb 2015
 * 
 * The Neural Network is defined as a connected network of nodes or neurons
 * that are separated into layers. Each node has a state of activation which
 * is just the value of the node. This state of activation can be defined as
 * "on" or "off" for simplicity. The nodes are also arranged in layers that,
 * in this case, are fully connected between adjacent layers but not across
 * layers. Information is propagated from layer to layer according to this
 * pattern. During propagation, the value of the initial node is multiplied
 * by a "weight" and that product is passed to the receiving node. Upon receiving
 * information from propagation, processing nodes will perform an activation function
 * on the given information to yield their own state of activation. Finally
 * the network requires a Learning rule which defines how the network
 * adjusts itself in order to achieve certain outputs for respective inputs.
 * 
 * module description:
 * 
 * The Network module handles all of the basic functions
 * of a Neural Network. When an instance of this class
 * is created, certain initial values are passed to the instance.
 * The instance can then calculate outputs of the network based on
 * the inputs and the weights. It also implements
 * minimization so that the weights can be adjusted
 * to fit the expected output values for a given input set.
 * 
 * It should be noted that although the network is fairly generalized (number of hidden nodes,
 * number and options for inputs, number and options for outputs, initial weight values, number of output nodes),
 * there are some aspects of the network that are constant. For example, the module
 * only supports three layer networks.
 * Also the pattern of connection is not generalized (see calculateOutput method) and the
 * activation function is not generalized.
 * 
 * 
 * An important design decision arose when confronted with the choice between
 * double-precision floating point format and Java's default floating point format.
 * Obviously the trade off would be speed compared to precision since floats are
 * 4 bytes and doubles are 8 bytes in total. Since this network is simple and it
 * will not need to handle calculations involving hundreds of neurons, doubles
 * were chosen as the data type. According to the source below, the speed
 * issue of doubles only becomes significant with networks involving thousands
 * of neurons
 * http://www.heatonresearch.com/content/choosing-between-java%E2%80%99s-float-and-double
 * 
 * 
 * UPDATE: After reconsidering the realistic number of nodes in the network, default floating point format
 * may actually be preferable when using input data such as images since there are so many input and hidden nodes. However,
 * For the purpose of consistency and compatibility this module will continue to use double-precision floating points.
 * 
 * methods:
 * Network (constructor)
 * calculateOutput
 * activationFunction
 * activationDerivative
 * getOutput
 * minimization
 * computeDeltaKJWeights
 * computeDeltaJIWeights
 * errorExceedsMin
 * printWeights
 * getKJWeights
 * getJIWeights
 * getTrainingSets
 * debug
 * stringifyLayer
 * computeError
 * 
 */
public class Network
{
   private static final double DEFAULT_MIN_ERROR = .06;                    // minimum error before declared satisfactory
   private double minError;
   
   private static double minTotalError;                                    // minimum total error before declared satisfactory
   
   private static final int PRINT_ITERATION_GAP = 10000;                   // default # of iterations between outputs to console
   private double printIterationGap = PRINT_ITERATION_GAP;
   
   private double[][] kJWeights;          									      // weights that join the k and j layers.
   private double[][] jIWeights;          									      // weights that join the j and i layers.
   private double[] hiddenLayer;        								            // values of j (hidden) layer nodes.
   private double[] outputLayer;         								            // values of i (output) layer nodes.
                                          									      // although outputLayer is an array, it should
   																			               // only have one value as of now.
   
   private TrainingSet[] trainingSets;   									         // training sets
   
   public static final double DEFAULT_LAMBDA_FACTOR = .1;
   private double lambdaFactor;           									      // arbitrary factor for computation of lambda value
   
   public static final int DEFAULT_MAX_ITERATIONS = 100000000;				 
   private int maxIterations;             									      // maximum iterations before minimization is declared pointless
   
   public static final double RAND_WEIGHT_RANGE = 0.25;                    // absolute value range of random weights for generator
   public static final double MIN_RAND_ERROR = 2.5;                        // minimum random error before the randomly generated weights are satisfactory.
   
   public static final double MAXIMUM_RAND_ITERATIONS = 5000;
   
   double[] iThetas;
   double[] jThetas;
   
   private double[][] kJDeltaWeights;     									      // momentum for KJ Weights
   private double[][] jIDeltaWeights;     									      // momentum for JI Weights
   
   
   Random rand = new Random();
   
   private double elapsedTime;                                             // the elapsed time in milliseconds
   
   private boolean adaptive;
   
   /**
    * method: Network
    * usage: new Network
    * This is the constructor method for the Neural Network.
    * The purpose of this method is to initialize the
    * instance variables. Instance variables are initialized 
    * with the given arguments so that the numbers of neurons,
    * inputs in the network, expected outputs, maximum iterations,
    * lambda scale, and number of outputs are all variable. 
    * The constructor also constructs the network by creating an array
    * of neurons which represents the j-layer
    * and an array that represents the i layer (which can have multiple outputs).
    * The arguments are simply set to their respective instance variables.
    * 
    * Note: You can choose to use only some parameters (many options are supported
    * by different constructors). However, you must at least provide the training sets
    * and the weights. The weights do not need to have values in them (then can just be 0)
    * but their dimensions determine the architecture of the network so that is why they must
    * be specified.
    * 
    * @param trainingSets        an array of the training sets which
    *                            data structures that match inputs to their
    *                            expected outputs that the network should
    *                            produce.
    * @param kJInitialWeights    initial weight values that are stored
    *                            in a two dimensional array. the columns
    *                            represent each input. the rows represent
    *                            which neuron in the next layer the inputs
    *                            are connected to. The value of the weight
    *                            between that connection is mapped to that
    *                            particular row and column.
    * @param jIInitialWeights     similar to the kJInitialWeights but instead
    *                            represents weights connected between the j
    *                            and i layers.
    * @param minError            The minimum error before the network stops minimization.
    * @param lambdaFactor        arbitrary value for calculating lambda to scale
    *                            partial derivatives of error function
    * @param maxIterations       the maximum iterations before minimization
    *                            is considered pointless.
    * @return void                   
    */
   public Network(
	         TrainingSet[] trainingSets,
	         double[][] kJInitialWeights,
	         double[][] jIInitialWeights,
	         double minError,
	         double lambdaFactor,
	         int maxIterations)
   {
      this.trainingSets = trainingSets;
      this.hiddenLayer = new double[jIInitialWeights.length];
      this.jThetas = new double[hiddenLayer.length];
		   
      this.outputLayer = new double[jIInitialWeights[0].length];      
		this.iThetas = new double[outputLayer.length];
	   this.kJWeights = kJInitialWeights;
	   this.jIWeights = jIInitialWeights;
	   this.minError = minError;
		   
	   this.minTotalError = minError * trainingSets.length / 5;
		   
	   this.lambdaFactor = lambdaFactor;
	   this.maxIterations = maxIterations;
		   
	   return;
		  
	      
   }  // public Network
   
   /**
    * method: Network
    * Alternate constructor: requires only the bare minimum
    * arguments.
    * @param trainingSets
    * @param kJInitialWeights
    * @param jIInitialWeights
    */
   public Network(TrainingSet[] trainingSets,
	         double[][] kJInitialWeights,
	         double[][] jIInitialWeights)
   {
	   this(trainingSets, kJInitialWeights, jIInitialWeights, DEFAULT_MIN_ERROR, DEFAULT_LAMBDA_FACTOR, DEFAULT_MAX_ITERATIONS);
	   	   
	   return;
   }
   
   /**
    * method: Network
    * Alternate constructor: minimum plus lambdaFactor
    * @param trainingSets
    * @param kJInitialWeights
    * @param jIInitialWeights
    * @param lambdaFactor
    */
   public Network(TrainingSet[] trainingSets,
	         double[][] kJInitialWeights,
	         double[][] jIInitialWeights,
	         double lambdaFactor)
   {
	   this(trainingSets, kJInitialWeights, jIInitialWeights, DEFAULT_MIN_ERROR, lambdaFactor, DEFAULT_MAX_ITERATIONS);
	   
	   return;
   }
   
   /**
    * method: Network
    * Alternate Constructor: minimum plus maxIterations
    * @param trainingSets
    * @param kJInitialWeights
    * @param jIInitialWeights
    * @param maxIterations
    */
   public Network(TrainingSet[] trainingSets,
	         double[][] kJInitialWeights,
	         double[][] jIInitialWeights,
	         int maxIterations)
   {
	   this(trainingSets, kJInitialWeights, jIInitialWeights, DEFAULT_MIN_ERROR, DEFAULT_LAMBDA_FACTOR, maxIterations);
	   
	   return;
   }
   
   /**
    * method: Network
    * Alternate constructor: minimum plus lambdaFactor and maxIterations
    * @param trainingSets
    * @param kJInitialWeights
    * @param jIInitialWeights
    * @param lambdaFactor
    * @param maxIterations
    */
   public Network(
	         TrainingSet[] trainingSets,
	         double[][] kJInitialWeights,
	         double[][] jIInitialWeights,
	         double lambdaFactor,
	         int maxIterations)
   {
	   this(trainingSets, kJInitialWeights, jIInitialWeights, DEFAULT_MIN_ERROR, lambdaFactor, maxIterations);
	   
	   return;
   }
   
   
   /**
    * method: generateRandomWeights
    * usage: program.generateRandomWeights()
    * The purpose of this method is to generate random weights that
    * can then be used to calculate outputs or minimize. A minimum random
    * is option, otherwise the default will be used which is calculated
    * (see alternate parameter-less method) 
    * Choosing a minRandError has an advantage because an small one
    * will reduce the time taken to train the network. However it also must
    * be large enough so that the method does not get hung the loop. The generator
    * will stop trying after a certain amount of iterations and just accept some random
    * weights.
    */
   public void generateRandomWeights(double minRandError)
   {
      boolean errorExceedsMin = true;
      int iterations = 0;
      
      while(errorExceedsMin && iterations < MAXIMUM_RAND_ITERATIONS)
      {
         populateWeights(kJWeights);
         populateWeights(jIWeights);
         
         double[] errors = new double[trainingSets.length];
         
         for(int t = 0; t < trainingSets.length; t++)
         {
            double[] expectedOutputs = trainingSets[t].getOutputs();
            calculateOutput(trainingSets[t].getInputs());
            errors[t] = this.computeError(expectedOutputs, this.outputLayer);
         }
         
         iterations++;
         errorExceedsMin = errorExceedsMin(errors, minRandError);
      }  // while(errorExceedsMin)
            
      return;
   }  // public void generateRandomWeights
   
   
   
   /**
    * method: generateRandomWeights
    * This method is the alternate which calculates its own minRandError
    * and then passes it to the original.
    */
   public void generateRandomWeights()
   {
      generateRandomWeights(MIN_RAND_ERROR);
      
      return;
   }
   
   
   /**
    * method: populateWeights
    * The purpose of this metho is to populate.
    * @param weights
    */
   public void populateWeights(double[][] weights)
   {
      for (int k = 0; k < weights.length; k++)
      {
         for (int j = 0; j < weights[0].length; j++)
         {
            weights[k][j] = (rand.nextDouble() - .5) * RAND_WEIGHT_RANGE;
         }
      }
      
      return;
   }
   
   /**
    * method: setIterationGap
    * The purpose of this method is to act 
    * as a setter method for the print iteration gap.
    * The different iteration gaps may be more appropriate
    * for different sized training sets. Use your own discretion
    * @param iterationGap
    */
   public void setIterationGap(int iterationGap)
   {
      this.printIterationGap = iterationGap;
      return;
   }
   
   /**
    * method: setMinError
    * The purpose of this method is to manually set the 
    * minimum error for the minimization process before
    * it is considered satisfactory.
    * @param minError
    */
   public void setMinError(double minError)
   {
      this.minError = minError;
      return;
   }
   
   /**
    * method: calculateOutput
    * usage: program.calculateOutput
    * In this network, information is passed from the k-layer to the
    * j-layer to the i-layer and the nodes of each layer are fully connected
    * to adjacent layers but not across layers. When propagated, the value of
    * a particular node is multiplied by its respective connecting weight
    * and then added to the products of other nodes and their respective weights
    * that are connected to the same receiving node. The receiving node
    * then applies a sigmoid function to that sum. See "Minimization of the Error
    * Function for a Single Output" By Dr. Eric R. Nelson for clarification.
    * The purpose of this method is to calculate the output of the network by
    * using this process. The output is defined as the values of the nodes in
    * the final layer (i-layer).
    * The method does this calculation by looping through all of the three layers. The traversing
    * of the k-layer becomes the inner most for-loop and so the propagated sum is first calculated
    * and then the sigmoid of that sum becomes the value of its respective hidden (j) neuron. The same
    * is done for the neurons of the i layer. The output is then returned as an array of values of nodes.
    * 
    * Several values are stored for reuse in backpropagation during evaluation.
    * The weighted sums are stored and the output and hidden layer calculations are stored in instance variables.
    * 
    * @param inputSet           the array of inputs to be used
    * @return                 the array of output for that specific input set
    */
   public double[] calculateOutput(double[] inputSet)
   {
      for (int i = 0; i < outputLayer.length; i++)
      {
         double propagationToOutput = 0.0;
         
         for (int j = 0; j < hiddenLayer.length; j++)
         {
            double propagationToHidden = 0.0;
            double[] inputs = inputSet;
            
            for (int k = 0; k < inputs.length; k++)
            {
               propagationToHidden += inputs[k] * kJWeights[k][j];
            }
            
            jThetas[j] = propagationToHidden;
            hiddenLayer[j] = activationFunction(propagationToHidden);
            propagationToOutput += hiddenLayer[j] * jIWeights[j][i];
         
         }   // for (int j = 0; j < jNodes.length; j++
         
         iThetas[i] = propagationToOutput;
         outputLayer[i] = activationFunction(propagationToOutput);
         
      }   // for (int i = 0; i < iNodes.length; i++)
      
      return outputLayer;
   }  // public double[] calculateOutput
   
   
   /**
    * method: activationFunction
    * usage: program.activationFunction(x)
    * mathematical sigmoid function.
    * The sigmoid function (or logistic function)
    * takes any double value as input, then
    * returns a double value between 1 and 0.
    * A clearly written sigmoid function can be found
    * in the document "Minimization of the Error
    * Function for a Single Output" By Dr. Eric R. Nelson.
    * @param x    the input value
    * @return     double value between 1 and 0
    */
   public double activationFunction(double x)
   {
      double sigmoid = 1.0/(1.0 + Math.exp(-x));
      return sigmoid;
   }
   
   /**
    * method: activationDerivative
    * The purpose of this method is to determine
    * the derivative of the activation function
    * 
    * In this case it is the derivative of the
    * sigmoid function which is f(x)(1 - f(x))
    * where f(x) is the sigmoid.
    * @param x
    * @return
    */
   public double activationDerivative(double x)
   {
      double sigmoid = activationFunction(x);
      return sigmoid * (1.0 - sigmoid);
   }
   
   
   
   /**
    * method: getOutput
    * usage: program.getOutpu()
    * This method is a simple getter method for the output layer.
    * This method just returns the output layer array.
    * @return  value of output node.
    */
   public double[] getOutput()
   {
      return outputLayer;
   }
   
   
   
   /**
    * method: minimization
    * usage: program.minimization()
    * Details about the theory of minimization 
    * can be found in "Rosenblatt Multi-layer perceptron"
    * by Jonathan Lee. The purpose of this method is to implement
    * that theory by applying changes to the weights based on steepest
    * descent on the error function. To achieve this, changes in weights
    * are applied continuously in a while-loop with gradient descent. 
    * Additionally, since there are multiple training sets, a for-loop is 
    * set inside the while loop so that separate changes in weights are applied 
    * to each training set and the error is reduced for each set. This step is 
    * done because reduction of the error for one set may raise the error for another set.
    * 
    * Inside the for-loop, the output for a particular training set is calculated,
    * then the expected output is retrieved from the training set instance variable
    * and the error can be calculated. From this information, the changes in weights
    * for both layers of weights can be calculated. Lastly the changes in weights
    * are applied to the weights.
    * 
    * Also, this method is designed to handle multiple outputs so the error
    * for a particular training set is actually the sum of errors for each
    * output.
    * 
    * The while-loop only continues under certain conditions. Firstly,
    * at least one error must exceed the minimum error value. Secondly,
    * an "iterations" local variable is kept and incremented after
    * each iteration. If it exceeds a certain value than the loop ends
    * and the minimization is considered pointless.
    *
    * postcondition: network weights adjusted to fit training sets
    */
   public void minimization()
   {
      int iterations = 0;                                    // iterations counter to be compared with max iterations
      double[] errors = new double[trainingSets.length];     // length = number of training sets
      
      
      do  // while(errorExceedsMin(errors) && iterations < MAX_ITERATIONS)
      {
         for (int t = 0; t < trainingSets.length; t++)
         {
            
            TrainingSet trainingSet = trainingSets[t];
            double[] inputLayer = trainingSet.getInputs();
            double[] expectedOutputLayer = trainingSet.getOutputs();
            
            calculateOutput(inputLayer);                            //outputLayer now set
            
            errors[t] = computeError(expectedOutputLayer, outputLayer);
 
            computeDeltaWeights(expectedOutputLayer, outputLayer, errors[t], inputLayer, iterations);
            
         }   // for (int t = 0; t < trainingSets.length; t++)
         
         
         if(iterations % printIterationGap == 0)
         {
            
            debug("iterations: " + iterations);
            printErrors(errors);
            
            debug("");
         }  // if(iterations % printIterationGap == 0)
         
         iterations++;
         
      }
      while(errorExceedsMin(errors, minError) && iterations < maxIterations);
      
      
      
      debug("iterations: " + iterations);
      
      for (int t = 0; t < errors.length; t++)
      {
         debug("errors[" + t + "]: " + errors[t]);
      }
            
      return;
   }  // public void minimization
   
   /**
    * method: printErrors
    * The purpose of this method is to print the errors to the console
    * for each training set
    * @param errors
    */
   public void printErrors(double[] errors)
   {
      double totalError = 0.0;
      for (int h = 0; h < errors.length; h++)
      {
         debug("errors[" + h + "]: " + errors[h]);
         totalError += errors[h];
      }
   }
   
   /**
    * method: computeError
    * usage: program.computeError(expected, actual)
    * The purpose of this method is to calculate
    * the error for a given training set. The error
    * for a training set is defined as the sum of the
    * square roots of the differences between the
    * expected output and the actual output all multiplied
    * by one half.
    * @param expectedOutput      expected values for output layer
    * @param outputLayer         actual values for output layer
    */
   public double computeError(double[] expectedOutputLayer, double[] outputLayer)
   {
      double error = 0.0;
      
      for (int i = 0; i < outputLayer.length; i++)
      {
         double difference = expectedOutputLayer[i] - outputLayer[i];
         error += 0.5 * difference * difference;
      }
      
      return error;
   }  // public void computeError
   
   /**
    * method: computeDeltaWeights
    * usage: program.computeDeltaWeights
    * BACKPROPAGATION
    * The purpose of this method is to compute the changes in weights of the network
    * by using gradient descent and then applying those changes in weights to the respective weights.
    * The goal is to reduce the error of the network by applying the changes. The calculation involves
    * solving for the direction and magnitude of the change in weight for each weight. This process
    * is done by calculating the partial derivative of the error function with respect to a given weight
    * and then multiplying that partial derivative by a negative scalar. The scalar is negative because
    * gradient gives the steepest ascent so its opposite direction is the steepest descent.
    * Details about these processes can be found in "Minimizing the Error
    * Function" By Dr. Eric R. Nelson and
    * "Rosenblatt Multi-Layer Perceptron" by Jonathan Lee.
    * 
    * Edit: backpropagation which is also detailed in the aforementioned papers is now implemented
    *       which cuts down on the number calculations by reusing values from the initial network evaluation
    * 
    * Edit: This method also implements momentum. Momentum involves saving the changes in weight for each
    *       weight in order to add it to the next calculated change in weight for that same weight. The purpose
    *       of momentum is to reduce the number of oscillations around the minimum error as detailed in R. Rojas's
    *       Fast Learning Algorithms: http://page.mi.fu-berlin.de/rojas/neural/chapter/K8.pdf
    * 
    * @param expectedOutputLayer
    * @param outputLayer
    * @param error
    * @param inputLayer
    * @param iterations
    */
   public void computeDeltaWeights(double[] expectedOutputLayer, double[] outputLayer, double error, double[] inputLayer, int iterations)
   {
      double lambda = lambdaFactor;
      //double lambda = (lambdaFactor);
      //lambda = lambdaFactor;
      if(adaptive)
         lambda *= error;
      double alpha = 1;
      
      double[][] kJMomentum = new double[kJWeights.length][kJWeights[0].length]; 
      double[][] jIMomentum = new double[jIWeights.length][jIWeights[0].length];
      
      for (int k = 0; k < kJWeights.length; k++)
      {
         for (int j = 0; j < kJWeights[0].length; j++)
         {
            double jOmega = 0.0;
            
            if(k == 0)                                         // This statement exists to limit the computation time by eliminating
            {                                                  // redundant calculations caused by the arrangement of the for-loops
                                                               // The variables within this if block do not depend on k
            
               for (int i = 0; i < outputLayer.length;i++)
               {
                  
                  double hAtJ = hiddenLayer[j];
                  double iOmega = expectedOutputLayer[i] - outputLayer[i];
                  double iTheta = this.iThetas[i];
                  
                  double iPsi = iOmega * activationDerivative(iTheta);
                  
                  jOmega += iPsi * jIWeights[j][i];
                  
                  double jIDelta = lambda * hAtJ * iPsi;
                  jIDelta += alpha * jIMomentum[j][i];
                  
                  jIMomentum[j][i] = jIDelta;
                  
                  jIWeights[j][i] += jIDelta;
                  
               }  // for (int i = 0; i < outputLayer.length;i++)
            }  // if(k == 0)
            
            double jPsi = jOmega * activationDerivative(jThetas[j]);
            
            double aAtK = inputLayer[k];
            
            double kJDelta = lambda * aAtK * jPsi;
            kJDelta += alpha * kJMomentum[k][j];
            
            kJMomentum[k][j] = kJDelta;
            
            kJWeights[k][j] += kJDelta;
            
         }  // for (int j = 0; j < kJWeights[0].length; j++)
      }  // for (int k = 0; k < kJWeights.length; k++)
      return;
   }  // public void computeDeltaKJWeights
   
   
   
   
   
   /**
    * method: errorExceedsMin
    * usage: program.errorExceedsMin(errors)
    * This method determines if any of the values
    * in a given array of errors exceeds
    * the minimum error which which is given
    * to the method. If all values
    * in the array are less than or 
    * equal to the minimum error
    * then the method returns false.
    * If one or more values exceed
    * the minimum then the method returns true. 
    * @param errors      array of error values of any length
    * @param minError    minimum error to be compared to
    * @return           true or false as described above
    */
   public boolean errorExceedsMin(double[] errors, double minError)
   {
      boolean result = false;
      
      for (int t = 0; t < errors.length; t++)
      {
         if(errors[t] > minError)
         {
            result = true;
         }
      }
      
      return result;
   }  // public boolean errorExceedsMin
   
   
   /**
    * method: printWeights
    * usage: program.printWeights
    * This method is primarily used for debugging purposes;
    * Thus it is a private method only accessible to instances
    * of the Network class. The purpose of this method is to
    * print out the current weight values of the network.
    * The method does this action by looping through
    * the two 2-dimensional arrays that represent the weights
    * in between the layers and printing the value of each
    * weight to the console.
    */
   public void printWeights()
   {
      System.out.println("kJWeights");
      for (int j = 0; j < kJWeights[0].length; j++)
      {
         for (int k = 0; k < kJWeights.length; k++)
         {
            debug("k: " + k + "; j: " + j + "; weight: " + kJWeights[k][j]);   
         }
      }
      
      System.out.println("jIWeights");
      
      for (int i = 0; i < jIWeights[0].length; i++)
      {
         for (int j = 0; j < jIWeights.length; j++)
         {
            debug("j: " + j + "; i: " + i + "; weight: " + jIWeights[j][i]);
         }
      }
      
      return;
   }  // public void printWeights
   
   /**
    * method: getKJWeights
    * usage: program.getKJWeights()
    * This method is a generic getter method for the 
    * 2d array of double values that
    * represent the values of the weights
    * on the connections between the K
    * and J layer weights.
    * @return kJWeights   2d array of kJWeights
    * 
    */
   public double[][] getKJWeights()
   {
      return kJWeights;
   }
   
   /**
    * method: getJIWeights
    * usage: program.getJIWeights()
    * This method is a generic getter method for the 
    * 2d array of double values that
    * represent the values of the weights
    * on the connections between the J
    * and I layer weights.
    * @return jIWeights   2d array of kJWeights
    * 
    */
   public double[][] getJIWeights()
   {
      return jIWeights;
   }
   
   
   /**
    * method: getTrainingSets
    * usage: program.getNumberTrainingSets
    * This method is a getter method
    * that simply returns the training
    * sets instance variable.
    * @return  training sets for this network
    */
   public TrainingSet[] getTrainingSets()
   {
      return trainingSets;
   }
   /**
    * method: debug
    * usage: program.debug(s)
    * This method is a debugging method used only for the Network
    * class that simplifies outputting to the console.
    * This method is used only for debugging
    * purposes hence the name.
    * @param s    the string that is to be printed
    */
   private void debug(String s)
   {
      System.out.println(s);
      return;
   }
   
   /**
    * method: stringifyLayer
    * usage: program.stringify
    * The purpose of this method is to convert a given layer
    * into a string by looping through the nodes of a layer
    * and constructing a string that displays the index of the
    * node and the value of the node. This method is basically
    * used just for debugging purposes (especially for displaying
    * output)
    * @param layer    the layer to be converted to a string
    * @return        the string representation of the layer
    */
   public static String stringifyLayer(double[] layer)
   {
      String s = "";
      
      for(int i = 0; i < layer.length; i++)
      {
         s += "node[" + i + "]: " + layer[i] + "\n";
      }
      
      s += "\n";        //for spacing
      
      return s;
   }  // public static String stringifyLayer
   
   /**
    * method: getElapsedTime
    * The purpose of this method is to return the elapsed
    * time during which the minimization process ran.
    * @return elapsedTime
    */
   public double getElapsedTime()
   {
      return elapsedTime;
   }
   
   /**
    * method: setLambdaFactor
    * This is a setter method for the lambda
    * factor which controls the learning rate
    * of the network.
    * @param lambdaFactor
    */
   public void setLambdaFactor(double lambdaFactor)
   {
      this.lambdaFactor = lambdaFactor;
   }
   
   /**
    * method: setAdaptive
    * This is a setter method for the adaptive instance variable.
    * Adaptive determines whether the learning rate is adaptive or not.
    * If not, it will not factor in the magnitude of error into the 
    * learning rate calculation. By default it is false.
    * @param adaptive
    */
   public void setAdaptive(boolean adaptive)
   {
      this.adaptive = adaptive;
   }
   
}
