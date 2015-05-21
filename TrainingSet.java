/**
 * 
 * @author JonathanLee
 * @created 19 Feb 2015
 * 
 * This class functions as a data structure
 * that associates inputs with expected outputs 
 * for a network. This class contains only constructor,
 * getter, setter and toString methods.
 * 
 * methods:
 * TrainingSet (constructor)
 * getInputs
 * getOutputs
 * setInputs
 * setOutputs
 */
public class TrainingSet
{
   private String rawInput;
   private double[] inputs;
   private double[] outputs;
   
   
   /**
    * method: TrainingSet
    * usage: new TrainingSet(inputs, outputs)
    * This constructor method essentially acts
    * as a setter method for the two instance
    * variables.
    * @param inputs   array of expected inputs
    * @param outputs   array of expected outputs
    */
   public TrainingSet(double[] inputs, double[] outputs)
   {
      this.inputs = inputs;
      this.outputs = outputs;
      this.rawInput = "no raw input";
      return;
   }
   
   /**
    * method: setRawInput
    * The purpose of this method is to set the raw input for the training
    * set. This raw input would be something like the file name of the 
    * input image. It can also be left blank. However, this functionality
    * is useful for readable classification of the input.
    * @param rawInput
    */
   public void setRawInput(String rawInput)
   {
      this.rawInput = rawInput;
   }
   
   /**
    * method: TrainingSet
    * usage: new TrainingSet()
    * This constructor is a more general constructor
    * that takes no arguments and leaves the instance
    * variables as not set. It does nothing but
    * create an instance of the training set
    */
   public TrainingSet()
   {
      return;
   }
   
   
   /**
    * method: getInputs
    * usage: program.getInputs()
    * This method is a generic getter
    * method for the inputs instance
    * variable which is an array.
    * @return inputs
    */
   public double[] getInputs()
   {
      return inputs;
   }
   
   /**
    * method: getOutputs
    * usage: program.getOutputs()
    * This method is a generic getter
    * method for the outputs instance
    * variable which is an array.
    * @return  outputs
    */
   public double[] getOutputs()
   {
      return outputs;
   }
   
   /**
    * method: setInputs
    * usage: program.setInputs(inputs)
    * This method is a generic setter
    * method for the inputs instance
    * variable which is an array.
    * @param inputs
    */
   public void setInputs(double[] inputs)
   {
      this.inputs = inputs;
   }
   
   /**
    * method: setOutputs
    * usage: program.setOutputs(outputs)
    * This method is a generic setter
    * method for the inputs instance
    * variable which is an array.
    * @param outputs
    */
   public void setOutputs(double[] outputs)
   {
      this.outputs = outputs;
   }
   
   /**
    * method: toString
    * usage: program.toString()
    * This method is just a generic
    * toString method that overrides
    * the Object toString method.
    * The purpose of this method is to provide
    * a simple way of returning the contents
    * of the training set.
    * @return string representation of TrainingSet
    */
   public String toString()
   {
      String s = "";
      s += "inputs\n";
      
      for (int k = 0; k < inputs.length; k++)
      {
         s += inputs[k] + ", ";
      }
      
      s += "\n expected outputs \n";
      
      for (int i = 0; i < outputs.length; i++)
      {
         s += outputs[i] + ", ";
      }
      return s;
   }  // public String toString
   
   /**
    * method: equalsOutput
    * The purpose of this method is to compare a given set of outputs
    * to the training set outputs. The method determines whether the outputs
    * are considered equal given a tolerance that accounts for any sort of deviation.
    * Output values are only between 0 and 1.
    * The higher the tolerance, the more likely the method will consider the sets equal.
    * @param givenOutputs     the given outputs that are being compared to the idealized training set
    * @param tolerance        the tolerance in the comparison of each output to determine if the sets are equal.
    * @return                 true if the outputs are considered equal, false if not.
    */
   public boolean equalsOutput(double[] givenOutputs, double tolerance)
   {
      boolean response = true;
      
      for (int i = 0; i < outputs.length; i++)
      {
         if(!(givenOutputs[i] + tolerance >= outputs[i] 
               && givenOutputs[i] - tolerance <= outputs[i]))
            response = false;
      }
      
      return response;
   }  // public boolean equalOutputs
   
   /**
    * method: getRawInput
    * usage: program.getRawInput
    * The purpose of this method is to return the
    * raw input which is a more readable version
    * of the network processed input. Processed input
    * is only represented quantitatively instead of as
    * something like a file name.
    * @return  raw input string
    */
   public String getRawInput()
   {
      return this.rawInput;
   }
}
