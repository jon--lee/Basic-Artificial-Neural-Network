import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;


public class Test
{
   public static void main(String[] args)
   {
      new Test();
   }
   
   public Test()
   {
      String fileName = "input.txt";
      try
      {
         TrainingSet[] sets = getTrainingSets(fileName);
         int jLength = 100;
         
         double[][] kjWeights = new double[sets[0].getInputs().length][jLength];
         double[][] jiWeights = new double[jLength][sets[0].getOutputs().length];
         
         Network net = new Network(sets, kjWeights, jiWeights);
         net.generateRandomWeights(.1);
         net.setMinError(.01);
         net.setLambdaFactor(.1);
         net.setIterationGap(1000);
         net.printWeights();
         net.minimization();
         
         TrainingSet[] tSets = net.getTrainingSets();
         
         System.out.println();
         
         for (int t = 0; t < tSets.length; t++)
         {
            net.calculateOutput(tSets[t].getInputs());
            String layerString = net.stringifyLayer(net.getOutput());
            System.out.println(layerString);
         }
         
         
         
      }
      catch(Exception e)
      {
         e.printStackTrace();
      }
   }
   
   public TrainingSet[] getTrainingSets(String fileName) throws FileNotFoundException
   {
      Scanner s = new Scanner(new File(fileName));
      
      s.next();
      
      int setsLength = s.nextInt();
      TrainingSet[] sets = new TrainingSet[setsLength];      
      
      int inputsLength = s.nextInt();
      
      
      for (int t = 0; t < setsLength; t++)
      {
         double[] inputs = new double[inputsLength];
         
         for (int k = 0; k < inputsLength; k++)
            inputs[k] = s.nextDouble();
         
         sets[t] =  new TrainingSet();
         sets[t].setInputs(inputs);
      }
      
      s.next();
      
      int outputsLength = s.nextInt();
      
      for (int t = 0; t < setsLength; t++)
      {
         double[] outputs = new double[outputsLength];
         
         for (int i = 0; i < outputsLength; i++)
            outputs[i] = s.nextDouble();
         
         sets[t].setOutputs(outputs);
      }
      
      return sets;
   }
   
   public void doSomething(int[] weights)
   {
      
   }
   
   public void print(int[] weights)
   {
      
   }
}
