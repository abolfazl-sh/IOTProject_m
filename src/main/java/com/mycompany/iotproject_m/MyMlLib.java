/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.iotproject_m;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.math3.util.Pair;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;



import org.apache.spark.mllib.linalg.Vectors;



import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel; 

import org.apache.spark.mllib.regression.LabeledPoint;



import scala.Tuple2;


/**
 *
 * @author AAA
 */
public class MyMlLib {
    
    private static int number = 1000;
    private static int split = 900;
    private static final double e = 2.71;
    private static Path path = FileSystems.getDefault().getPath("").toAbsolutePath();
    private static String dataFile = path + "\\Data\\train.csv";
    private static String dataFile1 = path + "\\Data\\train1.csv";
    private static String dataFile2 = path + "\\Data\\train2.csv";
    private static String dataFile3 = path + "\\Data\\train3.csv";
    private static String dataFile4 = path + "\\Data\\train4.csv";
    private static String testFile = path + "\\Data\\test.csv";
    
    private static Map<String, Integer> labels = new HashMap<>();
    private static Integer[] values = new Integer[number];
    private static Double[] weights = new Double[number];
    private static Map<LabeledPoint,Double> pointweights = new HashMap<>();
    private static Map<DecisionTreeModel, Double> errors = new HashMap<>();
    private static LearnPP learnpp = new LearnPP();
    
    static Integer count = -1;


    static double error = 0.0;
    static double beta = 0.0;
    //hypathasis error
    static double hserror = 0.0;
    
    public static void main(String[] args) throws FileNotFoundException, IOException{
        System.out.println("hi");
        SparkConf sparkConf = new SparkConf().setAppName("DecisionTreeExample")
                                        .setMaster("local[2]").set("spark.executor.memory","2g");
        labels.put("ack", 0);
        labels.put("benign_traffic", 1);
        labels.put("scaner", 2);
        labels.put("syn", 3);
        labels.put("udp", 4);
        labels.put("udpplain", 5);
        
        BufferedReader br = new BufferedReader(new FileReader(dataFile));
        while(br.ready()){
            String line = br.readLine();
            String[] parts = line.split(",");
            double[] points = new double[parts.length - 1];
            for (int j = 0; j < parts.length - 1; j++)
            points[j] = Double.parseDouble(parts[j]);
            pointweights.put(new LabeledPoint(labels.getOrDefault(parts[parts.length - 1],0),
                        Vectors.dense(points)), 1 / (double)number);
        }
        
        br.close();
        
        // start a spark context
        JavaSparkContext jsc = new JavaSparkContext(sparkConf); 
        
        for(int i = 0; i < number; i++){
            values[i] = i;
            
        }
        int z = 0;
        for(Double d : pointweights.values()){    
            weights[z] = d;
            z++;
        }

        ArrayList<DecisionTreeModel> models = new ArrayList<>();
        ArrayList<Double> accurecies = new ArrayList<>();

        for(int i = 0; i < 30; i++){
            BufferedReader br1 = new BufferedReader(new FileReader(dataFile));
            count = -1;        
            error = 0.0;
            beta = 0.0;
            hserror = 0.0;
            
            z = 0;

            for(Double d : pointweights.values()){

                weights[z] = d;
                z++;

            }
            br.close();
            
            List<LabeledPoint> t1 = new ArrayList<>();
            List<LabeledPoint> te1 = new ArrayList<>();
            Pair pair = learnpp.selectTrain(values, weights, split);
            ArrayList<Integer> first = (ArrayList<Integer>) pair.getFirst();
            ArrayList<Integer> second = (ArrayList<Integer>) pair.getSecond();
            while(br1.ready()){
            
            String line = br1.readLine();
            String[] parts = line.split(",");
            count++;
            double[] points = new double[parts.length - 1];
            for (int j = 0; j < parts.length - 1; j++)
            points[j] = Double.parseDouble(parts[j]);
            if(first.contains(count)){
                t1.add(new LabeledPoint(labels.getOrDefault(parts[parts.length - 1],0),
                        Vectors.dense(points)));
                te1.add(new LabeledPoint(labels.getOrDefault(parts[parts.length - 1],0),
                    Vectors.dense(points))); 
            }else if(second.contains(count)){
                te1.add(new LabeledPoint(labels.getOrDefault(parts[parts.length - 1],0),
                        Vectors.dense(points))); 
            }
        }
            JavaRDD<LabeledPoint> trainingData = jsc.parallelize(t1); 
            
            JavaRDD<LabeledPoint> testData = jsc.parallelize(te1);  
            
            int numClasses = 6;
            Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
            String impurity = "gini";
            //2 \n 4
            int maxDepth =3;
            int maxBins = 32; 

            DecisionTreeModel model = DecisionTree.trainClassifier(trainingData, numClasses,
                categoricalFeaturesInfo, impurity, maxDepth, maxBins); 
            
            JavaRDD<Tuple2<Double, Double>> predictionAndLabel = testData.map(
                new Function<LabeledPoint, Tuple2<Double, Double>>() {
                    @Override
                    public Tuple2<Double, Double> call(LabeledPoint point) throws Exception {

                        double prediction = model.predict(point.features());
                        if(prediction != point.label()){
                            error += pointweights.get(point);
                           
                        }
                        return new Tuple2<>(prediction, point.label());
                    }
                });
            
            
            double accuracy = predictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count() / (double) testData.count();        
            accurecies.add(accuracy);

            beta = error / (1 - error);
            double y = 1;
            double k = 5.0;
            //beta = (k/y) * Math.pow((error/y), k - 1) * Math.pow(e, Math.pow(-(error/y), k));
            //beta = 1 - Math.pow(e, Math.pow(-error/y, k));
            //beta = error/Math.sqrt(error);
            //beta = Math.pow(error, 2);
            errors.put(model, Math.log(1/beta));
            models.add(model);


            
            testData.map(new Function<LabeledPoint, Double>() {
                @Override
                public Double call(LabeledPoint point) throws Exception {

                    Map<Double, Double> votes = new HashMap<>();
                    
                    int c = 0;
                    for(DecisionTreeModel model1 : errors.keySet()){
                        double r = model1.predict(point.features());
                        if(votes.containsKey(r)){
                            votes.put(r,votes.get(r) + errors.getOrDefault(model1,0.0));
                        }else{
                            votes.put(r, errors.get(model1));
                        }
                        c++;
                    }
                    double key = votes.entrySet().stream().max((entry1, entry2) -> entry1.getValue() > entry2.getValue() ? 1 : -1).get().getKey();
                    if(key != point.label()){
                        hserror += pointweights.get(point);
                    }
                    return hserror;
                }
            }).collect();
           
            beta = hserror / (1 - hserror);
              y = 1.0;
              k = 5;
              //beta = (k/y) * Math.pow((hserror/y), k - 1) * Math.pow(e, Math.pow(-(hserror/y), k));
              //beta = 1 - Math.pow(e, Math.pow(-(hserror/y), k));
              //beta = hserror/(Math.sqrt(hserror));
              //beta = Math.pow(hserror, 2);
                          
              
             JavaRDD<Tuple2<Double, Double>> predictionAndLabel2 =  testData.map(new Function<LabeledPoint, Tuple2<Double, Double>>() {
                @Override
                public Tuple2<Double, Double> call(LabeledPoint point) throws Exception {
                    Map<Double, Double> votes = new HashMap<>();
                    int c = 0;
                     
                    for(DecisionTreeModel model1 : errors.keySet()){
                        double r = model1.predict(point.features());
                        if(votes.containsKey(r)){
                            votes.put(r,votes.get(r) + errors.getOrDefault(model1,0.0));
                        }else{
                            votes.put(r, errors.get(model1));
                        }
                        c++;
                    }
                    double key = votes.entrySet().stream().max((entry1, entry2) -> entry1.getValue() > entry2.getValue() ? 1 : -1).get().getKey();
                    if(key == point.label()){
                        
                        pointweights.put(point, pointweights.get(point) * beta);
                  
                    }
                    
                    return new Tuple2<>(key, point.label());
                }
            });
            
            accuracy = predictionAndLabel2.filter(pl -> pl._1().equals(pl._2())).count() / (double) testData.count();        
            accurecies.add(accuracy + 1);
        }

        //System.out.println("Trained Decision Tree model:\n" + model.toDebugString());
 

        
        for(Double a : accurecies)
            System.out.println(a);
        test(jsc,models);
        
        jsc.stop(); 
        }
        
    
    public static void test(JavaSparkContext jsc, ArrayList<DecisionTreeModel> models){
        List<LabeledPoint> t1 = new ArrayList<>();


            
            JavaRDD<LabeledPoint> testData = readData(jsc, testFile);
            
             JavaRDD<Tuple2<Double, Double>> predictionAndLabel2 =  testData.map(new Function<LabeledPoint, Tuple2<Double, Double>>() {
                @Override
                public Tuple2<Double, Double> call(LabeledPoint point) throws Exception {
                    Map<Double, Double> votes = new HashMap<>();
                    int c = 0;
                     
                    for(DecisionTreeModel model1 : errors.keySet()){
                        double r = model1.predict(point.features());
                        if(votes.containsKey(r)){
                            votes.put(r,votes.get(r) + errors.getOrDefault(model1,0.0));
                        }else{
                            votes.put(r, errors.get(model1));
                        }
                        c++;
                    }
                    double key = votes.entrySet().stream().max((entry1, entry2) -> entry1.getValue() > entry2.getValue() ? 1 : -1).get().getKey();

                    
                    return new Tuple2<>(key, point.label());
                }
            });
            
            double accuracy = predictionAndLabel2.filter(pl -> pl._1().equals(pl._2())).count() / (double) testData.count();        
            System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n" + accuracy + "\n\n\n\n\n\n\n");
            
            
            

    }
     
    
    public static JavaRDD<LabeledPoint> readData(JavaSparkContext jsc, String fileName){
        ArrayList<LabeledPoint> list = new ArrayList<>();
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(fileName));
            
            while(br.ready()){
                
                String[] parts = br.readLine().split(",");
                double[] points = new double[parts.length - 1];
                for(int i = 0; i < parts.length - 1; i++){
                    points[i] = Double.parseDouble(parts[i]);
                }
                list.add(new LabeledPoint(labels.getOrDefault(parts[parts.length - 1],0),
                        Vectors.dense(points)));
            }
            
        } catch (FileNotFoundException ex) {
            Logger.getLogger(MyMlLib.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(MyMlLib.class.getName()).log(Level.SEVERE, null, ex);
        }finally{
            try {
                br.close();
            } catch (IOException ex) {
                Logger.getLogger(MyMlLib.class.getName()).log(Level.SEVERE, null, ex);
            }
        }

        
        
        JavaRDD<LabeledPoint> result = jsc.parallelize(list);
        return result;
        
    }
    }
    

