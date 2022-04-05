/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.iotproject_m;

import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;

/**
 *
 * @author AAA
 */
public class MyMlLib2 {
    static long beforeUsedMem=Runtime.getRuntime().totalMemory()-Runtime.getRuntime().freeMemory();
    static long afterUsedMem;
    static long actualMemUsed;
    private static Integer nubmerofClassifire = 10;
    private static final double trainNum = 999d;
    private static final double evalNum = 300d;
    private static final double testNum = 999d;
    private static final int totalTest = 1;

    private static Integer numClasses = 6;
    private static Integer numTrees = 2;//10
    private static String featureSubsetStrategy = "auto";
    private static String impurity = "entropy";
    private static Integer maxDepth = 20;//20
    private static Integer maxBins = 34;//34

    private static Path path = FileSystems.getDefault().getPath("").toAbsolutePath();
    private static String trainFile = path + "\\Data\\train0.csv";
    private static String trainFile1 = path + "\\Data\\train1.csv";
    private static String trainFile2 = path + "\\Data\\train2.csv";
    private static String trainFile3 = path + "\\Data\\train3.csv";
    private static String trainFile4 = path + "\\Data\\train4.csv";
    private static String trainFile5 = path + "\\Data\\train5.csv";
    private static String trainFile6 = path + "\\Data\\train6.csv";
    private static String trainFile7 = path + "\\Data\\train7.csv";
    private static String trainFile8 = path + "\\Data\\train8.csv";
    private static String trainFile9 = path + "\\Data\\train9.csv";
    private static String[] trainFiles = {trainFile, trainFile1, trainFile2, trainFile3, trainFile4
    ,trainFile5, trainFile6, trainFile7, trainFile8, trainFile9};

    private static String testFile = path + "\\Data\\test0.csv";
    private static String testFile1 = path + "\\Data\\test1.csv";
    private static String testFile2 = path + "\\Data\\test2.csv";
    private static String testFile3 = path + "\\Data\\test3.csv";
    private static String testFile4 = path + "\\Data\\test4.csv";
    private static String testFile5 = path + "\\Data\\test5.csv";
    private static String testFile6 = path + "\\Data\\test6.csv";
    private static String testFile7 = path + "\\Data\\test7.csv";
    private static String testFile8 = path + "\\Data\\test8.csv";
    private static String testFile9 = path + "\\Data\\test9.csv";
    private static String[] testFiles = {testFile, testFile1, testFile2, testFile3, testFile4
    ,testFile5, testFile6, testFile7, testFile8, testFile9};

    private static Integer[][] cMatrix = new Integer[6][6];
    private static OneModel modelMaker = new OneModel();
    private static Map<RandomForestModel, Double> models = new HashMap<>();
    private static Map<RandomForestModel, Double> weights = new HashMap<>();
    private static SparkConf sparkConf = new SparkConf().setAppName("DecisionTreeExample")
            .setMaster("local[2]").set("spark.executor.memory", "2g");
    private static JavaSparkContext sc = new JavaSparkContext(sparkConf);
    private static int[] right = new int[nubmerofClassifire];
    private static List<Double> accuraces = new ArrayList<>();
    private static int accuracy = 0;
    private static List<Double> modelAccuraces = new ArrayList<>();
    private static double modelaccuracy = 0;
    private static List<Double> antiaccuraces = new ArrayList<>();
    private static int antiaccuracy = 0;

    private static List<List<Double>> taccuraces = new ArrayList<>();
    private static List<List<Double>> tmodelAccuraces = new ArrayList<>();

    private static double first = 0;
    private static double second = 0;
    private static double thirth = 0;
    private static double forth = 0;
    private static double fifth = 0;
    private static double sixth = 0;
    private static double seventh = 0;
    private static double eighth = 0;
    private static double ninth = 0;
    private static double tenth = 0;
    private static Long time = System.currentTimeMillis() / 1000;
    /*private static Object[] model = modelMaker.makeModel(trainFile, sc);
    private static Object[] model1 = modelMaker.makeModel(trainFile1, sc);
    private static Object[] model2 = modelMaker.makeModel(trainFile2, sc);
    private static Object[] model3 = modelMaker.makeModel(trainFile3, sc);
    private static Object[] model4 = modelMaker.makeModel(trainFile4, sc);
     */
    private static JavaRDD<LabeledPoint> test;

    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        
        for (int j = 0; j < totalTest; j++) {
            for (int i = 0; i < nubmerofClassifire; i++) {
                test = modelMaker.loadData(testFiles[i], sc);
                Object[] m = modelMaker.makeModel(trainFiles[i], sc);
                RandomForestModel model = (RandomForestModel) m[0];
                Double weihgt = (Double) m[1];
                modleAccuracyCal(model, testFiles[i]);
                models.put(model, vibol(weihgt / evalNum, i));
                //models.put(model,learnPP(weihgt / evalNum));
                accuracyCal(testFiles[i]);
                System.out.print(System.currentTimeMillis() / 1000 - time + ",");
            }
            models.clear();
            taccuraces.add(new ArrayList<>(accuraces));
            tmodelAccuraces.add(new ArrayList<>(modelAccuraces));
            modelAccuraces.clear();
            accuraces.clear();
            antiaccuraces.clear();
            
        }

        System.out.println("Single Model : " + tmodelAccuraces);
        System.out.println("Right : " + taccuraces);
        System.out.println("Wrong : " + antiaccuraces + "\n\n\n\n\n\n\n");

        for (List<Double> x : tmodelAccuraces) {
            first += x.get(0);
            second += x.get(1);
            thirth += x.get(2);
            forth += x.get(3);
            fifth += x.get(4);
            sixth += x.get(5);
            seventh += x.get(6);
            eighth += x.get(7);
            ninth += x.get(8);
            tenth += x.get(9);
        }
        System.out.println(first / totalTest + "," + second / totalTest + ","
                + thirth / totalTest + "," + forth / totalTest + "," + fifth / totalTest + ","
                + sixth / totalTest + "," + seventh / totalTest + "," + eighth / totalTest + ","
                + ninth / totalTest + "," + tenth / totalTest);
        first = 0;
        second = 0;
        thirth = 0;
        forth = 0;
        fifth = 0;
        sixth = 0;
        seventh = 0;
        eighth = 0;
        ninth = 0;
        tenth = 0;
        for (List<Double> x : taccuraces) {
            first += x.get(0);
            second += x.get(1);
            thirth += x.get(2);
            forth += x.get(3);
            fifth += x.get(4);
            sixth += x.get(5);
            seventh += x.get(6);
            eighth += x.get(7);
            ninth += x.get(8);
            tenth += x.get(9);
        }

        System.out.println(first / totalTest + "," + second / totalTest + ","
                + thirth / totalTest + "," + forth / totalTest + "," + fifth / totalTest + ","
                + sixth / totalTest + "," + seventh / totalTest + "," + eighth / totalTest + ","
                + ninth / totalTest + "," + tenth / totalTest + "\n\n\n\n\n\n\n");
        for(int i = 0; i < 6; i++){
            for(int j = 0; j< 6; j++){
                System.out.print(cMatrix[i][j] + " ");
            }
            System.out.println();
        }
        
        afterUsedMem=Runtime.getRuntime().totalMemory()-Runtime.getRuntime().freeMemory();
        actualMemUsed=afterUsedMem-beforeUsedMem;
        System.out.println((actualMemUsed/1024)/1024);
    }

    public static void accuracyCal(String testFile) {

        for(int i = 0; i < 6; i++){
            for(int j = 0; j<6; j++){
                cMatrix[i][j] = 0;
            }
        }
        test = modelMaker.loadData(testFile, sc);
        test.collect().forEach(x -> {
            Map<Integer, Double> votes = new HashMap<>();
            models.keySet().forEach((m) -> {

                if (votes.containsKey((int) m.predict(x.features()))) {
                    votes.put((int) m.predict(x.features()), votes.get((int) m.predict(x.features())) + models.get(m));
                } else {
                    votes.put((int) m.predict(x.features()), models.get(m));
                }

            });

            int vote = 0;
            double bigvote = 0;

            for (Integer l : votes.keySet()) {

                if (votes.get(l) > bigvote) {
                    vote = l;
                    bigvote = votes.get(l);
                }

            }
            
            cMatrix[(int)x.label()][vote] = cMatrix[(int)x.label()][vote] + 1;

            if ((double) vote == x.label()) {
                accuracy++;
            } else {
                antiaccuracy++;
            }

        });

        accuraces.add((accuracy / testNum));
        antiaccuraces.add(antiaccuracy / testNum);
        accuracy = 0;
        antiaccuracy = 0;

    }

    public static void modleAccuracyCal(RandomForestModel model, String testFile) {

        Random r = new Random();
        Integer seed = r.nextInt();
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        JavaRDD<LabeledPoint> train;
        ArrayList<LabeledPoint> data = new ArrayList<>();

        test.collect().forEach((x) -> {

            if (model.predict(x.features()) == x.label()) {
                modelaccuracy++;
            }
            data.add(new LabeledPoint(model.predict(x.features()), x.features()));
        });
        train = sc.parallelize(data);
        //RandomForestModel model1 = RandomForest.trainClassifier(train, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);
        //models.put(model1, (modelaccuracy) / evalNum);
        modelAccuraces.add((modelaccuracy / testNum));
        
        modelaccuracy = 0;

    }

    public static RandomForestModel modeltester(String path) {
        RandomForestModel result = null;

        return result;
    }
    
    public static Double vibol(Double x, int number){
        Double result = 0.0;
       
        Double h = 1d;
        Double k = 3d;
        Double e = 2.71;
        Double a = 1.04;
        result = (k/h)*Math.pow((x/h),(k-1))*Math.pow(e,Math.pow(-(x/h),k));
        return result * Math.pow(a, number);
        
    }
    
    public static Double learnPP(Double x){
        Double result = 0.0;
        result = Math.log10((x * 300) * (1/(x/(1-x))));
        
        return result;
    }
    
    public static Double x(Double x){
        Double result = 0.0;
        
        return result;
    }
}
