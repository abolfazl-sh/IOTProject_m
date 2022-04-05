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
import java.util.TreeMap;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.log4j.Logger;
import org.apache.log4j.Level;

/**
 *
 * @author AAA
 */
public class MyLearner {

    private static final double testnumber = 1000d;

    private static Path path = FileSystems.getDefault().getPath("").toAbsolutePath();
    private static String trainFile = path + "\\Data\\train\\train0.csv";
    private static String testFile = path + "\\Data\\test\\test";
    private static List<DecisionTreeModel> thmodel = new ArrayList<>();
    private static List<DecisionTreeModel> hmodel = new ArrayList<>();
    private static Map<List<DecisionTreeModel>, Double> Hmodel = new HashMap<>();

    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        SparkConf sparkConf = new SparkConf().setAppName("DecisionTreeExample")
                .setMaster("local[2]").set("spark.executor.memory", "2g");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        double result = 0;
        MyModel mymodel = new MyModel();
        int z = 0;
        for (int j = 0; j < 1; j++) {

            Object[] all = mymodel.makeModel(trainFile, jsc);
            DecisionTreeModel model = (DecisionTreeModel) all[0];
            if ((double) all[1] >= 166) {
                hmodel.add(model);
            }
        }
        System.out.println(mymodel.test(testFile + "0.csv", jsc, hmodel));
        thmodel = new ArrayList<>(hmodel);
        Hmodel.put(new ArrayList(hmodel), mymodel.test(trainFile, jsc, hmodel) / testnumber);
        for (int i = 0; i < 7; i++) {
            hmodel.clear();
            double acc = mymodel.test(trainFile, jsc, thmodel) / testnumber;
            for (int j = 0; j < 30; j++) {
                
                Object[] all = mymodel.makeModel(jsc, testFile + i + ".csv", trainFile, acc, thmodel);
                DecisionTreeModel model = (DecisionTreeModel) all[0];
                if ((double) all[1] >= 166) {
                    hmodel.add(model);
                }
                
            }
            System.out.println(mymodel.test(trainFile, jsc, hmodel));
            Hmodel.put(new ArrayList(hmodel), mymodel.test(trainFile, jsc, hmodel) / testnumber);
           
            // System.out.println(mymodel.Hmodeltest(trainFile, jsc, Hmodel));
        }

       
    }

}
