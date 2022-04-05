/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.iotproject_m;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;

/**
 *
 * @author AAA
 */
public class OneModel implements Serializable{

    Map<String, Integer> labels = new HashMap<>();

    Integer numClasses = 6;
    Integer numTrees = 2;//10
    String featureSubsetStrategy = "auto";
    String impurity = "entropy";
    Integer maxDepth = 20;//20
    Integer maxBins = 34;//34
    Double right = 0d;

    public OneModel() {
        //baiot
        labels.put("ack", 0);
        labels.put("benign_traffic", 1);
        labels.put("scaner", 2);
        labels.put("syn", 3);
        labels.put("udp", 4);
        labels.put("udpplain", 5);
        
        
       /* labels.put("DDoS", 0);
        labels.put("DoS", 1);
        labels.put("Normal", 2);
        labels.put("Reconnaissance", 3);
        labels.put("Theft", 4);*/
        
    }

    public Object[] makeModel(String address, JavaSparkContext sc) {
        right = 0d;
        JavaRDD<LabeledPoint>[] data = loadData(address, sc).randomSplit(new double[]{0.7, 0.3}, 0);
        
        JavaRDD<LabeledPoint> train = data[0];
        JavaRDD<LabeledPoint> test = data[1];
        
        Random r = new Random();
        Integer seed = r.nextInt();
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        RandomForestModel model = RandomForest.trainClassifier(train, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);
        
        test.collect().forEach(x -> {
            
            if(model.predict(x.features()) == x.label()){
                right++;
            }
        
        });
        
        Object[] result = new Object[]{model, right};
        
        return result;
    }
    
    

    public JavaRDD<LabeledPoint> loadData(String fileAddress, JavaSparkContext sc) {

        JavaRDD<String> basefile = sc.textFile(fileAddress);
        JavaRDD<LabeledPoint> data = basefile.map(l -> {
            String[] line = l.split(",");

            double[] features = new double[line.length - 1];
            for (int i = 0; i < line.length - 1; i++) {
                if(i == 0){
                    features[i] = 0;
                }else{
                features[i] = Math.abs(Double.parseDouble(line[i]));
                }
            }

            return new LabeledPoint(labels.get(line[line.length - 1]), Vectors.dense(features));
            //return new LabeledPoint(1, Vectors.dense(features));

        });

        return data;
    }

}
