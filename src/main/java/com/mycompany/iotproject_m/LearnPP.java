/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.iotproject_m;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.distribution.EnumeratedDistribution;
import org.apache.commons.math3.util.Pair;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

/**
 *
 * @author AAA
 */
public class LearnPP {
    
    
    
    public Pair selectTrain(Integer[] values, Double[] weights, int split){

        ArrayList<Integer> result1 = new ArrayList<>();
        ArrayList<Integer> result2 = new ArrayList<>();
        
        int count = 0;
        
        List<Pair<Integer, Double>> itemWeights = new ArrayList<>();
        Map<Integer, Double> index = new HashMap<>();
        for(int i = 0; i < weights.length; i++){
            itemWeights.add(new Pair(values[i], weights[i]));
            index.put(values[i], weights[i]);
        }
        
        while(count < split){
           
            int sample = new EnumeratedDistribution<>(itemWeights).sample();
            result1.add(sample);
            itemWeights.remove(new Pair(sample, index.get(sample)));
            count++;
        }
        

        count = 0;
        for(Pair p : itemWeights){
            
            result2.add((Integer) p.getFirst());
            count++;
        }
        


        return new Pair(result1, result2);
    }
    
}
