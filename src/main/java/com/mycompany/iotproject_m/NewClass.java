/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.iotproject_m;

import breeze.math.EnumeratedCoordinateField;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.math3.distribution.EnumeratedDistribution;
import org.apache.commons.math3.util.Pair;

/**
 *
 * @author AAA
 */
public class NewClass {
    public static void main(String[] args) throws FileNotFoundException, IOException{
        Path path = FileSystems.getDefault().getPath("").toAbsolutePath();
        String dataFile = path + "\\Data\\train.csv";
        editfile(path);
        BufferedReader br = new BufferedReader(new FileReader(dataFile));
        String[] a1 = br.readLine().split(",");
        System.out.println(a1.length);//116
        while(br.ready()){
            String val = br.readLine().split(",")[0];
           // System.out.println(val);
            Double a = Double.parseDouble(val);
        }
        
        String[] a = new String[] {"hi", "bye", "good"};
        String[] c = new String[a.length - 1];
        System.arraycopy(a, 0, c, 0, a.length - 1);
        for(String b : c)
            System.out.println(b);
        
        List<Pair<Double, Double>> itemWeights = new ArrayList<>();
        for(Double i = 0.0; i < 10; i++){
            itemWeights.add(new Pair(i * 100, i));
            
        }
        Double selectedItem = new EnumeratedDistribution<>(itemWeights).sample();
        System.out.println(selectedItem);
        
        System.out.println((samples(6, itemWeights)));
        
        LearnPP l = new LearnPP();
        
        //Integer[] values = new Integer[]{1,2,3,4,5};
        //Double[] weights = new Double[]{3.0,4.0,1.0,2.0,0.1};
        Integer[] values = new Integer[1000];
        Double[] weights = new Double[1000];
        for(int i = 0 ; i < 1000; i++){
            values[i] = i;
            weights[i] = i / 10.0;
        }
        Pair a11 = l.selectTrain(values, weights, 500);
        Integer[] b = (Integer[]) a11.getFirst();
        Integer[] c1 = (Integer[]) a11.getSecond();
        System.out.println(Arrays.toString(b) + b.length);
        System.out.println(Arrays.toString(c1) + c1.length);
        
        TreeMap<Integer, Double> treemap = new TreeMap<>();
        treemap.put(9, 10.0);
        treemap.put(12, 10.0);
        treemap.put(1, 10.0);
        System.out.println(treemap);
        
    }
    
    public static void editfile(Path path){
        
        try {
            BufferedReader br = new BufferedReader(new FileReader(path + "\\Data\\train.csv"));
            BufferedWriter bw = new BufferedWriter(new FileWriter(path + "\\Data\\train1.csv"));
            int count = 0;
            while(br.ready()){
                String line = br.readLine();
                bw.append(count + "," + line + "\n");
                count++;
            }
            bw.flush();
            bw.close();
            br.close();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(NewClass.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(NewClass.class.getName()).log(Level.SEVERE, null, ex);
        }
        
    }
    
    public static ArrayList<Double> samples(int numbers, List<Pair<Double, Double>> itemWeights){
        ArrayList<Double> result = new ArrayList<>();
        int count = 0;
        while(count < numbers){
            Double selectedItem = new EnumeratedDistribution<>(itemWeights).sample();
            if(!result.contains(selectedItem)){
                result.add(selectedItem);
                count++;
            }
        }
        
        return result;
    }
    
    
}
