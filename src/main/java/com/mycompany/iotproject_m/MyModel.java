/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.iotproject_m;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;

/**
 *
 * @author AAA
 */
public class MyModel implements Serializable {

    Map<String, Integer> labels = new HashMap<>();

    int numClasses = 6;
    Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
    String impurity = "gini";
    int maxDepth = 1;
    int maxBins = 6;
    Double right = 0d;
    int accurecy = 0;

    public MyModel() {

        labels.put("ack", 0);
        labels.put("benign_traffic", 1);
        labels.put("scaner", 2);
        labels.put("syn", 3);
        labels.put("udp", 4);
        labels.put("udpplain", 5);

    }

    public Object[] makeModel(String address, JavaSparkContext sc) {
        right = 0d;
        Random random = new Random();

        JavaRDD<LabeledPoint>[] data = loadData(address, sc).randomSplit(new double[]{0.5, 0.5}, random.nextInt());

        JavaRDD<LabeledPoint> train = data[0];
        JavaRDD<LabeledPoint> test = sc.union(data[0], data[1]);

        Random r = new Random();
        Integer seed = r.nextInt();
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        DecisionTreeModel model = DecisionTree.trainClassifier(train, numClasses,
                categoricalFeaturesInfo, impurity, maxDepth, maxBins);

        test.collect().forEach(x -> {

            if (model.predict(x.features()) == x.label()) {
                right++;
            }

        });

        Object[] result = new Object[]{model, right};

        return result;
    }

    public Object[] makeModel(JavaSparkContext sc, String f, String s, double trust, List<DecisionTreeModel> hmodel) {
        right = 0d;
        Random r = new Random();
        JavaRDD<LabeledPoint> first = labelTest(f, sc, hmodel).randomSplit(new double[]{0.0, 1.0}, r.nextInt())[0];
        JavaRDD<LabeledPoint> second = loadData(s, sc).randomSplit(new double[]{1, 0.0}, r.nextInt())[0];
        JavaRDD<LabeledPoint>[] all = (sc.union(first, second)).randomSplit(new double[]{0.5, 0.5}, r.nextInt());
        JavaRDD<LabeledPoint> train = all[0];
        JavaRDD<LabeledPoint> test = sc.union(all[0], all[1]);
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        DecisionTreeModel model = DecisionTree.trainClassifier(train, numClasses,
                categoricalFeaturesInfo, impurity, maxDepth, maxBins);

        test.collect().forEach(x -> {

            if (model.predict(x.features()) == x.label()) {
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
                features[i] = Double.parseDouble(line[i]);
            }

            return new LabeledPoint(labels.get(line[line.length - 1]), Vectors.dense(features));

        });

        return data;
    }

    public int test(String address, JavaSparkContext sc, List<DecisionTreeModel> hmodel) {
        accurecy = 0;
        JavaRDD<LabeledPoint> testdata = loadData(address, sc);

        testdata.collect().forEach((x) -> {

            // System.out.println(x.features() + " " + x.label());
            // System.out.println(votes);
            int vote = wVoting(hmodel, x);
            if (vote == (int) x.label()) {

                accurecy++;
            }

        });

        return accurecy;
    }

    public JavaRDD<LabeledPoint> labelTest(String fileAddress, JavaSparkContext sc, List<DecisionTreeModel> hmodel) {
        JavaRDD<LabeledPoint> test = loadData(fileAddress, sc);

        JavaRDD<LabeledPoint> result = test.map((x) -> {

            return new LabeledPoint(wVoting(hmodel, x), x.features());
        });

        return result;

    }

    public int wVoting(List<DecisionTreeModel> hmodel, LabeledPoint x) {
        Map<Integer, Integer> votes = new HashMap<>();
        for (DecisionTreeModel m : hmodel) {

            if (votes.containsKey((int) m.predict(x.features()))) {

                votes.put((int) m.predict(x.features()), votes.get((int) m.predict(x.features())) + 1);
            } else {

                votes.put((int) m.predict(x.features()), 1);
            }

        }

        double bigvote = 0;
        int vote = 0;
        for (Integer l : votes.keySet()) {

            if (votes.get(l) > bigvote) {
                vote = l;
                bigvote = votes.get(l);
            }

        }
        return vote;
    }

    public int Hmodeltest(String address, JavaSparkContext sc, Map<List<DecisionTreeModel>, Double> Hmodel) {
        accurecy = 0;

        JavaRDD<LabeledPoint> data = loadData(address, sc);
        data.collect().forEach((x) -> {
        HashMap<Integer, Double> votes = new HashMap<>();
            for (List<DecisionTreeModel> hmodel : Hmodel.keySet()) {
                int label = wVoting(hmodel, x);
                if (votes.containsKey(label)) {
                    votes.put(label, votes.get(label) + Hmodel.get(hmodel));
                } else {
                    votes.put(label, Hmodel.get(hmodel));
                }
            }
            double max = 0d;
            Integer label = 0;
            for (Integer l : votes.keySet()) {
                if (votes.get(l) > max) {
                    max = votes.get(l);
                    label = l;
                }
            }

            if (label == (int) x.label()) {
                accurecy++;
            }

        });
        return accurecy;
    }

}
