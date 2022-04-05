/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.iotproject_m;

import static com.mycompany.iotproject_m.MyMlLib2.afterUsedMem;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 *
 * @author AAA
 */
public class OtherLearners {

    static long beforeUsedMem = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
    static long afterUsedMem;
    static long actualMemUsed;

    private static SparkConf sparkConf = new SparkConf().setAppName("DecisionTreeExample")
            .setMaster("local[2]").set("spark.executor.memory", "2g");
    private static JavaSparkContext sc = new JavaSparkContext(sparkConf);
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
    private static OneModel modelMaker = new OneModel();
    private static List<Long> times = new ArrayList<>();

    private static JavaRDD<LabeledPoint> t = modelMaker.loadData(trainFile, sc);
    private static JavaRDD<LabeledPoint> t1 = t.union(modelMaker.loadData(trainFile1, sc));
    private static JavaRDD<LabeledPoint> t2 = t1.union(modelMaker.loadData(trainFile2, sc));
    private static JavaRDD<LabeledPoint> t3 = t2.union(modelMaker.loadData(trainFile3, sc));
    private static JavaRDD<LabeledPoint> t4 = t3.union(modelMaker.loadData(trainFile4, sc));
    private static JavaRDD<LabeledPoint> t5 = t4.union(modelMaker.loadData(trainFile5, sc));
    private static JavaRDD<LabeledPoint> t6 = t5.union(modelMaker.loadData(trainFile6, sc));
    private static JavaRDD<LabeledPoint> t7 = t6.union(modelMaker.loadData(trainFile7, sc));
    private static JavaRDD<LabeledPoint> t8 = t7.union(modelMaker.loadData(trainFile8, sc));
    private static JavaRDD<LabeledPoint> t9 = t8.union(modelMaker.loadData(trainFile9, sc));

    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        Logger.getLogger("LBFGS").setLevel(Level.OFF);
        long startTime = System.currentTimeMillis() / 1000;

        //for test you should use path + "\\Data\\test9.csv"
       /* mysvm(t, testFile, sc, startTime);
        mysvm(t1, testFile1, sc, startTime);
        mysvm(t2, testFile2, sc, startTime);
        mysvm(t3, testFile3, sc, startTime);
        mysvm(t4, testFile4, sc, startTime);
        mysvm(t5, testFile5, sc, startTime);
        mysvm(t6, testFile6, sc, startTime);
        mysvm(t7, testFile7, sc, startTime);
        mysvm(t8, testFile8, sc, startTime);
        mysvm(t9, testFile9, sc, startTime);*/

        /* mynaivebayse(t, testFile, sc, startTime);
        mynaivebayse(t1, testFile1, sc, startTime);
        mynaivebayse(t2, testFile2, sc, startTime);
        mynaivebayse(t3, testFile3, sc, startTime);
        mynaivebayse(t4, testFile4, sc, startTime);
        mynaivebayse(t5, testFile5, sc, startTime);
        mynaivebayse(t6, testFile6, sc, startTime);
        mynaivebayse(t7, testFile7, sc, startTime);
        mynaivebayse(t8, testFile8, sc, startTime);
        mynaivebayse(t9, testFile9, sc, startTime);*/
   mynural(sc, startTime,0);
        mynural(sc, startTime,1);
        mynural(sc, startTime,2);
        mynural(sc, startTime,3);
        mynural(sc, startTime,4);
        mynural(sc, startTime,5);
        mynural(sc, startTime,6);
        mynural(sc, startTime,7);
        mynural(sc, startTime,8);
        mynural(sc, startTime,9);
        System.out.println(times);
         afterUsedMem = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        actualMemUsed = afterUsedMem - beforeUsedMem;
        System.out.println((actualMemUsed / 1024) / 1024);
    }

    public static void mysvm(JavaRDD<LabeledPoint> train, String t, JavaSparkContext sc, Long startTime) {
        SVMModel m = SVMWithSGD.train(JavaRDD.toRDD(train), 100);

        JavaRDD<LabeledPoint> test = modelMaker.loadData(t, sc);
        test.collect().forEach((x) -> {

            m.predict(x.features());

        });
        



        //time usage
        System.out.print(System.currentTimeMillis() / 1000 - startTime + ",");
    }

    public static void mynural(JavaSparkContext sc, Long startTime, int howmany) {
        try {
            //115
            //learn++
            //int[] layers = new int[]{10, 5, 30, 6};
            int[] layers = new int[]{10, 100, 6};
            SparkSession spark = SparkSession
                    .builder()
                    .appName("OtherLearners")
                    .master("local[*]").getOrCreate();

            BufferedReader bf = new BufferedReader(new FileReader(path + "\\Data\\train\\train99.csv"));
            String[] tmp = bf.readLine().split(",");
            String[] features = new String[tmp.length - 1];
            System.arraycopy(tmp, 0, features, 0, tmp.length - 1);

            bf.close();

            VectorAssembler assembler = new VectorAssembler();
            assembler.setInputCols(features);
            assembler.setOutputCol("features");

            Dataset<Row> dataFrame = spark.read().option("header", "true").format("csv").load(path + "\\Data\\train\\train99.csv");
            Dataset<Row> base = spark.read().option("header", "true").format("csv").load(path + "\\Data\\train\\train99.csv");
            //learn++
            //for (int i = 0; i < 1; i++) {
            for (int i = 0; i < howmany; i++) {
                dataFrame.union(base);
            }

            for (String c : dataFrame.columns()) {

                if (!c.equals("label")) {
                    dataFrame = dataFrame.withColumn(c, dataFrame.col(c).cast("double"));
                }
                
                
            }

            StringIndexerModel labelIndexer = new StringIndexer().setHandleInvalid("skip").setInputCol("label").setOutputCol("indexedLabel").fit(dataFrame);
            dataFrame = labelIndexer.transform(dataFrame);
            dataFrame = assembler.transform(dataFrame);

            MultilayerPerceptronClassifier trainer = new MultilayerPerceptronClassifier();
            trainer.setLayers(layers);

            trainer.setLabelCol("indexedLabel");
            trainer.setFeaturesCol("features");
            trainer.setSeed(1234L);
            trainer.setMaxIter(200);
            MultilayerPerceptronClassificationModel model = trainer.fit(dataFrame);
            times.add(System.currentTimeMillis() / 1000 - startTime);
        } catch (IOException ex) {
            java.util.logging.Logger.getLogger(OtherLearners.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
    }

    public static void mynaivebayse(JavaRDD<LabeledPoint> train, String t, JavaSparkContext sc, Long startTime) {

        NaiveBayesModel model = NaiveBayes.train(train.rdd());

        JavaRDD<LabeledPoint> test = modelMaker.loadData(t, sc);
        test.collect().forEach((x) -> {

            model.predict(x.features());

        });
        System.out.print(System.currentTimeMillis() / 1000 - startTime + ",");
    }
}
